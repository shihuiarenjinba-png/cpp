"""
analytics.py
ポートフォリオのリスク、リターン、およびファクターエクスポージャーを計算するコア分析エンジン。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import warnings

# 分割した他のモジュールからのインポート
from config import MarketConfig, DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from data_engine import DataFetcher

# 📌 収縮推定（Shrinkage）による共分散計算用
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =========================================================
# 📊 高度な統計指標計算クラス
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, weights_dict=None, region="US"):
        """
        日次リターン系列から各種リスク指標、ダウンサイドリスク、分離リスクを計算する。
        """
        if returns.empty: return {}
        
        # 正規化・リスケール
        weights_dict = DataFetcher.normalize_weights(weights_dict)
        
        # 1. 集中投資ペナルティ(HHI)
        hhi = 0.0
        penalty = 1.0
        enc = 0.0 
        
        if weights_dict is not None and sum(weights_dict.values()) > 0:
            weights = np.array(list(weights_dict.values())) / 100.0
            hhi = np.sum(weights**2)
            enc = 1.0 / hhi if hhi > 0 else 1.0
            # 集中度が高いほどペナルティ係数が1.0から増加する
            penalty = 1.0 + (hhi ** 0.8) * 0.5 
            
        # 2. リスク（ボラティリティ）の計算
        # ⚠️ 【重要修正】日次リターンの年率換算に sqrt(252) を使用してリスクの過小評価を防止
        ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)
        raw_sigma = returns.std() * ann_factor
        lw_sigma = raw_sigma
        
        # 📌 収縮推定（Ledoit-Wolf）による堅牢なボラティリティ算出
        if HAS_SKLEARN and weights_dict:
            try:
                tickers = list(weights_dict.keys())
                raw_comp = DataFetcher.fetch_market_data(tickers)
                if not raw_comp.empty:
                    if isinstance(raw_comp, pd.Series):
                        raw_comp = raw_comp.to_frame(name=tickers[0])
                    comp_returns = raw_comp.pct_change().dropna()
                    
                    valid_tickers = [t for t in tickers if t in comp_returns.columns]
                    if len(valid_tickers) > 0:
                        comp_returns = comp_returns[valid_tickers]
                        new_weights = np.array([weights_dict[t] for t in valid_tickers])
                        new_weights = new_weights / np.sum(new_weights)
                        
                        # 外れ値やノイズに強い共分散行列を生成
                        lw = LedoitWolf().fit(comp_returns)
                        cov_matrix = lw.covariance_
                        port_var = np.dot(new_weights.T, np.dot(cov_matrix, new_weights))
                        lw_sigma = np.sqrt(port_var) * ann_factor
            except Exception as e:
                print(f"LedoitWolf Error: {e}")
        
        # 最終的なシグマは収縮推定値に集中投資ペナルティを加味したもの
        sigma = lw_sigma * penalty 
        
        # リターン計算 (ボラティリティ・ドラッグを考慮した幾何平均ベースの推定)
        arithmetic_mu = returns.mean() * TRADING_DAYS_PER_YEAR
        mu = arithmetic_mu - 0.5 * (sigma ** 2) 
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        ulcer_sq = (drawdown ** 2).mean()
        ulcer_index = np.sqrt(ulcer_sq)
        
        risk_free_rate = DEFAULT_RISK_FREE_RATE 
        sharpe = (mu - risk_free_rate) / sigma if sigma > 0 else 0
        
        # ダウンサイドリスク (下落時のみの標準偏差)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * ann_factor * penalty
        sortino = (mu - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        calmar = mu / abs(max_dd) if max_dd != 0 else 0
        
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else np.inf
        
        # CVaR (Expected Shortfall): 最悪5%の日の平均損失
        var_95 = np.percentile(returns, 5) * penalty 
        cvar_95 = returns[returns <= var_95].mean() * penalty
        
        kelly = mu / (sigma ** 2) if sigma > 0 else 0
        
        info_ratio = np.nan
        
        # 📌 固有リスク（Idiosyncratic Risk）とシステマティックリスクの分離
        systematic_risk = 0.0
        idiosyncratic_risk = 0.0
        portfolio_beta = 1.0
        
        try:
            if benchmark_returns is None:
                config = MarketConfig.get_config(region)
                bm_ticker = config["benchmark_ticker"]
                bm_prices = DataFetcher.fetch_market_data([bm_ticker])
                if not bm_prices.empty:
                    if isinstance(bm_prices, pd.Series):
                        bm_prices = bm_prices.to_frame(name=bm_ticker)
                    benchmark_returns = bm_prices.pct_change().iloc[:, 0].dropna()
                    
            if benchmark_returns is not None:
                if returns.index.tz is not None: returns.index = returns.index.tz_localize(None)
                if benchmark_returns.index.tz is not None: benchmark_returns.index = benchmark_returns.index.tz_localize(None)
                returns.index = returns.index.normalize()
                benchmark_returns.index = benchmark_returns.index.normalize()
                
                aligned = pd.concat([returns, benchmark_returns.rename("BM")], axis=1).dropna()
                if len(aligned) > 30:
                    cov_bm = np.cov(aligned.iloc[:, 0], aligned["BM"])
                    if cov_bm[1, 1] > 0:
                        portfolio_beta = cov_bm[0, 1] / cov_bm[1, 1]
                        sys_var = (portfolio_beta ** 2) * cov_bm[1, 1]
                        tot_var = cov_bm[0, 0]
                        idio_var = max(0, tot_var - sys_var)
                        systematic_risk = np.sqrt(sys_var) * ann_factor
                        idiosyncratic_risk = np.sqrt(idio_var) * ann_factor
                        
                        # Info Ratio
                        active_ret = aligned.iloc[:, 0] - aligned["BM"]
                        track_err = active_ret.std() * ann_factor
                        if track_err > 0:
                            info_ratio = (active_ret.mean() * TRADING_DAYS_PER_YEAR) / track_err
        except Exception as e:
            print(f"Risk Separation Error: {e}")

        clean_returns = returns.dropna()
        return {
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "omega": omega, "cvar_95": cvar_95, "ulcer_index": ulcer_index,
            "kelly_criterion": kelly, "max_dd": max_dd, "info_ratio": info_ratio,
            "skewness": skew(clean_returns) if len(clean_returns) > 0 else 0, 
            "kurtosis": kurtosis(clean_returns) if len(clean_returns) > 0 else 0,
            "hhi_index": hhi, "effective_n": enc, "risk_penalty_ratio": penalty,
            "volatility": sigma,  
            "raw_volatility": raw_sigma,
            "systematic_risk": systematic_risk,
            "idiosyncratic_risk": idiosyncratic_risk,
            "portfolio_beta": portfolio_beta
        }

# =========================================================
# 🧪 ファクター分析 (Fama-French)
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
        """
        Fama-French 3ファクターモデルを用いて、ポートフォリオのリターンの源泉を重回帰分析する。
        """
        if target_series.empty: return None
        # 月次リターンへ変換 ('ME' は Month End の意)
        target_monthly = target_series.resample('ME').last().pct_change().dropna()
        if len(target_monthly) < 6: return None
        
        start_date = target_monthly.index[0].strftime('%Y-%m-%d')
        config = MarketConfig.get_config(region)
        ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
        
        if ff_data.empty: return None
        
        # 期間（月）でインデックスを結合
        target_monthly.index = target_monthly.index.to_period('M')
        ff_data.index = ff_data.index.to_period('M')
        combined = pd.concat([target_monthly.rename("Target"), ff_data], axis=1).dropna()
        
        if len(combined) < 10: return None
        
        try:
            mkt = [c for c in combined.columns if 'Mkt' in c or 'MKT' in c][0]
            smb = [c for c in combined.columns if 'SMB' in c][0]
            hml = [c for c in combined.columns if 'HML' in c][0]
            rf  = [c for c in combined.columns if 'RF' in c][0]

            # 超過リターン = ポートフォリオリターン - 無リスク利子率
            y = combined["Target"] - combined[rf]
            X = combined[[mkt, smb, hml]]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            return {
                "beta_market": model.params.get(mkt, 1.0),
                "beta_size": model.params.get(smb, 0.0),
                "beta_value": model.params.get(hml, 0.0),
                "alpha": model.params.get("const", 0.0),
                "r_squared": model.rsquared,
                "region": region
            }
        except Exception as e:
            print(f"Factor Analyzer Error: {e}")
            return None
