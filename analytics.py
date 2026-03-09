"""
analytics.py
ポートフォリオのリスク、リターン、ファクターエクスポージャー、およびリスク寄与度を計算するコア分析エンジン。
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
        日次リターン系列から各種リスク指標、ダウンサイドリスク、分離リスク、およびリスク寄与度を計算する。
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
            penalty = 1.0 + (hhi ** 0.8) * 0.5 
            
        # 2. リスク（ボラティリティ）とリスク寄与度の計算
        ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)
        raw_sigma = returns.std() * ann_factor
        lw_sigma = raw_sigma
        
        # 【新規追加】銘柄ごとのリスク寄与度（Percentage Risk Contribution）
        risk_contribution = {}
        
        if weights_dict:
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
                        
                        # 共分散行列の計算 (Ledoit-Wolf または 通常の共分散)
                        if HAS_SKLEARN:
                            lw = LedoitWolf().fit(comp_returns)
                            cov_matrix = lw.covariance_
                        else:
                            cov_matrix = comp_returns.cov().values
                            
                        # ポートフォリオの分散
                        port_var = np.dot(new_weights.T, np.dot(cov_matrix, new_weights))
                        lw_sigma = np.sqrt(port_var) * ann_factor
                        
                        # 💡【重要追加】周辺リスク寄与度 (Marginal Contribution to Risk) の計算
                        # 各銘柄がポートフォリオ全体のリスク（分散）に何％寄与しているかを算出
                        mcr = np.dot(cov_matrix, new_weights) / np.sqrt(port_var)
                        component_risk = new_weights * mcr
                        percentage_risk = component_risk / np.sqrt(port_var)
                        
                        risk_contribution = dict(zip(valid_tickers, percentage_risk))
            except Exception as e:
                print(f"Risk Contribution/LedoitWolf Error: {e}")
        
        sigma = lw_sigma * penalty 
        
        # リターン計算 (ボラティリティ・ドラッグを考慮)
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
        
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * ann_factor * penalty
        sortino = (mu - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        calmar = mu / abs(max_dd) if max_dd != 0 else 0
        
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else np.inf
        
        var_95 = np.percentile(returns, 5) * penalty 
        cvar_95 = returns[returns <= var_95].mean() * penalty
        
        kelly = mu / (sigma ** 2) if sigma > 0 else 0
        info_ratio = np.nan
        
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
            "portfolio_beta": portfolio_beta,
            "risk_contribution": risk_contribution # 💡追加: 各銘柄のリスク寄与度(%)
        }

# =========================================================
# 🧪 ファクター分析 (Fama-French)
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
        """
        Fama-French 3ファクターモデルを用いて、超過収益率による厳密な重回帰分析を行う。
        """
        if target_series.empty: return None
        target_monthly = target_series.resample('ME').last().pct_change().dropna()
        if len(target_monthly) < 6: return None
        
        start_date = target_monthly.index[0].strftime('%Y-%m-%d')
        config = MarketConfig.get_config(region)
        ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
        
        if ff_data.empty: return None
        
        target_monthly.index = target_monthly.index.to_period('M')
        ff_data.index = ff_data.index.to_period('M')
        combined = pd.concat([target_monthly.rename("Target"), ff_data], axis=1).dropna()
        
        if len(combined) < 10: return None
        
        try:
            mkt = [c for c in combined.columns if 'Mkt' in c or 'MKT' in c][0]
            smb = [c for c in combined.columns if 'SMB' in c][0]
            hml = [c for c in combined.columns if 'HML' in c][0]
            rf  = [c for c in combined.columns if 'RF' in c][0]

            # 💡【超過収益率の算出】ポートフォリオリターンから無リスク金利を控除
            y = combined["Target"] - combined[rf]
            
            X = combined[[mkt, smb, hml]]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            
            monthly_alpha = model.params.get("const", 0.0)
            annualized_alpha = monthly_alpha * 12
            
            # 💡【重要追加】p値を取得し、回帰係数の信頼性をAIに渡せるようにする
            pvalues = model.pvalues
            
            return {
                "beta_market": model.params.get(mkt, 1.0),
                "beta_size": model.params.get(smb, 0.0),
                "beta_value": model.params.get(hml, 0.0),
                "alpha": annualized_alpha,
                "r_squared": model.rsquared,
                "p_value_market": pvalues.get(mkt, 1.0),
                "p_value_size": pvalues.get(smb, 1.0),
                "p_value_value": pvalues.get(hml, 1.0),
                "region": region
            }
        except Exception as e:
            print(f"Factor Analyzer Error: {e}")
            return None

    @staticmethod
    def get_factor_correlation(region="US", periods=60):
        """
        ファクター同士（市場、サイズ、バリュー）の相関行列を計算する。
        """
        try:
            config = MarketConfig.get_config(region)
            ff_data = DataFetcher.fetch_fama_french_factors(dataset_name=config["ff_dataset"])
            if ff_data.empty: return None
            
            ff_recent = ff_data.tail(periods)
            
            mkt = [c for c in ff_recent.columns if 'Mkt' in c or 'MKT' in c][0]
            smb = [c for c in ff_recent.columns if 'SMB' in c][0]
            hml = [c for c in ff_recent.columns if 'HML' in c][0]
            
            factors_only = ff_recent[[mkt, smb, hml]]
            return factors_only.corr().to_dict()
        except Exception as e:
            print(f"Factor Correlation Error: {e}")
            return None


# =========================================================
# 🤖 AIアドバイザープロンプト生成 (Step 3: AIの診断能力)
# =========================================================
class AIPromptBuilder:
    @staticmethod
    def generate_quant_prompt(stats_data, factor_data, target_name="現在のポートフォリオ"):
        """
        計算結果（数字）を元に、生成AIへ渡す「クオンツマネージャーの小言」作成用プロンプトを自動生成する。
        """
        # リスク寄与度の一番高い銘柄（真の支配者）を特定
        risk_contributions = stats_data.get("risk_contribution", {})
        top_risk_asset = "特定不能"
        top_risk_value = 0.0
        
        if risk_contributions:
            top_risk_asset = max(risk_contributions, key=risk_contributions.get)
            top_risk_value = risk_contributions[top_risk_asset] * 100

        # R2とアルファの取得（ファクター分析が成功している場合）
        r_squared = factor_data.get("r_squared", 0) * 100 if factor_data else 0.0
        alpha = factor_data.get("alpha", 0) * 100 if factor_data else 0.0

        # AIの脳内に叩き込む「秘密の指示書」
        prompt = f"""
あなたはウォール街で長年活躍する、非常に優秀だが少し辛口なクオンツ・ポートフォリオマネージャーです。
以下の計算結果データに基づいて、個人投資家向けに「プロの小言（診断レポート）」を作成してください。

【ポートフォリオの客観的データ】
- 分析対象: {target_name}
- リスクの支配者（最大リスク寄与銘柄）: {top_risk_asset} ({top_risk_value:.1f}%)
- 市場との連動性 (R2): {r_squared:.1f}% (※この数値が高いほど、ただのインデックスファンドと同じ動きをしています)
- 年率アルファ (超過収益): {alpha:.2f}%
- ボラティリティ: {stats_data.get("volatility", 0) * 100:.1f}%
- 最大ドローダウン: {stats_data.get("max_dd", 0) * 100:.1f}%

【指示】
1. 専門的なクオンツの視点から、厳しいが的確で愛のあるアドバイスをしてください。
2. もし「リスクの支配者」が50%を超えている場合、「分散投資になっているつもりか？実質的に{top_risk_asset}と心中しているだけだぞ」と厳しく指摘してください。
3. もしR2が90%を超えていて、かつアルファがマイナスまたはゼロ付近の場合は、「高い手数料（または手間）を払ってインデックス以下の成果を出す、典型的な『隠れインデックスファンド』だ」と警告してください。
4. 最後に、改善のための具体的なネクストアクションを1つ提示してください。
5. 出力はMarkdown形式で、見出しを使って読みやすくしてください。ですます調で構いませんが、プロとしての威厳を保ってください。
"""
        return prompt
