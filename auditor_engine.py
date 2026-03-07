import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import t, skew, kurtosis
import scipy.signal as signal
import pandas_datareader.data as web
from datetime import datetime, timedelta

# GARCHモデル用 (ライブラリがない場合はEWMAで代用するロジックが含まれています)
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# =========================================================
# 🌍 市場設定管理クラス
# =========================================================
class MarketConfig:
    REGIONS = {
        "US": {
            "name": "United States",
            "ff_dataset": "F-F_Research_Data_Factors",
            "benchmark_ticker": "^GSPC",   # S&P 500
            "risk_free_ticker": "^TNX",    # 米国10年国債利回り
            "vix_ticker": "^VIX"           # 米国VIX（恐怖指数）
        },
        "Japan": {
            "name": "Japan",
            "ff_dataset": "Japan_3_Factors",
            "benchmark_ticker": "^N225",   # 日経225
            "risk_free_ticker": "JPY=X",   # ドル円相場 (日本のマクロ指標として強力)
            "vix_ticker": "^JNIV"          # 日経VI (日本版の恐怖指数)
        }
    }

    @staticmethod
    def get_config(region="US"):
        return MarketConfig.REGIONS.get(region, MarketConfig.REGIONS["US"])

# =========================================================
# 📊 高度な統計指標計算クラス (STEP 1: 集中投資ペナルティ実装)
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, weights_dict=None):
        if returns.empty: return {}
        
        # 1. 集中投資ペナルティ(HHI)の計算
        hhi = 0.0
        penalty = 1.0
        if weights_dict is not None:
            weights = np.array(list(weights_dict.values())) / 100.0
            hhi = np.sum(weights**2)
            # HHIが1に近い(集中投資)ほどペナルティ増加（最大1.5倍のリスク評価）
            penalty = 1.0 + (hhi * 0.5) 
            
        mu = returns.mean() * 12
        # 分散不足の場合、ボラティリティを厳格に（高く）見積もる
        sigma = returns.std() * np.sqrt(12) * penalty 
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        sharpe = (mu - 0.02) / sigma if sigma > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(12) * penalty
        sortino = (mu - 0.02) / downside_dev if downside_dev > 0 else 0
        
        calmar = mu / abs(max_dd) if max_dd != 0 else 0
        
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else np.inf
        
        var_95 = np.percentile(returns, 5) * penalty # 下落リスクも厳しく
        cvar_95 = returns[returns <= var_95].mean() * penalty
        
        ulcer_sq = (drawdown ** 2).mean()
        ulcer_index = np.sqrt(ulcer_sq)
        
        kelly = mu / (sigma ** 2) if sigma > 0 else 0
        
        info_ratio = np.nan
        if benchmark_returns is not None:
            common = returns.index.intersection(benchmark_returns.index)
            if len(common) > 12:
                active_ret = returns.loc[common] - benchmark_returns.loc[common]
                track_err = active_ret.std() * np.sqrt(12)
                if track_err > 0:
                    info_ratio = (active_ret.mean() * 12) / track_err

        return {
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "omega": omega, "cvar_95": cvar_95, "ulcer_index": ulcer_index,
            "kelly_criterion": kelly, "max_dd": max_dd, "info_ratio": info_ratio,
            "skewness": skew(returns), "kurtosis": kurtosis(returns),
            "hhi_index": hhi, "risk_penalty_ratio": penalty
        }

# =========================================================
# 🛠️ データ取得 & ポートフォリオ合成クラス
# =========================================================
class DataFetcher:
    @staticmethod
    def fetch_market_data(tickers, start_date="2000-01-01"):
        try:
            if isinstance(tickers, str):
                tickers = [tickers]
                
            data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
            
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns: data = data['Close']
                elif 'Adj Close' in data.columns: data = data['Adj Close']
            
            if data.empty: return pd.DataFrame()
            return data.ffill().dropna(how='all')
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def validate_tickers(input_dict):
        valid_data = {}
        invalid_tickers = []
        for ticker, weight in input_dict.items():
            try:
                tick = yf.Ticker(ticker)
                hist = tick.history(period="5d")
                if not hist.empty: valid_data[ticker] = weight
                else: invalid_tickers.append(ticker)
            except: invalid_tickers.append(ticker)
        return valid_data, invalid_tickers

    @staticmethod
    def create_synthetic_portfolio(ticker_weights, region="US"):
        tickers = list(ticker_weights.keys())
        if not tickers: return None
        
        raw_prices = DataFetcher.fetch_market_data(tickers)
        if raw_prices.empty: return None
        if isinstance(raw_prices, pd.Series):
            raw_prices = raw_prices.to_frame(name=tickers[0])
            
        returns = raw_prices.pct_change()
        
        config = MarketConfig.get_config(region)
        bm_ticker = config["benchmark_ticker"]
        bm_prices = DataFetcher.fetch_market_data([bm_ticker])
        
        if bm_prices.empty:
            returns = returns.dropna()
            weighted_returns = pd.Series(0, index=returns.index)
            for ticker, weight in ticker_weights.items():
                if ticker in returns.columns:
                    weighted_returns += returns[ticker] * (weight / 100.0)
            return (1 + weighted_returns).cumprod() * 100

        if isinstance(bm_prices, pd.Series):
            bm_prices = bm_prices.to_frame(name=bm_ticker)
            
        bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")
        aligned_data = pd.concat([returns, bm_returns], axis=1)
        
        for ticker in tickers:
            if ticker not in aligned_data.columns: continue
            valid_mask = aligned_data[ticker].notna() & aligned_data["Benchmark"].notna()
            if valid_mask.sum() > 30:
                cov = np.cov(aligned_data.loc[valid_mask, ticker], aligned_data.loc[valid_mask, "Benchmark"])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            else:
                beta = 1.0
                
            missing_mask = aligned_data[ticker].isna() & aligned_data["Benchmark"].notna()
            aligned_data.loc[missing_mask, ticker] = aligned_data.loc[missing_mask, "Benchmark"] * beta
            
        aligned_data = aligned_data.dropna()
        weighted_returns = pd.Series(0, index=aligned_data.index)
        for ticker, weight in ticker_weights.items():
            if ticker in aligned_data.columns:
                weighted_returns += aligned_data[ticker] * (weight / 100.0)
                
        synthetic_price = (1 + weighted_returns).cumprod() * 100
        return synthetic_price

    @staticmethod
    def fetch_fama_french_factors(start_date, end_date=None, dataset_name="F-F_Research_Data_Factors"):
        try:
            ff_dict = web.DataReader(dataset_name, 'famafrench', start=start_date, end=end_date)
            if not ff_dict or 0 not in ff_dict: return pd.DataFrame()
            ff_data = ff_dict[0] / 100.0
            ff_data.index = ff_data.index.to_timestamp(freq='M')
            if ff_data.index.tz is not None: ff_data.index = ff_data.index.tz_localize(None)
            ff_data.columns = [c.strip() for c in ff_data.columns]
            return ff_data
        except Exception:
            return pd.DataFrame()

# =========================================================
# 🕰️ タイムマシン機能 (STEP 3: 真の実データバックテスト)
# =========================================================
class HistoryTimeMachine:
    SCENARIOS = {
        "IT Bubble Burst (2000)": {"start": "2000-03-01", "end": "2005-03-24", "desc": "ハイテクバブル崩壊と長期停滞"},
        "Lehman Shock (2008)": {"start": "2008-08-01", "end": "2013-09-15", "desc": "金融システム崩壊から量的緩和へ"},
        "Corona Shock (2020)": {"start": "2020-01-01", "end": "2022-02-19", "desc": "瞬間的な暴落と急激なバブル"},
        "Great Inflation (2022)": {"start": "2021-12-01", "end": "2024-01-03", "desc": "金利急騰による株債券同時安"}
    }
    
    @staticmethod
    def run_replay(current_price, current_beta, scenario_key, region="US", weights_dict=None):
        scenario = HistoryTimeMachine.SCENARIOS.get(scenario_key)
        if not scenario: return None
        
        config = MarketConfig.get_config(region)
        bm_ticker = config["benchmark_ticker"]
        start_date = scenario['start']
        end_date = scenario['end']
        
        try:
            # ベンチマーク（市場平均）の当時の実データを取得
            bm_data = yf.download(bm_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(bm_data, pd.DataFrame) and 'Close' in bm_data.columns: 
                bm_data = bm_data['Close']
            if bm_data.empty: return None
            
            market_returns = bm_data.pct_change().dropna()
            market_path = (1 + market_returns).cumprod() * current_price

            # 🆕 【真のバックテスト】構成銘柄の当時の実際の価格データを取得して合成する
            if weights_dict is not None:
                tickers = list(weights_dict.keys())
                port_data = DataFetcher.fetch_market_data(tickers, start_date=start_date)
                
                # 期間フィルタリング
                port_data = port_data.loc[:end_date]
                if not port_data.empty:
                    if isinstance(port_data, pd.Series):
                        port_data = port_data.to_frame(name=tickers[0])
                        
                    port_returns = port_data.pct_change()
                    
                    # プロキシ補完（当時上場していなかった銘柄は、ベンチマークの動きで埋める）
                    aligned = pd.concat([port_returns, market_returns.rename("BM")], axis=1)
                    for tkr in tickers:
                        if tkr not in aligned.columns: continue
                        missing = aligned[tkr].isna() & aligned["BM"].notna()
                        aligned.loc[missing, tkr] = aligned.loc[missing, "BM"] # Beta=1補完
                        
                    aligned = aligned.dropna(subset=["BM"])
                    
                    weighted_ret = pd.Series(0, index=aligned.index)
                    for tkr, w in weights_dict.items():
                        if tkr in aligned.columns:
                            weighted_ret += aligned[tkr] * (w / 100.0)
                            
                    price_path = (1 + weighted_ret).cumprod() * current_price
                    days = np.arange(len(price_path))
                    
                    return {"days": days, "prices": price_path.values, "market_prices": market_path.values, "desc": scenario['desc']}

            # weights_dictが渡されなかった場合のフォールバック（旧ベータ計算）
            simulated_returns = market_returns * current_beta
            days = np.arange(len(simulated_returns))
            price_path = (1 + simulated_returns).cumprod() * current_price
            return {"days": days, "prices": price_path.values, "market_prices": market_path.values, "desc": scenario['desc']}
            
        except Exception: 
            return None

# =========================================================
# 🧪 ファクター分析
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
        if target_series.empty: return None
        target_monthly = target_series.resample('M').last().pct_change().dropna()
        start_date = target_monthly.index[0]
        config = MarketConfig.get_config(region)
        ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
        if ff_data.empty: return None
        
        target_monthly.index = target_monthly.index.to_period('M')
        ff_data.index = ff_data.index.to_period('M')
        combined = pd.concat([target_monthly.rename("Target"), ff_data], axis=1).dropna()
        if len(combined) < 12: return None
        
        mkt_col = [c for c in combined.columns if "Mkt" in c or "MKT" in c][0]
        smb_col = [c for c in combined.columns if "SMB" in c][0]
        hml_col = [c for c in combined.columns if "HML" in c][0]
        rf_col = [c for c in combined.columns if "RF" in c][0]

        y = combined["Target"] - combined[rf_col]
        X = combined[[mkt_col, smb_col, hml_col]]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            return {
                "beta_market": model.params.get(mkt_col, 1.0),
                "beta_size": model.params.get(smb_col, 0.0),
                "beta_value": model.params.get(hml_col, 0.0),
                "alpha": model.params.get("const", 0.0),
                "r_squared": model.rsquared,
                "region": region
            }
        except: return None

# =========================================================
# 🔬 レジーム解析エンジン
# =========================================================
class RegimeAnalyzer:
    @staticmethod
    def analyze_periodicity(series, fs=12):
        try:
            clean_series = series.dropna()
            if len(clean_series) < 24: return 0, 0
            freqs, psd = signal.welch(clean_series, fs=fs, nperseg=min(len(clean_series), 60))
            valid_idx = (freqs > 1/60) & (freqs < 1/2) 
            if not any(valid_idx): return 0, 0
            target_freqs = freqs[valid_idx]
            target_psd = psd[valid_idx]
            dominant_freq = target_freqs[np.argmax(target_psd)]
            return round(1 / dominant_freq if dominant_freq > 0 else 0, 1), np.max(target_psd)
        except: return 0, 0

    @staticmethod
    def fit_garch_volatility(returns):
        scaled_returns = returns * 100.0
        result_data = {"current_vol": 0.0, "long_term_vol": 0.0, "residuals": [], "model_type": "GARCH" if HAS_ARCH else "EWMA"}
        try:
            if HAS_ARCH and len(returns) > 36:
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
                res = model.fit(disp='off')
                result_data["current_vol"] = res.conditional_volatility.iloc[-1] / 100.0 * np.sqrt(12) 
                result_data["long_term_vol"] = res.unconditional_volatility / 100.0 * np.sqrt(12)
                result_data["residuals"] = res.std_resid.dropna().values
            else:
                vol = returns.ewm(span=24).std() * np.sqrt(12)
                result_data["current_vol"] = vol.iloc[-1]
                result_data["long_term_vol"] = returns.std() * np.sqrt(12)
                result_data["residuals"] = ((returns - returns.mean()) / returns.std()).values
        except:
            std = returns.std() * np.sqrt(12)
            result_data.update({"current_vol": std, "long_term_vol": std, "residuals": np.random.normal(0, 1, len(returns))})
        return result_data

# =========================================================
# 🌊 確率的シナリオ生成エンジン (STEP 2: 連続性の確保)
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def _generate_single_factor_path(factor_returns, n_sims, horizon_months):
        period, power = RegimeAnalyzer.analyze_periodicity(factor_returns)
        garch_res = RegimeAnalyzer.fit_garch_volatility(factor_returns)
        
        current_vol = garch_res['current_vol'] / np.sqrt(12)
        long_run_vol = garch_res['long_term_vol'] / np.sqrt(12)
        
        residuals = garch_res['residuals']

        if len(residuals) > 36:
            # 🆕 Smoothed Bootstrap: 生の抽出に正規ノイズを足すことで、ヒストグラムの歯抜けを防ぐ
            sampled_shocks = np.random.choice(residuals, size=(horizon_months, n_sims))
            noise = np.random.normal(0, np.std(residuals) * 0.2, size=(horizon_months, n_sims))
            future_shocks = sampled_shocks + noise
        else:
            # 🆕 Fat-tailを極端に強調(df=3)し、最悪のシナリオを増やす
            DEGREES_OF_FREEDOM = 3 
            future_shocks = t.rvs(df=DEGREES_OF_FREEDOM, size=(horizon_months, n_sims))
            variance_adjustment = np.sqrt(DEGREES_OF_FREEDOM / (DEGREES_OF_FREEDOM - 2))
            future_shocks = future_shocks / variance_adjustment
            
        paths = np.zeros((horizon_months, n_sims))
        drift = factor_returns.mean()
        sim_vol = current_vol
        
        for i in range(horizon_months):
            cycle_multiplier = 1.0
            if period > 0:
                phase = (2 * np.pi * i) / period
                amp = min(power * 0.5, 0.5) 
                cycle_multiplier = 1.0 + amp * np.sin(phase)

            ret = drift + sim_vol * cycle_multiplier * future_shocks[i]
            paths[i] = ret
            
            alpha = 0.1
            sim_vol = np.sqrt((1 - alpha) * (long_run_vol**2) + alpha * (ret**2))
            
        return paths

    @staticmethod
    def generate_3factor_waves(start_date, region="US", n_sims=7500, horizon_months=60):
        config = MarketConfig.get_config(region)
        ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
        if ff_data.empty or len(ff_data) < 36: return None
            
        waves = {}
        target_map = {
            'Market': [c for c in ff_data.columns if "Mkt" in c or "MKT" in c],
            'SMB': [c for c in ff_data.columns if "SMB" in c],
            'HML': [c for c in ff_data.columns if "HML" in c]
        }
        
        for key, cols in target_map.items():
            if cols:
                paths = StochasticScenarioGenerator._generate_single_factor_path(
                    ff_data[cols[0]], n_sims, horizon_months
                )
                waves[key] = paths
        return waves

# =========================================================
# 🚀 プロジェクション・コア
# =========================================================
class ProjectionCore:
    @staticmethod
    def run_market_driven_projection(current_price, factor_waves, factor_profile, n_sims=7500, horizon_months=60):
        b_mkt = factor_profile.get('beta_market', 1.0)
        b_smb = factor_profile.get('beta_size', 0.0)
        b_hml = factor_profile.get('beta_value', 0.0)
        monthly_alpha = factor_profile.get('alpha', 0.0)

        simulated_returns = np.full((horizon_months, n_sims), monthly_alpha)

        if 'Market' in factor_waves:
            simulated_returns += b_mkt * factor_waves['Market']
            
        if 'SMB' in factor_waves:
            simulated_returns += b_smb * factor_waves['SMB']
            
        if 'HML' in factor_waves:
            simulated_returns += b_hml * factor_waves['HML']

        growth_factors = 1.0 + simulated_returns
        ones = np.ones((1, n_sims))
        cumulative_growth = np.vstack([ones, growth_factors])
        
        price_paths = np.cumprod(cumulative_growth, axis=0) * current_price

        return price_paths

# =========================================================
# 📋 最終監査レポート & 解析エンジン
# =========================================================
class AuditEngine:
    @staticmethod
    def analyze_recovery_probability(price_paths, threshold_dd=0.10):
        n_steps, n_sims = price_paths.shape
        recovery_months = []
        
        peaks = np.maximum.accumulate(price_paths, axis=0)
        drawdowns = (price_paths - peaks) / peaks
        max_dds = drawdowns.min(axis=0)
        
        crashed_indices = np.where(max_dds < -threshold_dd)[0]
        
        if len(crashed_indices) == 0:
            return {"probability": 1.0, "avg_months": 0, "desc": "指定閾値以上の暴落なし"}

        for idx in crashed_indices:
            path = price_paths[:, idx]
            peak_path = peaks[:, idx]
            
            dd_idx = np.argmin(drawdowns[:, idx])
            val_at_dd = path[dd_idx]
            peak_before_dd = peak_path[dd_idx]
            
            future_prices = path[dd_idx:]
            
            recovered = np.where(future_prices >= peak_before_dd)[0]
            
            if len(recovered) > 0:
                months_to_recover = recovered[0]
                recovery_months.append(months_to_recover)
                
        total_crashes = len(crashed_indices)
        success_count = len(recovery_months)
        prob_recovery = success_count / total_crashes if total_crashes > 0 else 0
        avg_recovery = np.mean(recovery_months) if recovery_months else 0
        
        return {
            "crashed_scenarios_count": total_crashes,
            "recovery_probability": round(prob_recovery * 100, 1),
            "avg_recovery_months": round(avg_recovery, 1),
            "median_recovery_months": np.median(recovery_months) if recovery_months else 0
        }
