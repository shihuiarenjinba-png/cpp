import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import t, skew, kurtosis
import scipy.signal as signal
import pandas_datareader.data as web
from datetime import datetime, timedelta
import warnings

# 📌 ログを埋め尽くすPandasの仕様変更警告(date_parser等)をミュートして動作を軽くする
warnings.filterwarnings("ignore", category=FutureWarning)

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
# 📊 高度な統計指標計算クラス
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, weights_dict=None):
        if returns.empty: return {}
        
        # 1. 集中投資ペナルティ(HHI)と有効銘柄数(ENC)の計算
        hhi = 0.0
        penalty = 1.0
        enc = 0.0 
        
        if weights_dict is not None and sum(weights_dict.values()) > 0:
            weights = np.array(list(weights_dict.values())) / 100.0
            hhi = np.sum(weights**2)
            enc = 1.0 / hhi if hhi > 0 else 1.0
            penalty = 1.0 + (hhi ** 0.8) * 0.5 
            
        # 2. リスク（ボラティリティ）の計算とペナルティ適用
        raw_sigma = returns.std() * np.sqrt(12)
        sigma = raw_sigma * penalty 
        
        # 3. リターンの計算 (ボラティリティ・ドラッグを考慮)
        arithmetic_mu = returns.mean() * 12
        mu = arithmetic_mu - 0.5 * (sigma ** 2) 
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        ulcer_sq = (drawdown ** 2).mean()
        ulcer_index = np.sqrt(ulcer_sq)
        
        risk_free_rate = 0.02 
        sharpe = (mu - risk_free_rate) / sigma if sigma > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(12) * penalty
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
        if benchmark_returns is not None:
            common = returns.index.intersection(benchmark_returns.index)
            if len(common) > 12:
                active_ret = returns.loc[common] - benchmark_returns.loc[common]
                track_err = active_ret.std() * np.sqrt(12)
                if track_err > 0:
                    info_ratio = (active_ret.mean() * 12) / track_err

        # 📌 修正: NaNを取り除いてからSkewnessとKurtosisを計算 (エラーや不正確な計算を防止)
        clean_returns = returns.dropna()
        return {
            "sharpe": sharpe, 
            "sortino": sortino, 
            "calmar": calmar,
            "omega": omega, 
            "cvar_95": cvar_95, 
            "ulcer_index": ulcer_index,
            "kelly_criterion": kelly, 
            "max_dd": max_dd, 
            "info_ratio": info_ratio,
            "skewness": skew(clean_returns) if len(clean_returns) > 0 else 0, 
            "kurtosis": kurtosis(clean_returns) if len(clean_returns) > 0 else 0,
            "hhi_index": hhi, 
            "effective_n": enc,
            "risk_penalty_ratio": penalty
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
        
        # 📌 修正: 文字列変換を廃止し、タイムゾーンを削除して日次ベースに揃える (確実な結合のため)
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        if bm_returns.index.tz is not None:
            bm_returns.index = bm_returns.index.tz_localize(None)
            
        returns.index = returns.index.normalize()
        bm_returns.index = bm_returns.index.normalize()
        
        aligned_data = pd.concat([returns, bm_returns], axis=1)
        
        for ticker in tickers:
            if ticker not in aligned_data.columns:
                aligned_data[ticker] = aligned_data["Benchmark"]
                continue
                
            valid_mask = aligned_data[ticker].notna() & aligned_data["Benchmark"].notna()
            if valid_mask.sum() > 30:
                cov = np.cov(aligned_data.loc[valid_mask, ticker], aligned_data.loc[valid_mask, "Benchmark"])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            else:
                beta = 1.0
                
            missing_mask = aligned_data[ticker].isna() & aligned_data["Benchmark"].notna()
            aligned_data.loc[missing_mask, ticker] = aligned_data.loc[missing_mask, "Benchmark"] * beta
            
        aligned_data = aligned_data.dropna(subset=["Benchmark"])
        
        weighted_returns = pd.Series(0, index=aligned_data.index)
        for ticker, weight in ticker_weights.items():
            if ticker in aligned_data.columns:
                weighted_returns += aligned_data[ticker].fillna(0) * (weight / 100.0)
                
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
# 🕰️ タイムマシン機能
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
            fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        except ValueError:
            fetch_start = start_date

        try:
            bm_data = yf.download(bm_ticker, start=fetch_start, end=end_date, progress=False, auto_adjust=True)
            if isinstance(bm_data, pd.DataFrame):
                if 'Close' in bm_data.columns: bm_data = bm_data['Close']
                elif 'Adj Close' in bm_data.columns: bm_data = bm_data['Adj Close']
            
            if bm_data.empty: return None
            
            market_returns = bm_data.pct_change().dropna()
            
            # 📌 修正: タイムゾーン削除
            if market_returns.index.tz is not None:
                market_returns.index = market_returns.index.tz_localize(None)
            market_returns.index = market_returns.index.normalize()
            
            market_returns = market_returns.loc[start_date:end_date]
            if market_returns.empty: return None

            if weights_dict is not None and sum(weights_dict.values()) > 0:
                tickers = list(weights_dict.keys())
                port_data = DataFetcher.fetch_market_data(tickers, start_date=fetch_start)
                
                port_data = port_data.loc[:end_date]
                if not port_data.empty:
                    if isinstance(port_data, pd.Series):
                        port_data = port_data.to_frame(name=tickers[0])
                        
                    port_returns = port_data.pct_change().dropna(how='all')
                    
                    # 📌 修正: タイムゾーン削除
                    if port_returns.index.tz is not None:
                        port_returns.index = port_returns.index.tz_localize(None)
                    port_returns.index = port_returns.index.normalize()
                    
                    aligned = pd.concat([port_returns, market_returns.rename("BM")], axis=1)
                    aligned = aligned.loc[aligned.index >= pd.to_datetime(start_date)]
                    
                    for tkr in tickers:
                        if tkr not in aligned.columns:
                            aligned[tkr] = aligned["BM"] * current_beta
                            continue
                            
                        valid_mask = aligned[tkr].notna() & aligned["BM"].notna()
                        if valid_mask.sum() > 30:
                            cov = np.cov(aligned.loc[valid_mask, tkr], aligned.loc[valid_mask, "BM"])
                            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
                        else:
                            beta = current_beta 
                            
                        missing = aligned[tkr].isna() & aligned["BM"].notna()
                        aligned.loc[missing, tkr] = aligned.loc[missing, "BM"] * beta
                        
                    aligned = aligned.dropna(subset=["BM"])
                    if aligned.empty: return None
                    
                    weighted_ret = pd.Series(0, index=aligned.index)
                    for tkr, w in weights_dict.items():
                        if tkr in aligned.columns:
                            weighted_ret += aligned[tkr].fillna(0) * (w / 100.0)
                            
                    price_path = (1 + weighted_ret).cumprod() * current_price
                    market_path = (1 + aligned["BM"]).cumprod() * current_price
                    
                    return {"dates": aligned.index, "prices": price_path.values, "market_prices": market_path.values, "desc": scenario['desc']}

            simulated_returns = market_returns * current_beta
            price_path = (1 + simulated_returns).cumprod() * current_price
            market_path = (1 + market_returns).cumprod() * current_price
            
            return {"dates": market_returns.index, "prices": price_path.values, "market_prices": market_path.values, "desc": scenario['desc']}
            
        except Exception as e: 
            return None

# =========================================================
# 🧪 ファクター分析
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
        if target_series.empty: return None
        # 📌 修正: 'M' から 'ME' へ変更し警告を防止
        target_monthly = target_series.resample('ME').last().pct_change().dropna()
        start_date = target_monthly.index[0]
        config = MarketConfig.get_config(region)
        ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
        if ff_data.empty: return None
        
        target_monthly.index = target_monthly.index.to_period('M')
        ff_data.index = ff_data.index.to_period('M')
        combined = pd.concat([target_monthly.rename("Target"), ff_data], axis=1).dropna()
        if len(combined) < 12: return None
        
        try:
            # 📌 修正: 該当カラムがFFデータになかった場合、クラッシュを避けるための安全機構
            mkt_col_list = [c for c in combined.columns if "Mkt" in c or "MKT" in c]
            smb_col_list = [c for c in combined.columns if "SMB" in c]
            hml_col_list = [c for c in combined.columns if "HML" in c]
            rf_col_list = [c for c in combined.columns if "RF" in c]
            
            if not (mkt_col_list and smb_col_list and hml_col_list and rf_col_list):
                return None
                
            mkt_col = mkt_col_list[0]
            smb_col = smb_col_list[0]
            hml_col = hml_col_list[0]
            rf_col = rf_col_list[0]

            y = combined["Target"] - combined[rf_col]
            X = combined[[mkt_col, smb_col, hml_col]]
            X = sm.add_constant(X)
            
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
# 🌊 確率的シナリオ生成エンジン
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_portfolio_paths(returns, n_sims=7500, horizon_months=60, stress_level="Stress"):
        if stress_level == "Extreme":
            df = 3   
        elif stress_level == "Stress":
            df = 5   
        else:
            df = 30  

        period, power = RegimeAnalyzer.analyze_periodicity(returns)
        garch_res = RegimeAnalyzer.fit_garch_volatility(returns)
        
        current_vol = garch_res['current_vol'] / np.sqrt(12)
        long_run_vol = garch_res['long_term_vol'] / np.sqrt(12)
        
        future_shocks = t.rvs(df=df, size=(horizon_months, n_sims))
        
        variance_adjustment = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        future_shocks = future_shocks / variance_adjustment
            
        paths = np.zeros((horizon_months, n_sims))
        drift = returns.mean()
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

# =========================================================
# 🚀 プロジェクション・コア
# =========================================================
class ProjectionCore:
    @staticmethod
    def run_projection(current_price, simulated_returns):
        growth_factors = 1.0 + simulated_returns
        n_sims = simulated_returns.shape[1]
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
