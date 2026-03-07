import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import t, skew, kurtosis
import scipy.signal as signal
import pandas_datareader.data as web
from datetime import datetime, timedelta
import warnings
import streamlit as st  # キャッシュ機能のために追加

# 📌 ログを埋め尽くすPandasの仕様変更警告をミュート
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# GARCHモデル用
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# 📌 [新規] 収縮推定（Shrinkage）による共分散計算用
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
            "vix_ticker": "^VIX"           # 米国VIX
        },
        "Japan": {
            "name": "Japan",
            "ff_dataset": "Japan_3_Factors",
            "benchmark_ticker": "^N225",   # 日経225
            "risk_free_ticker": "JPY=X",   # 日本株の場合は為替等をマクロ指標に
            "vix_ticker": "^JNIV"          # 日経VI
        }
    }

    @staticmethod
    def get_config(region="US"):
        return MarketConfig.REGIONS.get(region, MarketConfig.REGIONS["US"])

# =========================================================
# 🛠️ データ取得 & ポートフォリオ合成クラス
# =========================================================
class DataFetcher:
    
    @staticmethod
    def normalize_weights(weights_dict):
        """
        📌 [新規] CSV解析ロジックの柔軟化
        ティッカー名の正規化（BRK-B -> BRK.B）と、合計が100でない場合の自動リスケールを行う
        """
        if not weights_dict: return {}
        normalized = {}
        total = 0.0
        for k, v in weights_dict.items():
            # Yahoo Finance形式への正規化
            new_k = str(k).strip().upper().replace('-', '.') 
            normalized[new_k] = float(v)
            total += float(v)
        
        # 自動リスケール (100%基準)
        if total > 0 and abs(total - 100.0) > 1e-4:
            for k in normalized:
                normalized[k] = (normalized[k] / total) * 100.0
                
        return normalized

    @staticmethod
    @st.cache_data(ttl=3600) # 1時間のキャッシュ化で通信を大幅削減
    def fetch_market_data(tickers, start_date="2000-01-01"):
        if not tickers: return pd.DataFrame()
        try:
            if isinstance(tickers, str): tickers = [tickers]
            
            # threads=False でマルチスレッドを無効化しAPI制限を回避
            data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, threads=False)
            
            if data.empty: return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]: data = data['Close']
                elif 'Adj Close' in data.columns.levels[0]: data = data['Adj Close']
            
            # タイムゾーンを削除してインデックスを「日付」のみに標準化
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            data.index = data.index.normalize()

            # 前日値で埋めて歯抜けを防止
            return data.ffill().dropna(how='all')
        except Exception as e:
            print(f"Fetch Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def validate_tickers(input_dict):
        valid_data = {}
        invalid_tickers = []
        # ここでも正規化を通す
        input_dict = DataFetcher.normalize_weights(input_dict)
        for ticker, weight in input_dict.items():
            try:
                tick = yf.Ticker(ticker)
                hist = tick.history(period="1d") 
                if not hist.empty: valid_data[ticker] = weight
                else: invalid_tickers.append(ticker)
            except: invalid_tickers.append(ticker)
        return valid_data, invalid_tickers

    @staticmethod
    def create_synthetic_portfolio(ticker_weights, region="US"):
        # 📌 自動正規化・リスケールを通す
        ticker_weights = DataFetcher.normalize_weights(ticker_weights)
        tickers = list(ticker_weights.keys())
        if not tickers: return None
        
        raw_prices = DataFetcher.fetch_market_data(tickers)
        if raw_prices.empty: return None
        
        config = MarketConfig.get_config(region)
        bm_ticker = config["benchmark_ticker"]
        bm_prices = DataFetcher.fetch_market_data([bm_ticker])
        
        if isinstance(raw_prices, pd.Series):
            raw_prices = raw_prices.to_frame(name=tickers[0])
        if isinstance(bm_prices, pd.Series):
            bm_prices = bm_prices.to_frame(name=bm_ticker)

        returns = raw_prices.pct_change()
        bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")
        
        # 銘柄とベンチマークの日付を完全に結合（ズレを吸収）
        master_index = returns.index.union(bm_returns.index).sort_values()
        aligned_data = pd.DataFrame(index=master_index)
        aligned_data = aligned_data.join(returns).join(bm_returns)
        
        aligned_data = aligned_data.dropna(subset=["Benchmark"])
        
        for ticker in tickers:
            # その時代に全く上場していなかった場合のフォールバック
            if ticker not in aligned_data.columns or aligned_data[ticker].isna().all():
                aligned_data[ticker] = aligned_data["Benchmark"] 
                continue
            
            # ベータ値の算出
            valid_mask = aligned_data[ticker].notna() & aligned_data["Benchmark"].notna()
            if valid_mask.sum() > 30:
                cov = np.cov(aligned_data.loc[valid_mask, ticker], aligned_data.loc[valid_mask, "Benchmark"])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            else:
                beta = 1.0
            
            # 上場前の空白期間を「BMリターン * ベータ」でバックフィル生成
            missing_mask = aligned_data[ticker].isna()
            aligned_data.loc[missing_mask, ticker] = aligned_data.loc[missing_mask, "Benchmark"] * beta
        
        weighted_returns = pd.Series(0, index=aligned_data.index)
        for ticker, weight in ticker_weights.items():
            # 最後の細かい休場のズレは 0 (変動なし) として処理
            weighted_returns += aligned_data[ticker].fillna(0) * (weight / 100.0)
                
        return (1 + weighted_returns).cumprod() * 100

    @staticmethod
    @st.cache_data(ttl=86400) # ファクターデータは1日キャッシュ
    def fetch_fama_french_factors(start_date, end_date=None, dataset_name="F-F_Research_Data_Factors"):
        try:
            ff_dict = web.DataReader(dataset_name, 'famafrench', start=start_date, end=end_date)
            if not ff_dict: return pd.DataFrame()
            
            # 日米で異なるインデックス構造を安全に日付型へ変換
            ff_data = ff_dict[0] / 100.0
            ff_data.index = ff_data.index.to_timestamp()
            if ff_data.index.tz is not None: ff_data.index = ff_data.index.tz_localize(None)
            ff_data.columns = [c.strip() for c in ff_data.columns]
            return ff_data
        except Exception as e:
            print(f"FF Data Error: {e}")
            return pd.DataFrame()

# =========================================================
# 📊 高度な統計指標計算クラス
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, weights_dict=None, region="US"):
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
            
        # 2. リスク（ボラティリティ）の計算
        raw_sigma = returns.std() * np.sqrt(12)
        lw_sigma = raw_sigma
        
        # 📌 [新規] 収縮推定（Ledoit-Wolf）による堅牢なボラティリティ算出
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
                        
                        lw = LedoitWolf().fit(comp_returns)
                        cov_matrix = lw.covariance_
                        port_var = np.dot(new_weights.T, np.dot(cov_matrix, new_weights))
                        lw_sigma = np.sqrt(port_var) * np.sqrt(12)
            except Exception as e:
                print(f"LedoitWolf Error: {e}")
        
        # 最終的なシグマは収縮推定値にペナルティを加味したもの
        sigma = lw_sigma * penalty 
        
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
        
        # 📌 [新規] 固有リスク（Idiosyncratic Risk）とシステマティックリスクの分離
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
                # タイムゾーン等の調整
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
                        systematic_risk = np.sqrt(sys_var) * np.sqrt(12)
                        idiosyncratic_risk = np.sqrt(idio_var) * np.sqrt(12)
                        
                        # Info Ratio
                        active_ret = aligned.iloc[:, 0] - aligned["BM"]
                        track_err = active_ret.std() * np.sqrt(12)
                        if track_err > 0:
                            info_ratio = (active_ret.mean() * 12) / track_err
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
            "volatility": sigma,  # 📌 App側でこれを使えるようエクスポート
            "raw_volatility": raw_sigma,
            "systematic_risk": systematic_risk,
            "idiosyncratic_risk": idiosyncratic_risk,
            "portfolio_beta": portfolio_beta
        }

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
            bm_data = DataFetcher.fetch_market_data([bm_ticker], start_date="1999-01-01")
            bm_data = bm_data.loc[start_date:end_date]
            if bm_data.empty: return None
            
            market_returns = bm_data.pct_change().iloc[:, 0].dropna()
            
            weights_dict = DataFetcher.normalize_weights(weights_dict)
            if weights_dict and sum(weights_dict.values()) > 0:
                tickers = list(weights_dict.keys())
                port_data = DataFetcher.fetch_market_data(tickers, start_date="1999-01-01")
                port_returns = port_data.pct_change()
                
                master_idx = market_returns.index
                aligned = pd.DataFrame(index=master_idx).join(port_returns).join(market_returns.rename("BM"))
                
                for tkr in tickers:
                    valid = aligned[tkr].notna() & aligned["BM"].notna()
                    if valid.sum() > 20:
                        c = np.cov(aligned.loc[valid, tkr], aligned.loc[valid, "BM"])
                        beta = c[0, 1] / c[1, 1] if c[1, 1] != 0 else current_beta
                    else:
                        beta = current_beta
                    
                    aligned[tkr] = aligned[tkr].fillna(aligned["BM"] * beta)
                
                weighted_ret = pd.Series(0, index=aligned.index)
                for tkr, w in weights_dict.items():
                    weighted_ret += aligned[tkr] * (w / 100.0)
            else:
                weighted_ret = market_returns * current_beta

            price_path = (1 + weighted_ret).cumprod() * current_price
            market_path = (1 + market_returns).cumprod() * current_price
            
            return {
                "dates": weighted_ret.index, 
                "prices": price_path.values, 
                "market_prices": market_path.values, 
                "desc": scenario['desc']
            }
        except Exception: 
            return None

# =========================================================
# 🧪 ファクター分析
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
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
            dominant_freq = freqs[valid_idx][np.argmax(psd[valid_idx])]
            return round(1 / dominant_freq if dominant_freq > 0 else 0, 1), np.max(psd)
        except: return 0, 0

    @staticmethod
    def fit_garch_volatility(returns):
        scaled_returns = returns * 100.0
        result_data = {"current_vol": 0.0, "long_term_vol": 0.0, "model_type": "GARCH" if HAS_ARCH else "EWMA"}
        try:
            if HAS_ARCH and len(returns) > 36:
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
                res = model.fit(disp='off')
                result_data["current_vol"] = res.conditional_volatility.iloc[-1] / 100.0 * np.sqrt(12) 
                result_data["long_term_vol"] = res.unconditional_volatility / 100.0 * np.sqrt(12)
            else:
                vol = returns.ewm(span=24).std() * np.sqrt(12)
                result_data["current_vol"] = vol.iloc[-1]
                result_data["long_term_vol"] = returns.std() * np.sqrt(12)
        except:
            std = returns.std() * np.sqrt(12)
            result_data.update({"current_vol": std, "long_term_vol": std})
        return result_data

# =========================================================
# 🌊 確率的シナリオ生成エンジン
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_portfolio_paths(returns, n_sims=10000, horizon_months=60, stress_level="Stress"):
        # 📌 [新規] サンプル数を10,000回に増強して分布の密度を上げる
        df = {"Extreme": 3, "Stress": 5}.get(stress_level, 30)
        period, power = RegimeAnalyzer.analyze_periodicity(returns)
        garch_res = RegimeAnalyzer.fit_garch_volatility(returns)
        
        current_vol = garch_res['current_vol'] / np.sqrt(12)
        long_run_vol = garch_res['long_term_vol'] / np.sqrt(12)
        
        future_shocks = t.rvs(df=df, size=(horizon_months, n_sims))
        if df > 2: future_shocks /= np.sqrt(df / (df - 2))
            
        paths = np.zeros((horizon_months, n_sims))
        drift = returns.mean()
        sim_vol = current_vol
        
        for i in range(horizon_months):
            cycle = 1.0 + (min(power * 0.5, 0.5) * np.sin(2 * np.pi * i / period)) if period > 0 else 1.0
            ret = drift + sim_vol * cycle * future_shocks[i]
            paths[i] = ret
            sim_vol = np.sqrt(0.9 * (sim_vol**2) + 0.1 * (long_run_vol**2))
            
        return paths

# =========================================================
# 🚀 プロジェクション・コア
# =========================================================
class ProjectionCore:
    @staticmethod
    def run_projection(current_price, simulated_returns):
        cumulative_growth = np.vstack([np.ones((1, simulated_returns.shape[1])), 1.0 + simulated_returns])
        return np.cumprod(cumulative_growth, axis=0) * current_price

# =========================================================
# 📋 最終監査レポート & 解析エンジン
# =========================================================
class AuditEngine:
    @staticmethod
    def calculate_optimal_bins(data_array):
        """
        📌 [新規] 動的ビン（Bin）計算: 
        Freedman-Diaconis準則を用いて、ヒストグラムの歯抜けを防ぐ最適な棒の幅を計算
        """
        data = np.asarray(data_array)
        data = data[~np.isnan(data)]
        if len(data) < 2: return 30
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0: return 30
        bin_width = 2 * iqr * (len(data) ** (-1/3))
        if bin_width == 0: return 30
        bins = int(np.ceil((data.max() - data.min()) / bin_width))
        return min(max(bins, 10), 150) # 10から150の間に制限

    @staticmethod
    def analyze_recovery_probability(price_paths, threshold_dd=0.10):
        peaks = np.maximum.accumulate(price_paths, axis=0)
        drawdowns = (price_paths - peaks) / peaks
        max_dds = drawdowns.min(axis=0)
        
        crashed_indices = np.where(max_dds < -threshold_dd)[0]
        if len(crashed_indices) == 0: return {"probability": 100.0, "avg_months": 0}

        recovery_months = []
        for idx in crashed_indices:
            path = price_paths[:, idx]
            dd_idx = np.argmin(drawdowns[:, idx])
            peak_before = peaks[dd_idx, idx]
            recovered = np.where(path[dd_idx:] >= peak_before)[0]
            if len(recovered) > 0: recovery_months.append(recovered[0])
                
        return {
            "recovery_probability": round(len(recovery_months) / len(crashed_indices) * 100, 1),
            "avg_recovery_months": round(np.mean(recovery_months), 1) if recovery_months else 0
        }
