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
# 🌍 市場設定管理クラス (STEP 4 Implementation) [NO CHANGE]
# =========================================================
class MarketConfig:
    REGIONS = {
        "US": {
            "name": "United States",
            "ff_dataset": "F-F_Research_Data_Factors",
            "benchmark_ticker": "^GSPC",
            "risk_free_ticker": "^TNX",
            "vix_ticker": "^VIX"
        },
        "Japan": {
            "name": "Japan",
            "ff_dataset": "Japan_3_Factors",
            "benchmark_ticker": "^N225",
            "risk_free_ticker": "^TNX",
            "vix_ticker": "^VIX"
        },
        "Developed": {
            "name": "Developed Markets",
            "ff_dataset": "Developed_3_Factors",
            "benchmark_ticker": "URTH",
            "risk_free_ticker": "^TNX",
            "vix_ticker": "^VIX"
        }
    }

    @staticmethod
    def get_config(region="US"):
        return MarketConfig.REGIONS.get(region, MarketConfig.REGIONS["US"])

# =========================================================
# 📊 高度な統計指標計算クラス [NO CHANGE]
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None):
        if returns.empty: return {}
        
        mu = returns.mean() * 12
        sigma = returns.std() * np.sqrt(12)
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        sharpe = (mu - 0.02) / sigma if sigma > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(12)
        sortino = (mu - 0.02) / downside_dev if downside_dev > 0 else 0
        
        calmar = mu / abs(max_dd) if max_dd != 0 else 0
        
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else np.inf
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
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
            "skewness": skew(returns), "kurtosis": kurtosis(returns)
        }

# =========================================================
# 🛠️ データ取得 & ポートフォリオ合成クラス [MODIFIED]
# =========================================================
class DataFetcher:
    @staticmethod
    def fetch_market_data(tickers, start_date="2000-01-01"):
        try:
            # yfinanceの仕様変更に対応 (listで渡す)
            if isinstance(tickers, str):
                tickers = [tickers]
                
            data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
            
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns: data = data['Close']
                elif 'Adj Close' in data.columns: data = data['Adj Close']
            
            if data.empty: return pd.DataFrame()
            # MultiIndexの場合の処理はpandasの仕様に委ねる（ffillで対応）
            return data.ffill().dropna()
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
    def create_synthetic_portfolio(ticker_weights):
        tickers = list(ticker_weights.keys())
        if not tickers: return None
        raw_prices = DataFetcher.fetch_market_data(tickers)
        if raw_prices.empty: return None
        returns = raw_prices.pct_change().dropna()
        weighted_returns = pd.Series(0, index=returns.index)
        for ticker, weight in ticker_weights.items():
            if ticker in returns.columns:
                weighted_returns += returns[ticker] * (weight / 100.0)
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

    @staticmethod
    def fetch_benchmark_and_macro(start_date, region="US"):
        """
        [NEW] 画面側の app_re.py から呼び出されるメソッド。
        指定されたリージョンのベンチマークとマクロ指標（金利等）を取得する。
        """
        try:
            # 設定からティッカーを取得
            config = MarketConfig.get_config(region)
            benchmark = config["benchmark_ticker"]
            risk_free = config["risk_free_ticker"]
            
            target_tickers = [benchmark, risk_free]
            
            # 既存のfetch_market_dataを再利用
            data = DataFetcher.fetch_market_data(target_tickers, start_date=start_date)
            return data
        except Exception:
            return pd.DataFrame()

# =========================================================
# 🕰️ タイムマシン機能 [NO CHANGE]
# =========================================================
class HistoryTimeMachine:
    SCENARIOS = {
        "IT Bubble Burst (2000)": {"start": "2000-03-24", "end": "2005-03-24", "desc": "ハイテクバブル崩壊と長期停滞"},
        "Lehman Shock (2008)": {"start": "2008-09-15", "end": "2013-09-15", "desc": "金融システム崩壊から量的緩和へ"},
        "Corona Shock (2020)": {"start": "2020-02-19", "end": "2022-02-19", "desc": "瞬間的な暴落と急激なバブル"},
        "Great Inflation (2022)": {"start": "2022-01-03", "end": "2024-01-03", "desc": "金利急騰による株債券同時安"}
    }
    @staticmethod
    def run_replay(current_price, current_beta, scenario_key):
        scenario = HistoryTimeMachine.SCENARIOS.get(scenario_key)
        if not scenario: return None
        try:
            sp500 = yf.download("^GSPC", start=scenario['start'], end=scenario['end'], progress=False, auto_adjust=True)
            if isinstance(sp500, pd.DataFrame) and 'Close' in sp500.columns: sp500 = sp500['Close']
            if sp500.empty: return None
            market_returns = sp500.pct_change().dropna()
            simulated_returns = market_returns * current_beta
            days = np.arange(len(simulated_returns))
            price_path = (1 + simulated_returns).cumprod() * current_price
            market_path = (1 + market_returns).cumprod() * current_price
            return {"days": days, "prices": price_path, "market_prices": market_path, "desc": scenario['desc']}
        except Exception: return None

# =========================================================
# 🧪 ファクター分析 [NO CHANGE]
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
# 🔬 レジーム解析エンジン [NO CHANGE]
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
# 🌊 確率的シナリオ生成エンジン (STEP 3 Modified) [NO CHANGE]
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def _generate_single_factor_path(factor_returns, n_sims, horizon_months):
        """
        [修正点]
        正規分布(Normal)ではなく、t分布(Student's t)を使用してショックを生成。
        これにより、現実の市場で見られる「ファットテール(極端な暴落/急騰)」を再現する。
        """
        period, power = RegimeAnalyzer.analyze_periodicity(factor_returns)
        garch_res = RegimeAnalyzer.fit_garch_volatility(factor_returns)
        
        current_vol = garch_res['current_vol'] / np.sqrt(12)
        long_run_vol = garch_res['long_term_vol'] / np.sqrt(12)
        
        # t分布の自由度 (df=5 は金融資産によく適合する)
        DEGREES_OF_FREEDOM = 5 
        residuals = garch_res['residuals']

        # ショック生成 (ヒストリカル or t分布)
        if len(residuals) > 36:
            # 過去の残差からランダムサンプリング（ブートストラップ）
            future_shocks = np.random.choice(residuals, size=(horizon_months, n_sims))
        else:
            # t分布から乱数生成
            future_shocks = t.rvs(df=DEGREES_OF_FREEDOM, size=(horizon_months, n_sims))
            # t分布は分散が df/(df-2) になるため、標準偏差1に正規化する
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

            # リターン生成
            ret = drift + sim_vol * cycle_multiplier * future_shocks[i]
            paths[i] = ret
            
            # GARCH的ボラティリティ更新
            alpha = 0.1
            sim_vol = np.sqrt((1 - alpha) * (long_run_vol**2) + alpha * (ret**2))
            
        return paths

    @staticmethod
    def generate_3factor_waves(start_date, region="US", n_sims=7500, horizon_months=60):
        """Market, SMB, HMLの3つの独立した波を生成"""
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
# 🚀 プロジェクション・コア (STEP 5 Finalized) [NO CHANGE]
# =========================================================
class ProjectionCore:
    """
    [STEP 5] マルチファクター合成エンジン
    t分布で生成された3つの「市場のうねり」に対し、ポートフォリオのベータを掛け合わせて将来価格を決定する。
    追加の残差項（Unexplained Epsilon）は排除済み。
    """
    
    @staticmethod
    def run_market_driven_projection(current_price, factor_waves, factor_profile, n_sims=7500, horizon_months=60):
        """
        市場波形(waves)とポートフォリオ感応度(beta)を合成して価格パスを生成
        """
        # 1. ファクターベータの展開
        b_mkt = factor_profile.get('beta_market', 1.0)
        b_smb = factor_profile.get('beta_size', 0.0)
        b_hml = factor_profile.get('beta_value', 0.0)
        monthly_alpha = factor_profile.get('alpha', 0.0)

        # 2. 合成リターンの計算 (初期値はAlphaのみ)
        # shape: (horizon_months, n_sims)
        simulated_returns = np.full((horizon_months, n_sims), monthly_alpha)

        # 市場要因 (Market Wave)
        if 'Market' in factor_waves:
            simulated_returns += b_mkt * factor_waves['Market']
            
        # サイズ要因 (SMB Wave)
        if 'SMB' in factor_waves:
            simulated_returns += b_smb * factor_waves['SMB']
            
        # バリュー要因 (HML Wave)
        if 'HML' in factor_waves:
            simulated_returns += b_hml * factor_waves['HML']

        # NOTE: 修正点 - ノイズ(epsilon)加算処理は削除済み

        # 3. 価格パスへの変換
        growth_factors = 1.0 + simulated_returns
        ones = np.ones((1, n_sims))
        cumulative_growth = np.vstack([ones, growth_factors])
        
        price_paths = np.cumprod(cumulative_growth, axis=0) * current_price

        return price_paths

    @staticmethod
    def calculate_prob_metrics(price_paths, target_price=None):
        final_prices = price_paths[-1, :]
        mean_final = np.mean(final_prices)
        median_final = np.median(final_prices)
        
        start_price = price_paths[0, 0]
        prob_up = np.mean(final_prices > start_price)
        
        prob_target = 0.0
        if target_price:
            prob_target = np.mean(final_prices >= target_price)
            
        var_95 = np.percentile(final_prices, 5)
        cvar_95 = final_prices[final_prices <= var_95].mean() if len(final_prices[final_prices <= var_95]) > 0 else var_95

        return {
            "mean_price": mean_final,
            "median_price": median_final,
            "prob_gain": prob_up,
            "prob_target": prob_target,
            "VaR_95": var_95,
            "CVaR_95": cvar_95
        }

# =========================================================
# 📋 最終監査レポート & 解析エンジン (STEP 6 New) [NO CHANGE]
# =========================================================
class AuditEngine:
    """
    [STEP 6] 7,500通りのパラレルワールドを監査し、意思決定用データを生成する
    """

    @staticmethod
    def analyze_recovery_probability(price_paths, threshold_dd=0.10):
        """
        回復確率の算出:
        特定の深さ(threshold_dd, 例:10%)以上の下落をしたシナリオにおいて、
        期間内に元の最高値を回復できた割合と、回復にかかった平均月数。
        """
        n_steps, n_sims = price_paths.shape
        recovery_months = []
        
        # 最高値の履歴 (Cummax)
        peaks = np.maximum.accumulate(price_paths, axis=0)
        # ドローダウン
        drawdowns = (price_paths - peaks) / peaks
        # 各シナリオの最大ドローダウン
        max_dds = drawdowns.min(axis=0)
        
        # 指定した閾値より深く沈んだシナリオを抽出
        crashed_indices = np.where(max_dds < -threshold_dd)[0]
        
        if len(crashed_indices) == 0:
            return {"probability": 1.0, "avg_months": 0, "desc": "指定閾値以上の暴落なし"}

        for idx in crashed_indices:
            path = price_paths[:, idx]
            peak_path = peaks[:, idx]
            
            # 最大ドローダウン発生地点
            dd_idx = np.argmin(drawdowns[:, idx])
            val_at_dd = path[dd_idx]
            peak_before_dd = peak_path[dd_idx]
            
            # ドローダウン以降のデータ
            future_prices = path[dd_idx:]
            
            # 回復判定: ドローダウン前の最高値を超えたか
            recovered = np.where(future_prices >= peak_before_dd)[0]
            
            if len(recovered) > 0:
                # 最初の回復地点までの距離（月数）
                months_to_recover = recovered[0]
                recovery_months.append(months_to_recover)
                
        # 統計算出
        total_crashes = len(crashed_indices)
        success_count = len(recovery_months)
        prob_recovery = success_count / total_crashes if total_crashes > 0 else 0
        avg_recovery = np.mean(recovery_months) if recovery_months else 0
        
        return {
            "crashed_scenarios_count": total_crashes,
            "recovery_probability": round(prob_recovery * 100, 1), # %表記
            "avg_recovery_months": round(avg_recovery, 1),
            "median_recovery_months": np.median(recovery_months) if recovery_months else 0
        }

    @staticmethod
    def stress_test_regimes(price_paths, factor_waves):
        """
        レジーム別耐性スコア:
        生成された波形の特徴に基づいてシナリオを分類し、
        特定の「悪い状況」におけるポートフォリオの平均パフォーマンスを算出する。
        """
        final_returns = (price_paths[-1] / price_paths[0]) - 1
        results = {}
        
        # 1. 高VIX局面 (High Volatility Regime)
        # Market波形の標準偏差が高い上位10%のシナリオを抽出
        if 'Market' in factor_waves:
            mkt_vol = np.std(factor_waves['Market'], axis=0)
            threshold = np.percentile(mkt_vol, 90) # 上位10%
            high_vol_indices = np.where(mkt_vol >= threshold)[0]
            
            if len(high_vol_indices) > 0:
                avg_ret_stress = np.mean(final_returns[high_vol_indices])
                prob_loss_stress = np.mean(final_returns[high_vol_indices] < 0)
                
                results["High_VIX_Regime"] = {
                    "avg_return": avg_ret_stress,
                    "win_rate": 1.0 - prob_loss_stress,
                    "desc": "市場変動が激しい上位10%のシナリオ"
                }

        # 2. インフレ/バリュー局面 (High HML Regime)
        # HML(Value)が高く、Marketが低い（または普通）シナリオ
        if 'HML' in factor_waves and 'Market' in factor_waves:
            hml_sum = np.sum(factor_waves['HML'], axis=0)
            mkt_sum = np.sum(factor_waves['Market'], axis=0)
            
            # バリューが市場をアウトパフォームしたシナリオ
            value_dominant_indices = np.where(hml_sum > mkt_sum)[0]
            
            if len(value_dominant_indices) > 0:
                avg_ret_val = np.mean(final_returns[value_dominant_indices])
                results["Inflation_Value_Regime"] = {
                    "avg_return": avg_ret_val,
                    "count": len(value_dominant_indices),
                    "desc": "バリュー株が市場平均を上回るシナリオ"
                }

        return results

    @staticmethod
    def generate_histogram_data(price_paths):
        """可視化用の分布データを生成"""
        final_prices = price_paths[-1, :]
        hist, bin_edges = np.histogram(final_prices, bins=50, density=True)
        return {
            "bins": bin_edges.tolist(),
            "frequency": hist.tolist(),
            "raw_final_prices": final_prices.tolist()
        }
