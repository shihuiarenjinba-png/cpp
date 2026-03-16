"""
simulation.py
GARCHモデル、t分布を用いたモンテカルロ・シミュレーション、および過去の危機のタイムマシンテストを行うモジュール
【アップデート】過去危機のSurvivor Weighting（動的再配分）、GARCH最尤法（MLE）、ローリング回帰（動的エクスポージャー）、
および トラッキングエラー（TE）を反映した確率的シナリオ生成を実装。
※ 修正版(v13): モンテカルロのドリフト項を幾何平均(対数リターン)ベースに修正し、長期シミュレーションの異常発散を防止。
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import t
import statsmodels.api as sm
import warnings

# データ取得エンジン等のインポート
from data_engine import DataFetcher
from config import TRADING_DAYS_PER_YEAR, MarketConfig

# 📌 GARCHモデルによる動的ボラティリティ予測用
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 🔄 動的ファクター・エクスポージャー解析クラス (ローリング回帰)
# =========================================================
class DynamicFactorAnalyzer:
    @staticmethod
    def calculate_rolling_exposure(target_series, region="US", window_months=36):
        """
        過去36ヶ月の窓をスライドさせながら、超過収益率によるFF5回帰を繰り返す。
        因果的安定性（危機時にファクター関係が崩壊していないか、シンプソンのパラドックスが起きていないか）を検証する。
        """
        if target_series is None or target_series.empty: return None
        
        try:
            # ターゲットを対数リターンで月次化
            target_monthly = np.log(target_series.resample('ME').last() / target_series.resample('ME').last().shift(1)).dropna()
            
            if len(target_monthly) < window_months + 1: return None
            
            start_date = target_monthly.index[0].strftime('%Y-%m-%d')
            config = MarketConfig.get_config(region)
            ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
            
            if ff_data is None or ff_data.empty: return None
            
            # インデックスの型を明示的にDatetimeに変換してからPeriodにすることで、マージの空振りを防ぐ
            target_monthly.index = pd.to_datetime(target_monthly.index).to_period('M')
            ff_data.index = pd.to_datetime(ff_data.index).to_period('M')
            
            # Inner Mergeによる完全一致データの抽出
            combined = pd.merge(target_monthly.to_frame(name="Target"), ff_data, left_index=True, right_index=True, how='inner')
            
            if len(combined) < window_months + 1: return None
            
            # カラム取得を安全な関数に切り出し、[0]での IndexError (即死) を回避
            def _get_col(search_terms):
                cols = [c for c in combined.columns if any(term in c.upper() for term in search_terms)]
                return cols[0] if cols else None

            mkt = _get_col(['MKT'])
            smb = _get_col(['SMB'])
            hml = _get_col(['HML'])
            rmw = _get_col(['RMW'])
            cma = _get_col(['CMA'])
            rf  = _get_col(['RF', 'RISKFREE'])

            # 市場要因か無リスク利回りのカラムが見つからない場合は計算不可として安全に終了
            if not mkt or not rf:
                return None

            # 超過収益率（Y）と説明変数（X）の設定
            y = combined["Target"] - combined[rf]
            
            # 存在するファクターのみを動的に追加（日本市場の3ファクター等にも対応）
            factors = [mkt]
            if smb: factors.append(smb)
            if hml: factors.append(hml)
            if rmw: factors.append(rmw)
            if cma: factors.append(cma)
            
            X = combined[factors]
            X = sm.add_constant(X)
            
            rolling_results = []
            
            # 【手動ローリングOLS】 各ウィンドウで厳密に Adjusted R2 と P値を算出し続ける
            for i in range(window_months, len(combined) + 1):
                y_win = y.iloc[i - window_months : i]
                X_win = X.iloc[i - window_months : i]
                
                try:
                    model = sm.OLS(y_win, X_win).fit()
                    row_data = {
                        "Date": combined.index[i - 1].to_timestamp(), # 窓の「最終月」を日付として記録
                        "Market_Beta": model.params.get(mkt, np.nan),
                        "Size_Beta": model.params.get(smb, np.nan) if smb else np.nan,
                        "Value_Beta": model.params.get(hml, np.nan) if hml else np.nan,
                        "Quality_Beta": model.params.get(rmw, np.nan) if rmw else np.nan,
                        "Invest_Beta": model.params.get(cma, np.nan) if cma else np.nan,
                        "Alpha": model.params.get("const", 0.0) * 12, # 年率化アルファ
                        "Adjusted_R2": model.rsquared_adj * 100
                    }
                    rolling_results.append(row_data)
                except:
                    # 逆行列が計算できない等、特異な期間はスキップ
                    pass
            
            if not rolling_results: return None
            
            # DataFrame化して返す
            df_rolling = pd.DataFrame(rolling_results).set_index("Date")
            return df_rolling
            
        except Exception as e:
            print(f"Rolling Exposure Error: {e}")
            return None


# =========================================================
# 🌀 市場周期・レジーム解析クラス
# =========================================================
class RegimeAnalyzer:
    @staticmethod
    def detect_cycle(returns):
        """
        ウェルチ法（パワースペクトル密度）を用いて、市場のボラティリティ周期（レジーム）を検知する。
        """
        if returns is None or len(returns) < 252: return None
        
        # 20日ローリングボラティリティを算出し、その波の周期を解析
        rolling_vol = returns.rolling(window=20).std().dropna()
        if len(rolling_vol) < 100: return None
        
        # Pandas Seriesを純粋な1次元NumPy配列に変換してから渡す
        freqs, psd = signal.welch(rolling_vol.to_numpy().flatten())
        dominant_freq = freqs[np.argmax(psd)]
        
        # 周波数から周期（日数）へ変換
        cycle_days = int(1 / dominant_freq) if dominant_freq > 0 else 0
        return cycle_days


# =========================================================
# ⏳ 過去の危機リプレイクラス
# =========================================================
class HistoryTimeMachine:
    
    # 過去の主要な金融危機の期間定義
    CRISES = {
        "リーマン・ショック (2007-2009)": ("2007-10-01", "2009-03-31"),
        "コロナ・ショック (2020)": ("2020-02-19", "2020-03-23"),
        "ドットコム・バブル崩壊 (2000-2002)": ("2000-03-01", "2002-10-09")
    }
    
    @staticmethod
    def replay_crisis(weights_dict, crisis_name, region="US"):
        """
        現在のポートフォリオ構成のまま過去の暴落期にタイムスリップし、
        最大下落幅（ドローダウン）をシミュレーションする。
        """
        if crisis_name not in HistoryTimeMachine.CRISES: return None
        start_date, end_date = HistoryTimeMachine.CRISES[crisis_name]
        
        # 変化率計算のため、開始日より少し前（30日前）からデータを取得
        fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        
        normalized_weights = DataFetcher.normalize_weights(weights_dict)
        tickers = list(normalized_weights.keys())
        
        # 過去の生データを直接取得
        raw_data = DataFetcher.fetch_market_data(tickers, start_date=fetch_start)
        if raw_data is None or raw_data.empty: return None

        # 日次リターンを計算
        daily_returns = raw_data.pct_change().dropna(how='all')
        
        # 危機期間中のデータを抽出
        mask = (daily_returns.index >= start_date) & (daily_returns.index <= end_date)
        crisis_returns = daily_returns.loc[mask]
        
        if len(crisis_returns) < 5: return None
        
        # 動的ウェイト再配分（Survivor Weighting）ロジック
        w_series = pd.Series(normalized_weights)
        common_assets = [c for c in crisis_returns.columns if c in w_series.index]
        if not common_assets: return None
        
        crisis_returns = crisis_returns[common_assets]
        w_series = w_series[common_assets]
        
        # その日存在している（NaNではない）銘柄を判定
        is_alive = crisis_returns.notna()
        active_weights = is_alive.multiply(w_series, axis=1)
        
        # ウェイト合計が0になる日（全銘柄データなし）のゼロ除算を防ぐガード処理
        weight_sums = active_weights.sum(axis=1).replace(0, np.nan)
        normalized_active_weights = active_weights.div(weight_sums, axis=0)
        
        # ポートフォリオの合成リターンを計算
        port_ret = (crisis_returns.fillna(0) * normalized_active_weights.fillna(0)).sum(axis=1)
        port_ret.loc[weight_sums.isna()] = np.nan
        port_ret = port_ret.dropna()
        
        if len(port_ret) < 5: return None
        
        # ドローダウンと累積パフォーマンスの計算 (初期値を100とする)
        cumulative = 100.0 * (1 + port_ret).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min() * 100  # %表記
        
        return {
            "max_drawdown_pct": max_dd,
            "start_value": 100.0,
            "end_value": cumulative.iloc[-1]
        }


# =========================================================
# 🎲 確率的シナリオ生成クラス
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_paths(returns, n_scenarios=10000, n_days=252, tracking_error_annual=0.0):
        """
        GARCH(1,1)とt分布を組み合わせたモンテカルロ・シミュレーション。
        ポートフォリオ固有のトラッキングエラー（TE）を分散に加味する。
        【修正 G】期待リターンを算術平均から幾何平均(対数リターン)ベースに変更。
        """
        if returns is None or len(returns) < 30: return None
        
        # 💡 修正 G: 異常な発散を防ぐため、算術平均ではなく対数リターンの平均(幾何平均ベース)を使用
        log_returns = np.log(1 + returns)
        base_drift = log_returns.mean()
        
        # デフォルトのパラメータ
        current_vol = returns.std()
        df = 5.0 # デフォルトの自由度
        
        # 1. 現在のボラティリティ・レジームと分布の厚みをGARCHモデルから取得
        if HAS_ARCH and len(returns) > 252:
            try:
                # GARCHモデルの収束失敗（Iteration limit等）を防ぐため rescale=False 等の安全パラメータを追加
                am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', rescale=False)
                res = am.fit(disp='off', show_warning=False)
                
                # archのバージョンによる警告を防ぐため reindex=False を追加
                forecasts = res.forecast(horizon=1, reindex=False)
                
                # 直近の予測ボラティリティ (ilocを使用して安全にアクセス)
                current_vol = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0
                
                # 最尤推定されたt分布の自由度(nu)を取得 (ファットテールの厚み)
                if 'nu' in res.params:
                    df = res.params['nu']
            except Exception as e:
                # 収束しなかった場合は通常の標準偏差へ静かにフォールバック
                print(f"GARCH fallback triggered: {e}")
                pass
        
        # トラッキングエラー（アクティブリスク）を分散に加算
        # TE(年率)を日次化し、市場ボラティリティと合成 (σ_total^2 = σ_market^2 + σ_TE^2)
        te_daily = tracking_error_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
        adjusted_vol = np.sqrt(current_vol**2 + te_daily**2)
        
        # 自由度が2以下になると分散が無限大になるため、安全装置を設ける
        df = max(df, 2.1)
        
        # 2. t分布による乱数生成（ファットテールの考慮）
        # t分布の分散は df/(df-2) になるため、合成された標準偏差(adjusted_vol)に合わせるようスケーリング
        scale_factor = np.sqrt((df - 2) / df) * adjusted_vol
        
        # Z_t ~ t(df)
        random_shocks = t.rvs(df, loc=0, scale=scale_factor, size=(n_days, n_scenarios))
        
        # 3. シナリオパスの生成 (ベクトル化演算で高速化)
        # S_t = S_{t-1} * exp(drift + shock)
        # 💡 修正 G: すでに対数リターン平均(base_drift)に元のボラティリティ減価が含まれているため、
        # 追加で加味したTE分のペナルティ(- 0.5 * te_daily^2)のみを差し引く
        te_penalty = 0.5 * (te_daily**2)
        drift = base_drift - te_penalty
        
        daily_log_returns = drift + random_shocks
        
        # 累積リターンを計算 (初期値 1.0 = 100%)
        cumulative_returns = np.exp(np.cumsum(daily_log_returns, axis=0))
        
        # 初期値を先頭に追加
        starting_point = np.ones((1, n_scenarios))
        paths = np.vstack([starting_point, cumulative_returns])
        
        return paths


# =========================================================
# 🔮 プロジェクション統合クラス
# =========================================================
class ProjectionCore:
    @staticmethod
    def run_projection(returns, bm_returns=None, n_scenarios=10000, n_years=1):
        """
        シナリオジェネレータを呼び出し、TEを加味した上で最終的なパーセンタイル値などを算出する。
        """
        n_days = int(n_years * TRADING_DAYS_PER_YEAR)
        tracking_error_annual = 0.0

        # TEの自動計算ロジック
        if bm_returns is not None:
            # ポートフォリオとベンチマークの日付を同期して差分を計算
            aligned = pd.concat([returns.rename("Port"), bm_returns.rename("BM")], axis=1).dropna()
            if len(aligned) > 30:
                active_ret = aligned["Port"] - aligned["BM"]
                # 標準偏差を年率換算してTEとする
                tracking_error_annual = active_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # TEパラメータをジェネレータに渡す
        paths = StochasticScenarioGenerator.generate_paths(
            returns, 
            n_scenarios=n_scenarios, 
            n_days=n_days,
            tracking_error_annual=tracking_error_annual
        )
        
        if paths is None: return None
        
        final_values = paths[-1, :]
        
        # 中央値、ワースト5%、トップ5%などの抽出
        results = {
            "paths": paths, # 描画用に全パスを返す
            "median": np.percentile(final_values, 50),
            "worst_5th": np.percentile(final_values, 5),
            "worst_1st": np.percentile(final_values, 1),
            "best_5th": np.percentile(final_values, 95),
            "prob_loss": (final_values < 1.0).mean() * 100 # 元本割れ確率
        }
        return results
