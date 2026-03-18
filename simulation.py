"""
simulation.py
ヒストリカル・ブートストラップ法を用いたシミュレーション、および過去の危機のタイムマシンテストを行うモジュール
【アップデート】過去危機のSurvivor Weighting（動的再配分）、ローリング回帰（動的エクスポージャー）、
および トラッキングエラー（TE）を反映した確率的シナリオ生成を実装。
※ 修正版(v20): 
   - 30日窓ブロック・ブートストラップ法への移行
   - 【修正 STEP 2】 期待値のセンタリング補正（伊藤のレンマに基づくボラティリティ・ドラッグの控除による過大評価の抑制）
   - 【修正 STEP 3】 長期の重力・平均回帰の導入（異常な上振れパスの抑制）
   - 【新規・修正】 CAGR上限(15%)の厳格化とNaN回避の安全装置、ベンチマークパスの独立生成
   - 【新規・修正】 インデックスのDatetime正規化とInner Joinによる厳格な日付紐付け（バグ修正）
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

        # --- 【修正】 日付ズレ防止 ---
        # インデックスを確実にDatetime型に変換し、タイムゾーンを消去
        daily_returns = raw_data.pct_change().dropna(how='all')
        daily_returns.index = pd.to_datetime(daily_returns.index).tz_localize(None)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 危機期間中のデータを抽出（Datetime型同士で正確に比較）
        mask = (daily_returns.index >= start_dt) & (daily_returns.index <= end_dt)
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
# 🎲 確率的シナリオ生成クラス (ブートストラップ版)
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_paths(returns, n_scenarios=10000, n_days=252, tracking_error_annual=0.0, target_annual_return=0.06, mean_reversion_strength=0.1):
        """
        過去の生データをそのままサンプリングする「30日窓ブロック・ブートストラップ法」。
        【修正】NaN回避のため、対数を取る前にリターンを -0.99 にクリッピングする安全装置を追加。
        """
        if returns is None or len(returns) < 30: return None
        
        # --- 【修正】 NaN対策（対数計算エラーの完全防止） ---
        # -1.0（-100%）以下のリターンが発生すると対数が計算できずNaNになるため、下限を-99%でクリップ
        safe_returns = np.clip(returns, -0.99, None)
        log_returns = np.log1p(safe_returns).to_numpy()
        n_history = len(log_returns)
        
        # --- 【修正】 期待値のセンタリング補正（過大評価の抑制） ---
        # 伊藤の補題に基づくボラティリティ・ドラッグ（分散の半分）を引くことで、
        # シミュレーションによる将来資産額が算術平均的に上振れ（現実離れして高く計算）するのを防ぐ
        variance_annual = np.var(log_returns) * TRADING_DAYS_PER_YEAR
        target_daily_log_ret = (np.log1p(target_annual_return) - 0.5 * variance_annual) / TRADING_DAYS_PER_YEAR
        
        historical_mean_log_ret = np.mean(log_returns)
        adjusted_log_returns = log_returns - historical_mean_log_ret + target_daily_log_ret

        # ブロック・ブートストラップの窓幅
        window_size = 30
        if n_history < window_size:
            window_size = 1
            
        n_blocks = int(np.ceil(n_days / window_size))
        max_start_idx = max(0, n_history - window_size)
        
        np.random.seed(42)  # 再現性
        random_start_indices = np.random.randint(0, max_start_idx + 1, size=(n_blocks, n_scenarios))
        
        simulated_log_returns = np.zeros((n_blocks * window_size, n_scenarios))
        current_cum_log_returns = np.zeros(n_scenarios)
        
        # 抽出したブロックをつなぎ合わせる
        for i in range(n_blocks):
            start_row = i * window_size
            end_row = start_row + window_size
            
            indices = random_start_indices[i, :]
            broadcast_indices = indices + np.arange(window_size)[:, None]
            
            block_log_returns = adjusted_log_returns[broadcast_indices]
            
            # --- 平均回帰（Mean Reversion）の適用 ---
            ideal_cum_log_return = start_row * target_daily_log_ret
            deviation = current_cum_log_returns - ideal_cum_log_return
            
            penalty_per_day = np.where(deviation > 0, deviation * mean_reversion_strength / window_size, 0)
            block_log_returns = block_log_returns - penalty_per_day
            
            simulated_log_returns[start_row:end_row, :] = block_log_returns
            current_cum_log_returns += np.sum(block_log_returns, axis=0)
        
        simulated_log_returns = simulated_log_returns[:n_days, :]
        
        # トラッキングエラー（運用ブレ）のノイズ追加
        if tracking_error_annual > 0:
            te_daily = tracking_error_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
            noise = np.random.normal(0, te_daily, size=(n_days, n_scenarios))
            simulated_log_returns += noise
            
        cumulative_returns = np.exp(np.cumsum(simulated_log_returns, axis=0))
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
        【修正】CAGR上限(15%)の厳格化、ゼロ除算防止、およびベンチマーク用パスの独立生成。
        """
        n_days = int(n_years * TRADING_DAYS_PER_YEAR)
        tracking_error_annual = 0.0

        # --- 【修正】 インデックスのDatetime正規化 ---
        # タイムゾーン等の違いによる結合時の日付ズレを防ぐ
        returns.index = pd.to_datetime(returns.index).tz_localize(None)

        # TEの自動計算ロジック
        if bm_returns is not None:
            bm_returns.index = pd.to_datetime(bm_returns.index).tz_localize(None)
            # inner joinで確実に日付が一致するデータのみを比較する
            aligned = pd.concat([returns.rename("Port"), bm_returns.rename("BM")], axis=1, join='inner').dropna()
            if len(aligned) > 30:
                active_ret = aligned["Port"] - aligned["BM"]
                tracking_error_annual = active_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # --- 【修正】 CAGRの計算とゼロ除算防止 ---
        valid_returns = returns.dropna()
        if len(valid_returns) > 0:
            cum_ret = (1 + valid_returns).prod()
            # len()ではなく非NaN数(count)を用いて正しい運用期間を算出する
            years = valid_returns.count() / TRADING_DAYS_PER_YEAR
            # 年数が極端に短い場合のゼロ除算エラーを防ぐため years > 0.1 を条件とする
            cagr = cum_ret ** (1 / years) - 1 if years > 0.1 else 0.0
        else:
            cagr = 0.0
            
        target_return = cagr
        alert_msg = None
        
        # --- 【修正】 CAGRの厳格な保守的キャップ処理（上限15%） ---
        MAX_CAGR_CAP = 0.15 
        if target_return > MAX_CAGR_CAP:
            conservative_return = MAX_CAGR_CAP
            alert_msg = f"【アラート】過去の実績CAGR（{target_return*100:.1f}%）が非現実的なため、将来予測には保守的シナリオ（{MAX_CAGR_CAP*100:.1f}%上限）を適用しました。"
            print(alert_msg) # バックエンドのログ用
            target_return = conservative_return

        # ポートフォリオのパス生成
        paths = StochasticScenarioGenerator.generate_paths(
            returns, 
            n_scenarios=n_scenarios, 
            n_days=n_days,
            tracking_error_annual=tracking_error_annual,
            target_annual_return=target_return,
            mean_reversion_strength=0.1
        )
        
        if paths is None: return None

        # --- 【追加】 ベンチマーク専用のシミュレーションパスを生成（グラフ分離用） ---
        bm_paths = None
        if bm_returns is not None:
            valid_bm_returns = bm_returns.dropna()
            if len(valid_bm_returns) > 30:
                bm_cum_ret = (1 + valid_bm_returns).prod()
                bm_years = valid_bm_returns.count() / TRADING_DAYS_PER_YEAR
                bm_cagr = bm_cum_ret ** (1 / bm_years) - 1 if bm_years > 0.1 else 0.0
                
                # ベンチマークも異常値が出ないよう同様にキャップをかける
                bm_target_return = min(bm_cagr, MAX_CAGR_CAP)
                
                bm_paths = StochasticScenarioGenerator.generate_paths(
                    valid_bm_returns, 
                    n_scenarios=n_scenarios, 
                    n_days=n_days,
                    tracking_error_annual=0.0, # ベンチマーク自身なのでTEは0
                    target_annual_return=bm_target_return,
                    mean_reversion_strength=0.1
                )
        # -------------------------------------------------------------
        
        final_values = paths[-1, :]
        
        # 中央値、ワースト5%、トップ5%などの抽出
        results = {
            "paths": paths,       # ポートフォリオの描画用パス
            "bm_paths": bm_paths, # 【追加】ベンチマークの描画用独立パス
            "median": np.percentile(final_values, 50),
            "worst_5th": np.percentile(final_values, 5),
            "worst_1st": np.percentile(final_values, 1),
            "best_5th": np.percentile(final_values, 95),
            "prob_loss": (final_values < 1.0).mean() * 100,
            "applied_cagr": target_return,
            "alert_message": alert_msg
        }
        return results
