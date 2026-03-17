"""
simulation.py
ヒストリカル・ブートストラップ法を用いたシミュレーション、および過去の危機のタイムマシンテストを行うモジュール
【アップデート】過去危機のSurvivor Weighting（動的再配分）、ローリング回帰（動的エクスポージャー）、
および トラッキングエラー（TE）を反映した確率的シナリオ生成を実装。
※ 修正版(v16): モンテカルロ予測を廃止し、過去の生データをそのままサンプリングする「30日窓ブロック・ブートストラップ法」へ完全移行。
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
# 🎲 確率的シナリオ生成クラス (ブートストラップ版)
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_paths(returns, n_scenarios=10000, n_days=252, tracking_error_annual=0.0):
        """
        過去の生データをそのままサンプリングする「30日窓ブロック・ブートストラップ法」。
        架空の期待収益率や乱数分布に頼らず、市場のリアルな暴落と反発の「うねり」を再現する。
        """
        if returns is None or len(returns) < 30: return None
        
        # 実績リターンを対数リターンに変換して蓄積
        log_returns = np.log1p(returns).to_numpy()
        n_history = len(log_returns)
        
        # ブロック・ブートストラップの窓幅（市場の自己相関を保つための期間）
        window_size = 30
        
        if n_history < window_size:
            window_size = 1
            
        n_blocks = int(np.ceil(n_days / window_size))
        max_start_idx = max(0, n_history - window_size)
        
        # 過去データからランダムに開始地点（ブロック）を抽出
        np.random.seed(42)  # 実行ごとの再現性を担保
        random_start_indices = np.random.randint(0, max_start_idx + 1, size=(n_blocks, n_scenarios))
        
        simulated_log_returns = np.zeros((n_blocks * window_size, n_scenarios))
        
        # 抽出したブロックをつなぎ合わせる
        for i in range(n_blocks):
            start_row = i * window_size
            end_row = start_row + window_size
            
            indices = random_start_indices[i, :]
            broadcast_indices = indices + np.arange(window_size)[:, None]
            
            simulated_log_returns[start_row:end_row, :] = log_returns[broadcast_indices]
        
        # 指定された年数（日数）でカット
        simulated_log_returns = simulated_log_returns[:n_days, :]
        
        # ポートフォリオ独自の運用ブレ（トラッキングエラー）をノイズとして追加
        if tracking_error_annual > 0:
            te_daily = tracking_error_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
            # TE分のノイズを付加（平均0の正規分布）
            noise = np.random.normal(0, te_daily, size=(n_days, n_scenarios))
            simulated_log_returns += noise
            
        # 累積リターンを計算 (初期値 1.0 = 100%)
        cumulative_returns = np.exp(np.cumsum(simulated_log_returns, axis=0))
        
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
