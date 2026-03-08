"""
simulation.py
GARCHモデル、t分布を用いたモンテカルロ・シミュレーション、および過去の危機のタイムマシンテストを行うモジュール
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import t
import warnings

# データ取得エンジンのインポート
from data_engine import DataFetcher
from config import TRADING_DAYS_PER_YEAR

# 📌 GARCHモデルによる動的ボラティリティ予測用
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 🌀 市場周期・レジーム解析クラス
# =========================================================
class RegimeAnalyzer:
    @staticmethod
    def detect_cycle(returns):
        """
        ウェルチ法（パワースペクトル密度）を用いて、市場のボラティリティ周期（レジーム）を検知する。
        """
        if len(returns) < 252: return None
        
        # 20日ローリングボラティリティを算出し、その波の周期を解析
        rolling_vol = returns.rolling(window=20).std().dropna()
        if len(rolling_vol) < 100: return None
        
        # 🚨【修正箇所】Pandas Seriesを純粋な1次元NumPy配列に変換してから渡す
        # これによりSciPy内部でのKeyError(tuple)を完全に回避
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
        
        # 対象期間のデータを少し広めに取得（計算の安定化のため）
        fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        
        tickers = list(DataFetcher.normalize_weights(weights_dict).keys())
        raw_data = DataFetcher.fetch_market_data(tickers, start_date=fetch_start)
        if raw_data.empty: return None
        
        # 合成ポートフォリオの生成（上場前銘柄のバックフィル含む）
        simulated_prices = DataFetcher.create_synthetic_portfolio(weights_dict, region=region)
        if simulated_prices is None: return None
        
        # 危機期間中のデータを抽出
        mask = (simulated_prices.index >= start_date) & (simulated_prices.index <= end_date)
        crisis_data = simulated_prices.loc[mask]
        
        if len(crisis_data) < 5: return None
        
        # ドローダウンの計算
        peak = crisis_data.cummax()
        drawdown = (crisis_data - peak) / peak
        max_dd = drawdown.min() * 100  # %表記
        
        return {
            "max_drawdown_pct": max_dd,
            "start_value": crisis_data.iloc[0],
            "end_value": crisis_data.iloc[-1]
        }

# =========================================================
# 🎲 確率的シナリオ生成クラス
# =========================================================
class StochasticScenarioGenerator:
    @staticmethod
    def generate_paths(returns, n_scenarios=10000, n_days=252):
        """
        GARCH(1,1)とt分布を組み合わせたモンテカルロ・シミュレーション。
        幾何ブラウン運動(GBM)の拡張版。
        """
        if len(returns) < 30: return None
        
        mu_daily = returns.mean()
        
        # 1. 現在のボラティリティ・レジームの取得（GARCHモデル）
        current_vol = returns.std()
        if HAS_ARCH and len(returns) > 252:
            try:
                # GARCH(1,1)でボラティリティ・クラスタリングをモデル化
                # scale=100 で計算を安定化させ、後で戻す
                am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
                res = am.fit(disp='off')
                forecasts = res.forecast(horizon=1)
                current_vol = np.sqrt(forecasts.variance.values[-1, :][0]) / 100.0
            except Exception as e:
                print(f"GARCH Error: {e}, falling back to standard std.")
        
        # 2. t分布による乱数生成（ファットテールの考慮）
        # 自由度(df) 5 は金融資産のリターン分布によく適合する
        df = 5 
        # t分布の分散は df/(df-2) になるため、標準偏差に合わせるようスケーリング
        scale_factor = np.sqrt((df - 2) / df) * current_vol
        
        # Z_t ~ t(df)
        random_shocks = t.rvs(df, loc=0, scale=scale_factor, size=(n_days, n_scenarios))
        
        # 3. シナリオパスの生成 (ベクトル化演算で高速化)
        # S_t = S_{t-1} * exp((mu - sigma^2/2) + shock)
        drift = mu_daily - (0.5 * current_vol**2)
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
    def run_projection(returns, n_scenarios=10000, n_years=1):
        """
        シナリオジェネレータを呼び出し、最終的なパーセンタイル値などを算出する。
        """
        n_days = int(n_years * TRADING_DAYS_PER_YEAR)
        paths = StochasticScenarioGenerator.generate_paths(returns, n_scenarios=n_scenarios, n_days=n_days)
        
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
