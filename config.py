"""
config.py
市場環境、ベンチマーク、およびシステム全体の静的設定を管理するモジュール
※ 修正版(v5): FF5ファクターデータセットを「純粋な5ファクター（Daily）」に厳格指定
"""

# =========================================================
# 🌍 市場設定管理クラス
# =========================================================
class MarketConfig:
    """
    地域ごとの参照インデックスやファクターデータセットを定義するクラス。
    ここで定義されたティッカーを用いて、DataFetcherが市場データを取得する。
    """
    
    REGIONS = {
        "US": {
            "name": "United States",
            # 💡修正ポイント: 2x3ポートフォリオ(従属変数)などを排除し、純粋な5ファクター独立変数(Daily)を厳格指定
            "ff_dataset": "North_America_5_Factors_Daily",
            "benchmark_ticker": "^GSPC",   # S&P 500 (米国市場全体のベンチマーク)
            "risk_free_ticker": "^TNX",    # 米国10年国債利回り (無リスク資産)
            "vix_ticker": "^VIX",          # 米国VIX (恐怖指数)
            "fallback_rf_rate": 0.02       # データ取得失敗時の固定利回り (年利2%)
        },
        "Japan": {
            "name": "Japan",
            # 💡修正ポイント: 旧来のJapan_3_Factorsを廃止し、日本の純粋な5ファクター独立変数(Daily)を厳格指定
            "ff_dataset": "Japan_5_Factors_Daily",
            "benchmark_ticker": "^N225",   # 日経225 (日本市場全体のベンチマーク)
            "risk_free_ticker": "^JGBS",   # 日本の10年国債利回り
            "vix_ticker": "^JNIV",         # 日経VI (日本の恐怖指数)
            "fallback_rf_rate": 0.001      # 日本の超低金利環境を反映した固定利回り (年利0.1%)
        }
    }

    @staticmethod
    def get_config(region="US"):
        """
        指定された地域の市場設定辞書を取得する。
        未定義の地域が指定された場合は、デフォルトでUS（米国）の設定を返す。
        
        Args:
            region (str): "US" または "Japan"
            
        Returns:
            dict: 該当地域のティッカーやデータセット名を含む辞書
        """
        return MarketConfig.REGIONS.get(region, MarketConfig.REGIONS["US"])


# =========================================================
# 📝 用語マッピング・システム定数
# =========================================================
# カラム名の揺れ（米国 Mkt-RF vs 日本 Mkt 等）を吸収するためのエイリアスマッピング。
# UIと分析エンジンの両方で、どの名称が来ても統一した日本語表示ができるようにする。
FACTOR_TRANSLATION = {
    # 市場全体
    "Mkt-RF": "市場全体 (Market)",
    "Mkt": "市場全体 (Market)",
    "MKT": "市場全体 (Market)",
    "Market": "市場全体 (Market)",
    
    # 小型株効果
    "SMB": "小型株効果 (Size)",
    "Size": "小型株効果 (Size)",
    
    # 割安株効果
    "HML": "割安株効果 (Value)",
    "Value": "割安株効果 (Value)",
    
    # 収益性
    "RMW": "収益性 (Profitability)",
    "Profitability": "収益性 (Profitability)",
    
    # 投資態度
    "CMA": "投資態度 (Investment)",
    "Investment": "投資態度 (Investment)",
    
    # モメンタム (地域によってMom, WML等と表記が揺れる)
    "Mom": "モメンタム (Trend)",
    "WML": "モメンタム (Trend)",
    "Momentum": "モメンタム (Trend)",
    
    # 無リスク資産
    "RF": "無リスク資産 (Risk-Free)"
}

# 分析のデフォルト設定（将来的な一括変更を容易にするため）
DEFAULT_RISK_FREE_RATE = 0.02  # システム全体の絶対的なフォールバック無リスク利回り (年利2%)
TRADING_DAYS_PER_YEAR = 252    # 1年間の営業日数 (年率換算に使用)
