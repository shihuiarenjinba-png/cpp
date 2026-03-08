"""
config.py
市場環境、ベンチマーク、およびシステム全体の静的設定を管理するモジュール
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
            "ff_dataset": "F-F_Research_Data_Factors",
            "benchmark_ticker": "^GSPC",   # S&P 500 (米国市場全体のベンチマーク)
            "risk_free_ticker": "^TNX",    # 米国10年国債利回り (無リスク資産)
            "vix_ticker": "^VIX"           # 米国VIX (恐怖指数)
        },
        "Japan": {
            "name": "Japan",
            "ff_dataset": "Japan_3_Factors",
            "benchmark_ticker": "^N225",   # 日経225 (日本市場全体のベンチマーク)
            "risk_free_ticker": "JPY=X",   # 日本株の場合は為替レート等をマクロ指標の代替に
            "vix_ticker": "^JNIV"          # 日経VI (日本の恐怖指数)
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
# これまで app.py に直接書かれていた翻訳辞書を config に分離。
# UIと分析エンジンの両方で統一した用語を使用するための基盤。
FACTOR_TRANSLATION = {
    "Mkt-RF": "市場全体 (Market)",
    "SMB": "小型株効果 (Size)",
    "HML": "割安株効果 (Value)",
    "RMW": "収益性 (Profitability)",
    "CMA": "投資態度 (Investment)",
    "Mom": "モメンタム (Trend)"
}

# 分析のデフォルト設定（将来的な一括変更を容易にするため）
DEFAULT_RISK_FREE_RATE = 0.02  # デフォルトの無リスク利回り (年利2%)
TRADING_DAYS_PER_YEAR = 252    # 1年間の営業日数 (年率換算に使用)
