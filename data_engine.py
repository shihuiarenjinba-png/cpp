"""
data_engine.py
市場データの取得、ティッカーの正規化、および合成ポートフォリオの生成を行うモジュール
※ 修正版: タイムゾーン同期、生存銘柄による動的ウェイト再配分、カレンダーアライメントの徹底
※ 修正版(v2): FF5ファクターへの拡張、無リスク利子率(RF)の厳密分離
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import warnings
import streamlit as st

# 先ほど作成したconfigからMarketConfigを読み込む
from config import MarketConfig

# 📌 ログを埋め尽くすPandasの仕様変更警告をミュート
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 🛠️ データ取得 & ポートフォリオ合成クラス
# =========================================================
class DataFetcher:
    
    @staticmethod
    def normalize_weights(weights_dict):
        """
        ティッカー名の正規化と、ウェイトの自動リスケール（合計100%化）を行う。
        例: 'BRK-B' -> 'BRK.B' (Yahoo Finance形式)
        """
        if not weights_dict: return {}
        normalized = {}
        total = 0.0
        
        for k, v in weights_dict.items():
            # Yahoo Finance形式への正規化 (ハイフンをドットに変換)
            new_k = str(k).strip().upper().replace('-', '.') 
            normalized[new_k] = float(v)
            total += float(v)
        
        # 自動リスケール (100%基準に満たない・超える場合への安全装置)
        if total > 0 and abs(total - 100.0) > 1e-4:
            for k in normalized:
                normalized[k] = (normalized[k] / total) * 100.0
                
        return normalized

    @staticmethod
    @st.cache_data(ttl=3600)  # 1時間のキャッシュ化で通信API制限を大幅削減
    def fetch_market_data(tickers, start_date="2000-01-01"):
        """
        指定されたティッカーのヒストリカルデータを取得し、クレンジングする。
        """
        if not tickers: return pd.DataFrame()
        try:
            if isinstance(tickers, str): tickers = [tickers]
            
            # threads=False でマルチスレッドを無効化しAPI制限(Rate Limit)を回避
            data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, threads=False)
            
            if data.empty: return pd.DataFrame()
            
            # データ構造の確実なクレンジング（不要な階層の削ぎ落とし）
            # MultiIndex（複数階層）で返ってきた場合、純粋なティッカー名だけを残す
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    data = data['Close']
                elif 'Adj Close' in data.columns.get_level_values(0):
                    data = data['Adj Close']
            else:
                # 古い形式の場合の対応
                if 'Close' in data.columns:
                    data = data[['Close']]
                elif 'Adj Close' in data.columns:
                    data = data[['Adj Close']]
            
            # 1銘柄・複数銘柄の取得パターンの統一
            if isinstance(data, pd.Series):
                data = data.to_frame()
                
            # 列名が 'Close' 等の一般名詞になってしまっている場合は、ティッカー名に強制上書き
            if len(data.columns) == 1 and len(tickers) == 1:
                data.columns = [tickers[0]]
                
            # 列名を確実に文字列化
            data.columns = [str(c).strip() for c in data.columns]
            
            # タイムゾーンの完全剥奪 (Tz-Naive)
            # インデックスを「日付」のみに標準化し、結合時のズレを根絶する
            if data.index.tz is not None: 
                data.index = data.index.tz_localize(None)
            data.index = data.index.normalize()

            # 前日値で埋めて祝日等の不整合を吸収
            return data.ffill().dropna(how='all')
        except Exception as e:
            print(f"Fetch Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def validate_tickers(input_dict):
        """
        入力されたティッカーがYahoo Financeに存在するか1日分のデータで検証する。
        """
        valid_data = {}
        invalid_tickers = []
        
        input_dict = DataFetcher.normalize_weights(input_dict)
        for ticker, weight in input_dict.items():
            try:
                tick = yf.Ticker(ticker)
                hist = tick.history(period="1d") 
                if not hist.empty: 
                    valid_data[ticker] = weight
                else: 
                    invalid_tickers.append(ticker)
            except: 
                invalid_tickers.append(ticker)
                
        return valid_data, invalid_tickers

    @staticmethod
    def create_synthetic_portfolio(ticker_weights, region="US"):
        """
        各銘柄の生データを取得し、「生存銘柄のみ」による動的ウェイト再配分を行い、
        正確な合成ポートフォリオの累積リターンを算出する。
        """
        ticker_weights = DataFetcher.normalize_weights(ticker_weights)
        tickers = list(ticker_weights.keys())
        if not tickers: return None
        
        raw_prices = DataFetcher.fetch_market_data(tickers)
        if raw_prices.empty: return None
        
        # config.py からベンチマーク情報を動的に取得
        config = MarketConfig.get_config(region)
        bm_ticker = config["benchmark_ticker"]
        bm_prices = DataFetcher.fetch_market_data([bm_ticker])
        
        # 合成計算前の安全化（列名の確実な文字列化と次元統一）
        if isinstance(raw_prices, pd.Series):
            raw_prices = raw_prices.to_frame(name=tickers[0])
        if isinstance(bm_prices, pd.Series):
            bm_prices = bm_prices.to_frame(name=bm_ticker)

        # リターンの算出
        returns = raw_prices.pct_change()
        bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")
        
        # カレンダー・アライメント
        # ベンチマークの営業日カレンダーにポートフォリオ側を強制適合させる
        aligned_returns = returns.reindex(bm_returns.index)
        
        # 「生存銘柄のみ」による動的重み再配分 (Survivor Weighting)
        w_series = pd.Series(ticker_weights) / 100.0
        is_alive = aligned_returns.notna()
        active_weights = is_alive.multiply(w_series, axis=1)
        weight_sums = active_weights.sum(axis=1)
        
        # ゼロ割り防止（全銘柄が存在しない日はNaNにする）
        weight_sums = weight_sums.replace(0, np.nan)
        normalized_weights = active_weights.div(weight_sums, axis=0)
        
        # ポートフォリオのリターン = Σ(各銘柄リターン * 再正規化ウェイト)
        weighted_returns = (aligned_returns.fillna(0) * normalized_weights.fillna(0)).sum(axis=1)
        weighted_returns.loc[weight_sums.isna()] = np.nan
        weighted_returns = weighted_returns.dropna()
        
        if weighted_returns.empty: return None

        # 100をベースとした累積リターン(価格推移)を返す
        return (1 + weighted_returns).cumprod() * 100

    @staticmethod
    @st.cache_data(ttl=86400)
    # 💡【修正】デフォルトをFF5（5ファクター）のデータセットに変更
    def fetch_fama_french_factors(start_date, end_date=None, dataset_name="F-F_Research_Data_5_Factors_2x3"):
        """
        Fama-Frenchの5ファクターデータ（Mkt-RF, SMB, HML, RMW, CMA）と
        無リスク金利（RF）をKenneth Frenchのライブラリから取得する。
        """
        try:
            ff_dict = web.DataReader(dataset_name, 'famafrench', start=start_date, end=end_date)
            if not ff_dict: return pd.DataFrame()
            
            # 日米で異なるインデックス構造を安全に日付型へ変換（％表示を実数に変換）
            ff_data = ff_dict[0] / 100.0
            ff_data.index = ff_data.index.to_timestamp()
            
            # タイムゾーンの完全剥奪 (Tz-Naive)
            if ff_data.index.tz is not None: 
                ff_data.index = ff_data.index.tz_localize(None)
                
            ff_data.columns = [c.strip() for c in ff_data.columns]
            
            # 💡【修正】無リスク利子率（RF）の確実な分離と保証
            if 'RF' not in ff_data.columns:
                # 取得データにRFが含まれない場合の安全装置（実務上はほぼ無いが堅牢化のため）
                ff_data['RF'] = 0.0
            
            return ff_data
        except Exception as e:
            print(f"FF Data Error: {e}")
            return pd.DataFrame()
