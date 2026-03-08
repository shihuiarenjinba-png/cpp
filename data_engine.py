"""
data_engine.py
市場データの取得、ティッカーの正規化、および合成ポートフォリオの生成を行うモジュール
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
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]: data = data['Close']
                elif 'Adj Close' in data.columns.levels[0]: data = data['Adj Close']
            
            # タイムゾーンを削除してインデックスを「日付」のみに標準化（結合時のズレを防止）
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            data.index = data.index.normalize()

            # 前日値で埋めて歯抜けを防止（祝日等の不整合を吸収）
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
        各銘柄の生データを取得し、上場前期間を「ベンチマークリターン × ベータ値」で
        バックフィルした上で、合成ポートフォリオの累積リターンを算出する。
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
            # その時代に全く上場していなかった場合（完全欠損）のフォールバック
            if ticker not in aligned_data.columns or aligned_data[ticker].isna().all():
                aligned_data[ticker] = aligned_data["Benchmark"] 
                continue
            
            # 有効な重複期間からベータ値の算出
            valid_mask = aligned_data[ticker].notna() & aligned_data["Benchmark"].notna()
            if valid_mask.sum() > 30:
                cov = np.cov(aligned_data.loc[valid_mask, ticker], aligned_data.loc[valid_mask, "Benchmark"])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            else:
                beta = 1.0
            
            # 上場前の空白期間を「ベンチマークリターン * ベータ」でバックフィル生成
            missing_mask = aligned_data[ticker].isna()
            aligned_data.loc[missing_mask, ticker] = aligned_data.loc[missing_mask, "Benchmark"] * beta
        
        # ウェイトに基づく加重平均リターンの算出
        weighted_returns = pd.Series(0, index=aligned_data.index)
        for ticker, weight in ticker_weights.items():
            # 最後の細かい休場のズレは 0 (変動なし) として処理
            weighted_returns += aligned_data[ticker].fillna(0) * (weight / 100.0)
                
        # 100をベースとした累積リターン(価格推移)を返す
        return (1 + weighted_returns).cumprod() * 100

    @staticmethod
    @st.cache_data(ttl=86400)  # ファクターデータは更新頻度が低いため1日キャッシュ
    def fetch_fama_french_factors(start_date, end_date=None, dataset_name="F-F_Research_Data_Factors"):
        """
        Fama-FrenchのファクターデータをKenneth Frenchのライブラリから取得する。
        """
        try:
            ff_dict = web.DataReader(dataset_name, 'famafrench', start=start_date, end=end_date)
            if not ff_dict: return pd.DataFrame()
            
            # 日米で異なるインデックス構造を安全に日付型へ変換（％表示を実数に変換）
            ff_data = ff_dict[0] / 100.0
            ff_data.index = ff_data.index.to_timestamp()
            if ff_data.index.tz is not None: ff_data.index = ff_data.index.tz_localize(None)
            ff_data.columns = [c.strip() for c in ff_data.columns]
            
            return ff_data
        except Exception as e:
            print(f"FF Data Error: {e}")
            return pd.DataFrame()
