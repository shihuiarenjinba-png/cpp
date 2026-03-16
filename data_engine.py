"""
data_engine.py
市場データの取得、ティッカーの正規化、および合成ポートフォリオの生成を行うモジュール
※ 修正版(v13): 銘柄名正規化の徹底、インナージョインによる期間同期の強制、時価総額取得エラーの明示化を追加。
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import warnings
import streamlit as st
import time  # リトライ待機用に新規追加
import sqlite3 # DBキャッシュ用
import requests # FMP API通信用
import os

# 先ほど作成したconfigからMarketConfigを読み込む
from config import MarketConfig

# 📌 ログを埋め尽くすPandasの仕様変更警告をミュート
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 🛠️ データベース初期化・APIキー設定
# =========================================================
DB_PATH = "market_data.db"

def get_fmp_api_key():
    try:
        if hasattr(st, "secrets") and "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    return os.environ.get("FMP_API_KEY")

FMP_API_KEY = get_fmp_api_key()

def init_db():
    """SQLiteデータベースとテーブルの初期化"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS market_caps
                     (ticker TEXT PRIMARY KEY, market_cap REAL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Init Error: {e}")

# =========================================================
# 🛠️ データ取得 & ポートフォリオ合成クラス
# =========================================================
class DataFetcher:
    
    @staticmethod
    def normalize_weights(weights_dict):
        """
        ティッカー名の正規化と、ウェイトの自動リスケール（合計100%化）を行う。
        空白除去、大文字化、ドットのハイフン変換(BRK.B -> BRK-B)を統合して徹底する。
        """
        if not weights_dict: return {}
        normalized = {}
        total = 0.0
        
        for k, v in weights_dict.items():
            # 空白削除と大文字化
            new_k = str(k).strip().upper()
            
            # yfinance特有のクラス株表記(BRK.B -> BRK-B)への一律置換
            new_k = new_k.replace('.', '-')
            new_k = new_k.replace('_', '-')
            
            # ただし、日本株のサフィックス(-T)になってしまった場合は(.T)に修復する
            if new_k.endswith('-T'):
                new_k = new_k[:-2] + '.T'
                
            normalized[new_k] = float(v)
            total += float(v)
        
        # 自動リスケール (100%基準に満たない・超える場合への安全装置)
        if total > 0 and abs(total - 100.0) > 1e-4:
            for k in normalized:
                normalized[k] = (normalized[k] / total) * 100.0
                
        return normalized

    @staticmethod
    @st.cache_data(ttl=3600)  # 1時間のキャッシュ化で通信API制限を大幅削減
    def fetch_market_data(tickers, start_date="2000-01-01", max_retries=3):
        """
        指定されたティッカーのヒストリカルデータを取得し、クレンジングする。
        Rate Limit対策として、指数バックオフを伴うリトライロジックを追加。
        """
        if not tickers: return pd.DataFrame()
        if isinstance(tickers, str): tickers = [tickers]
        
        for attempt in range(max_retries):
            try:
                # threads=False でマルチスレッドを無効化しAPI制限(Rate Limit)を回避
                data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, threads=False)
                
                # 取得データが空の場合は一時的なブロックの可能性があるのでリトライへ
                if data.empty:
                    time.sleep(2 ** attempt)
                    continue
                
                # データ構造の確実なクレンジング（不要な階層の削ぎ落とし）
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
                data.columns = [str(c).strip().upper() for c in data.columns]
                
                # 💡【重要】タイムゾーンの完全剥奪と日付の正規化
                if data.index.tz is not None: 
                    data.index = data.index.tz_localize(None)
                data.index = pd.to_datetime(data.index).normalize()

                # ffillに制限(limit=5)を設け、上場廃止・取引停止銘柄が「永遠に同じ価格で生き残る」ことを防ぐ
                return data.ffill(limit=5).dropna(how='all')
                
            except Exception as e:
                # エラー発生時は待機して再試行 (例: 1秒 -> 2秒 -> 4秒)
                print(f"Fetch Error for {tickers} on attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
                
        # 全リトライ失敗時は空のDataFrameを返す
        return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=86400)  # 時価総額は日次キャッシュ(24時間)で十分
    def fetch_market_caps(tickers, region="US", **kwargs):
        """
        指定されたティッカーの時価総額(Market Cap)を取得する。
        DBキャッシュ -> yfinance -> FMP API の多層フォールバックで堅牢に取得。
        """
        if not tickers: return {}
        if isinstance(tickers, str): tickers = [tickers]
        
        init_db()
        fetched_caps = {}
        
        for ticker in tickers:
            mcap = None
            
            # 1. まずローカルDBキャッシュを確認（API呼び出しの節約と安定化）
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT market_cap FROM market_caps WHERE ticker=?", (ticker,))
                row = c.fetchone()
                conn.close()
                if row and row[0] is not None and row[0] > 0:
                    fetched_caps[ticker] = row[0]
                    continue # DBにあれば次へ
            except Exception as e:
                print(f"DB Read Error for {ticker}: {e}")

            # 2. キャッシュがなければ yfinance を試す（基本ルート）
            try:
                tick = yf.Ticker(ticker)
                mcap = tick.info.get('marketCap')
                if mcap is None or not isinstance(mcap, (int, float)) or mcap <= 0:
                    mcap = None
            except Exception as e:
                print(f"yfinance Market Cap Error for {ticker}: {e}")
                mcap = None

            # 3. yfinance がダメなら FMP API でフォールバック
            if mcap is None and FMP_API_KEY:
                try:
                    url = f"https://financialmodelingprep.com/api/v3/market-capitalization/{ticker}?apikey={FMP_API_KEY}"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0 and 'marketCap' in data[0]:
                            mcap = data[0]['marketCap']
                except Exception as e:
                    print(f"FMP API Error for {ticker}: {e}")

            # 4. 取得できた場合は結果を辞書に格納し、次回のためにDBへ保存
            if mcap is not None and isinstance(mcap, (int, float)) and mcap > 0:
                fetched_caps[ticker] = mcap
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    # 既に存在する場合は上書き更新 (INSERT OR REPLACE)
                    c.execute("INSERT OR REPLACE INTO market_caps (ticker, market_cap, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)", (ticker, float(mcap)))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"DB Write Error for {ticker}: {e}")
            else:
                print(f"Warning: Market Cap data missing or invalid for {ticker} across all sources.")
                
        # --- 取得結果の検証とエラー明示化 ---
        valid_caps = [float(v) for v in fetched_caps.values() if v > 0]
        
        if not valid_caps:
            # 全滅した場合は明確なエラーメッセージを画面に出す
            st.error("🚨 【エラー】全銘柄の時価総額データの取得に失敗しました。経済（時価総額加重）ポートフォリオは正確に計算されません。")
            fallback_cap = 1.0  # 計算クラッシュを防ぐため最低限の均等配分値を入れる
        else:
            fallback_cap = float(np.median(valid_caps))
            # 一部だけ取得できなかった場合は警告を出す
            if len(valid_caps) < len(tickers):
                st.warning(f"⚠️ 一部銘柄の時価総額が取得できなかったため、中央値による補完を行いました。")
        
        # 最終的に全ての要求ティッカーに対して確実な float 値を返す
        final_market_caps = {}
        for ticker in tickers:
            if ticker in fetched_caps and fetched_caps[ticker] > 0:
                final_market_caps[ticker] = float(fetched_caps[ticker])
            else:
                final_market_caps[ticker] = fallback_cap
                
        return final_market_caps

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
    def create_synthetic_portfolio(ticker_weights, region="US", rebalance_freq="M"):
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

        # 欠損値(NaN)のまま pct_change を計算させ、存在しない日のリターンを誤って0にしない (fill_method=None)
        returns = raw_prices.pct_change(fill_method=None)
        bm_returns = bm_prices.pct_change(fill_method=None).iloc[:, 0].rename("Benchmark")
        
        # 💡【重要】結合前に日付フォーマット(YYYY-MM-DD)とタイムゾーン(None)を強制的に揃える
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        returns.index = pd.to_datetime(returns.index).normalize()
        
        if bm_returns.index.tz is not None:
            bm_returns.index = bm_returns.index.tz_localize(None)
        bm_returns.index = pd.to_datetime(bm_returns.index).normalize()
        
        # 💡【重要】日付同期の強制化: ベンチマークとポートフォリオの共通期間のみを抽出 (Inner Join)
        aligned_df = pd.merge(returns, bm_returns, left_index=True, right_index=True, how='inner')
        
        # 結合後にデータがごっそり消えていないか監査
        if len(aligned_df) < 20:
            st.error(f"⚠️ データ結合後の有効日数が {len(aligned_df)} 日しかありません。期間設定やティッカーの地域設定が合っているか確認してください。")
            return None

        # 完全に日付が揃った状態でポートフォリオ側の列だけを抽出して計算に進む
        aligned_returns = aligned_df[tickers]
        
        # --- リバランス・ドリフト計算ロジック ---
        w_series = pd.Series(ticker_weights) / 100.0
        is_alive = aligned_returns.notna()
        # 💡 エラーハンドリング: 計算途中の欠損は0リターンとして扱い、グラフの不自然な切断を防ぐ
        clean_returns = aligned_returns.fillna(0)

        if rebalance_freq == "D":
            # 毎日リバランス（従来のドリフトなしロジック）
            active_weights = is_alive.multiply(w_series, axis=1)
            weight_sums = active_weights.sum(axis=1).replace(0, np.nan)
            normalized_weights = active_weights.div(weight_sums, axis=0)
            weighted_returns = (clean_returns * normalized_weights.fillna(0)).sum(axis=1)
            weighted_returns.loc[weight_sums.isna()] = np.nan
        else:
            # ドリフト（放置）を考慮するロジック (定期リバランス or Buy & Hold)
            rebalance_mask = pd.Series(False, index=aligned_returns.index)
            rebalance_mask.iloc[0] = True  # 運用初日は必ず設定

            # 定期リバランス日のマーキング
            if rebalance_freq == "M":
                period_ends = aligned_returns.resample('ME').last().index
                rebalance_mask.loc[aligned_returns.index.intersection(period_ends)] = True
            elif rebalance_freq == "Y":
                period_ends = aligned_returns.resample('YE').last().index
                rebalance_mask.loc[aligned_returns.index.intersection(period_ends)] = True
            
            # 生存銘柄の増減があった日も強制再配分（Survivor Weightingの維持）
            alive_changed = is_alive.astype(int).diff().fillna(0).abs().sum(axis=1) > 0
            rebalance_mask = rebalance_mask | alive_changed

            # 期間ごとのID（リバランスのたびにインクリメント）
            period_ids = rebalance_mask.cumsum()

            # 期間ごとの累積乗数 (Drift) を計算
            gross_returns = 1 + clean_returns
            drift_multipliers = gross_returns.groupby(period_ids).cumprod()

            # 期首のターゲットウェイト（その時点で生存している銘柄のみで100%になるよう再正規化）
            target_weights = is_alive.multiply(w_series, axis=1)
            target_weights = target_weights.div(target_weights.sum(axis=1).replace(0, np.nan), axis=0)

            # 各日における「その期間の期首ウェイト」を前方に埋める(ffill)
            period_start_weights = target_weights.where(rebalance_mask).ffill()

            # ドリフト計算：各日の「開始時点」でのウェイトを出すため、前日の累積乗数を使用
            prev_drift = drift_multipliers.shift(1)
            prev_drift[rebalance_mask] = 1.0  # リバランス日はドリフトをリセット
            prev_drift.iloc[0] = 1.0

            # 価格変動で崩れた後の未正規化ウェイト
            unnormalized_w = period_start_weights * prev_drift

            # ドリフト後の正規化ウェイト（現実のポートフォリオ内の保有比率）
            actual_weights = unnormalized_w.div(unnormalized_w.sum(axis=1), axis=0)

            # ポートフォリオの日次リターン = Σ( 当日リターン * 当日開始時点のドリフト後ウェイト )
            weighted_returns = (clean_returns * actual_weights.fillna(0)).sum(axis=1)
            
            # 全滅している日はNaNにする
            weighted_returns.loc[is_alive.sum(axis=1) == 0] = np.nan
        
        weighted_returns = weighted_returns.dropna()
        if weighted_returns.empty: return None

        # 100をベースとした累積リターン(価格推移)を返す
        return (1 + weighted_returns).cumprod() * 100

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_fama_french_factors(start_date, end_date=None, dataset_name="North_America_5_Factors_Daily"):
        """
        Fama-Frenchの純粋な5ファクターデータ（Mkt-RF, SMB, HML, RMW, CMA）と
        無リスク金利（RF）をKenneth Frenchのライブラリから取得する。
        """
        try:
            # 5ファクターデータセットの取得を試行
            ff_dict = web.DataReader(dataset_name, 'famafrench', start=start_date, end=end_date)
            if not ff_dict: 
                print(f"Warning: Empty dictionary returned for dataset {dataset_name}")
                return pd.DataFrame()
            
            # データ辞書の最初の要素（通常は日次または月次ファクター）を取得
            ff_data = ff_dict[0]
            
            # カラムの確実な抽出と正規化
            ff_data.columns = [c.strip() for c in ff_data.columns]
            
            # 必要なカラムが含まれているか確認 (Mkt-RF or Mkt, SMB, HML, RMW, CMA, RF)
            required_cols = ['SMB', 'HML', 'RMW', 'CMA', 'RF']
            # MktはMkt-RFだったりMktだったりするので柔軟に対応
            has_mkt = any('MKT' in c.upper() for c in ff_data.columns)
            
            if not has_mkt or not all(any(req in c.upper() for c in ff_data.columns) for req in required_cols):
                print(f"Warning: Dataset {dataset_name} does not contain all required 5 factors and RF.")
            
            # スケール統一の厳格化
            # Ken Frenchのデータは原則パーセント（1.0 = 1%）。
            # 最大値が0.5を超える場合、パーセント表記とみなして100で割り小数(0.01 = 1%)に統一する
            if ff_data.max().max() > 0.5:
                ff_data = ff_data / 100.0
            
            # 💡【重要】インデックスの厳密な正規化 (予測モデルとの完全同期のため)
            if isinstance(ff_data.index, pd.PeriodIndex):
                ff_data.index = ff_data.index.to_timestamp()
            else:
                ff_data.index = pd.to_datetime(ff_data.index)
            
            # タイムゾーンの完全剥奪と時刻リセット
            ff_data.index = pd.to_datetime(ff_data.index).normalize()
            if ff_data.index.tz is not None: 
                ff_data.index = ff_data.index.tz_localize(None)
            
            # 無リスク利子率（RF）の確実な分離と保証（万が一欠損していた場合）
            if not any('RF' in c.upper() for c in ff_data.columns):
                ff_data['RF'] = 0.0
            
            # エラーハンドリング: ファクターデータの欠損を前後の値で線形補間して埋める
            return ff_data.interpolate(method='linear').ffill().bfill()
            
        except Exception as e:
            # エラー時は何が失敗したのかログに残す
            print(f"FF Data Error for dataset '{dataset_name}': {e}")
            return pd.DataFrame()
