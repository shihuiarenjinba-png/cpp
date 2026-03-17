"""
data_engine.py
市場データの取得、ティッカーの正規化、および合成ポートフォリオの生成を行うモジュール
※ 修正版(v15): 複雑な補完計算を削除。「オール・オア・ナッシング」ロジックを導入し、時価総額が1つでも欠損した場合は即座に株価平均に切り替える堅牢な設計に変更。
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import warnings
import streamlit as st
import time  # リトライ待機用
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
        """
        if not tickers: return pd.DataFrame()
        if isinstance(tickers, str): tickers = [tickers]
        
        batch_size = 50
        all_data = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            batch_data = pd.DataFrame()
            
            for attempt in range(max_retries):
                try:
                    # threads=False でマルチスレッドを無効化しAPI制限(Rate Limit)を回避
                    data = yf.download(batch_tickers, start=start_date, progress=False, auto_adjust=True, threads=False)
                    
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
                        if 'Close' in data.columns:
                            data = data[['Close']]
                        elif 'Adj Close' in data.columns:
                            data = data[['Adj Close']]
                    
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                        
                    if len(data.columns) == 1 and len(batch_tickers) == 1:
                        data.columns = [batch_tickers[0]]
                        
                    data.columns = [str(c).strip().upper() for c in data.columns]
                    
                    if data.index.tz is not None: 
                        data.index = data.index.tz_localize(None)
                    data.index = pd.to_datetime(data.index).normalize()

                    batch_data = data.ffill(limit=5).dropna(how='all')
                    break  # 成功したらリトライループを抜ける
                    
                except Exception as e:
                    print(f"Fetch Error for batch {i} on attempt {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
                    
            if not batch_data.empty:
                all_data.append(batch_data)
            
            # バッチ間に少し待機時間を設け、APIからの完全ブロックを防ぐ
            if i + batch_size < len(tickers):
                time.sleep(0.5)
                
        if not all_data:
            return pd.DataFrame()
            
        # 全バッチを列方向に結合して一つにする
        final_data = pd.concat(all_data, axis=1)
        # 重複してしまった列名があれば除外
        final_data = final_data.loc[:, ~final_data.columns.duplicated()]
        return final_data

    @staticmethod
    @st.cache_data(ttl=86400)  # 日次キャッシュ(24時間)
    def fetch_market_caps(tickers, region="US", **kwargs):
        """
        指定されたティッカーの時価総額(Market Cap)を取得する。
        💡 修正内容: オール・オア・ナッシング方式。
        1つでも時価総額が欠落している場合、全体を「株価」ベースに切り替える。
        """
        if not tickers: return {}
        if isinstance(tickers, str): tickers = [tickers]
        
        init_db()
        fetched_caps = {}
        
        # --- 💡 1. 準備: 万が一のフォールバック用に株価(Price)を取得しておく ---
        latest_prices = {}
        try:
            recent_start = (pd.Timestamp.today() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            recent_data = DataFetcher.fetch_market_data(tickers, start_date=recent_start)
            if not recent_data.empty:
                latest_series = recent_data.ffill().iloc[-1]
                for t in tickers:
                    if t in latest_series and pd.notna(latest_series[t]) and latest_series[t] > 0:
                        latest_prices[t] = float(latest_series[t])
        except Exception as e:
            print(f"Error fetching recent prices for fallback: {e}")

        # --- 💡 2. FMP APIで一括(バルク)取得を試みる ---
        if FMP_API_KEY:
            batch_size = 20
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]
                ticker_str = ",".join(batch_tickers)
                try:
                    url = f"https://financialmodelingprep.com/api/v3/market-capitalization/{ticker_str}?apikey={FMP_API_KEY}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data:
                            if 'symbol' in item and 'marketCap' in item and item['marketCap'] > 0:
                                fetched_caps[item['symbol'].upper()] = item['marketCap']
                except Exception as e:
                    print(f"FMP API Batch Error: {e}")
                time.sleep(0.2) 
        
        # --- 💡 3. 残りの銘柄をDBキャッシュとyfinanceで穴埋め ---
        for ticker in tickers:
            if ticker in fetched_caps: continue
                
            mcap = None
            # DBの確認
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT market_cap FROM market_caps WHERE ticker=?", (ticker,))
                row = c.fetchone()
                conn.close()
                if row and row[0] is not None and row[0] > 0:
                    fetched_caps[ticker] = row[0]
                    continue
            except Exception: pass

            # yfinanceへの個別リクエスト
            try:
                tick = yf.Ticker(ticker)
                mcap = tick.info.get('marketCap')
                if mcap is None or not isinstance(mcap, (int, float)) or mcap <= 0:
                    mcap = None
            except Exception: pass

            if mcap is not None:
                fetched_caps[ticker] = mcap
                
        # 取得できた分をDBへ保存
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            for t, mc in fetched_caps.items():
                c.execute("INSERT OR REPLACE INTO market_caps (ticker, market_cap, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)", (t, float(mc)))
            conn.commit()
            conn.close()
        except Exception: pass

        # --- 💡 4. 【新機能】オール・オア・ナッシングの切り替え判定 ---
        missing_tickers = [t for t in tickers if t not in fetched_caps]
        
        if not missing_tickers:
            # 全ての時価総額が揃った場合 → 理想の【時価総額加重】
            return fetched_caps
        else:
            # 1つでも欠落がある場合 → 強制的に【株価加重(株価平均)】へ切り替え（混ぜない）
            missing_names = ", ".join(missing_tickers[:3])
            if len(missing_tickers) > 3: missing_names += " 等"
            st.warning(f"⚠️ 一部の銘柄（{missing_names}）の時価総額が取得できないため、ポートフォリオ全体を**【株価平均（Price-Weighted）】**に切り替えて計算します。")
            
            fallback_weights = {}
            for t in tickers:
                if t in latest_prices:
                    fallback_weights[t] = latest_prices[t]
                else:
                    # 株価すら取れない極端なエラー時の最終安全装置
                    fallback_weights[t] = 1.0
            
            # 値として株価の配列を返すことで、呼び出し元で自動的に「株価加重」として処理される
            return fallback_weights

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
