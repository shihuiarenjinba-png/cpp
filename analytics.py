"""
analytics.py
ポートフォリオのリスク、リターン、ファクターエクスポージャー、およびリスク寄与度を計算するコア分析エンジン。
※修正版(v13): 回帰分析を「対数リターン」に完全統一し累積誤差を最小化。期待収益率にボラティリティ・ドラッグ(幾何近似)を導入。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 💡 VIF計算用
from scipy.stats import skew, kurtosis
import warnings
import plotly.graph_objects as go  # 💡 モンテカルロ可視化用に追加

# 分割した他のモジュールからのインポート
from config import MarketConfig, DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from data_engine import DataFetcher

# 📌 収縮推定（Shrinkage）による共分散計算用
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =========================================================
# 🕰️ タイムマシン・シミュレーション
# =========================================================
class HistoryTimeMachine:
    @staticmethod
    def calculate_historical_portfolio(returns_df, weights_dict):
        """
        過去の危機時などで「まだ存在しない銘柄」がある場合、
        データが存在する（生きている）銘柄だけでウェイトを100%に再正規化して計算する（Survivor Weighting）。
        """
        if returns_df.empty or not weights_dict:
            return pd.Series(dtype=float)

        df = returns_df.copy()
        w_series = pd.Series(weights_dict)

        # ポートフォリオに含まれる銘柄のみ抽出
        common_assets = [c for c in df.columns if c in w_series.index]
        if not common_assets:
            return pd.Series(dtype=float)

        df = df[common_assets]
        w_series = w_series[common_assets]

        # 生存フラグ（NaNでないならTrue）
        is_alive = df.notna()

        # 生きている銘柄のみに元のウェイトを掛ける
        active_weights = is_alive.multiply(w_series, axis=1)

        # その日の「生きている銘柄のウェイト合計」を算出
        weight_sums = active_weights.sum(axis=1)
        weight_sums = weight_sums.replace(0, np.nan) # 合計0（全滅）の日はNaNにする

        # ウェイトが100%になるように再正規化（動的リスケール）
        normalized_weights = active_weights.div(weight_sums, axis=0)

        # ポートフォリオリターン = Σ(各銘柄リターン * 正規化ウェイト)
        port_ret = (df.fillna(0) * normalized_weights.fillna(0)).sum(axis=1)

        # データが全く存在しない日はNaNに戻す
        port_ret.loc[weight_sums.isna()] = np.nan

        return port_ret


# =========================================================
# 📊 高度な統計指標計算クラス
# =========================================================
class AdvancedStats:
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, weights_dict=None, region="US"):
        """
        日次リターン系列から各種リスク指標とトラッキングエラー(TE)を計算する。
        """
        default_metrics = {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "omega": 0.0, "cvar_95": 0.0,
            "ulcer_index": 0.0, "kelly_criterion": 0.0, "tracking_error": 0.0, "info_ratio": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "hhi_index": 0.0, "effective_n": 1.0,
            "risk_penalty_ratio": 1.0, "volatility": 0.0, "raw_volatility": 0.0,
            "systematic_risk": 0.0, "idiosyncratic_risk": 0.0, "portfolio_beta": 1.0,
            "risk_contribution": {}
        }

        if returns.empty or returns.dropna().empty:
            return default_metrics
        
        try:
            weights_dict = DataFetcher.normalize_weights(weights_dict)
            
            # 1. 集中投資ペナルティ(HHI)
            hhi, penalty, enc = 0.0, 1.0, 0.0 
            if weights_dict is not None and sum(weights_dict.values()) > 0:
                weights = np.array(list(weights_dict.values())) / 100.0
                hhi = np.sum(weights**2)
                enc = 1.0 / hhi if hhi > 0 else 1.0
                penalty = 1.0 + (hhi ** 0.8) * 0.5 
                
            # 2. リスク（ボラティリティ）とリスク寄与度の計算
            ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)
            # 統計学的に正確な対数リターンへ変換 ln(1 + R)
            log_returns = np.log1p(returns.dropna())
            raw_sigma = log_returns.std() * ann_factor
            lw_sigma = raw_sigma
            risk_contribution = {}
            
            if weights_dict:
                try:
                    tickers = list(weights_dict.keys())
                    raw_comp = DataFetcher.fetch_market_data(tickers)
                    if not raw_comp.empty:
                        if isinstance(raw_comp, pd.Series):
                            raw_comp = raw_comp.to_frame(name=tickers[0])
                        
                        # 各構成銘柄の対数リターン ln(P_t / P_t-1)
                        comp_log_returns = np.log(raw_comp / raw_comp.shift(1)).fillna(0.0)
                        
                        valid_tickers = [t for t in tickers if t in comp_log_returns.columns]
                        if len(valid_tickers) > 0:
                            comp_log_returns = comp_log_returns[valid_tickers]
                            new_weights = np.array([weights_dict[t] for t in valid_tickers])
                            new_weights = new_weights / np.sum(new_weights)
                            
                            # Shrinkage共分散行列の採用 (Ledoit-Wolf)
                            if HAS_SKLEARN and len(comp_log_returns) > 10:
                                lw = LedoitWolf().fit(comp_log_returns)
                                cov_matrix = lw.covariance_
                            else:
                                cov_matrix = comp_log_returns.cov().values
                                
                            port_var = np.dot(new_weights.T, np.dot(cov_matrix, new_weights))
                            if port_var > 0:
                                lw_sigma = np.sqrt(port_var) * ann_factor
                                mcr = np.dot(cov_matrix, new_weights) / np.sqrt(port_var)
                                component_risk = new_weights * mcr
                                percentage_risk = component_risk / np.sqrt(port_var)
                                risk_contribution = dict(zip(valid_tickers, percentage_risk))
                except Exception as e:
                    print(f"Risk Contribution Calculation Fallback: {e}")
            
            sigma = lw_sigma * penalty 
            
            # 💡 【修正】ボラティリティ・ドラッグの算入 (幾何平均近似)
            # 単純な算術平均から、ボラティリティによる目減り分(0.5 * σ^2)を差し引いて期待収益率(mu)を現実化
            arithmetic_mu = returns.dropna().mean() * TRADING_DAYS_PER_YEAR
            mu = arithmetic_mu - 0.5 * (sigma ** 2) 
            
            # 最大ドローダウン計算 (内部のリスク指標計算用であり、外部へ時系列データとしては出力しない)
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            current_max_dd = drawdown.min()
            
            ulcer_sq = (drawdown ** 2).mean()
            ulcer_index = np.sqrt(ulcer_sq)
            
            risk_free_rate = DEFAULT_RISK_FREE_RATE 
            sharpe = (mu - risk_free_rate) / sigma if sigma > 0 else 0
            
            downside_returns = log_returns[log_returns < 0]
            downside_dev = downside_returns.std() * ann_factor * penalty if not downside_returns.empty else 0
            sortino = (mu - risk_free_rate) / downside_dev if downside_dev > 0 else 0
            
            calmar = mu / abs(current_max_dd) if current_max_dd != 0 else 0
            
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            omega = gains / losses if losses > 0 else np.inf
            
            var_95 = np.percentile(log_returns.dropna(), 5) * penalty 
            cvar_95 = log_returns[log_returns <= var_95].mean() * penalty if len(log_returns[log_returns <= var_95]) > 0 else 0
            
            kelly = mu / (sigma ** 2) if sigma > 0 else 0
            
            # --- トラッキングエラー(TE)算出ロジック ---
            info_ratio = 0.0
            tracking_error = 0.0
            systematic_risk, idiosyncratic_risk, portfolio_beta = 0.0, 0.0, 1.0
            
            try:
                # ベンチマークが渡されていない場合は取得を試みる
                if benchmark_returns is None:
                    config = MarketConfig.get_config(region)
                    bm_prices = DataFetcher.fetch_market_data([config["benchmark_ticker"]])
                    if not bm_prices.empty:
                        bm_prices = bm_prices.to_frame() if isinstance(bm_prices, pd.Series) else bm_prices
                        benchmark_returns = bm_prices.pct_change().iloc[:, 0].dropna()
                        
                if benchmark_returns is not None and not benchmark_returns.empty:
                    log_ret_df = log_returns.to_frame(name="Port")
                    log_bm_df = np.log1p(benchmark_returns.dropna()).to_frame(name="BM")
                    
                    # 日付・タイムゾーンの厳密な同期
                    if log_ret_df.index.tz is not None: log_ret_df.index = log_ret_df.index.tz_localize(None)
                    if log_bm_df.index.tz is not None: log_bm_df.index = log_bm_df.index.tz_localize(None)
                    log_ret_df.index = log_ret_df.index.normalize()
                    log_bm_df.index = log_bm_df.index.normalize()
                    
                    # インナー結合による日付完全一致データの抽出
                    aligned = pd.merge(log_ret_df, log_bm_df, left_index=True, right_index=True, how='inner')
                    
                    if len(aligned) > 30:
                        # 共分散とベータの計算
                        cov_bm = np.cov(aligned["Port"], aligned["BM"])
                        if cov_bm[1, 1] > 0:
                            portfolio_beta = cov_bm[0, 1] / cov_bm[1, 1]
                            sys_var = (portfolio_beta ** 2) * cov_bm[1, 1]
                            idio_var = max(0, cov_bm[0, 0] - sys_var)
                            systematic_risk = np.sqrt(sys_var) * ann_factor
                            idiosyncratic_risk = np.sqrt(idio_var) * ann_factor
                            
                            # アクティブ・リターンとTEの計算
                            active_ret = aligned["Port"] - aligned["BM"]
                            tracking_error = active_ret.std() * ann_factor
                            
                            # インフォメーション・レシオ (IR) の計算
                            if tracking_error > 0:
                                info_ratio = (active_ret.mean() * TRADING_DAYS_PER_YEAR) / tracking_error
            except Exception as e:
                print(f"Risk Separation Fallback: {e}")

            clean_returns = log_returns.dropna()
            
            return {
                "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
                "omega": omega, "cvar_95": cvar_95, "ulcer_index": ulcer_index,
                "kelly_criterion": kelly, "tracking_error": tracking_error, "info_ratio": info_ratio,
                "skewness": skew(clean_returns) if len(clean_returns) > 3 else 0, 
                "kurtosis": kurtosis(clean_returns) if len(clean_returns) > 3 else 0,
                "hhi_index": hhi, "effective_n": enc, "risk_penalty_ratio": penalty,
                "volatility": sigma, "raw_volatility": raw_sigma,
                "systematic_risk": systematic_risk, "idiosyncratic_risk": idiosyncratic_risk,
                "portfolio_beta": portfolio_beta,
                "risk_contribution": risk_contribution
            }
        except Exception as e:
            print(f"AdvancedStats Critical Fallback: {e}")
            return default_metrics


# =========================================================
# 🧪 ファクター分析 (Fama-French)
# =========================================================
class FactorAnalyzer:
    @staticmethod
    def analyze_style(target_series, region="US"):
        """
        Fama-French 5ファクターモデルを用いた重回帰分析。
        💡 予測リターンと残差の算出、および実績vs予測の累積リターン計算(100スタート)。
        """
        fallback_result = {
            "beta_market": 1.0, "beta_size": 0.0, "beta_value": 0.0,
            "beta_quality": 0.0, "beta_invest": 0.0,
            "alpha": 0.0, "r_squared": 0.0, "vif": {},
            "p_value_market": 1.0, "p_value_size": 1.0, "p_value_value": 1.0,
            "region": region, "status": "insufficient_data",
            "actual_cumulative": None, "predicted_cumulative": None, "residuals": None
        }

        if target_series.empty: return fallback_result
        
        try:
            # 💡 【修正】日次リターンを「対数リターン」に統一して回帰分析を行う
            # 対数リターンの二乗誤差最小化は、期間全体の「累積リターンのズレ最小化」と等価になる
            target_daily = np.log1p(target_series.pct_change().dropna())
            if len(target_daily) < 30: return fallback_result
            
            start_date = target_daily.index[0].strftime('%Y-%m-%d')
            config = MarketConfig.get_config(region)
            
            ff_data = DataFetcher.fetch_fama_french_factors(start_date, dataset_name=config["ff_dataset"])
            if ff_data.empty: return fallback_result
            
            # 💡 【修正】FFファクターも対数リターン化してスケールを完全に一致させる
            ff_data = np.log1p(ff_data)
            
            # 日付のアライメント（タイムゾーンと時刻の除去）
            target_daily.index = pd.to_datetime(target_daily.index).normalize()
            if target_daily.index.tz is not None:
                target_daily.index = target_daily.index.tz_localize(None)
            
            # 日次でインナー結合
            combined = pd.merge(target_daily.to_frame(name="Target"), ff_data, left_index=True, right_index=True, how='inner')
            combined = combined.dropna()

            if len(combined) < 30: return fallback_result
            
            cols_upper = {c: c.strip().upper() for c in combined.columns}
            
            mkt = next((c for c, up in cols_upper.items() if up in ['MKT-RF', 'MKT_RF', 'MKT - RF']), None)
            smb = next((c for c, up in cols_upper.items() if up == 'SMB'), None)
            hml = next((c for c, up in cols_upper.items() if up == 'HML'), None)
            rmw = next((c for c, up in cols_upper.items() if up == 'RMW'), None)
            cma = next((c for c, up in cols_upper.items() if up == 'CMA'), None)
            rf  = next((c for c, up in cols_upper.items() if up == 'RF'), None)

            if not all([mkt, smb, hml, rmw, cma, rf]):
                print(f"Alert: FF5 Variables are missing. Found: {combined.columns.tolist()}")
                fallback_result["status"] = "missing_factors"
                return fallback_result

            # 従属変数：超過対数リターン
            y = combined["Target"] - combined[rf]
            # 独立変数：FF5ファクター (対数)
            X = combined[[mkt, smb, hml, rmw, cma]]
            X = sm.add_constant(X)
            
            # 回帰モデルの構築と当てはめ
            model = sm.OLS(y, X).fit()
            annualized_alpha = model.params.get("const", 0.0) * TRADING_DAYS_PER_YEAR
            pvalues = model.pvalues
            
            # 予測リターンと残差の算出
            predicted_excess = model.predict(X)
            predicted_returns = predicted_excess + combined[rf]
            actual_returns = combined["Target"]
            
            # 残差（対数リターンの差分）
            residuals = actual_returns - predicted_returns

            # 💡 【修正】累積リターン推移の100スタート共通化
            # 対数リターンを np.exp() で指数関数的に戻し、算術の累積（価格）ベースに変換する
            actual_cum = np.exp(actual_returns.cumsum()) * 100
            pred_cum = np.exp(predicted_returns.cumsum()) * 100
            
            # グラフ描画用に初期日(t=0)を100として先頭に挿入
            base_date = actual_cum.index[0] - pd.Timedelta(days=1)
            actual_cum = pd.concat([pd.Series([100.0], index=[base_date]), actual_cum])
            pred_cum = pd.concat([pd.Series([100.0], index=[base_date]), pred_cum])
            
            vifs = {}
            for i, col in enumerate(X.columns):
                if col != "const":
                    try:
                        vif_val = variance_inflation_factor(X.values, i)
                        vifs[col] = vif_val
                    except:
                        vifs[col] = np.nan
            
            return {
                "beta_market": model.params.get(mkt, 1.0),
                "beta_size": model.params.get(smb, 0.0),
                "beta_value": model.params.get(hml, 0.0),
                "beta_quality": model.params.get(rmw, 0.0),
                "beta_invest": model.params.get(cma, 0.0),
                "alpha": annualized_alpha,
                "r_squared": model.rsquared_adj,
                "r_squared_raw": model.rsquared,
                "vif": vifs,
                "p_value_market": pvalues.get(mkt, 1.0),
                "p_value_size": pvalues.get(smb, 1.0),
                "p_value_value": pvalues.get(hml, 1.0),
                "p_value_quality": pvalues.get(rmw, 1.0),
                "p_value_invest": pvalues.get(cma, 1.0),
                "region": region,
                "status": "success",
                "actual_cumulative": actual_cum,
                "predicted_cumulative": pred_cum,
                "residuals": residuals
            }
        except Exception as e:
            print(f"Factor Analyzer Fallback Triggered: {e}")
            return fallback_result

    @staticmethod
    def get_factor_correlation(region="US", periods=60):
        try:
            config = MarketConfig.get_config(region)
            ff_data = DataFetcher.fetch_fama_french_factors("2000-01-01", dataset_name=config["ff_dataset"])
            if ff_data.empty: return None
            
            ff_monthly = ff_data.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            ff_recent = ff_monthly.tail(periods)
            
            factors_cols = [c for c in ff_recent.columns if c.strip().upper() != 'RF']
            factors_only = ff_recent[factors_cols]
            
            return factors_only.corr().to_dict()
        except Exception as e:
            print(f"Factor Correlation Fallback Triggered: {e}")
            return None


# =========================================================
# 🤖 AIアドバイザープロンプト生成
# =========================================================
class AIPromptBuilder:
    @staticmethod
    def generate_quant_prompt(stats_data, factor_data, target_name="現在のポートフォリオ"):
        risk_contributions = stats_data.get("risk_contribution", {})
        top_risk_asset = "特定不能"
        top_risk_value = 0.0
        
        if risk_contributions:
            top_risk_asset = max(risk_contributions, key=risk_contributions.get)
            top_risk_value = risk_contributions[top_risk_asset] * 100

        r_squared = factor_data.get("r_squared", 0) * 100 if factor_data else 0.0
        alpha = factor_data.get("alpha", 0) * 100 if factor_data else 0.0
        te = stats_data.get("tracking_error", 0) * 100

        prompt = f"""
あなたはウォール街で長年活躍する、非常に優秀だが少し辛口なクオンツ・ポートフォリオマネージャーです。
以下の計算結果データに基づいて、個人投資家向けに「プロの小言（診断レポート）」を作成してください。

【ポートフォリオの客観的データ】
- 分析対象: {target_name}
- リスクの支配者（最大リスク寄与銘柄）: {top_risk_asset} ({top_risk_value:.1f}%)
- モデル適合度 (Adjusted R2): {r_squared:.1f}% (※この数値が高いほど、自身のファクター戦略通りに動いています)
- 年率トラッキングエラー (TE): {te:.2f}% (※対ベンチマークのズレ。低いほど市場のコピー、高いほど独自路線です)
- 年率アルファ (超過収益): {alpha:.2f}%
- ボラティリティ: {stats_data.get("volatility", 0) * 100:.1f}%

【指示】
1. 専門的なクオンツの視点から、厳しいが的確で愛のあるアドバイスをしてください。
2. もし「リスクの支配者」が50%を超えている場合、「分散投資になっているつもりか？実質的に{top_risk_asset}と心中しているだけだぞ」と厳しく指摘してください。
3. トラッキングエラー(TE)が非常に低い(例: 2%未満)のにアルファがマイナスまたはゼロ付近の場合は、「高い手数料（または手間）を払ってインデックス以下の成果を出す、典型的な『隠れインデックスファンド』だ」と警告してください。
4. モデル適合度(R2)にも触れ、戦略通りに運用できているかを評価してください。
5. 最後に、改善のための具体的なネクストアクションを1つ提示してください。
6. 出力はMarkdown形式で、見出しを使って読みやすくしてください。ですます調で構いませんが、プロとしての威厳を保ってください。
"""
        return prompt


# =========================================================
# ⚖️ ポートフォリオウェイト算出
# =========================================================
class PortfolioBuilder:
    @staticmethod
    def calculate_market_cap_weights(market_caps_dict, backtest_start_date=None):
        """
        現在の時価総額に基づくウェイトを算出する。
        """
        if not market_caps_dict:
            return {}
        
        total_cap = sum(market_caps_dict.values())
        if total_cap <= 0:
            # 時価総額が不正な場合は均等ウェイトにフォールバック
            return {k: 100.0 / len(market_caps_dict) for k in market_caps_dict}
        
        # 各銘柄の時価総額比率（%）を計算
        weights = {k: (v / total_cap) * 100.0 for k, v in market_caps_dict.items()}
        return weights


# =========================================================
# 🎲 モンテカルロ・シミュレーション可視化
# =========================================================
class MonteCarloVisualizer:
    @staticmethod
    def plot_histogram(final_values, title="モンテカルロ・シミュレーション結果 (最終評価額の分布)"):
        """
        モンテカルロシミュレーションの最終結果分布をPlotlyのヒストグラムとして描画。
        バーの透過度を調整し、下位10%（悲観シナリオ）と期待値（Mean）を垂直線で明示して視認性を高める。
        """
        if not final_values or len(final_values) == 0:
            return go.Figure()

        mean_val = np.mean(final_values)
        # 下位10% (P90: 90%の確率でこれ以上になるライン、あるいは悲観シナリオ)
        p10_val = np.percentile(final_values, 10)

        fig = go.Figure()
        
        # バーの透過度設定 (opacity=0.6) で色が濃すぎる問題を解消
        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            marker=dict(color='royalblue', opacity=0.6),
            name='最終評価額'
        ))

        # 期待値（Mean）の垂直線とラベル
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color="green", 
            line_width=2,
            annotation_text=f"期待値(Mean): {mean_val:,.0f}", 
            annotation_position="top right",
            annotation_font=dict(color="green", size=12)
        )
        
        # 下位10%（悲観シナリオ）の垂直線とラベル
        fig.add_vline(
            x=p10_val, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text=f"下位10%(悲観): {p10_val:,.0f}", 
            annotation_position="top left",
            annotation_font=dict(color="red", size=12)
        )

        fig.update_layout(
            title=title,
            xaxis_title="最終評価額",
            yaxis_title="頻度 (シミュレーション回数)",
            showlegend=False,
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
