"""
app_re.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
※ 修正版(v10): 役割の完全分離
  - Page 1: 現在のポートフォリオの診断（ファクター要因分解、実績vs予測、市場相対評価）
  - Page 2: 未来に向けたウェイトの最適化提案（効率的フロンティア、将来予測）
  - ドローダウン関連コンポーネントを完全削除
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re 

# これまでに作成したモジュールのインポート
from config import MarketConfig, FACTOR_TRANSLATION
from data_engine import DataFetcher
from analytics import AdvancedStats, FactorAnalyzer, AIPromptBuilder
from simulation import RegimeAnalyzer, HistoryTimeMachine, ProjectionCore, DynamicFactorAnalyzer

# =========================================================
# 📊 監査・可視化エンジンクラス (Plotly インタラクティブ版)
# =========================================================
class AuditEngine:
    
    @staticmethod
    def plot_actual_vs_predicted(actual_cum, predicted_cum):
        """[1ページ目用]: 実績累積リターン vs モデル予測累積リターンの比較"""
        if actual_cum is None or predicted_cum is None: return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_cum.index, y=actual_cum.values, mode='lines', name='Actual Portfolio', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=predicted_cum.index, y=predicted_cum.values, mode='lines', name='Predicted by Factor Model', line=dict(color='#ff7f0e', dash='dash', width=2)))
        
        fig.update_layout(
            title="Model Fit: Actual vs Predicted Cumulative Return (Base=100)",
            xaxis_title="Date", yaxis_title="Cumulative Return",
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_residuals(residuals):
        """[1ページ目用]: モデルで説明できない残差（アルファ＋誤差）の推移"""
        if residuals is None or residuals.empty: return
        
        cum_residuals = residuals.cumsum() * 100 # %表記
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_residuals.index, y=cum_residuals.values, mode='lines', name='Cumulative Residuals', line=dict(color='#2ca02c'), fill='tozeroy'))
        fig.add_hline(y=0, line_dash="solid", line_color="black")
        
        fig.update_layout(
            title="Cumulative Residuals (Unexplained by Model / Alpha)",
            xaxis_title="Date", yaxis_title="Residuals (%)",
            height=250, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_active_return(port_returns, bm_returns):
        """[1ページ目用]: アクティブ・リターン（ベンチマークとの乖離）の推移を可視化"""
        if port_returns.empty or bm_returns.empty:
            st.info("💡 データ不足のため表示できません（ベンチマークデータがありません）。")
            return
            
        aligned = pd.concat([port_returns.rename("Portfolio"), bm_returns.rename("Benchmark")], axis=1).dropna()
        if len(aligned) < 2: return
            
        active_returns = aligned["Portfolio"] - aligned["Benchmark"]
        cum_active_returns = active_returns.cumsum() * 100
        
        fig = px.area(
            x=cum_active_returns.index, y=cum_active_returns.values,
            labels={'x': 'Date', 'y': 'Cumulative Active Return (%)'},
            title="Cumulative Active Return Spread (vs Benchmark)",
            color_discrete_sequence=["#17becf"]
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Benchmark Level")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_factor_correlation(region="US"):
        """ファクター同士の相関関係をヒートマップで可視化"""
        factor_corr = FactorAnalyzer.get_factor_correlation(region=region)
        if not factor_corr: return
            
        corr_df = pd.DataFrame(factor_corr)
        fig = px.imshow(
            corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Factor Correlation ({region} Region)"
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_rolling_correlation(port_returns, bm_returns, window=60):
        """市場との連動性（ローリング相関）の推移を可視化"""
        if port_returns.empty or bm_returns.empty: return
            
        aligned = pd.concat([port_returns.rename("Portfolio"), bm_returns], axis=1).dropna()
        if len(aligned) < window: return
            
        rolling_corr = aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1]).dropna()
        
        fig = px.line(
            x=rolling_corr.index, y=rolling_corr.values,
            labels={'x': 'Date', 'y': 'Correlation'},
            title=f"Rolling {window}-Day Market Correlation"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_crisis_replays(crisis_results):
        """過去の危機における最大下落幅の表示"""
        if not crisis_results:
            st.info("💡 指定された過去の危機期間に該当銘柄のデータが存在しません。")
            return
            
        names = list(crisis_results.keys())
        for i, name in enumerate(names):
            data = crisis_results[name]
            st.markdown(f"**{name}**")
            st.markdown(f"- 最大下落幅 (Max Drawdown): **{data['max_drawdown_pct']:.2f}%**")

    @staticmethod
    def plot_rolling_exposure(rolling_df):
        """ローリング回帰による動的エクスポージャーの推移を可視化"""
        if rolling_df is None or rolling_df.empty: return
        
        factors = ["Market_Beta", "Size_Beta", "Value_Beta", "Quality_Beta", "Invest_Beta"]
        valid_factors = [col for col in factors if col in rolling_df.columns and not rolling_df[col].isna().all()]
        has_r2 = "Adjusted_R2" in rolling_df.columns
        
        n_rows = len(valid_factors) + (1 if has_r2 else 0)
        if n_rows == 0: return
        
        titles = valid_factors + (["Adjusted R-Squared (%)"] if has_r2 else [])
        
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=titles)
        
        row_idx = 1
        for col in valid_factors:
            fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df[col], name=col, mode='lines'), row=row_idx, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=row_idx, col=1)
            row_idx += 1
            
        if has_r2:
            fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["Adjusted_R2"], name="Adj R2", line=dict(color='purple')), row=row_idx, col=1)
        
        fig.update_layout(height=150 * n_rows, title_text="Dynamic Factor Exposure", margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def optimize_and_plot_frontier(asset_returns, current_weights_dict):
        """💡 新設 [2ページ目用]: 効率的フロンティアと最適ウェイトの算出"""
        if asset_returns is None or asset_returns.empty or len(current_weights_dict) < 2:
            st.info("💡 最適化には2銘柄以上のデータが必要です。")
            return

        tickers = list(current_weights_dict.keys())
        available_tickers = [t for t in tickers if t in asset_returns.columns]
        if len(available_tickers) < 2:
            st.warning("💡 有効な銘柄データが不足しているため、最適化をスキップします。")
            return
            
        returns_df = asset_returns[available_tickers]
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # モンテカルロ近似による最適ポートフォリオ探索
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        np.random.seed(42)
        for i in range(num_portfolios):
            weights = np.random.random(len(available_tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            p_ret = np.sum(mean_returns * weights)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results[0,i] = p_std
            results[1,i] = p_ret
            results[2,i] = p_ret / p_std if p_std > 0 else 0
            
        max_sharpe_idx = np.argmax(results[2])
        opt_ret = results[1, max_sharpe_idx]
        opt_std = results[0, max_sharpe_idx]
        opt_weights = weights_record[max_sharpe_idx]
        
        curr_weights = np.array([current_weights_dict.get(t, 0) for t in available_tickers])
        if np.sum(curr_weights) > 0:
            curr_weights /= np.sum(curr_weights)
        curr_ret = np.sum(mean_returns * curr_weights)
        curr_std = np.sqrt(np.dot(curr_weights.T, np.dot(cov_matrix, curr_weights)))
        
        fig = go.Figure()
        # Random Portfolios
        fig.add_trace(go.Scatter(
            x=results[0], y=results[1], mode='markers',
            marker=dict(color=results[2], colorscale='Viridis', showscale=True, size=4, colorbar=dict(title="Sharpe Ratio")),
            name='Simulated Portfolios'
        ))
        # Max Sharpe
        fig.add_trace(go.Scatter(
            x=[opt_std], y=[opt_ret], mode='markers',
            marker=dict(color='red', size=16, symbol='star', line=dict(color='black', width=1)),
            name='Max Sharpe Portfolio'
        ))
        # Current
        fig.add_trace(go.Scatter(
            x=[curr_std], y=[curr_ret], mode='markers',
            marker=dict(color='orange', size=14, symbol='x', line=dict(color='black', width=1)),
            name='Current Portfolio'
        ))
        
        fig.update_layout(
            title="Efficient Frontier (Risk-Return Tradeoff)",
            xaxis_title="Expected Annual Volatility (Risk)",
            yaxis_title="Expected Annual Return",
            height=450, margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### ⚖️ Proposed Optimal Weights (Max Sharpe Ratio)")
        comp_df = pd.DataFrame({
            "Ticker": available_tickers,
            "Current Weight (%)": curr_weights * 100,
            "Optimal Weight (%)": opt_weights * 100,
            "Difference (%)": (opt_weights - curr_weights) * 100
        })
        
        st.dataframe(comp_df.style.format({
            "Current Weight (%)": "{:.1f}%",
            "Optimal Weight (%)": "{:.1f}%",
            "Difference (%)": "{:+.1f}%"
        }).background_gradient(subset=["Difference (%)"], cmap="RdYlGn"), use_container_width=True)

    @staticmethod
    def plot_monte_carlo_fanchart(paths):
        """1万回のシミュレーション推移を扇状で可視化"""
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        days = np.arange(paths.shape[0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=percentiles[4], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=percentiles[0], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.2)', line=dict(width=0), name='5th-95th Percentile'))
        fig.add_trace(go.Scatter(x=days, y=percentiles[3], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=percentiles[1], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.4)', line=dict(width=0), name='25th-75th Percentile'))
        fig.add_trace(go.Scatter(x=days, y=percentiles[2], mode='lines', line=dict(color='darkblue', width=2), name='Median (50th)'))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(title="Monte Carlo Projection (1-Year Fan Chart)", xaxis_title="Trading Days", yaxis_title="Portfolio Value", height=450)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_monte_carlo_histogram(final_values):
        """最終資産分布の正確なヒストグラム"""
        fig = px.histogram(final_values, nbins=50, title="Final Value Distribution", labels={'value': 'Final Portfolio Value', 'count': 'Frequency'}, color_discrete_sequence=["steelblue"])
        median_val = np.median(final_values)
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.add_vline(x=median_val, line_dash="solid", line_color="green", annotation_text=f"Median: {median_val*100:.1f}%")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 🚀 Streamlit メインロジック
# =========================================================
def main():
    st.set_page_config(page_title="Institutional Portfolio Auditor", layout="wide", page_icon="🏦")
    st.title("🏦 Institutional Portfolio Auditor")
    st.markdown("インタラクティブなリスク評価とAI診断システム")
    st.divider()

    # --- サイドバー (動的スプレッドシート機能 & API設定) ---
    st.sidebar.header("⚙️ 設定 & ポートフォリオ")
    
    st.sidebar.markdown("**🤖 AI診断用 API設定**")
    ai_api_key = st.sidebar.text_input("API Key (現在プレースホルダー)", type="password", help="ここにOpenAI等のAPIキーを入れると本物のAIが動くようになります")
    st.sidebar.divider()
    
    region = st.sidebar.selectbox("Market Region", ["US", "Japan"], help="対象市場を選択してください。バックエンドの参照ファイルや無リスク金利が切り替わります。")
    config = MarketConfig.get_config(region)
    
    rebalance_options = {"Monthly (月次)": "M", "Annually (年次)": "Y", "Daily (日次)": "D", "Buy & Hold (放置)": None}
    rebalance_choice = st.sidebar.selectbox("Rebalance Frequency", list(rebalance_options.keys()), help="ポートフォリオの比率を元に戻す頻度。放置すると強い銘柄の比率が勝手に増えます(ドリフト)。")
    rebalance_freq = rebalance_options[rebalance_choice]

    target_te = st.sidebar.slider("Target Tracking Error (%)", min_value=0.5, max_value=15.0, value=3.0, step=0.5, help="運用目標とするトラッキングエラー（乖離リスク）の目安。")

    st.sidebar.caption(f"📌 **設定情報:**\n- データセット: `{config['ff_dataset']}`\n- ベンチマーク: `{config['benchmark_ticker']}`")
    
    st.sidebar.markdown("**📤 1. ポートフォリオ一括読込 (オプション)**")
    uploaded_file = st.sidebar.file_uploader("CSVファイル", type=["csv"], help="ティッカーと比率が書かれたCSVを読み込みます。")
    
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL"] if region == "US" else ["7203.T", "8306.T", "9984.T"],
            "Weight": [40.0, 40.0, 20.0]
        })

    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
            ticker_col = next((c for c in df_csv.columns if re.search(r'(ticker|symbol|code|銘柄|コード)', str(c), re.IGNORECASE)), None)
            weight_col = next((c for c in df_csv.columns if re.search(r'(weight|ratio|percent|比率|割合|ウェイト|%)', str(c), re.IGNORECASE)), None)

            if ticker_col and weight_col:
                clean_weights = df_csv[weight_col].astype(str).str.replace(r'[%,]', '', regex=True)
                df_csv["Weight"] = pd.to_numeric(clean_weights, errors='coerce').fillna(0)
                df_csv["Ticker"] = df_csv[ticker_col].astype(str).str.strip().str.upper()
                
                st.session_state.portfolio_data = df_csv[["Ticker", "Weight"]]
                st.sidebar.success(f"CSVを読み込みました（{len(df_csv)}銘柄）")
            else:
                st.sidebar.error("CSVエラー: 「銘柄」と「比率」を示す列が自動認識できませんでした。")
        except Exception as e:
            st.sidebar.error(f"CSV読み込みエラー: {e}")

    st.sidebar.markdown("**✏️ 2. 銘柄とウェイト調整（直接編集可能）**")
    edited_df = st.sidebar.data_editor(
        st.session_state.portfolio_data, 
        num_rows="dynamic", width="stretch", hide_index=True
    )
    
    weights_dict = {}
    for _, row in edited_df.dropna().iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        if ticker and pd.notna(row["Weight"]):
            weights_dict[ticker] = float(row["Weight"])

    # --- 実行ボタン ---
    if st.sidebar.button("Run Advanced Analysis", type="primary", use_container_width=True):
        if not weights_dict:
            st.error("有効なティッカーとウェイトを入力してください。")
            return
            
        mismatch = False
        for ticker in weights_dict.keys():
            if region == "Japan" and not ticker.endswith('.T'): mismatch = True; break
            elif region == "US" and ticker.endswith('.T'): mismatch = True; break
                
        if mismatch:
            st.markdown("<p style='color:red; font-weight:bold; font-size:1.1em;'>⚠️ 地域設定と銘柄が一致していないため、データ取得や分析に失敗する可能性があります。</p>", unsafe_allow_html=True)
            
        with st.spinner(f"Initializing Quantitative Engine ({region} Market) & Fetching Data..."):
            
            norm_weights = DataFetcher.normalize_weights(weights_dict)
            input_ticker_count = len(norm_weights)
            
            raw_input_data = DataFetcher.fetch_market_data(list(norm_weights.keys()))
            valid_ticker_count = len(raw_input_data.columns) if not raw_input_data.empty else 0
            
            synthetic_portfolio = DataFetcher.create_synthetic_portfolio(norm_weights, region=region, rebalance_freq=rebalance_freq)
            
            if synthetic_portfolio is None or synthetic_portfolio.empty:
                st.error("データの構築に失敗しました。ティッカー記号や期間、通信制限(Rate Limit)を確認してください。")
                return
            
            if valid_ticker_count == input_ticker_count:
                st.success(f"✅ **データ網羅性:** 入力された全 {input_ticker_count} 銘柄のデータを取得しました。")
            else:
                st.warning(f"⚠️ **データ網羅性:** 入力された {input_ticker_count} 銘柄中、**{valid_ticker_count} 銘柄** のみが計算に適用されました。")

            returns = synthetic_portfolio.pct_change().dropna()
            
            bm_prices = DataFetcher.fetch_market_data([config["benchmark_ticker"]])
            if not bm_prices.empty:
                bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")
            else:
                st.warning("⚠️ ベンチマークデータの取得に失敗しました（API制限等）。市場比較やTEの計算はスキップされます。")
                bm_returns = pd.Series(dtype=float)

            # 解析実行
            metrics = AdvancedStats.calculate_metrics(returns, benchmark_returns=bm_returns if not bm_returns.empty else None, weights_dict=norm_weights, region=region)
            style = FactorAnalyzer.analyze_style(synthetic_portfolio, region=region)
            cycle_days = RegimeAnalyzer.detect_cycle(returns)
            rolling_exposure_df = DynamicFactorAnalyzer.calculate_rolling_exposure(synthetic_portfolio, region=region)
            
            crises = ["リーマン・ショック (2007-2009)", "コロナ・ショック (2020)", "ドットコム・バブル崩壊 (2000-2002)"]
            crisis_results = {}
            for crisis in crises:
                res = HistoryTimeMachine.replay_crisis(norm_weights, crisis, region)
                if res: crisis_results[crisis] = res
                
            projection = ProjectionCore.run_projection(
                returns=returns, bm_returns=bm_returns if not bm_returns.empty else None, 
                n_scenarios=10000, n_years=1
            )

            # ==========================================
            # 🗂️ 描画レイヤー (2つのタブ構成へ完全分離)
            # ==========================================
            tab1, tab2 = st.tabs([
                "📊 Page 1: ポートフォリオ診断 (要因分解・モデル適合度)", 
                "🎯 Page 2: 未来への最適化提案 (フロンティア・予測)"
            ])

            # ---------------------------------------------------------
            # --- Page 1: 診断 (現状分析, ファクター要因分解, 市場相対) ---
            # ---------------------------------------------------------
            with tab1:
                st.header(f"1. Current Portfolio Diagnosis ({region} Market)")
                st.subheader("🤖 クオンツマネージャーの辛口診断")
                
                ai_prompt = AIPromptBuilder.generate_quant_prompt(metrics, style, target_name="現在のポートフォリオ")
                
                if ai_api_key:
                    st.info("APIキーが認識されました。（※実際の実装時はここでLLM APIを呼び出します）")
                else:
                    st.markdown("""
                    > **【AI診断ダミー表示】**
                    > あなたのポートフォリオを拝見しました。分散投資をしているつもりかもしれませんが、
                    > リスクの大半が特定の1銘柄に集中しており、実質的にその銘柄と心中している状態です。
                    > **[ネクストアクション]** 早急に最大リスク寄与銘柄のウェイトを下げ、他セクターへの分散を図りなさい。
                    """)
                with st.expander("🔍 裏で生成されたAIへの命令書（プロンプト）を見る"):
                    st.text(ai_prompt)
                    
                st.divider()
                
                r2_score = style.get('r_squared', 0.0) * 100
                
                # ドローダウン関連を削除し、3つの指標のみ表示
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
                col2.metric("Portfolio Volatility", f"{metrics.get('volatility', 0) * 100:.2f}%")
                col3.metric("Model Fit (Adjusted R²)", f"{r2_score:.1f}%", help="自身のファクター戦略でどれだけ動きを説明できているか")
                
                st.divider()
                
                st.subheader("📈 Model Fit: Actual vs Predicted")
                if style and style.get('status') == 'success':
                    AuditEngine.plot_actual_vs_predicted(style.get("actual_cumulative"), style.get("predicted_cumulative"))
                    AuditEngine.plot_residuals(style.get("residuals"))
                else:
                    st.info("💡 ファクターモデルによる予測データが不足しているため、実績値のみを表示します。")
                    st.line_chart(synthetic_portfolio)
                
                st.divider()

                # --- ファクター要因分解 (旧Page 4を統合) ---
                st.subheader(f"🔬 Causal Factor Analysis ({region} Fama-French 5-Factor)")
                st.markdown("ポートフォリオの背後にある「リスクの源泉」を統計的に分解し、その因果的妥当性と安定性を評価します。")
                
                if style and style.get('status') != 'insufficient_data':
                    high_vif_factors = [f for f, v in style.get('vif', {}).items() if v > 10]
                    if high_vif_factors:
                        st.warning(f"⚠️ **多重共線性の警告 (VIF > 10):** ファクター [{', '.join(high_vif_factors)}] の間で強い相関が検出されました。")

                    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                    mkt_beta = style.get('beta_market', 1.0)
                    f_col1.metric("Market-RF (市場超過ベータ)", f"{mkt_beta:.2f}", delta="ハイリスク" if mkt_beta > 1.2 else ("ローリスク" if mkt_beta < 0.8 else ""), delta_color="inverse")
                    size_beta = style.get('beta_size', 0.0)
                    f_col2.metric("SMB (企業規模)", f"{size_beta:.2f}", delta="小型株寄り" if size_beta > 0 else "大型株寄り", delta_color="off")
                    val_beta = style.get('beta_value', 0.0)
                    f_col3.metric("HML (割安性)", f"{val_beta:.2f}", delta="割安株寄り" if val_beta > 0 else "成長株寄り", delta_color="off")
                    alpha_val = style.get('alpha', 0.0) * 100
                    f_col4.metric("Alpha (年率固有・超過収益)", f"{alpha_val:.3f}%", delta=f"{alpha_val:.3f}%", delta_color="normal")
                    
                    q_col1, q_col2, _, _ = st.columns(4)
                    quality_beta = style.get('beta_quality', 0.0)
                    q_col1.metric("RMW (クオリティ/収益力)", f"{quality_beta:.2f}", delta="高収益企業寄り" if quality_beta > 0 else "低収益企業寄り", delta_color="off")
                    invest_beta = style.get('beta_invest', 0.0)
                    q_col2.metric("CMA (インベストメント/堅実性)", f"{invest_beta:.2f}", delta="堅実投資寄り" if invest_beta > 0 else "過剰投資寄り", delta_color="off")

                    p_val_market = style.get('p_value_market', 1.0)
                    if p_val_market < 0.05 and not high_vif_factors:
                        if r2_score >= 80:
                            insight = f"決定係数が **{r2_score:.1f}%** と極めて高く、市場要因のP値も **{p_val_market:.3f}** で有意です。実質的に市場平均と強く連動しています（隠れインデックスの可能性）。"
                        else:
                            insight = f"決定係数は **{r2_score:.1f}%**。市場全体のリスクを受けつつも、独自の要素を持つアクティブ・ポートフォリオと推論されます。"
                    else:
                        insight = f"P値が **{p_val_market:.3f}** と高く有意でないか、VIF警告が出ています。特定期間の見せかけの相関を疑い、下の動的エクスポージャーを確認してください。"
                    st.info(f"🧠 **分析インサイト:** {insight}")
                    
                    if rolling_exposure_df is not None:
                        AuditEngine.plot_rolling_exposure(rolling_exposure_df)
                else:
                    st.info("💡 ファクター分析を行うためのデータが不足しています（最低36ヶ月分のデータが推奨されます）。")

                st.divider()

                # --- 市場相対リスク (旧Page 2を統合) ---
                st.subheader("🎯 Market Relative Performance & Tracking Error")
                if bm_returns.empty:
                    st.info("💡 データ不足のため表示できません（ベンチマークデータの取得に失敗しました）。")
                else:
                    te_actual = metrics.get('tracking_error', 0) * 100
                    te_col1, te_col2, te_col3 = st.columns(3)
                    te_col1.metric("Tracking Error (実績TE)", f"{te_actual:.2f}%")
                    te_col2.metric(f"Target TE (目標)", f"{target_te:.1f}%", delta=f"{te_actual - target_te:.2f}% (目標との差)", delta_color="inverse")
                    te_col3.metric("Information Ratio (情報レシオ)", f"{metrics.get('info_ratio', 0):.2f}")
                    
                    aligned_growth = pd.concat([synthetic_portfolio.rename("Portfolio"), (1+bm_returns).cumprod()*100], axis=1).dropna()
                    fig_rel = px.line(aligned_growth, labels={'value': 'Cumulative Return (Base=100)', 'index': 'Date'}, title="Portfolio vs Benchmark Growth")
                    fig_rel.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_rel, use_container_width=True)
                    
                    AuditEngine.plot_active_return(returns, bm_returns)
                    
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        AuditEngine.plot_factor_correlation(region=region)
                    with c_col2:
                        AuditEngine.plot_rolling_correlation(returns, bm_returns)

                if cycle_days:
                    st.caption(f"⏳ ボラティリティ周期 (ウェルチ法): 約 {cycle_days} 日")


            # ---------------------------------------------------------
            # --- Page 2: 最適化提案 (ウェイト最適化, 予測, ストレス) ---
            # ---------------------------------------------------------
            with tab2:
                st.header("2. Future Portfolio Optimization & Projections")
                
                # --- 新機能: ウェイト最適化と効率的フロンティア ---
                st.subheader("⚖️ Efficient Frontier & Weight Optimization")
                st.markdown("現在のポートフォリオから、**シャープレシオ（リスク・リターン比）を最大化**する未来の最適ウェイトを提案します。")
                
                # 個別銘柄の収益率データを算出
                asset_returns = raw_input_data.pct_change().dropna()
                AuditEngine.optimize_and_plot_frontier(asset_returns, norm_weights)
                
                st.divider()

                # --- 将来シミュレーション ---
                st.subheader("🔮 Stochastic Projection (Monte Carlo Simulation)")
                if projection:
                    AuditEngine.plot_monte_carlo_fanchart(projection["paths"])
                    
                    final_values = projection["paths"][-1, :]
                    worst_5th = projection['worst_5th']
                    cvar = final_values[final_values <= worst_5th].mean()
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Median (中央値)", f"{projection['median'] * 100:.1f}%")
                    p_col2.metric("Worst 5% (下位5%)", f"{worst_5th * 100:.1f}%")
                    p_col3.metric("CVaR (下位5%の平均)", f"{cvar * 100:.1f}%", help="テールリスク")
                    p_col4.metric("Prob of Loss (元本割れ確率)", f"{projection['prob_loss']:.1f}%")
                    
                    AuditEngine.plot_monte_carlo_histogram(final_values)
                
                st.divider()

                # --- ストレステスト ---
                st.subheader("⚡ Stress Tests (Crash Replays)")
                st.markdown("過去の主要な金融危機の際、現在のポートフォリオ構成がどの程度の下落を経験したかを追体験します。")
                AuditEngine.plot_crisis_replays(crisis_results)

if __name__ == "__main__":
    main()
