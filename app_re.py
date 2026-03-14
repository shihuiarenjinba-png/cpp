"""
app_re.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
※ 修正版(v9): 役割の完全分離（1ページ目：モデル適合度[実績vs予測]、2ページ目：市場相対評価[実績vsBM]）
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
        """💡 新設 [1ページ目用]: 実績累積リターン vs モデル予測累積リターンの比較"""
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
        """💡 新設 [1ページ目用]: モデルで説明できない残差（アルファ＋誤差）の推移"""
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
        """💡 [2ページ目用]: アクティブ・リターン（ベンチマークとの乖離）の推移を可視化"""
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
    def plot_underwater_drawdown(portfolio_prices):
        """アンダーウォーター・プロット（Plotly面積グラフ）"""
        if portfolio_prices.empty: return
        peak = portfolio_prices.cummax()
        drawdown = (portfolio_prices - peak) / peak * 100
        
        fig = px.area(
            x=drawdown.index, y=drawdown,
            labels={'x': 'Date', 'y': 'Drawdown (%)'},
            title="Underwater Plot (Historical Drawdowns)",
            color_discrete_sequence=["firebrick"]
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
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
        
    @staticmethod
    def plot_monte_carlo_drawdown_hist(paths):
        """1万回のシナリオにおける最大ドローダウンの分布"""
        peaks = np.maximum.accumulate(paths, axis=0)
        drawdowns = (paths - peaks) / peaks
        max_dds = drawdowns.min(axis=0) * 100
        fig = px.histogram(max_dds, nbins=50, title="Simulated Max Drawdown Distribution", labels={'value': 'Maximum Drawdown (%)', 'count': 'Frequency'}, color_discrete_sequence=["darkorange"])
        median_dd = np.median(max_dds)
        fig.add_vline(x=median_dd, line_dash="dash", line_color="red", annotation_text=f"Median: {median_dd:.1f}%")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def analyze_recovery(paths):
        drawdown_paths_idx = np.where((paths < 1.0).any(axis=0))[0]
        if len(drawdown_paths_idx) == 0: return 100.0
        recovered_count = np.sum(paths[-1, drawdown_paths_idx] >= 1.0)
        return (recovered_count / len(drawdown_paths_idx)) * 100


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
            # 🗂️ 描画レイヤー (4つのタブ構成)
            # ==========================================
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 概要 & モデル適合度",   # 💡 名称変更 (1ページ目)
                "🎯 市場相対 & 乖離リスク", # 💡 名称変更 (2ページ目)
                "🔮 予測 & ストレス", 
                "🔬 ファクター解析 (FF5)"
            ])

            # --- タブ1: 概要 & モデル適合度 (内省的分析) ---
            with tab1:
                st.header(f"1. Core Metrics & Model Fit ({region} Market)")
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
                
                peak = synthetic_portfolio.cummax()
                calc_dd = ((synthetic_portfolio - peak) / peak).min() * 100
                r2_score = style.get('r_squared', 0.0) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
                col2.metric("Portfolio Volatility", f"{metrics.get('volatility', 0) * 100:.2f}%")
                # 💡 1ページ目にモデル適合度(R2)をハイライト
                col3.metric("Model Fit (Adjusted R²)", f"{r2_score:.1f}%", help="自身のファクター戦略でどれだけ動きを説明できているか")
                col4.metric("Max Drawdown", f"{calc_dd:.2f}%") 
                
                st.divider()
                
                # 💡 1ページ目のメイン: 実績 vs 予測 (ベンチマーク比較は削除)
                if style and style.get('status') == 'success':
                    AuditEngine.plot_actual_vs_predicted(style.get("actual_cumulative"), style.get("predicted_cumulative"))
                    AuditEngine.plot_residuals(style.get("residuals"))
                else:
                    st.info("💡 ファクターモデルによる予測データが不足しているため、実績値のみを表示します。")
                    st.line_chart(synthetic_portfolio)
                
                AuditEngine.plot_underwater_drawdown(synthetic_portfolio)

            # --- タブ2: 市場相対 & 乖離リスク (対外的評価) ---
            with tab2:
                st.header("2. Relative Market Performance & Tracking Error")
                
                if bm_returns.empty:
                    st.info("💡 データ不足のため表示できません（ベンチマークデータの取得に失敗しました）。")
                else:
                    te_actual = metrics.get('tracking_error', 0) * 100
                    
                    st.subheader("🎯 Tracking Error Overview")
                    te_col1, te_col2, te_col3 = st.columns(3)
                    te_col1.metric("Tracking Error (実績TE)", f"{te_actual:.2f}%", help="この数値が高いほど、市場平均から外れた独自の値動きをしています。")
                    te_col2.metric(f"Target TE (目標)", f"{target_te:.1f}%", delta=f"{te_actual - target_te:.2f}% (目標との差)", delta_color="inverse", help="目標とするトラッキングエラー（目安）。")
                    te_col3.metric("Information Ratio (情報レシオ)", f"{metrics.get('info_ratio', 0):.2f}", help="取った乖離リスク(TE)に対して、どれだけ超過リターンを稼げたかを示す「アクティブ運用のコスパ」です。")
                    
                    st.divider()
                    
                    # 💡 2ページ目のメイン: 対ベンチマークの比較グラフ
                    st.subheader("📊 Market Relative Performance (Actual vs Benchmark)")
                    aligned_growth = pd.concat([synthetic_portfolio.rename("Portfolio"), (1+bm_returns).cumprod()*100], axis=1).dropna()
                    
                    fig_rel = px.line(aligned_growth, labels={'value': 'Cumulative Return (Base=100)', 'index': 'Date'}, title="Portfolio vs Benchmark Growth")
                    fig_rel.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_rel, use_container_width=True)
                    
                    st.divider()
                    st.subheader("📈 Active Return Spread")
                    AuditEngine.plot_active_return(returns, bm_returns)
                    
                    st.divider()
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.subheader(f"Factor Correlation Matrix ({region})")
                        AuditEngine.plot_factor_correlation(region=region)
                    with c_col2:
                        st.subheader("Market Correlation (Rolling 60-Day)")
                        AuditEngine.plot_rolling_correlation(returns, bm_returns)
                
            # --- タブ3: 将来シミュレーション & ストレステスト ---
            with tab3:
                st.header("3. Stochastic Projection & Stress Tests")
                if projection:
                    AuditEngine.plot_monte_carlo_fanchart(projection["paths"])
                    
                    final_values = projection["paths"][-1, :]
                    worst_5th = projection['worst_5th']
                    cvar = final_values[final_values <= worst_5th].mean()
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Median (中央値)", f"{projection['median'] * 100:.1f}%")
                    p_col2.metric("Worst 5% (下位5%)", f"{worst_5th * 100:.1f}%")
                    p_col3.metric("CVaR (下位5%の平均)", f"{cvar * 100:.1f}%", help="テールリスク")
                    p_col4.metric("Prob of Loss (元本割れ)", f"{projection['prob_loss']:.1f}%")
                    
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        AuditEngine.plot_monte_carlo_histogram(final_values)
                    with g_col2:
                        AuditEngine.plot_monte_carlo_drawdown_hist(projection["paths"])
                    
                    recovery_rate = AuditEngine.analyze_recovery(projection["paths"])
                    st.info(f"📉 **回復力監査:** ドローダウン発生後、1年以内に元本を回復する確率: **{recovery_rate:.1f}%**")
                
                st.divider()
                st.subheader("⏳ History Time Machine (Crash Recovery Paths)")
                st.markdown("過去の主要な金融危機の際、当ポートフォリオがどのように下落し、**どの程度の期間で回復したか**を追体験します。")
                AuditEngine.plot_crisis_replays(crisis_results)

            # --- タブ4: ファクター解析 (FF5 プロフェッショナル・ダッシュボード化) ---
            with tab4:
                st.header(f"4. Causal Factor Analysis ({region} Fama-French 5-Factor)")
                st.markdown("ポートフォリオの背後にある「リスクの源泉」を統計的に分解し、その因果的妥当性と安定性を評価します。無リスク金利(RF)を控除した**超過リターンベース**で計算されています。")
                
                if style and style.get('status') != 'insufficient_data':
                    high_vif_factors = [f for f, v in style.get('vif', {}).items() if v > 10]
                    if high_vif_factors:
                        st.warning(f"⚠️ **多重共線性の警告 (VIF > 10):** ファクター [{', '.join(high_vif_factors)}] の間で強い相関が検出されました。")

                    st.subheader("📊 Factor Sensitivity (市場・スタイル感度)")
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

                    st.divider()
                    st.subheader("📈 Model Reliability (統計的信頼性と因果的インサイト)")
                    s_col1, s_col2, s_col3 = st.columns(3)
                    
                    r2 = style.get('r_squared', 0.0) * 100
                    p_val_market = style.get('p_value_market', 1.0)
                    
                    s_col1.metric("Adjusted R-Squared", f"{r2:.1f}%")
                    s_col2.metric("Market P-Value", f"{p_val_market:.3f}")
                    s_col3.metric("Quality/Invest P-Value", f"{style.get('p_value_quality', 1.0):.3f} / {style.get('p_value_invest', 1.0):.3f}")

                    st.markdown("#### 🧠 統計的因果推論による解釈")
                    
                    if p_val_market < 0.05 and not high_vif_factors:
                        significance = "統計的に有意かつ多重共線性の問題もない"
                        if r2 >= 80:
                            insight = f"自由度調整済み決定係数が **{r2:.1f}%** と極めて高く、市場要因のP値も **{p_val_market:.3f}** であり、{significance}モデルです。このポートフォリオの変動は、実質的にFF5のシステム（特に市場平均）によって**構造的かつ因果的に説明される**と推論されます（隠れインデックスの可能性）。"
                        elif r2 >= 50:
                            insight = f"自由度調整済み決定係数は **{r2:.1f}%**、市場P値は **{p_val_market:.3f}** であり、{significance}状態です。市場全体のシステマティック・リスクに一定の影響を受けつつも、独自の非システマティックな要素（固有アルファ）を併せ持つ健全なアクティブ・ポートフォリオと推論されます。"
                        else:
                            insight = f"自由度調整済み決定係数は **{r2:.1f}%** と低く、市場の動きだけでは説明できない**独自の変動因果**を持っています。P値（**{p_val_market:.3f}**）は有意であるため、市場とは無関係な特定セクターやテーマへの集中投資が値動きを支配していると推測されます。"
                    else:
                        significance = "統計的に有意でない、あるいは多重共線性のノイズを含んでいる"
                        insight = f"P値が **{p_val_market:.3f}** と高い、もしくはVIF警告が出ているため、現在の結果は{significance}と判定されます。決定係数（**{r2:.1f}%**）を鵜呑みにせず、シンプソンのパラドックス（特定期間だけの見せかけの相関）を疑う必要があります。下の「動的エクスポージャー（ローリング回帰）」チャートで、時間的な安定性を確認してください。"

                    st.info(insight)
                    
                    if rolling_exposure_df is not None:
                        AuditEngine.plot_rolling_exposure(rolling_exposure_df)

                else:
                    st.info("💡 ファクター分析を行うためのデータが不足しています（最低36ヶ月分のデータが推奨されます）。")
                
                st.divider()
                st.subheader("⏳ Volatility Cycle Detection")
                if cycle_days:
                    st.info(f"ウェルチ法による現在のボラティリティ周期: **約 {cycle_days} 日**")

if __name__ == "__main__":
    main()
