"""
app.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
※ 修正版(v5): 地域(US/JP)の動的切り替えと、超過リターンベースのAlpha/Betaの正確なUI表示に対応
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
    def plot_factor_correlation(region="US"):
        """ファクター同士の相関関係をヒートマップで可視化"""
        factor_corr = FactorAnalyzer.get_factor_correlation(region=region)
        if not factor_corr:
            st.info("💡 ファクター相関データが取得できませんでした（データ期間が短い、またはAPI制限の可能性があります）。")
            return
            
        corr_df = pd.DataFrame(factor_corr)
        fig = px.imshow(
            corr_df, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            title=f"Factor Correlation ({region} Region - クラウディング・重複リスクの確認)"
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_rolling_correlation(port_returns, bm_returns, window=60):
        """市場との連動性（ローリング相関）の推移を可視化"""
        aligned = pd.concat([port_returns.rename("Portfolio"), bm_returns], axis=1).dropna()
        if len(aligned) < window:
            st.info(f"💡 ローリング相関を描画するための期間データ（{window}日分）が不足していますが、全体の分析には影響しません。")
            return
            
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
        names = list(crisis_results.keys())
        n_crises = len(names)
        
        if n_crises == 0:
            st.info("💡 このポートフォリオを構成する銘柄は、指定された過去の危機期間にデータが存在しないため、シミュレーションをスキップしました。")
            return
            
        for i, name in enumerate(names):
            data = crisis_results[name]
            st.markdown(f"**{name}**")
            st.markdown(f"- 最大下落幅 (Max Drawdown): **{data['max_drawdown_pct']:.2f}%**")
        
        st.info("📌 **仕様メモ:** 指定期間にまだ上場していなかった銘柄は自動的に除外され、当時存在していた銘柄のみでポートフォリオ比率を再配分（100%に正規化）してシミュレーションを行っています。")

    @staticmethod
    def plot_rolling_exposure(rolling_df):
        """ローリング回帰による動的エクスポージャーの推移を可視化（シンプソンのパラドックス検証用）"""
        if rolling_df is None or rolling_df.empty: return
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, subplot_titles=("Factor Betas (36-Month Rolling)", "Adjusted R-Squared (%)"))
        
        # Betas (超過リターンベース)
        for col in ["Market_Beta", "Size_Beta", "Value_Beta", "Quality_Beta", "Invest_Beta"]:
            if col in rolling_df.columns and not rolling_df[col].isna().all():
                fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df[col], name=col, mode='lines'), row=1, col=1)
                
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
        
        # Adjusted R2
        if "Adjusted_R2" in rolling_df.columns:
            fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["Adjusted_R2"], name="Adj R2", line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(height=500, title_text="Dynamic Factor Exposure (Regime Stability Check)", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_monte_carlo_fanchart(paths):
        """1万回のシミュレーション推移を扇状（ファンチャート）で可視化"""
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
        fig = px.histogram(
            final_values, nbins=50, 
            title="Final Value Distribution",
            labels={'value': 'Final Portfolio Value (1.0 = Initial)', 'count': 'Frequency'},
            color_discrete_sequence=["steelblue"]
        )
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
        
        fig = px.histogram(
            max_dds, nbins=50, 
            title="Simulated Max Drawdown Distribution",
            labels={'value': 'Maximum Drawdown (%)', 'count': 'Frequency'},
            color_discrete_sequence=["darkorange"]
        )
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
    
    # 💡修正ポイント: 地域(Region)の選択。ここで選択された内容が以降のすべての計算ロジックに波及する。
    region = st.sidebar.selectbox("Market Region", ["US", "Japan"], help="対象市場を選択してください。バックエンドの参照ファイルや無リスク金利が切り替わります。")
    config = MarketConfig.get_config(region)
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
    # 💡修正ポイント: ログの警告(use_container_width deprecated)に対応し width="stretch" を使用
    edited_df = st.sidebar.data_editor(
        st.session_state.portfolio_data, 
        num_rows="dynamic",
        width="stretch",
        hide_index=True
    )
    
    weights_dict = {}
    for _, row in edited_df.dropna().iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        if ticker and pd.notna(row["Weight"]):
            weights_dict[ticker] = float(row["Weight"])

    # --- 実行ボタン ---
    # 💡修正ポイント: ボタンも同様に警告対応
    if st.sidebar.button("Run Advanced Analysis", type="primary", use_container_width=True):
        if not weights_dict:
            st.error("有効なティッカーとウェイトを入力してください。")
            return
            
        with st.spinner(f"Initializing Quantitative Engine ({region} Market) & Fetching Data..."):
            
            # 1. データの準備
            norm_weights = DataFetcher.normalize_weights(weights_dict)
            input_ticker_count = len(norm_weights)
            
            raw_input_data = DataFetcher.fetch_market_data(list(norm_weights.keys()))
            valid_ticker_count = len(raw_input_data.columns) if not raw_input_data.empty else 0
            
            synthetic_portfolio = DataFetcher.create_synthetic_portfolio(norm_weights, region=region)
            
            if synthetic_portfolio is None or synthetic_portfolio.empty:
                st.error("データの構築に失敗しました。ティッカー記号（特に日本株の場合は .T の付与など）や期間を確認してください。")
                return
            
            if valid_ticker_count == input_ticker_count:
                st.success(f"✅ **データ網羅性:** 入力された全 {input_ticker_count} 銘柄のデータを取得し、シミュレーションに適用しました。")
            else:
                st.warning(f"⚠️ **データ網羅性:** 入力された {input_ticker_count} 銘柄中、**{valid_ticker_count} 銘柄** のみが計算に適用されました。")

            returns = synthetic_portfolio.pct_change().dropna()
            
            bm_prices = DataFetcher.fetch_market_data([config["benchmark_ticker"]])
            bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")

            # 2. 解析 (regionを渡して超過リターン等を含めて厳密に計算)
            metrics = AdvancedStats.calculate_metrics(returns, weights_dict=norm_weights, region=region)
            style = FactorAnalyzer.analyze_style(synthetic_portfolio, region=region)
            cycle_days = RegimeAnalyzer.detect_cycle(returns)
            rolling_exposure_df = DynamicFactorAnalyzer.calculate_rolling_exposure(synthetic_portfolio, region=region)
            
            # 3. タイムマシン＆シミュレーション
            crises = ["リーマン・ショック (2007-2009)", "コロナ・ショック (2020)", "ドットコム・バブル崩壊 (2000-2002)"]
            crisis_results = {}
            for crisis in crises:
                res = HistoryTimeMachine.replay_crisis(norm_weights, crisis, region)
                if res: crisis_results[crisis] = res
                
            projection = ProjectionCore.run_projection(returns, n_scenarios=10000, n_years=1)

            # ==========================================
            # 🗂️ 描画レイヤー (4つのタブ構成)
            # ==========================================
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 概要＆AI診断", 
                "⚠️ リスク＆連動性", 
                "🔮 将来シミュレーション", 
                "🔬 ファクター解析 (FF5)"
            ])

            # --- タブ1: 概要 ---
            with tab1:
                st.header(f"1. Core Risk Metrics & AI Diagnosis ({region} Market)")
                st.subheader("🤖 クオンツマネージャーの辛口診断")
                
                ai_prompt = AIPromptBuilder.generate_quant_prompt(metrics, style, target_name="現在のポートフォリオ")
                
                if ai_api_key:
                    st.info("APIキーが認識されました。（※実際の実装時はここでLLM APIを呼び出します）")
                else:
                    st.markdown("""
                    > **【AI診断ダミー表示】**
                    > あなたのポートフォリオを拝見しました。分散投資をしているつもりかもしれませんが、
                    > リスクの大半が特定の1銘柄に集中しており、実質的にその銘柄と心中している状態です。
                    > また、市場との連動性（Adjusted R-Squared）が非常に高く、高い手数料を払ってインデックスファンドと
                    > 同じ動きをしている「隠れインデックス」の兆候が見られます。
                    > **[ネクストアクション]** 早急に最大リスク寄与銘柄のウェイトを下げ、他セクターへの分散を図りなさい。
                    """)
                
                with st.expander("🔍 裏で生成されたAIへの命令書（プロンプト）を見る"):
                    st.text(ai_prompt)
                    
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
                col2.metric("Portfolio Volatility (Shrunk)", f"{metrics.get('volatility', 0) * 100:.2f}%")
                col3.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                col4.metric("Max Drawdown", f"{metrics.get('max_dd', 0) * 100:.2f}%")
                
                st.subheader("Historical Cumulative Growth")
                aligned_growth = pd.concat([synthetic_portfolio.rename("Portfolio"), (1+bm_returns).cumprod()*100], axis=1).dropna()
                st.line_chart(aligned_growth)
                AuditEngine.plot_underwater_drawdown(synthetic_portfolio)

            # --- タブ2: リスクと連動性 ---
            with tab2:
                st.header("2. Risk Diversification & Stress Tests")
                c_col1, c_col2 = st.columns(2)
                
                with c_col1:
                    st.subheader(f"Factor Correlation Matrix ({region})")
                    AuditEngine.plot_factor_correlation(region=region)
                    
                with c_col2:
                    st.subheader("Market Correlation (Rolling 60-Day)")
                    AuditEngine.plot_rolling_correlation(returns, bm_returns)
                
                st.divider()
                st.subheader("History Time Machine (Crash Recovery Paths)")
                st.markdown("過去の主要な金融危機の際、当ポートフォリオがどのように下落し、**どの程度の期間で回復したか**を追体験します。")
                AuditEngine.plot_crisis_replays(crisis_results)

            # --- タブ3: 将来シミュレーション ---
            with tab3:
                st.header("3. Stochastic Projection (GARCH & Student's t)")
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

            # --- タブ4: ファクター解析 (FF5 プロフェッショナル・ダッシュボード化) ---
            with tab4:
                st.header(f"4. Causal Factor Analysis ({region} Fama-French 5-Factor)")
                st.markdown("ポートフォリオの背後にある「リスクの源泉」を統計的に分解し、その因果的妥当性と安定性を評価します。無リスク金利(RF)を控除した**超過リターンベース**で計算されています。")
                
                if style and style.get('status') != 'insufficient_data':
                    
                    high_vif_factors = [f for f, v in style.get('vif', {}).items() if v > 10]
                    if high_vif_factors:
                        st.warning(f"⚠️ **多重共線性の警告 (VIF > 10):** ファクター [{', '.join(high_vif_factors)}] の間で強い相関（似たような動き）が検出されました。これらのベータ値は統計的に不安定（信頼性が低い）可能性があります。")

                    st.subheader("📊 Factor Sensitivity (市場・スタイル感度)")
                    
                    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                    
                    # 💡修正ポイント: 超過リターンであることをUI上で明示
                    mkt_beta = style.get('beta_market', 1.0)
                    f_col1.metric("Market-RF (市場超過ベータ)", f"{mkt_beta:.2f}", delta="ハイリスク" if mkt_beta > 1.2 else ("ローリスク" if mkt_beta < 0.8 else ""), delta_color="inverse")
                    
                    size_beta = style.get('beta_size', 0.0)
                    f_col2.metric("SMB (企業規模)", f"{size_beta:.2f}", delta="小型株寄り" if size_beta > 0 else "大型株寄り", delta_color="off")
                    
                    val_beta = style.get('beta_value', 0.0)
                    f_col3.metric("HML (割安性)", f"{val_beta:.2f}", delta="割安株寄り" if val_beta > 0 else "成長株寄り", delta_color="off")
                    
                    alpha_val = style.get('alpha', 0.0) * 100
                    f_col4.metric("Alpha (年率固有・超過収益)", f"{alpha_val:.3f}%", delta=f"{alpha_val:.3f}%", delta_color="normal", help="無リスク金利と市場・スタイルの要因を差し引いた後の、ポートフォリオ固有の純粋な超過リターン（年率）です。")
                    
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
                    
                    s_col1.metric("Adjusted R-Squared", f"{r2:.1f}%", help="変数の数によるペナルティを課した「真の決定係数」。この数値が高いほど、モデルの当てはまりが論理的に正しいことを示します。")
                    s_col2.metric("Market P-Value", f"{p_val_market:.3f}", help="0.05未満であれば統計的に有意（偶然ではない）です。")
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
