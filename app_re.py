"""
app.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
※ Plotly対応 ＆ サイドバー即時編集（st.data_editor）統合版
※ Step 3: AIプロンプト連携 ＆ ファクター解析詳細化 追加版
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# これまでに作成したモジュールのインポート
from config import MarketConfig, FACTOR_TRANSLATION
from data_engine import DataFetcher
from analytics import AdvancedStats, FactorAnalyzer, AIPromptBuilder
from simulation import RegimeAnalyzer, HistoryTimeMachine, ProjectionCore

# =========================================================
# 📊 監査・可視化エンジンクラス (Plotly インタラクティブ版)
# =========================================================
class AuditEngine:
    
    @staticmethod
    def plot_correlation_matrix(returns_df):
        """銘柄間の相関関係をヒートマップで可視化（Plotly）"""
        corr = returns_df.corr()
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            title="Asset Correlation Matrix (Interactive)"
        )
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_rolling_correlation(port_returns, bm_returns, window=60):
        """市場との連動性（ローリング相関）の推移を可視化（Plotly）"""
        aligned = pd.concat([port_returns.rename("Portfolio"), bm_returns], axis=1).dropna()
        if len(aligned) < window:
            st.warning("ローリング相関を描画するための十分な期間データがありません。")
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
    def plot_time_machine_chart(crisis_results):
        """過去の危機における最大ドローダウンを棒グラフで表示（Plotly）"""
        names = list(crisis_results.keys())
        dd_values = [crisis_results[n]['max_drawdown_pct'] for n in names]
        
        fig = px.bar(
            x=dd_values, y=names, orientation='h',
            text=[f"{v:.1f}%" for v in dd_values],
            labels={'x': 'Max Drawdown (%)', 'y': 'Crisis Event'},
            title="Historical Stress Tests (Max Drawdown)",
            color=dd_values, color_continuous_scale="Reds_r"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_monte_carlo_fanchart(paths):
        """1万回のシミュレーション推移を扇状（ファンチャート）で可視化（Plotly）"""
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        days = np.arange(paths.shape[0])
        
        fig = go.Figure()
        
        # 5th to 95th Percentile
        fig.add_trace(go.Scatter(x=days, y=percentiles[4], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=percentiles[0], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.2)', line=dict(width=0), name='5th-95th Percentile'))
        
        # 25th to 75th Percentile
        fig.add_trace(go.Scatter(x=days, y=percentiles[3], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=percentiles[1], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.4)', line=dict(width=0), name='25th-75th Percentile'))
        
        # Median
        fig.add_trace(go.Scatter(x=days, y=percentiles[2], mode='lines', line=dict(color='darkblue', width=2), name='Median (50th)'))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(title="Monte Carlo Projection (1-Year Fan Chart)", xaxis_title="Trading Days", yaxis_title="Portfolio Value", height=450)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_monte_carlo_histogram(final_values):
        """最終資産分布の正確なヒストグラム（Plotly）"""
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
        """1万回のシナリオにおける最大ドローダウンの分布（Plotly）"""
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
    
    # 将来的にAIと連携するためのAPIキー入力欄
    st.sidebar.markdown("**🤖 AI診断用 API設定**")
    ai_api_key = st.sidebar.text_input("API Key (現在プレースホルダー)", type="password", help="ここにOpenAI等のAPIキーを入れると本物のAIが動くようになります（次ステップ以降）")
    st.sidebar.divider()
    
    region = st.sidebar.selectbox("Market Region", ["US", "Japan"])
    
    st.sidebar.markdown("**📊 銘柄とウェイトを入力（直接編集可能）**")
    
    # セッションステートを使って初期データを保持
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL"],
            "Weight": [40.0, 40.0, 20.0]
        })
    
    # st.data_editorでエクセルライクな入力UIを提供
    edited_df = st.sidebar.data_editor(
        st.session_state.portfolio_data, 
        num_rows="dynamic", # 行の追加・削除を許可
        use_container_width=True,
        hide_index=True
    )
    
    # 辞書型に変換
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
            
        with st.spinner("Initializing Quantitative Engine & Fetching Data..."):
            
            # 1. データの準備と配線
            norm_weights = DataFetcher.normalize_weights(weights_dict)
            synthetic_portfolio = DataFetcher.create_synthetic_portfolio(norm_weights, region=region)
            
            if synthetic_portfolio is None or synthetic_portfolio.empty:
                st.error("データの構築に失敗しました。ティッカー記号を確認してください。")
                return
            
            returns = synthetic_portfolio.pct_change().dropna()
            raw_prices = DataFetcher.fetch_market_data(list(norm_weights.keys()))
            raw_returns = raw_prices.pct_change().dropna()
            
            config = MarketConfig.get_config(region)
            bm_prices = DataFetcher.fetch_market_data([config["benchmark_ticker"]])
            bm_returns = bm_prices.pct_change().iloc[:, 0].rename("Benchmark")

            # 2. リスク・アルファ解析
            metrics = AdvancedStats.calculate_metrics(returns, weights_dict=norm_weights, region=region)
            style = FactorAnalyzer.analyze_style(synthetic_portfolio, region=region)
            cycle_days = RegimeAnalyzer.detect_cycle(returns)
            
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
                "⚠️ リスク＆連動性分析", 
                "🔮 将来シミュレーション", 
                "🔬 ファクター解析"
            ])

            # --- タブ1: 概要 ---
            with tab1:
                st.header("1. Core Risk Metrics & AI Diagnosis")
                
                # 💡【新規追加】AIによる「プロの小言」セクション
                st.subheader("🤖 クオンツマネージャーの辛口診断")
                
                # analytics.pyで追加したクラスを使ってプロンプト（命令書）を生成
                ai_prompt = AIPromptBuilder.generate_quant_prompt(metrics, style, target_name="現在のポートフォリオ")
                
                if ai_api_key:
                    # TODO: 実際のAPI呼び出し（OpenAI, Geminiなど）をここに実装予定
                    st.info("APIキーが認識されました。（※実際の実装時はここでLLM APIを呼び出します）")
                else:
                    # APIキーがない場合のプレースホルダー表示
                    st.markdown("""
                    > **【AI診断ダミー表示】**
                    > あなたのポートフォリオを拝見しました。分散投資をしているつもりかもしれませんが、
                    > リスクの大半が特定の1銘柄に集中しており、実質的にその銘柄と心中している状態です。
                    > また、市場との連動性（R2）が非常に高く、高い手数料を払ってインデックスファンドと
                    > 同じ動きをしている「隠れインデックス」の兆候が見られます。
                    > **[ネクストアクション]** 早急に最大リスク寄与銘柄のウェイトを下げ、他セクターへの分散を図りなさい。
                    """)
                
                # 実際にどんな命令が裏で作られたのかを見せるアコーディオン
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
                    st.subheader("Asset Correlation Matrix")
                    if not raw_returns.empty:
                        AuditEngine.plot_correlation_matrix(raw_returns)
                with c_col2:
                    st.subheader("Market Correlation (Rolling 60-Day)")
                    AuditEngine.plot_rolling_correlation(returns, bm_returns)
                
                st.divider()
                st.subheader("History Time Machine (Crash Replay)")
                if crisis_results:
                    AuditEngine.plot_time_machine_chart(crisis_results)
                else:
                    st.warning("危機期間のシミュレーションに必要なデータが合成できませんでした。")

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

            # --- タブ4: ファクター解析 ---
            with tab4:
                st.header("4. Market Regime & Factor Exposure")
                st.subheader("Fama-French 3-Factor Model")
                if style:
                    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                    f_col1.metric("Market Beta (市場連動性)", f"{style.get('beta_market', 1.0):.2f}")
                    f_col2.metric("Size (小型株効果)", f"{style.get('beta_size', 0.0):.2f}")
                    f_col3.metric("Value (割安株効果)", f"{style.get('beta_value', 0.0):.2f}")
                    f_col4.metric("Alpha", f"{style.get('alpha', 0.0) * 100:.3f}%")
                    
                    # 💡【新規追加】統計的信頼性（決定係数とP値）の表示
                    st.markdown("##### 📈 統計的信頼性 (Statistical Reliability)")
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("R-Squared (決定係数)", f"{style.get('r_squared', 0.0) * 100:.1f}%", help="この数値が高いほど、市場全体の動きだけでポートフォリオの動きが説明できる（インデックスに近い）ことを示します。")
                    s_col2.metric("Market P-Value", f"{style.get('p_value_market', 1.0):.3f}")
                    s_col3.metric("Size/Value P-Value", f"{style.get('p_value_size', 1.0):.3f} / {style.get('p_value_value', 1.0):.3f}")
                else:
                    st.warning("ファクター分析に必要な期間のデータが不足しています。")
                
                st.divider()
                
                # 💡【新規追加】ファクター同士の相関クラウディング確認
                st.subheader("Factor Correlation (ファクターのクラウディング確認)")
                factor_corr = FactorAnalyzer.get_factor_correlation(region=region)
                if factor_corr:
                    corr_df = pd.DataFrame(factor_corr)
                    fig_corr = px.imshow(
                        corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", 
                        zmin=-1, zmax=1, title="Fama-French Factors Correlation"
                    )
                    fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                st.divider()
                st.subheader("Volatility Cycle Detection")
                if cycle_days:
                    st.info(f"ウェルチ法による現在のボラティリティ周期: **約 {cycle_days} 日**")

if __name__ == "__main__":
    main()
