import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

# 将来の警告を無視する設定
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# 🔗 モジュール読み込みチェック
# =========================================================
try:
    from simulation_engine import MarketDataEngine, PortfolioAnalyzer, PortfolioDiagnosticEngine
    from pdf_generator import create_pdf_report
except ImportError as e:
    st.error(f"❌ 重要ファイルが見つかりません: {e}")
    st.info("app.py と同じフォルダに 'simulation_engine.py' と 'pdf_generator.py' があるか確認してください。")
    st.stop()

# =========================================================
# ⚙️ 定数・設定
# =========================================================

# 🎨 カラーパレット
COLORS = {
    'main': '#00FFFF',      # Neon Cyan
    'benchmark': '#FF69B4', # Hot Pink
    'principal': '#FFFFFF', # White
    'median': '#32CD32',    # Lime Green
    'mean': '#FFD700',      # Gold
    'p10': '#FF6347',       # Pessimistic
    'p90': '#00BFFF',       # Optimistic
    'hist_bar': '#42A5F5',  # Mid Blue
    'cost_net': '#FF6347',  # Tomato Red
    'bg_fill': 'rgba(0, 255, 255, 0.1)'
}

st.set_page_config(page_title="Factor Simulator V17.2", layout="wide", page_icon="🧬")

# CSSスタイリング
st.markdown("""
<style>
    .metric-card { background-color: #262730; border: 1px solid #444; padding: 15px; border-radius: 8px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #00FFFF; color: black; font-weight: bold; }
    .report-box { border-left: 5px solid #00FFFF; padding-left: 15px; margin-top: 10px; background-color: rgba(0, 255, 255, 0.05); }
    .factor-box { border-left: 5px solid #FF69B4; padding-left: 15px; margin-top: 10px; background-color: rgba(255, 105, 180, 0.05); }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Helvetica', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Factor & Stress Test Simulator V17.2")
st.caption("Professional Edition: Portfolio Diagnosis, Monte Carlo, Risk Analysis (Stable Version)")

# =========================================================
# 🛠️ セッション状態の初期化
# =========================================================
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'payload' not in st.session_state:
    st.session_state.payload = None
if 'figs' not in st.session_state:
    st.session_state.figs = {}

# =========================================================
# 🏗️ サイドバー: ポートフォリオ設定
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings Panel")

    st.markdown("### 1. Portfolio Composition")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="Required columns: 'Ticker', 'Weight'")
    
    default_input = "SPY: 40, VWO: 20, 7203.T: 20, GLD: 20"
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if df_upload.shape[1] >= 2:
                tickers_up = df_upload.iloc[:, 0].astype(str)
                weights_up = df_upload.iloc[:, 1].astype(str)
                formatted_list = [f"{t}: {w}" for t, w in zip(tickers_up, weights_up)]
                default_input = ", ".join(formatted_list)
                st.success("✅ CSV Loaded")
            else:
                st.error("CSV must have at least 2 columns (Ticker, Weight).")
        except Exception as e:
            st.error(f"Load Error: {e}")

    input_text = st.text_area("Ticker: Weight (Input)", value=default_input, height=100)

    st.markdown("### 2. Analysis Model & Benchmark")
    target_region = st.selectbox("Analysis Region", ["US (United States)", "Japan", "Global"], index=0)
    region_code = target_region.split()[0]
    
    bench_options = {
        'US': {'S&P 500 (^GSPC)': '^GSPC', 'NASDAQ 100 (^NDX)': '^NDX'},
        'Japan': {'TOPIX (1306 ETF)': '1306.T', 'Nikkei 225 (^N225)': '^N225'},
        'Global': {'VT (Total World)': 'VT', 'MSCI ACWI (Index)': 'ACWI'}
    }
    selected_bench_label = st.selectbox("Benchmark", list(bench_options[region_code].keys()) + ["Custom"])

    if selected_bench_label == "Custom":
        bench_ticker = st.text_input("Benchmark Ticker", value="^GSPC")
    else:
        bench_ticker = bench_options[region_code][selected_bench_label]

    st.markdown("### 3. Cost Settings")
    cost_tier = st.select_slider("Management Cost", options=["Low", "Medium", "High"], value="Medium")

    st.markdown("### 4. Advisor's Note")
    st.caption("✍️ Add your personal message. This appears at the top of the PDF.")
    
    default_note = "Based on our strategy session, I recommend maintaining this allocation to balance growth and stability."
    advisor_note = st.text_area("Message to Client (English Only):", 
                                value=default_note,
                                height=100)

    st.markdown("---")
    analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)


# =========================================================
# 🚀 メインロジック (計算実行)
# =========================================================

if analyze_btn:
    with st.spinner("⏳ Fetching data & running 7,500 simulations..."):
        try:
            # 1. 入力解析
            raw_items = [item.strip() for item in input_text.split(',')]
            parsed_dict = {}
            for item in raw_items:
                try:
                    k, v = item.split(':')
                    parsed_dict[k.strip()] = float(v.strip())
                except: pass

            if not parsed_dict: st.stop()

            # 🚀 Engine 呼び出し
            engine = MarketDataEngine()
            valid_assets, _ = engine.validate_tickers(parsed_dict)
            if not valid_assets:
                st.error("有効なティッカーが見つかりませんでした。")
                st.stop()

            tickers = list(valid_assets.keys())
            hist_returns = engine.fetch_historical_prices(tickers)

            if hist_returns.empty:
                 st.error("価格データの取得に失敗しました。")
                 st.stop()

            weights_clean = {k: v['weight'] for k, v in valid_assets.items()}
            port_series, final_weights = PortfolioAnalyzer.create_synthetic_history(hist_returns, weights_clean)

            # 2. ベンチマーク取得
            is_jpy_bench = True if bench_ticker in ['^TPX', '^N225', '1306.T'] or bench_ticker.endswith('.T') else False
            bench_series = engine.fetch_benchmark_data(bench_ticker, is_jpy_asset=is_jpy_bench)

            # 3. ファクター取得
            french_factors = engine.fetch_french_factors(region_code)

            # データ保存
            st.session_state.portfolio_data = {
                'returns': port_series,
                'benchmark': bench_series,
                'components': hist_returns,
                'weights': final_weights,
                'factors': french_factors,
                'asset_info': valid_assets,
                'cost_tier': cost_tier,
                'bench_name': selected_bench_label,
            }
            
            # 再計算時にキャッシュをクリア
            st.session_state.pdf_bytes = None
            st.session_state.analysis_done = False

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.stop()


# =========================================================
# 📊 ダッシュボード表示 & PDF用データ準備
# =========================================================

if st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    analyzer = PortfolioAnalyzer()
    port_ret = data['returns']
    bench_ret = data['benchmark']

    # --- 1. 基本指標 ---
    total_ret_cum = (1 + port_ret).cumprod()
    cagr = (total_ret_cum.iloc[-1])**(12/len(port_ret)) - 1
    vol = port_ret.std() * np.sqrt(12)
    max_dd = (total_ret_cum / total_ret_cum.cummax() - 1).min()
    calmar = analyzer.calculate_calmar_ratio(port_ret)
    omega = analyzer.calculate_omega_ratio(port_ret, threshold=0.0) 
    info_ratio, track_err = analyzer.calculate_information_ratio(port_ret, bench_ret)
    sharpe_ratio = (cagr - 0.02) / vol # Simplified Sharpe

    # --- 2. 高度計算 ---
    params, r_sq = analyzer.perform_factor_regression(port_ret, data['factors'])
    if params is not None:
        factor_comment = PortfolioDiagnosticEngine.generate_factor_report(params)
    else:
        factor_comment = "No factor data available."

    # モンテカルロ
    sim_years = 20
    init_inv = 1000000
    df_stats, final_values = analyzer.run_monte_carlo_simulation(port_ret, n_years=sim_years, n_simulations=7500, initial_investment=init_inv)
    
    final_median = np.median(final_values)
    final_p10 = np.percentile(final_values, 10)
    final_p90 = np.percentile(final_values, 90)
    
    # 相関行列
    corr_matrix = analyzer.calculate_correlation_matrix(data['components'])
    fig_corr_report = None
    if not corr_matrix.empty:
        fig_corr_report = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    # AI診断
    pca_ratio, _ = analyzer.perform_pca(data['components'])
    report = PortfolioDiagnosticEngine.generate_report(data['weights'], pca_ratio, port_ret)

    # ▼▼▼ 詳細レビュー生成 ▼▼▼
    detailed_review = []
    
    # 効率性評価
    if sharpe_ratio > 1.0:
        detailed_review.append(f"✅ Efficiency: The portfolio demonstrates excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f}). You are getting well-compensated for the risk taken.")
    elif sharpe_ratio > 0.6:
        detailed_review.append(f"ℹ️ Efficiency: The portfolio has a balanced risk/return profile (Sharpe: {sharpe_ratio:.2f}), typical for a diversified equity strategy.")
    else:
        detailed_review.append(f"⚠️ Efficiency: Risk-adjusted returns are lower than ideal (Sharpe: {sharpe_ratio:.2f}). Consider increasing diversification or reducing volatile assets.")

    # ボラティリティ評価
    if vol < 0.12:
        detailed_review.append(f"🛡️ Stability: Volatility is low ({vol:.2%}), suggesting a defensive posture suitable for capital preservation.")
    elif vol < 0.18:
        detailed_review.append(f"⚖️ Stability: Volatility is moderate ({vol:.2%}), aligning with standard market fluctuations.")
    else:
        detailed_review.append(f"🔥 Stability: Volatility is high ({vol:.2%}). Ensure your risk tolerance matches this potential variance.")

    # ドローダウン評価
    detailed_review.append(f"📉 Stress Test: The historical maximum drawdown was {max_dd:.2%}. In future bear markets, expect temporary declines of similar magnitude.")

    detailed_review_str = "\n".join(detailed_review)

    # --- 3. Payload 作成 ---
    analysis_payload = {
        'metrics': {
            'CAGR': f"{cagr:.2%}",
            'Volatility': f"{vol:.2%}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Calmar Ratio': f"{calmar:.2f}",
            'Information Ratio': f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "N/A"
        },
        'factor_comment': factor_comment,
        'ai_diagnosis': {
            'status': report['diversification_comment'],
            'risk': report['risk_comment'],
            'action': report['action_plan']
        },
        'detailed_review': detailed_review_str,
        'mc_stats': f"Median Outlook: {final_median:,.0f} JPY | "
                    f"Pessimistic (10%): {final_p10:,.0f} JPY | "
                    f"Optimistic (90%): {final_p90:,.0f} JPY\n\n"
    }

    figs_for_report = {}
    if fig_corr_report:
        figs_for_report['correlation'] = fig_corr_report

    # --- 4. ビジュアライゼーション表示 ---
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{cagr:.2%}")
    c2.metric("Vol (Risk)", f"{vol:.2%}")
    c3.metric("Max DD", f"{max_dd:.2%}", delta_color="inverse")
    c4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    c5.metric("Omega Ratio", f"{omega:.2f}")

    if not np.isnan(info_ratio):
        st.caption(f"📊 vs {data['bench_name']} | Information Ratio: **{info_ratio:.2f}** (Tracking Error: {track_err:.2%})")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧬 DNA", "🌊 Factors", "⏳ History", "💸 Cost", "🏆 Attribution", "🔮 Future"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Diversification Quality")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = pca_ratio * 100, 
                title = {'text': "1st PCA Component Dominance (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['main']},
                         'steps': [{'range': [0, 60], 'color': "#333"}, {'range': [60, 100], 'color': "#555"}],
                         'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Asset Allocation")
            fig_pie = px.pie(values=list(data['weights'].values()), names=list(data['weights'].keys()), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            figs_for_report['pie'] = fig_pie

        with c2:
            st.subheader("🩺 Portfolio Diagnosis")
            st.markdown(f"""
            <div class="report-box">
                <h3 style="color: #00FFFF; margin-bottom:0px;">{report['type']}</h3>
                <hr style="margin-top:5px; margin-bottom:10px; border-color: #555;">
                <p><b>🧐 Status:</b><br>{report['diversification_comment']}</p>
                <p><b>⚠️ Risk Alert:</b><br>{report['risk_comment']}</p>
                <p><b>💡 Action Plan:</b><br>{report['action_plan']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"🤖 **AI Analysis:**\n\n{detailed_review_str}")

            st.markdown("---")
            st.subheader("🔥 Correlation Heatmap")
            if fig_corr_report:
                st.plotly_chart(fig_corr_report, use_container_width=True)

    with tab2:
        if data['factors'].empty:
            st.error("🚫 Failed to fetch factor data.")
        else:
            st.subheader("📊 Style Analysis (Regression)")
            if params is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    beta_df = params.drop('const') if 'const' in params else params
                    colors = ['#00CC96' if x > 0 else '#FF4B4B' for x in beta_df.values]
                    fig_beta = go.Figure(go.Bar(
                        x=beta_df.values, y=beta_df.index, orientation='h', 
                        marker_color=colors, text=[f"{x:.2f}" for x in beta_df.values], textposition='auto'
                    ))
                    fig_beta.update_layout(title="Factor Beta Sensitivity", xaxis_title="Sensitivity", height=300)
                    st.plotly_chart(fig_beta, use_container_width=True)
                    st.caption(f"R-Squared (R²): {r_sq:.2%} (Model explains {r_sq*100:.0f}% of movement)")
                    figs_for_report['factor_beta'] = fig_beta
                
                with c2:
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4 style="color: #FF69B4; margin-bottom:10px;">🧠 AI Style Analysis</h4>
                        <div style="white-space: pre-wrap;">{factor_comment}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📈 Rolling Beta Analysis")
            rolling_betas = analyzer.rolling_beta_analysis(port_ret, data['factors'])
            if not rolling_betas.empty:
                fig_roll = go.Figure()
                if 'Mkt-RF' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['Mkt-RF'], name='Market (Beta)', line=dict(width=3, color=COLORS['main'])))
                if 'SMB' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['SMB'], name='Size (SMB)', line=dict(dash='dot', color='orange')))
                if 'HML' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['HML'], name='Value (HML)', line=dict(dash='dot', color='yellow')))
                st.plotly_chart(fig_roll, use_container_width=True)

    with tab3:
        st.subheader("Historical Stress Test")
        cum_ret = (1 + port_ret).cumprod() * 10000
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=[10000]*len(cum_ret), mode='lines', name='Principal (10,000)', line=dict(color=COLORS['principal'], width=1, dash='dot')))

        if not bench_ret.empty:
            bench_cum = (1 + bench_ret).cumprod()
            common_idx = cum_ret.index.intersection(bench_cum.index)
            bench_cum = bench_cum.loc[common_idx]
            bench_cum = bench_cum / bench_cum.iloc[0] * 10000
            fig_hist.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, mode='lines', name=f"Benchmark ({data['bench_name']})", line=dict(color=COLORS['benchmark'], width=1.5)))

        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, fill='tozeroy', fillcolor=COLORS['bg_fill'], mode='lines', name='My Portfolio', line=dict(color=COLORS['main'], width=2.5)))
        st.plotly_chart(fig_hist, use_container_width=True)
        figs_for_report['history'] = fig_hist

        st.markdown("---")
        st.subheader("📊 Return Distribution")
        mu, std = port_ret.mean(), port_ret.std()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=port_ret, histnorm='probability density', name='Actual', marker_color=COLORS['hist_bar'], opacity=0.8, nbinsx=50))
        x_range = np.linspace(port_ret.min(), port_ret.max(), 100)
        y_norm = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_range - mu) / std) ** 2)
        fig_dist.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', name='Normal Dist (Theory)', line=dict(color='white', dash='dash', width=2)))
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        st.subheader("Cost Drag Analysis")
        gross, net, loss, cost_pct = analyzer.cost_drag_simulation(port_ret, data['cost_tier'])
        loss_amount = 1000000 * loss
        final_amount_net = 1000000 * net.iloc[-1]
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(x=gross.index, y=gross, name='Gross (Ideal)', line=dict(color='gray', dash='dot')))
            fig_cost.add_trace(go.Scatter(x=net.index, y=net, name=f'Net (Actual)', fill='tonexty', line=dict(color=COLORS['cost_net'])))
            st.plotly_chart(fig_cost, use_container_width=True)
        with c2:
            st.error(f"💸 Lost Value: ▲{loss_amount:,.0f} JPY")
            st.markdown(f"Final Value (1M Investment): **{final_amount_net:,.0f} JPY**")

    with tab5:
        st.subheader("Strict Attribution Analysis")
        attrib = analyzer.calculate_strict_attribution(data['components'], data['weights'])
        if not attrib.empty:
            colors = ['#FF4B4B' if x < 0 else '#00CC96' for x in attrib.values]
            fig_attr = go.Figure(go.Bar(
                x=attrib.values, y=attrib.index, orientation='h', marker_color=colors,
                text=[f"{x:.2%}" for x in attrib.values], textposition='auto'
            ))
            fig_attr.update_layout(xaxis_title="Contribution", yaxis_title="Asset")
            st.plotly_chart(fig_attr, use_container_width=True)
            figs_for_report['attribution'] = fig_attr

    with tab6:
        st.subheader("🎲 Monte Carlo Simulation (7,500 runs / Fat-Tail)")
        if df_stats is not None:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p50'], mode='lines', name='Median', line=dict(color=COLORS['median'], width=3)))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p10'], mode='lines', name='Bottom 10%', line=dict(color=COLORS['p10'], width=1, dash='dot')))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p90'], mode='lines', name='Top 10%', line=dict(color=COLORS['p90'], width=1, dash='dot')))
            fig_mc.update_layout(title=f"20-Year Forecast (Principal: {init_inv:,} JPY)", yaxis_title="Value (JPY)", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)

            st.markdown("### 🏁 Final Outcome Distribution")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("P10 (Bear)", f"{final_p10:,.0f}", delta_color="inverse")
            mc2.metric("Median", f"{final_median:,.0f}")
            mc3.metric("Mean", f"{np.mean(final_values):,.0f}")
            mc4.metric("P90 (Bull)", f"{final_p90:,.0f}")

            fig_mc_hist = go.Figure()
            counts, _ = np.histogram(final_values, bins=100)
            y_max_freq = counts.max()
            x_max_view = np.percentile(final_values, 98)

            fig_mc_hist.add_trace(go.Histogram(
                x=final_values, nbinsx=100, name='Freq', 
                marker_color=COLORS['hist_bar'], opacity=0.85
            ))
            lines_config = [
                (final_p10, COLORS['p10'], "P10", 1.05, "dash", 2),
                (final_median, COLORS['median'], "Median", 1.15, "solid", 3),
                (final_p90, COLORS['p90'], "P90", 1.05, "dash", 2),
            ]
            for val, color, label, h_rate, dash, width in lines_config:
                fig_mc_hist.add_vline(x=val, line_width=width, line_dash=dash, line_color=color)

            fig_mc_hist.update_layout(
                xaxis_title="Final Value (JPY)", yaxis_title="Count", showlegend=False,
                xaxis=dict(range=[0, x_max_view]), yaxis=dict(range=[0, y_max_freq * 1.4])
            )
            st.plotly_chart(fig_mc_hist, use_container_width=True)
            figs_for_report['mc'] = fig_mc_hist
            st.success(f"✅ Simulation Complete: **7,500 scenarios** generated.")

    # --- 5. データ保存 ---
    st.session_state.payload = analysis_payload
    st.session_state.figs = figs_for_report
    st.session_state.analysis_done = True


# =========================================================
# 📄 PDF ダウンロードセクション
# =========================================================
st.markdown("---")

if st.session_state.analysis_done:
    st.header("📄 Generate Report")
    st.caption("Download the analysis results as a PDF.")

    col_gen, col_dl = st.columns([1, 1])

    with col_gen:
        if st.button("📥 Create PDF Report"):
            with st.spinner("📄 Generating PDF..."):
                try:
                    final_payload = st.session_state.payload.copy()
                    final_payload['advisor_note'] = advisor_note 

                    if final_payload and st.session_state.figs:
                        pdf_data = create_pdf_report(final_payload, st.session_state.figs)
                        
                        if pdf_data and len(pdf_data) > 0:
                            st.session_state.pdf_bytes = pdf_data
                            st.success(f"✅ Report Ready! Size: {len(pdf_data)} bytes")
                        else:
                            st.error("⚠️ PDF generation returned empty data.")
                            st.session_state.pdf_bytes = None
                    else:
                        st.error("⚠️ Data missing. Please run simulation again.")
                        
                except Exception as e:
                    st.error(f"PDF Error: {e}")

    with col_dl:
        if st.session_state.pdf_bytes is not None:
            st.download_button(
                label="⬇️ Download PDF File",
                data=st.session_state.pdf_bytes,
                file_name="Portfolio_Analysis_Report.pdf",
                mime="application/pdf",
                type="primary"
            )

else:
    st.info("ℹ️ To generate a PDF report, please run the simulation first.")
