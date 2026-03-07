import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# 🔄 [Phase 1: 修正] エンジン読み込み (New Engine Integration)
# =========================================================
try:
    from auditor_engine import (
        DataFetcher, AdvancedStats, FactorAnalyzer, 
        StochasticScenarioGenerator, ProjectionCore, 
        AuditEngine, HistoryTimeMachine
    )
except ImportError as e:
    st.error(f"エンジンの読み込みに失敗しました: {e}")
    st.stop()

# =========================================================
# 📝 [Phase 1: 追加] 用語のマッピング辞書 (Soft Language)
# =========================================================
FACTOR_TRANSLATION = {
    "Mkt-RF": "市場全体 (Market)",
    "SMB": "小型株効果 (Size)",
    "HML": "割安株効果 (Value)",
    "RMW": "収益性 (Profitability)",
    "CMA": "投資態度 (Investment)",
    "Mom": "モメンタム (Trend)"
}

# =========================================================
# 🎨 ページ設定 & CSSデザイン
# =========================================================
st.set_page_config(page_title="Portfolio Auditor Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* 全体のフォントと背景調整 */
    .main { background-color: #0E1117; color: #FAFAFA; }
    
    /* KPIカードのデザイン */
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 0.9em; color: #A0A0A0; margin-bottom: 5px; }
    .metric-value { font-size: 1.8em; font-weight: bold; color: #FFFFFF; }
    .metric-delta { font-size: 0.8em; font-weight: bold; }
    
    /* 🚦 信号機カラー定義 */
    .delta-pos { color: #00CC96; } /* Green: 安全 */
    .delta-warn { color: #F5A623; } /* Yellow: 注意 */
    .delta-neg { color: #EF553B; } /* Red: 警告 */
    
    /* アドバイザーカード */
    .advisor-card {
        background-color: #16213E;
        border-left: 5px solid #4B7BFF;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* ボタンカスタマイズ */
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 8px; }
    
    /* タブのスタイル */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 5px 5px 0 0;
        padding: 0 20px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🏗️ セッション状態の管理
# =========================================================
if 'audit_result' not in st.session_state:
    st.session_state.audit_result = None
if 'simulation_result' not in st.session_state:
    st.session_state.simulation_result = None
if 'target_series' not in st.session_state:
    st.session_state.target_series = None
if 'region_code' not in st.session_state:
    st.session_state.region_code = "US"

# =========================================================
# 📂 サイドバー: データ入力 (Input)
# =========================================================
with st.sidebar:
    st.title("🛡️ Auditor Pro")
    st.caption("Professional Risk Analysis System")
    st.markdown("---")
    
    input_tab, settings_tab = st.tabs(["📂 ポートフォリオ", "⚙️ 設定"])
    
    input_weights_dict = None
    
    with input_tab:
        st.subheader("🌍 Market & Assets")
        
        market_choice = st.radio("分析対象マーケット", ["🇺🇸 米国市場 (US)", "🇯🇵 日本市場 (Japan)"], horizontal=True)
        st.session_state.region_code = "US" if "US" in market_choice else "Japan"
        
        if st.session_state.region_code == "US":
            default_input = "SPY: 60\nTLT: 40"
        else:
            default_input = "1321.T: 60\n2510.T: 40"

        input_str = st.text_area("Ticker: Weight (%)", value=default_input, height=150)
        
        if input_str:
            try:
                weights = {}
                for line in input_str.split('\n'):
                    if ':' in line: k, v = line.split(':')
                    elif ',' in line: k, v = line.split(',')
                    else: continue
                    weights[k.strip()] = float(v.strip())
                input_weights_dict = weights
                
                total_w = sum(weights.values())
                if abs(total_w - 100) > 1:
                    st.warning(f"⚠️ 合計が {total_w:.1f}% です (100%推奨)")
                else:
                    st.caption(f"✅ 合計: {total_w:.1f}%")
            except:
                pass

    with settings_tab:
        st.subheader("Analysis Config")
        scenario_mode = st.selectbox("ストレス強度 (Tail Risk)", ["Standard (Normal)", "Stress (Fat Tail)", "Extreme (Crisis)"], index=1)
        n_sims = st.slider("MC試行回数", 1000, 10000, 5000)
        months = st.selectbox("予測期間 (月)", [12, 36, 60, 120], index=2)

    st.markdown("---")
    run_btn = st.button("🚀 診断を実行 (Run Audit)")

# =========================================================
# 🧠 メインロジック (Execution)
# =========================================================
if run_btn:
    if not input_weights_dict:
        st.error("ポートフォリオを入力してください")
    else:
        with st.spinner(f"🔍 {market_choice} の市場構造を分析中... (Applying Factor Model & Tail Risk)"):
            try:
                # 🆕 [Step 4-2] 地域を渡して未上場銘柄のプロキシ補完を有効化
                target = DataFetcher.create_synthetic_portfolio(input_weights_dict, region=st.session_state.region_code)
                if target is None or target.empty:
                    st.error("データ取得失敗。ティッカーを確認してください。")
                    st.stop()
                
                st.session_state.target_series = target
                returns = target.pct_change().dropna()

                metrics = AdvancedStats.calculate_metrics(returns)
                metrics['annual_return'] = returns.mean() * 12
                metrics['volatility'] = returns.std() * np.sqrt(12)
                
                factor_profile = FactorAnalyzer.analyze_style(target, region=st.session_state.region_code)
                if not factor_profile:
                    factor_profile = {'beta_market': 1.0, 'beta_size': 0.0, 'beta_value': 0.0, 'alpha': 0.0}

                audit_res = {
                    'metrics': metrics,
                    'betas': {
                        'Mkt-RF': factor_profile.get('beta_market', 1.0),
                        'SMB': factor_profile.get('beta_size', 0.0),
                        'HML': factor_profile.get('beta_value', 0.0)
                    },
                    'current_regime': 'Normal',
                    'region': st.session_state.region_code
                }
                st.session_state.audit_result = audit_res

                start_date = target.index[0]
                waves = StochasticScenarioGenerator.generate_3factor_waves(
                    start_date=start_date, region=st.session_state.region_code, n_sims=n_sims, horizon_months=months
                )

                if waves:
                    price_paths_arr = ProjectionCore.run_market_driven_projection(
                        current_price=target.iloc[-1], factor_waves=waves,
                        factor_profile=factor_profile, n_sims=n_sims, horizon_months=months
                    )
                else:
                    st.warning("ファクターデータが不足しています。簡易予測に切り替えます。")
                    price_paths_arr = np.tile(target.iloc[-1], (months, n_sims))

                sim_paths = pd.DataFrame(price_paths_arr)

                recovery_metrics = AuditEngine.analyze_recovery_probability(price_paths_arr)
                crashed = recovery_metrics.get('crashed_scenarios_count', 0)
                recovery_metrics['survival_prob'] = 1.0 - (crashed / n_sims) if n_sims > 0 else 1.0

                st.session_state.simulation_result = {
                    "paths": sim_paths,
                    "recovery": recovery_metrics,
                    "final_values": price_paths_arr[-1, :]
                }
                
                st.success("✅ 分析完了")

            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

# =========================================================
# 📊 結果ダッシュボード (Dashboard)
# =========================================================
if st.session_state.audit_result is not None:
    res = st.session_state.audit_result
    sim_res = st.session_state.simulation_result
    metrics = res.get('metrics', {})
    
    st.header("📊 Executive Summary")
    
    regime = res.get('current_regime', 'Normal')
    region_display = "🇺🇸 米国市場" if res.get('region', 'US') == "US" else "🇯🇵 日本市場"
    
    advisor_text = f"**{region_display}** の現在の市場環境は**平時**と判定されています。"
    if regime == 'Crisis':
        advisor_text = f"⚠️ **{region_display}** の市場は**不安定な局面（Crisis Regime）**にあります。ボラティリティの上昇に警戒が必要です。"
    
    betas = res.get('betas', {})
    hml_val = betas.get('HML', 0)
    factor_msg = ""
    
    if hml_val > 0.1:
        factor_msg = "あなたのポートフォリオは**「割安株（バリュー）」**の傾向が強いです。インフレや金利上昇局面には強いですが、景気後退の初期には値動きが重くなる傾向があります。"
    elif hml_val < -0.1:
        factor_msg = "あなたのポートフォリオは**「成長株（グロース）」**の傾向が強いです。市場上昇時の爆発力はありますが、金利上昇には弱いため、債券等でのヘッジが推奨されます。"
    else:
        factor_msg = "あなたのポートフォリオは、バリュー（割安）とグロース（成長）のバランスが取れた構成です。"
        
    st.markdown(f"""
    <div class="advisor-card">
        <b>🤖 AI Risk Advisor:</b><br>
        {advisor_text} <br>
        {factor_msg}<br>
        <hr style="border-top: 1px solid #4B7BFF; margin: 10px 0;">
        設定された期間({months}ヶ月)での推定元本回復期間は <b>{sim_res['recovery'].get('avg_recovery_months', 0):.1f}ヶ月</b> です。
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    def kpi_card(label, value, delta=None, color=""):
        delta_html = f"<span class='metric-delta {color}'>{delta}</span>" if delta else ""
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """

    with col1:
        ann_ret = metrics.get('annual_return', 0) * 100
        st.markdown(kpi_card("Expected Return", f"{ann_ret:.1f}%", "年率期待値", "delta-pos"), unsafe_allow_html=True)
    with col2:
        es_95 = metrics.get('cvar_95', 0) * 100
        st.markdown(kpi_card("Expected Shortfall", f"{es_95:.1f}%", "暴落時の平均損失", "delta-neg"), unsafe_allow_html=True)
    with col3:
        survival = sim_res['recovery'].get('survival_prob', 0) * 100
        st.markdown(kpi_card("Survival Prob", f"{survival:.1f}%", f"{months}ヶ月生存率", "delta-pos" if survival>80 else "delta-neg"), unsafe_allow_html=True)
    
    with col4:
        rec_months = sim_res['recovery'].get('avg_recovery_months', 99)
        if rec_months <= 18:
            color = "delta-pos"
            suffix = "🟢 早期回復 (Safe)"
        elif rec_months <= 36:
            color = "delta-warn"
            suffix = "🟡 注意 (Caution)"
        else:
            color = "delta-neg"
            suffix = "🔴 警告 (Danger)"
        st.markdown(kpi_card("Recovery Speed", f"{rec_months:.1f} M", suffix, color), unsafe_allow_html=True)

    st.markdown("---")

    t1, t2, t3, t4, t5 = st.tabs([
        "🔮 未来予測 (Projection)", 
        "🛡️ リスク詳細 (Downside)", 
        "🧠 メンタル指標 (Stress)",
        "🕰️ タイムマシン (History)",
        "🧪 ファクター (Style)"
    ])

    with t1:
        st.subheader(f"📈 {months}-Month Projection (Fan Chart)")
        paths = sim_res['paths']
        x_axis = np.arange(len(paths))
        
        p10 = paths.apply(lambda x: np.percentile(x, 10), axis=1)
        p50 = paths.apply(lambda x: np.percentile(x, 50), axis=1)
        p90 = paths.apply(lambda x: np.percentile(x, 90), axis=1)
        
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=x_axis, y=p90, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_fan.add_trace(go.Scatter(x=x_axis, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(75, 123, 255, 0.2)', name='80% Confidence'))
        fig_fan.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', line=dict(color='#4B7BFF', width=2), name='Median Forecast'))
        
        current_val = st.session_state.target_series.iloc[-1]
        fig_fan.add_hline(y=current_val, line_dash="dash", line_color="gray", annotation_text="Start")

        fig_fan.update_layout(
            title="Portfolio Value Projection (Monte Carlo)", 
            xaxis_title="Months Ahead", 
            yaxis_title="Portfolio Value", 
            template="plotly_dark", 
            height=500,
            font=dict(color="#FAFAFA")
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    with t2:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Risk Metrics")
            st.table(pd.DataFrame({
                "Metric": ["Volatility (Ann.)", "Max Drawdown", "Skewness (歪度)", "Kurtosis (尖度)"],
                "Value": [
                    f"{metrics.get('volatility', 0)*100:.1f}%",
                    f"{metrics.get('max_dd', 0)*100:.1f}%",
                    f"{metrics.get('skewness', 0):.2f}",
                    f"{metrics.get('kurtosis', 0):.2f}"
                ]
            }))
            st.caption("※ 尖度(Kurtosis)が高いほど、極端な暴落(Fat Tail)が起きやすいことを示唆します。")

        with c2:
            st.subheader("Recovery Time Distribution")
            paths_arr = sim_res['paths'].values
            start_price = paths_arr[0, 0]
            
            recovery_months_list = []
            sample_indices = np.random.choice(paths_arr.shape[1], min(1000, paths_arr.shape[1]), replace=False)
            
            for i in sample_indices:
                path = paths_arr[:, i]
                underwater = path < start_price
                if not np.any(underwater):
                    recovery_months_list.append(0)
                else:
                    if path[-1] < start_price:
                        recovery_months_list.append(months)
                    else:
                        first_under = np.argmax(underwater)
                        recovered_after = np.argmax(path[first_under:] >= start_price)
                        recovery_months_list.append(recovered_after + 1)
            
            df_rec = pd.DataFrame(recovery_months_list, columns=['Months'])
            fig_rec = px.histogram(df_rec, x='Months', nbins=30, title="Probability of Recovery Time (Months)", color_discrete_sequence=['#00CC96'], labels={'Months': 'Months to Recover'})
            fig_rec.add_vline(x=12, line_dash="dash", line_color="yellow", annotation_text="1 Year")
            fig_rec.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="3 Years")
            
            fig_rec.update_layout(
                template="plotly_dark", 
                bargap=0.1, 
                xaxis_title="Months to Recover Principal",
                font=dict(color="#FAFAFA")
            )
            st.plotly_chart(fig_rec, use_container_width=True)

    with t3:
        st.subheader("🧠 総合ストレス・メーター (Comprehensive Stress Score)")
        
        ulcer = metrics.get('ulcer_index', 0) * 100
        historical_stress = min(ulcer, 100) 
        
        final_values = sim_res['final_values']
        start_price = st.session_state.target_series.iloc[-1]
        var_95 = np.percentile(final_values, 5)
        
        if len(final_values[final_values <= var_95]) > 0:
            cvar_95_val = final_values[final_values <= var_95].mean()
        else:
            cvar_95_val = var_95
            
        cvar_drop_pct = max(0, (start_price - cvar_95_val) / start_price * 100)
        
        total_stress = (historical_stress * 0.4) + (cvar_drop_pct * 0.6)
        total_stress = min(total_stress, 100)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = total_stress,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ポートフォリオ危険度 (0-100)", 'font': {'size': 20, 'color': '#FAFAFA'}},
            number = {'font': {'color': '#FAFAFA'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "rgba(255,255,255,0.4)"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#00CC96'},   # 緑: 安全
                    {'range': [30, 60], 'color': '#F5A623'},  # 黄: 注意
                    {'range': [60, 100], 'color': '#EF553B'}  # 赤: 危険
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': total_stress
                }
            }
        ))
        
        fig_gauge.update_layout(
            template="plotly_dark", 
            font=dict(color="#FAFAFA"),
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            st.markdown("### 📊 スコアの内訳")
            st.markdown(f"- **過去のストレス度 (胃潰瘍指数):** `{historical_stress:.1f}`")
            st.markdown(f"- **未来の最悪ケース (推定最大下落率):** `{cvar_drop_pct:.1f}%`")
            st.info("""
            **見方:** - 🟢 **0-30:** 安定しています。急落の可能性は低めです。
            - 🟡 **30-60:** 注意が必要です。相場急変時に胃が痛くなる下落が想定されます。
            - 🔴 **60-100:** 危険水域です。ファットテール（想定外の暴落）が発生すると致命傷になる可能性があります。
            """)

    with t4:
        st.subheader("🕰️ Stress Testing with History")
        scenario_key = st.selectbox("Select Historical Crisis", list(HistoryTimeMachine.SCENARIOS.keys()))
        
        # 🆕 [Step 4-2] エンジンに地域情報を渡し、ベンチマークの実データも取得する
        current_region = res.get('region', 'US')
        replay_res = HistoryTimeMachine.run_replay(
            current_price=st.session_state.target_series.iloc[-1],
            current_beta=res['betas'].get('Mkt-RF', 1.0),
            scenario_key=scenario_key,
            region=current_region
        )
        
        if replay_res:
            st.caption(f"Scenario: {replay_res['desc']}")
            fig_tm = go.Figure()
            
            # 🆕 [Step 4-2] ベンチマーク（市場平均）のリアルな動きをグレーの点線で追加
            bm_name = "S&P 500 (US)" if current_region == "US" else "Nikkei 225 (Japan)"
            fig_tm.add_trace(go.Scatter(
                x=replay_res['days'], y=replay_res['market_prices'], 
                mode='lines', name=bm_name, 
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            # ポートフォリオの動きを緑の実線で追加
            fig_tm.add_trace(go.Scatter(
                x=replay_res['days'], y=replay_res['prices'], 
                mode='lines', name='Your Portfolio', 
                line=dict(color='#00CC96', width=3)
            ))
            
            fig_tm.update_layout(
                template="plotly_dark", 
                title=f"Replay: {scenario_key}",
                xaxis_title="Days since Crisis Start",
                yaxis_title="Portfolio Value",
                font=dict(color="#FAFAFA"),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_tm, use_container_width=True)

    with t5:
        f_res = res.get('betas', {})
        if f_res:
            c1, c2 = st.columns([1, 1])
            with c1:
                categories = [FACTOR_TRANSLATION.get(k, k) for k in f_res.keys()]
                values = list(f_res.values())
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values, theta=categories, fill='toself', name='Factor Exposure'
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, color="#FAFAFA")),
                    template="plotly_dark",
                    title="Factor Exposure Radar",
                    font=dict(color="#FAFAFA")
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            with c2:
                st.subheader("診断レポート")
                st.markdown(f"""
                - **{FACTOR_TRANSLATION.get('Mkt-RF', 'Market')}**: {f_res.get('Mkt-RF', 1.0):.2f} (市場連動性)
                - **{FACTOR_TRANSLATION.get('SMB', 'Size')}**: {f_res.get('SMB', 0.0):.2f} (小型株要素)
                - **{FACTOR_TRANSLATION.get('HML', 'Value')}**: {f_res.get('HML', 0.0):.2f} (割安株要素)
                
                **解説:**
                各数値がプラスであればその要素の恩恵を受けやすく、マイナスであれば逆の動きをする傾向があります。
                """)
        else:
            st.warning("ファクター分析データが不足しています。")

else:
    st.info("👈 左側のサイドバーからポートフォリオを入力し、「診断を実行」ボタンを押してください。")
    st.markdown("""
    ### 🛡️ What is Portfolio Auditor Pro?
    新エンジン(V2)搭載のプロフェッショナル診断ツールです：
    1.  **Probability Projection:** t分布を用いた「テールリスク（極端な暴落）」を考慮した未来予測。
    2.  **Recovery Analysis:** 暴落した際に、どれくらいの期間で回復可能かを算出。
    3.  **Factor X-Ray:** 専門用語を噛み砕き、あなたの資産の「癖」を可視化。
    """)
