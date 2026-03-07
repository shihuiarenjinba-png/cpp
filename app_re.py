import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# 🔄 [Phase 1: 修正] エンジン読み込み (New Engine Integration)
# 旧: AuditorCore, ScenarioSimulator -> 新: AuditEngine, ProjectionCore
# =========================================================
try:
    from auditor_engine import DataFetcher, AuditEngine, ProjectionCore, HistoryTimeMachine
except ImportError as e:
    # 開発用ダミーモード（エンジンがない場合でもUI確認できるようにする措置）
    # 本番では st.error を出して stop してください
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
# 🎨 ページ設定 & CSSデザイン (変更なし)
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
    .metric-delta { font-size: 0.8em; }
    .delta-pos { color: #00CC96; }
    .delta-neg { color: #EF553B; }
    
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
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None

# =========================================================
# 📂 サイドバー: データ入力 (Input)
# =========================================================
with st.sidebar:
    st.title("🛡️ Auditor Pro")
    st.caption("Professional Risk Analysis System")
    st.markdown("---")
    
    # 入力モード切り替え
    input_tab, settings_tab = st.tabs(["📂 ポートフォリオ", "⚙️ 設定"])
    
    input_weights_dict = None
    macro_file = None
    
    with input_tab:
        st.subheader("Asset Allocation")
        
        # ファイルアップロード
        port_file = st.file_uploader("構成CSV (Ticker, Weight)", type=['csv'], key="port_up")
        
        # デフォルト値の設定
        default_input = "SPY: 60\nTLT: 40"
        
        if port_file:
            try:
                df_port = pd.read_csv(port_file)
                if df_port.shape[1] >= 2:
                    tickers = df_port.iloc[:, 0].astype(str).tolist()
                    weights = df_port.iloc[:, 1].astype(str).tolist()
                    default_input = "\n".join(f"{t}: {w}" for t, w in zip(tickers, weights))
                    st.success(f"✅ {len(tickers)} 銘柄ロード完了")
            except Exception as e:
                st.error(f"読込エラー: {e}")

        # テキストエリア (入力・編集用)
        input_str = st.text_area("Ticker: Weight (%)", value=default_input, height=150)
        
        # 入力解析
        if input_str:
            try:
                weights = {}
                for line in input_str.split('\n'):
                    if ':' in line: k, v = line.split(':')
                    elif ',' in line: k, v = line.split(',')
                    else: continue
                    weights[k.strip()] = float(v.strip())
                input_weights_dict = weights
                
                # 合計チェック
                total_w = sum(weights.values())
                if abs(total_w - 100) > 1:
                    st.warning(f"⚠️ 合計が {total_w:.1f}% です (100%推奨)")
                else:
                    st.caption(f"✅ 合計: {total_w:.1f}%")
            except:
                pass

        st.markdown("---")
        st.subheader("Macro Indicators (Optional)")
        macro_file = st.file_uploader("経済指標CSV", type=['csv'], key="macro_up")

    with settings_tab:
        st.subheader("Analysis Config")
        
        # [Phase 2: 修正] 設定項目のデフォルト値を長期・プロ向けに変更
        scenario_mode = st.selectbox("ストレス強度 (Tail Risk)", ["Standard (Normal)", "Stress (Fat Tail)", "Extreme (Crisis)"], index=1)
        
        n_sims = st.slider("MC試行回数", 1000, 10000, 5000) # デフォルトを増やしました
        
        # [Phase 2: 修正] デフォルト期間を60ヶ月に変更
        months = st.selectbox("予測期間 (月)", [12, 36, 60, 120], index=2) # default: 60 months

    st.markdown("---")
    run_btn = st.button("🚀 診断を実行 (Run Audit)")

# =========================================================
# 🧠 メインロジック (Execution) - [Phase 3: 全面刷新]
# =========================================================
if run_btn:
    if not input_weights_dict:
        st.error("ポートフォリオを入力してください")
    else:
        with st.spinner("🔍 市場構造を分析中... (Applying 3-Factor Model & t-Distribution)"):
            try:
                # 1. データ取得とPF組成 (既存ロジック維持)
                target = DataFetcher.create_synthetic_portfolio(input_weights_dict)
                if target is None or target.empty:
                    st.error("データ取得失敗。ティッカーを確認してください。")
                    st.stop()
                
                st.session_state.target_series = target

                # 2. マクロデータ結合 (既存ロジック維持)
                std_macro = DataFetcher.fetch_benchmark_and_macro(start_date=target.index[0])
                user_macro = None
                if macro_file:
                    macro_file.seek(0)
                    user_macro = DataFetcher.load_macro_csv(macro_file)
                
                if user_macro is not None and not user_macro.empty:
                    macro_combined = pd.merge(std_macro, user_macro, left_index=True, right_index=True, how='outer').ffill().dropna()
                else:
                    macro_combined = std_macro

                # ---------------------------------------------------------
                # [Phase 3: 変更点] 新しいエンジンの呼び出し
                # ---------------------------------------------------------
                
                # 3. 監査実行 (AuditEngine)
                engine = AuditEngine()
                audit_res = engine.analyze(target, macro_combined)
                st.session_state.audit_result = audit_res

                # 4. 未来予測 (ProjectionCore)
                if audit_res:
                    projector = ProjectionCore(engine_result=audit_res)
                    
                    # t分布などを用いた高度なシミュレーション
                    sim_paths = projector.simulate(
                        months=months, 
                        n_scenarios=n_sims, 
                        use_t_dist=(scenario_mode != "Standard (Normal)")
                    )
                    
                    # 回復力分析
                    recovery_metrics = projector.analyze_recovery(
                        paths=sim_paths, 
                        target_return=0.0  # 元本回復
                    )
                    
                    st.session_state.simulation_result = {
                        "paths": sim_paths,
                        "recovery": recovery_metrics,
                        "final_values": sim_paths.iloc[-1, :].values
                    }
                
                st.success("✅ 分析完了")

            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

# =========================================================
# 📊 結果ダッシュボード (Dashboard) - [Phase 4: 表示調整]
# =========================================================
if st.session_state.audit_result is not None:
    res = st.session_state.audit_result
    sim_res = st.session_state.simulation_result
    metrics = res.get('metrics', {}) # 基本指標
    
    # --- A. エグゼクティブ・サマリー ---
    st.header("📊 Executive Summary")
    
    # 1. アドバイザーメッセージ
    regime = res.get('current_regime', 'Normal')
    
    advisor_text = "現在の市場環境は**平時**と判定されています。"
    if regime == 'Crisis':
        advisor_text = "⚠️ 市場は**不安定な局面（Crisis Regime）**にあります。ボラティリティの上昇に警戒が必要です。"
    
    # ---------------------------------------------------------
    # [修正箇所] ファクター翻訳ロジックの強化
    # ---------------------------------------------------------
    betas = res.get('betas', {})
    hml_val = betas.get('HML', 0)
    factor_msg = ""
    
    if hml_val > 0.1:
        # Value（割安）が強い場合
        factor_msg = "あなたのポートフォリオは**「割安株（バリュー）」**の傾向が強いです。インフレや金利上昇局面には強いですが、景気後退の初期には値動きが重くなる傾向があります。"
    elif hml_val < -0.1:
        # Growth（成長）が強い場合
        factor_msg = "あなたのポートフォリオは**「成長株（グロース）」**の傾向が強いです。市場上昇時の爆発力はありますが、金利上昇には弱いため、債券等でのヘッジが推奨されます。"
    else:
        # どちらでもない場合（バランス型）
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

    # 2. KPI Cards
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
        # 信号機ロジック: 18ヶ月以内=安全、36ヶ月以内=注意、それ以上=危険
        color = "delta-pos"
        if rec_months > 36: color = "delta-neg"
        elif rec_months > 18: color = "text-warning" # 黄色的な扱い（CSS未定義なら白になる）
        
        st.markdown(kpi_card("Recovery Speed", f"{rec_months:.1f} M", "平均回復期間", color), unsafe_allow_html=True)

    st.markdown("---")

    # --- B. 詳細分析タブ ---
    t1, t2, t3, t4, t5 = st.tabs([
        "🔮 未来予測 (Projection)", 
        "🛡️ リスク詳細 (Downside)", 
        "🧠 メンタル指標 (Stress)",
        "🕰️ タイムマシン (History)",
        "🧪 ファクター (Style)"
    ])

    # Tab 1: 未来予測
    with t1:
        st.subheader(f"📈 {months}-Month Projection (Fan Chart)")
        
        paths = sim_res['paths']
        x_axis = np.arange(len(paths))
        
        p10 = paths.apply(lambda x: np.percentile(x, 10), axis=1)
        p50 = paths.apply(lambda x: np.percentile(x, 50), axis=1)
        p90 = paths.apply(lambda x: np.percentile(x, 90), axis=1)
        
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(
            x=x_axis, y=p90, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_fan.add_trace(go.Scatter(
            x=x_axis, y=p10, mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor='rgba(75, 123, 255, 0.2)', name='80% Confidence'
        ))
        fig_fan.add_trace(go.Scatter(
            x=x_axis, y=p50, mode='lines', line=dict(color='#4B7BFF', width=2), name='Median Forecast'
        ))
        
        current_val = st.session_state.target_series.iloc[-1]
        fig_fan.add_hline(y=current_val, line_dash="dash", line_color="gray", annotation_text="Start")

        fig_fan.update_layout(
            title="Portfolio Value Projection (Monte Carlo with t-Distribution)",
            xaxis_title="Months Ahead",
            yaxis_title="Portfolio Value",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    # Tab 2: リスク詳細 (Modified for Recovery Histogram)
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
            
            # --- ここから分布生成ロジック ---
            # シミュレーションパスから、元本回復にかかった月数を計算
            # (エンジンがこのリストを返さない場合を想定し、ここで軽量計算を行う)
            paths_arr = sim_res['paths'].values  # numpy array (months, sims)
            start_price = paths_arr[0, 0]
            
            recovery_months = []
            # パフォーマンスのため、最大1000シナリオ程度をサンプリングして描画
            sample_indices = np.random.choice(paths_arr.shape[1], min(1000, paths_arr.shape[1]), replace=False)
            
            for i in sample_indices:
                path = paths_arr[:, i]
                # 元本割れしている期間を探す
                underwater = path < start_price
                if not np.any(underwater):
                    recovery_months.append(0) # そもそも割れてない
                else:
                    # 元本割れした後、初めて元本に戻ったインデックスを探す
                    # 簡易的に、最終的に戻っていない場合は max_months とする
                    if path[-1] < start_price:
                        recovery_months.append(months) # 期間内に回復せず
                    else:
                        # 最初に割れた地点以降で、回復した地点を探す
                        first_under = np.argmax(underwater)
                        recovered_after = np.argmax(path[first_under:] >= start_price)
                        recovery_months.append(recovered_after + 1) # +1 for approximate month count
            
            # ヒストグラムの作成
            df_rec = pd.DataFrame(recovery_months, columns=['Months'])
            
            # 
            fig_rec = px.histogram(
                df_rec, x='Months', nbins=30,
                title="Probability of Recovery Time (Months)",
                color_discrete_sequence=['#00CC96'], # Greenish for recovery
                labels={'Months': 'Months to Recover'}
            )
            
            # 1年、3年のライン
            fig_rec.add_vline(x=12, line_dash="dash", line_color="yellow", annotation_text="1 Year")
            fig_rec.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="3 Years")
            
            fig_rec.update_layout(
                template="plotly_dark",
                bargap=0.1,
                xaxis_title="Months to Recover Principal"
            )
            st.plotly_chart(fig_rec, use_container_width=True)
            
            st.info("""
            **見方:** 左側に山があるほど「すぐ戻る」健全なポートフォリオです。
            右端（期間内未回復）に柱がある場合、長期塩漬けリスクがあります。
            """)

    # Tab 3: メンタル指標
    with t3:
        st.subheader("🧠 Investor Psychology Metrics")
        c1, c2, c3 = st.columns(3)
        
        ulcer = metrics.get('ulcer_index', 0) * 100
        c1.metric("Ulcer Index (胃潰瘍指数)", f"{ulcer:.1f}", help="下落の深さと期間の長さを組み合わせたストレス指数。")
        
        pain_idx = metrics.get('pain_index', 0) * 100 if 'pain_index' in metrics else 0
        c2.metric("Pain Index", f"{pain_idx:.1f}", help="投資家が感じる痛みの平均値。")
        
        st.progress(min(ulcer/20, 1.0), text="Stress Level (Visual)")

    # Tab 4: タイムマシン
    with t4:
        st.subheader("🕰️ Stress Testing with History")
        scenario_key = st.selectbox("Select Historical Crisis", list(HistoryTimeMachine.SCENARIOS.keys()))
        
        replay_res = HistoryTimeMachine.run_replay(
            current_value=st.session_state.target_series.iloc[-1],
            beta=metrics.get('beta_market', 1.0),
            scenario_name=scenario_key
        )
        
        if replay_res:
            st.caption(f"Scenario: {replay_res['desc']}")
            fig_tm = go.Figure()
            fig_tm.add_trace(go.Scatter(x=replay_res['days'], y=replay_res['prices'], mode='lines', name='Your Portfolio', line=dict(color='#00CC96', width=3)))
            fig_tm.update_layout(template="plotly_dark", title=f"Replay: {scenario_key}")
            st.plotly_chart(fig_tm, use_container_width=True)

    # Tab 5: ファクター分析
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
                    polar=dict(radialaxis=dict(visible=True)),
                    template="plotly_dark",
                    title="Factor Exposure Radar"
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
    # 初期画面
    st.info("👈 左側のサイドバーからポートフォリオを入力し、「診断を実行」ボタンを押してください。")
    st.markdown("""
    ### 🛡️ What is Portfolio Auditor Pro?
    新エンジン(V2)搭載のプロフェッショナル診断ツールです：
    1.  **Probability Projection:** t分布を用いた「テールリスク（極端な暴落）」を考慮した未来予測。
    2.  **Recovery Analysis:** 暴落した際に、どれくらいの期間で回復可能かを算出。
    3.  **Factor X-Ray:** 専門用語を噛み砕き、あなたの資産の「癖」を可視化。
    """)    'bg_fill': 'rgba(0, 255, 255, 0.1)'
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

    # 494行目あたり: ボタンのレイアウト作成
    col_gen, col_dl = st.columns([1, 1])

    with col_gen:
        # ⚠️ ここから下は「4スペース」インデントを入れます
        if st.button("📥 Create PDF Report"):
            with st.spinner("📄 Generating PDF..."):
                try:
                    # payloadの作成
                    final_payload = st.session_state.payload.copy()
                    
                    # サイドバーのコメントを反映 (変数名がadvisor_noteであることを確認してください)
                    if 'advisor_note' in locals() or 'advisor_note' in globals():
                        final_payload['advisor_note'] = advisor_note
                    
                    if final_payload and st.session_state.figs:
                        # pdf_generator.py の関数を呼び出し
                        pdf_data = create_pdf_report(final_payload, st.session_state.figs)
                        
                        if pdf_data:
                            st.session_state.pdf_bytes = pdf_data
                            st.success(f"✅ Report Ready! ({len(pdf_data)} bytes)")
                        else:
                            st.error("⚠️ Failed to generate PDF data.")
                    else:
                        st.error("⚠️ No simulation data found. Please run analysis first.")
                        
                except Exception as e:
                    st.error(f"PDF Error: {e}")

    with col_dl:
        # ⚠️ ここも同様に「4スペース」インデント
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
