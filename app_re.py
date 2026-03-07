import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
import re
import unicodedata

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
# 🧹 [Phase B: 追加] データクリーニング関数
# =========================================================
def clean_ticker(t_str):
    """ティッカー名の前後の空白を消し、すべて大文字に統一する"""
    return str(t_str).strip().upper()

def clean_weight(w_str):
    """ウェイト値に混ざった記号(%, $, 全角など)を掃除して数値(float)にする"""
    try:
        # 全角を半角に変換
        w_str = unicodedata.normalize('NFKC', str(w_str))
        # 数字とピリオド以外の記号(%, $, 円, コンマ等)を削除
        w_str = re.sub(r'[%$円,]', '', w_str).strip()
        return float(w_str)
    except ValueError:
        return None

# =========================================================
# 🎨 ページ設定 & CSSデザイン
# =========================================================
st.set_page_config(page_title="Portfolio Auditor Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* 全体のフォントと背景調整 */
    .main { background-color: #0E1117; color: #FAFAFA; }
    
    /* ▼▼ [Phase B-1] 文字の視認性向上（徹底調整） ▼▼ */
    /* 入力ボックス、テキストエリア、セレクトボックス、ファイルアップローダー等の文字色を白に強制 */
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div,
    .stFileUploader, 
    .stFileUploader div,
    .stFileUploader span,
    .stFileUploader p {
        color: #FAFAFA !important;
    }
    
    /* 入力要素の背景色を少し明るいダークグレーに */
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div {
        background-color: #262730 !important;
    }

    /* セレクトボックスのドロップダウンメニュー */
    ul[data-baseweb="menu"] {
        background-color: #262730 !important;
    }
    ul[data-baseweb="menu"] li {
        color: #FAFAFA !important;
    }
    /* ▲▲ 追加ここまで ▲▲ */
    
    /* KPIカードのデザイン */
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333333;
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
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .alert-card {
        background-color: #3E1616;
        border-left: 5px solid #EF553B;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #FAFAFA;
    }
    
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
if 'input_weights_dict' not in st.session_state:
    st.session_state.input_weights_dict = None

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
        
        # 🆕 CSVアップロード機能を追加
        uploaded_file = st.file_uploader("📂 CSVアップロード (Ticker, Weight列を含む)", type=["csv"])
        
        if st.session_state.region_code == "US":
            default_input = "SPY: 60\nTLT: 40"
        else:
            default_input = "1885.T: 10\n5449.T: 10\n8078.T: 10\n7241.T: 10\n3105.T: 10"

        st.caption("またはテキストで入力:")
        input_str = st.text_area("Ticker: Weight (%)", value=default_input, height=150)
        
        weights = {}
        
        # 1. CSVがアップロードされている場合は優先して読み込む
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                skipped_rows = 0
                if 'Ticker' in df.columns and 'Weight' in df.columns:
                    for _, row in df.iterrows():
                        # 📌 [Phase B-3] TickerとWeightのノイズを自動でクリーニング
                        tkr = clean_ticker(row['Ticker'])
                        w = clean_weight(row['Weight'])
                        
                        if tkr and w is not None:
                            weights[tkr] = w
                        else:
                            skipped_rows += 1
                            
                    st.success(f"✅ CSVから {len(weights)} 銘柄を読み込みました")
                    if skipped_rows > 0:
                        st.warning(f"⚠️ {skipped_rows} 銘柄をスキップしました (空欄または不正なフォーマット)")
                else:
                    st.error("❌ CSVに 'Ticker' と 'Weight' の列がありません")
            except Exception as e:
                st.error(f"❌ CSVの読み込みエラー: {e}")
                
        # 2. CSVがない（またはエラー）場合はテキスト入力を使用する
        if not weights and input_str:
            skipped_rows = 0
            for line in input_str.split('\n'):
                if ':' in line: k, v = line.split(':', 1)
                elif ',' in line: k, v = line.split(',', 1)
                else: continue
                
                # 📌 [Phase B-3] 手入力データもクリーニング
                tkr = clean_ticker(k)
                w = clean_weight(v)
                
                if tkr and w is not None:
                    weights[tkr] = w
                else:
                    skipped_rows += 1
                    
            if skipped_rows > 0:
                st.warning(f"⚠️ {skipped_rows} 行の入力をスキップしました (不正なフォーマット)")
                
        # ウェイトの集計と検証
        if weights:
            input_weights_dict = weights
            total_w = sum(weights.values())
            if abs(total_w - 100) > 1:
                st.warning(f"⚠️ 合計ウェイトが {total_w:.1f}% です (100%を推奨)")
            else:
                st.caption(f"✅ 合計ウェイト: {total_w:.1f}%")

    with settings_tab:
        st.subheader("Analysis Config")
        scenario_mode = st.selectbox("ストレス強度 (Tail Risk)", ["Standard (Normal)", "Stress (Fat Tail)", "Extreme (Crisis)"], index=1)
        n_sims = st.slider("MC試行回数", 1000, 10000, 5000)
        months = st.selectbox("予測期間 (月)", [12, 36, 60, 120], index=2)

    # 📌 [Phase B-2] run_btn (実行ボタン) を完全に廃止し、自動実行のトリガーへ移行

# =========================================================
# 🧠 メインロジック (Execution - 自動実行化)
# =========================================================
# 📌 [Phase B-2] ボタンが押されたかではなく、「有効なポートフォリオがあるか」で処理を開始する
if input_weights_dict:
    st.session_state.input_weights_dict = input_weights_dict
    
    with st.spinner(f"🔍 {market_choice} の市場構造を分析中... (Applying Strict Factor Model)"):
        try:
            # 1. データの取得と合成
            target = DataFetcher.create_synthetic_portfolio(input_weights_dict, region=st.session_state.region_code)
            if target is None or target.empty:
                st.error("データ取得失敗。ティッカーを確認してください。")
                st.stop()
            
            st.session_state.target_series = target
            returns = target.pct_change().dropna()

            # 2. 厳格化された評価ロジックを呼び出し
            metrics = AdvancedStats.calculate_metrics(returns, weights_dict=input_weights_dict)
            metrics['annual_return'] = returns.mean() * 12
            
            # 3. ファクター分析
            factor_profile = FactorAnalyzer.analyze_style(target, region=st.session_state.region_code)
            safe_profile = factor_profile if factor_profile else {'beta_market': 1.0, 'beta_size': 0.0, 'beta_value': 0.0, 'alpha': 0.0}

            audit_res = {
                'metrics': metrics,
                'betas': {
                    'Mkt-RF': safe_profile.get('beta_market', 1.0),
                    'SMB': safe_profile.get('beta_size', 0.0),
                    'HML': safe_profile.get('beta_value', 0.0)
                },
                'current_regime': 'Normal',
                'region': st.session_state.region_code,
                'factor_success': factor_profile is not None  
            }
            st.session_state.audit_result = audit_res

            # 4. モンテカルロシミュレーション
            stress_level_key = scenario_mode.split(" ")[0] 
            
            simulated_returns = StochasticScenarioGenerator.generate_portfolio_paths(
                returns=returns, 
                n_sims=n_sims, 
                horizon_months=months, 
                stress_level=stress_level_key
            )
            
            price_paths_arr = ProjectionCore.run_projection(
                current_price=target.iloc[-1], 
                simulated_returns=simulated_returns
            )

            sim_paths = pd.DataFrame(price_paths_arr)

            # 5. リカバリー解析
            recovery_metrics = AuditEngine.analyze_recovery_probability(price_paths_arr)
            crashed = recovery_metrics.get('crashed_scenarios_count', 0)
            recovery_metrics['survival_prob'] = 1.0 - (crashed / n_sims) if n_sims > 0 else 1.0

            st.session_state.simulation_result = {
                "paths": sim_paths,
                "recovery": recovery_metrics,
                "final_values": price_paths_arr[-1, :]
            }
            
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
    
    # 集中投資リスク（ペナルティ）の警告表示
    penalty_ratio = metrics.get('risk_penalty_ratio', 1.0)
    if penalty_ratio > 1.05:
        st.markdown(f"""
        <div class="alert-card">
            <b>⚠️ 集中投資アラート (Concentration Risk)</b><br>
            銘柄数が少ない、または一部の資産に比率が偏っているため、分散効果が十分に機能していません。<br>
            現実のドローダウンを過小評価しないよう、推定リスク（ボラティリティ等）を <b>{penalty_ratio:.2f} 倍に厳格化</b> して評価しています。
        </div>
        """, unsafe_allow_html=True)
    
    regime = res.get('current_regime', 'Normal')
    region_display = "🇺🇸 米国市場" if res.get('region', 'US') == "US" else "🇯🇵 日本市場"
    
    advisor_text = f"**{region_display}** の過去データを基にした診断を実行しました。"
    
    betas = res.get('betas', {})
    hml_val = betas.get('HML', 0)
    factor_msg = ""
    
    if not res.get('factor_success'):
        factor_msg = "⚠️ ファクターデータの取得サーバーが応答しなかったため、詳細なスタイル分析をスキップしました。（※未来予測は自身のボラティリティとt分布を使用して正常に完了しています）"
    elif hml_val > 0.1:
        factor_msg = "あなたのポートフォリオは**「割安株（バリュー）」**の傾向が強いです。インフレや金利上昇局面には強いですが、景気後退の初期には値動きが重くなる傾向があります。"
    elif hml_val < -0.1:
        factor_msg = "あなたのポートフォリオは**「成長株（グロース）」**の傾向が強いです。市場上昇時の爆発力はありますが、金利上昇には弱いため、債券等でのヘッジが推奨されます。"
    else:
        factor_msg = "あなたのポートフォリオは、バリュー（割安）とグロース（成長）のバランスが取れた構成です。"
        
    # 📌 [Phase B-4] アドバイザーテキストのレイアウトを洗練（タイトル追加、メリハリ向上）
    st.markdown(f"""
    <div class="advisor-card">
        <h4 style="margin-top: 0; margin-bottom: 10px; color: #4B7BFF;">🤖 AI Risk Advisor</h4>
        <p style="margin-bottom: 5px;">{advisor_text}</p>
        <p style="margin-bottom: 10px;">{factor_msg}</p>
        <hr style="border-top: 1px solid rgba(75, 123, 255, 0.3); margin: 15px 0;">
        <p style="margin-bottom: 0; font-size: 1.05em;">
            ⏳ 設定期間での推定元本回復期間: 
            <strong style="font-size: 1.2em; color: #FAFAFA;">{sim_res['recovery'].get('avg_recovery_months', 0):.1f} ヶ月</strong>
        </p>
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
        st.markdown(kpi_card("Expected Shortfall", f"{es_95:.1f}%", "暴落時の平均損失 (ペナルティ込)", "delta-neg"), unsafe_allow_html=True)
    with col3:
        survival = sim_res['recovery'].get('survival_prob', 0) * 100
        st.markdown(kpi_card("Survival Prob", f"{survival:.1f}%", "生存率", "delta-pos" if survival>80 else "delta-neg"), unsafe_allow_html=True)
    
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
        st.subheader("📈 Projection (Fat-Tail Engine)")
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
        # 📌 [Phase B-4] Y軸をデータに合わせて自動調整し、グラフが平坦に見えるのを防ぐ
        fig_fan.update_yaxes(autorange=True, fixedrange=False)
        st.plotly_chart(fig_fan, width="stretch")

    with t2:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Risk Metrics")
            st.table(pd.DataFrame({
                "Metric": ["Volatility (Ann.)", "Max Drawdown", "Skewness (歪度)", "Kurtosis (尖度)", "Concentration (HHI)"],
                "Value": [
                    f"{metrics.get('volatility', raw_sigma:=returns.std()*np.sqrt(12))*100:.1f}%",
                    f"{metrics.get('max_dd', 0)*100:.1f}%",
                    f"{metrics.get('skewness', 0):.2f}",
                    f"{metrics.get('kurtosis', 0):.2f}",
                    f"{metrics.get('hhi_index', 0):.2f}"
                ]
            }))
            st.caption("※ 集中度(HHI)が0.3以上だと分散不足のリスクが高まります。")

        with c2:
            st.subheader("Recovery Time Distribution (Smoothed)")
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
                        selected_months = int(st.session_state.get('months', paths_arr.shape[0]-1))
                        recovery_months_list.append(selected_months)
                    else:
                        first_under = np.argmax(underwater)
                        recovered_after = np.argmax(path[first_under:] >= start_price)
                        recovery_months_list.append(recovered_after + 1)
            
            fig_rec = go.Figure()
            df_rec = pd.DataFrame(recovery_months_list, columns=['Months'])
            
            fig_rec.add_trace(go.Histogram(
                x=df_rec['Months'],
                histnorm='probability density',
                nbinsx=30,
                name='Simulated Frequency',
                marker_color='rgba(0, 204, 150, 0.5)'
            ))
            
            if len(recovery_months_list) > 1 and np.var(recovery_months_list) > 0:
                try:
                    kde = gaussian_kde(recovery_months_list)
                    x_range = np.linspace(0, max(recovery_months_list), 200)
                    fig_rec.add_trace(go.Scatter(
                        x=x_range, y=kde(x_range),
                        mode='lines', name='Density Curve',
                        line=dict(color='#FAFAFA', width=2)
                    ))
                except Exception:
                    pass 
            
            fig_rec.add_vline(x=12, line_dash="dash", line_color="yellow", annotation_text="1 Year")
            fig_rec.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="3 Years")
            
            fig_rec.update_layout(
                template="plotly_dark", 
                bargap=0.05, 
                xaxis_title="Months to Recover Principal",
                yaxis_title="Probability Density",
                font=dict(color="#FAFAFA"),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_rec, width="stretch")

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
            st.plotly_chart(fig_gauge, width="stretch")
            
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
        st.subheader("🕰️ Stress Testing (Real Historical Data)")
        scenario_key = st.selectbox("Select Historical Crisis", list(HistoryTimeMachine.SCENARIOS.keys()))
        
        current_region = res.get('region', 'US')
        
        replay_res = HistoryTimeMachine.run_replay(
            current_price=st.session_state.target_series.iloc[-1],
            current_beta=res['betas'].get('Mkt-RF', 1.0),
            scenario_key=scenario_key,
            region=current_region,
            weights_dict=st.session_state.input_weights_dict
        )
        
        if replay_res:
            st.caption(f"Scenario: {replay_res['desc']}")
            fig_tm = go.Figure()
            
            x_data = replay_res.get('dates', replay_res.get('days', np.arange(len(replay_res['prices']))))
            
            bm_name = "S&P 500 (US)" if current_region == "US" else "Nikkei 225 (Japan)"
            fig_tm.add_trace(go.Scatter(
                x=x_data, y=replay_res['market_prices'], 
                mode='lines', name=bm_name, 
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig_tm.add_trace(go.Scatter(
                x=x_data, y=replay_res['prices'], 
                mode='lines', name='Your Portfolio (Real Data)', 
                line=dict(color='#00CC96', width=3)
            ))
            
            fig_tm.update_layout(
                template="plotly_dark", 
                title=f"Replay: {scenario_key}",
                xaxis_title="Time",
                yaxis_title="Portfolio Value",
                font=dict(color="#FAFAFA"),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            # 📌 [Phase B-4] Y軸スケールの自動調整を適用
            fig_tm.update_yaxes(autorange=True, fixedrange=False)
            st.plotly_chart(fig_tm, width="stretch")
        else:
            st.warning("⚠️ この期間のシミュレーションに必要なデータが不足しています。")

    with t5:
        f_res = res.get('betas', {})
        if res.get('factor_success'):
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
                st.plotly_chart(fig_radar, width="stretch")
                
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
            st.warning("⚠️ ファクター分析の外部データ（Fama-French）が取得できなかったため、この項目の表示をスキップしています。")

else:
    st.info("👈 左側のサイドバーからポートフォリオを入力するか、CSVファイルをアップロードしてください。")
    st.markdown("""
    ### 🛡️ What is Portfolio Auditor Pro?
    新エンジン(V2)搭載のプロフェッショナル診断ツールです：
    1.  **Probability Projection:** t分布を用いた「テールリスク（極端な暴落）」を考慮した未来予測。
    2.  **Recovery Analysis:** 暴落した際に、どれくらいの期間で回復可能かを算出。
    3.  **Factor X-Ray:** 専門用語を噛み砕き、あなたの資産の「癖」を可視化。
    """)
