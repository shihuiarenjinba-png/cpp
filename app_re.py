"""
app_re.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
※ 【修正版(v21)】過去データの異常な跳ね上がりを防ぐ幾何平均近似式(μ - σ²/2)の導入、
   およびベンチマークとポートフォリオのグラフ波形混線問題を解決する完全分離描画を実装。
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import random

# =========================================================
# 🔑 APIキーと環境変数の集中管理
# =========================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def setup_api_keys():
    """Streamlit Secrets または 環境変数 からAPIキーを安全に取得・設定する"""
    fmp_key = None
    
    try:
        if hasattr(st, "secrets") and "FMP_API_KEY" in st.secrets:
            fmp_key = st.secrets["FMP_API_KEY"]
    except Exception:
        pass

    if not fmp_key:
        fmp_key = os.environ.get("FMP_API_KEY")

    if fmp_key:
        os.environ["FMP_API_KEY"] = fmp_key

setup_api_keys()

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
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_portfolio_vs_benchmark(port_returns, bm_returns):
        """
        [1ページ目用・修正版]: ポートフォリオとベンチマークの累積リターン（対数スケール）比較
        💡 修正: 変数の混線を防ぐため、明示的に別々のSeriesとして独立させ、厳格に結合します。
        """
        if port_returns.empty or bm_returns.empty:
            st.info("💡 データ不足のため表示できません（ベンチマークデータの取得に失敗しました）。")
            return
            
        # 安全装置: 物理的に独立したSeriesに変換
        s_port = pd.Series(port_returns.squeeze(), index=port_returns.index)
        s_bm = pd.Series(bm_returns.squeeze(), index=bm_returns.index)
        
        df = pd.concat([s_port, s_bm], axis=1).dropna()
        df.columns = ["Portfolio", "Benchmark"] # 列名を強制的に上書きして混線を防止
            
        if df.empty:
            st.info("💡 共通のデータ期間がありません。")
            return
            
        # 独立して複利の累積計算を行う
        port_growth = (1 + df["Portfolio"]).cumprod() * 100
        bm_growth = (1 + df["Benchmark"]).cumprod() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_growth.index, y=port_growth.values, mode='lines', name='Portfolio', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=bm_growth.index, y=bm_growth.values, mode='lines', name='Benchmark', line=dict(color='#ff7f0e', dash='dash', width=2)))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Growth (Base=100)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Log Scale)",
            yaxis_type="log",
            height=400, 
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_residuals(residuals):
        """[2ページ目用]: モデルで説明できない残差（アルファ＋誤差）の推移"""
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
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_active_return(port_returns, bm_returns):
        """[3ページ目用]: アクティブ・リターン（ベンチマークとの乖離）の推移を可視化"""
        if port_returns.empty or bm_returns.empty:
            st.info("💡 データ不足のため表示できません（ベンチマークデータがありません）。")
            return
            
        # 安全装置: 物理的に独立したSeriesに変換
        s_port = pd.Series(port_returns.squeeze(), index=port_returns.index)
        s_bm = pd.Series(bm_returns.squeeze(), index=bm_returns.index)
        
        aligned = pd.concat([s_port, s_bm], axis=1).dropna()
        aligned.columns = ["Portfolio", "Benchmark"]

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
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_factor_correlation(region="US"):
        """[3ページ目用]: ファクター同士の相関関係をヒートマップで可視化"""
        factor_corr = FactorAnalyzer.get_factor_correlation(region=region)
        if not factor_corr: return
            
        corr_df = pd.DataFrame(factor_corr)
        fig = px.imshow(
            corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Factor Correlation ({region} Region)"
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_rolling_correlation(port_returns, bm_returns, window=60):
        """[3ページ目用]: 市場との連動性（ローリング相関）の推移を可視化"""
        if port_returns.empty or bm_returns.empty: return
            
        s_port = pd.Series(port_returns.squeeze(), index=port_returns.index)
        s_bm = pd.Series(bm_returns.squeeze(), index=bm_returns.index)
        aligned = pd.concat([s_port, s_bm], axis=1).dropna()
        aligned.columns = ["Portfolio", "Benchmark"]

        if len(aligned) < window: return
            
        rolling_corr = aligned["Portfolio"].rolling(window=window).corr(aligned["Benchmark"]).dropna()
        
        fig = px.line(
            x=rolling_corr.index, y=rolling_corr.values,
            labels={'x': 'Date', 'y': 'Correlation'},
            title=f"Rolling {window}-Day Market Correlation"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_crisis_replays(crisis_results):
        """[4ページ目用]: 過去の危機における最大下落幅の表示"""
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
        """[2ページ目用]: ローリング回帰による動的エクスポージャーの推移を可視化"""
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
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def optimize_and_plot_frontier(asset_returns, current_weights_dict, market_caps=None):
        """[5ページ目用]: 効率的フロンティアと最適ウェイトの算出"""
        
        clean_weights_dict = {str(k).strip().upper(): float(v) for k, v in current_weights_dict.items()}
        
        tickers = list(clean_weights_dict.keys())
        available_tickers = [t for t in tickers if t in asset_returns.columns]
        
        if len(available_tickers) < 3:
            st.info("💡 3銘柄以上で最適化機能が有効になります。")
            return
            
        returns_df = asset_returns[available_tickers]
        
        if returns_df.empty or len(returns_df) < 10:
            st.warning("⚠️ 最適化に必要なデータが不足しています。")
            return
        
        trading_days = 252
        
        # 💡【修正】算術平均から分散の半分を差し引く幾何平均近似式 (μ - σ^2 / 2) を使用
        arithmetic_mean = returns_df.mean() * trading_days
        variance = returns_df.var() * trading_days
        mean_returns = arithmetic_mean - (variance / 2.0)
        
        cov_matrix = returns_df.cov() * trading_days
        
        # ② あなた (Self)
        curr_weights = np.array([clean_weights_dict.get(t, 0.0) for t in available_tickers], dtype=np.float64)
        if np.sum(curr_weights) > 0:
            curr_weights /= np.sum(curr_weights)
            
        curr_ret = np.sum(mean_returns * curr_weights)
        curr_std = np.sqrt(np.dot(curr_weights.T, np.dot(cov_matrix, curr_weights)))

        # ③ 経済 (Economy)
        if market_caps is not None and isinstance(market_caps, dict):
            clean_caps = {str(k).strip().upper(): v for k, v in market_caps.items()}
            valid_caps = [v for v in clean_caps.values() if isinstance(v, (int, float)) and v > 0]
            default_cap = np.median(valid_caps) if valid_caps else 1.0
            econ_weights = np.array([clean_caps.get(t, default_cap) for t in available_tickers], dtype=np.float64)
        else:
            econ_weights = np.ones(len(available_tickers), dtype=np.float64)
            
        if np.sum(econ_weights) > 0:
            econ_weights /= np.sum(econ_weights)
            
        econ_ret = np.sum(mean_returns * econ_weights)
        econ_std = np.sqrt(np.dot(econ_weights.T, np.dot(cov_matrix, econ_weights)))

        num_portfolios = 5000
        np.random.seed(42)
        
        random_weights = np.random.dirichlet(np.ones(len(available_tickers)), num_portfolios)
        
        weights_list = list(random_weights)
        weights_list.append(curr_weights)
        weights_list.append(econ_weights)
        
        total_portfolios = len(weights_list)
        results = np.zeros((3, total_portfolios))
        
        for i, weights in enumerate(weights_list):
            p_ret = np.sum(mean_returns * weights)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results[0,i] = p_std
            results[1,i] = p_ret
            results[2,i] = p_ret / p_std if p_std > 0 else 0
            
        max_sharpe_idx = np.argmax(results[2])
        opt_ret = results[1, max_sharpe_idx]
        opt_std = results[0, max_sharpe_idx]
        
        # ① 理論 (Finance) : Max Sharpe
        opt_weights = np.array(weights_list[max_sharpe_idx])

        distance_economy = 0.5 * np.sum(np.abs(curr_weights - econ_weights)) * 100
        distance_finance = 0.5 * np.sum(np.abs(curr_weights - opt_weights)) * 100

        if len(available_tickers) <= 1 or distance_economy < 0.1:
            distance_economy = 0.0
        if len(available_tickers) <= 1 or distance_finance < 0.1:
            distance_finance = 0.0

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results[0], y=results[1], mode='markers',
            marker=dict(color=results[2], colorscale='Viridis', showscale=True, size=4, opacity=0.3, colorbar=dict(title="Sharpe Ratio")),
            name='Simulated Portfolios'
        ))
        
        fig.add_trace(go.Scatter(
            x=[opt_std], y=[opt_ret], mode='markers',
            marker=dict(color='red', size=16, symbol='star', line=dict(color='black', width=1)),
            name='Max Sharpe Portfolio'
        ))
        fig.add_trace(go.Scatter(
            x=[curr_std], y=[curr_ret], mode='markers',
            marker=dict(color='orange', size=14, symbol='x', line=dict(color='black', width=1)),
            name='Current Portfolio'
        ))
        fig.add_trace(go.Scatter(
            x=[econ_std], y=[econ_ret], mode='markers',
            marker=dict(color='blue', size=14, symbol='square', line=dict(color='black', width=1)),
            name='Economy'
        ))
        
        fig.update_layout(
            title="Efficient Frontier (Risk-Return Tradeoff) - Volatility Drag Adjusted",
            xaxis_title="Expected Annual Volatility (Risk)",
            yaxis_title="Expected Annual Return (CAGR)",
            height=450, margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("#### 📏 Positioning & Active Share")
        st.markdown("現在のあなたのポートフォリオが、「市場全体（経済）」と「最適解（理論）」からどの程度乖離しているか（アクティブ・シェア）を示します。")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Distance vs Economy (経済からの乖離)", f"{distance_economy:.1f}%")
        m_col2.metric("Distance vs Finance (理論からの乖離)", f"{distance_finance:.1f}%")

        st.markdown("#### ⚖️ Proposed Optimal Weights")
        comp_df = pd.DataFrame({
            "Ticker": available_tickers,
            "🏢 経済": econ_weights * 100,
            "👤 あなた (現在)": curr_weights * 100,
            "⚖️ 理論 (Max Sharpe)": opt_weights * 100
        })
        
        styled_df = comp_df.style.format({
            "🏢 経済": "{:.1f}%",
            "👤 あなた (現在)": "{:.1f}%",
            "⚖️ 理論 (Max Sharpe)": "{:.1f}%"
        }).background_gradient(subset=["👤 あなた (現在)"], cmap="Wistia") 
        
        st.dataframe(styled_df, width="stretch")

    @staticmethod
    def plot_bootstrap_fanchart(paths):
        """[4ページ目用]: 1万回のシミュレーション推移を扇状で可視化（ブートストラップ対応）"""
        
        profit_paths = (paths - 1.0) * 100
        
        percentiles = np.percentile(profit_paths, [5, 25, 50, 75, 95], axis=1)
        days = np.arange(profit_paths.shape[0])
        years = days / 252.0 
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=percentiles[4], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=percentiles[0], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.2)', line=dict(width=0), name='5th-95th Percentile'))
        fig.add_trace(go.Scatter(x=years, y=percentiles[3], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=percentiles[1], fill='tonexty', fillcolor='rgba(70, 130, 180, 0.4)', line=dict(width=0), name='25th-75th Percentile'))
        fig.add_trace(go.Scatter(x=years, y=percentiles[2], mode='lines', line=dict(color='darkblue', width=2), name='Median (50th)'))
        
        fig.add_hline(y=0.0, line_dash="dash", line_color="red", annotation_text="Break-even (0%)")
        
        fig.update_layout(title="Historical Bootstrap Projection (10-Year Fan Chart)", xaxis_title="Years", yaxis_title="Total Return (%)", height=450)
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def plot_bootstrap_histogram(final_values):
        """[4ページ目用]: 最終資産分布の正確なヒストグラム（外れ値トリミング・損益率ベース版）"""
        
        profit_pct = (final_values - 1.0) * 100
        
        # 外れ値をカットしたX軸の範囲設定 (上下1%をトリミングして視認性向上)
        pct_01 = np.percentile(profit_pct, 1)
        pct_99 = np.percentile(profit_pct, 99)
        
        fig = px.histogram(
            profit_pct, nbins=100, title="Final Value Distribution (Historical Bootstrap - 10 Years)", 
            labels={'value': 'Total Return (%)', 'count': 'Frequency'}, 
            color_discrete_sequence=["rgba(70, 130, 180, 0.6)"],
            range_x=[pct_01, pct_99] # X軸の描画範囲を制限
        )
        
        median_val = np.median(profit_pct)
        mean_val = np.mean(profit_pct)
        pct_10 = np.percentile(profit_pct, 10)
        pct_90 = np.percentile(profit_pct, 90)

        fig.add_vline(x=0.0, line_dash="dash", line_color="red")
        fig.add_vline(x=pct_10, line_dash="dash", line_color="orange")
        fig.add_vline(x=median_val, line_dash="solid", line_color="green")
        fig.add_vline(x=mean_val, line_dash="dot", line_color="black")
        fig.add_vline(x=pct_90, line_dash="dash", line_color="purple")
        
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("##### 📊 10年後の予想損益サマリー（投資元本 = 0%）")
        l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns(5)
        l_col1.markdown(f"**🔴 元本割れライン**<br>0.0%", unsafe_allow_html=True)
        l_col2.markdown(f"**🟠 下位10%**<br>{pct_10:+.1f}%", unsafe_allow_html=True)
        l_col3.markdown(f"**🟢 中央値**<br>{median_val:+.1f}%", unsafe_allow_html=True)
        l_col4.markdown(f"**⚫ 平均値**<br>{mean_val:+.1f}%", unsafe_allow_html=True)
        l_col5.markdown(f"**🟣 上位10%**<br>{pct_90:+.1f}%", unsafe_allow_html=True)

        st.caption("💡 **プロの視点:** 平均値(Mean)が中央値(Median)より右に大きくズレている場合、一部の大当たりシナリオに引っ張られた「一発逆転狙い」のリスキーなポートフォリオ特性を示唆します。")

# =========================================================
# 🚀 Streamlit メインロジック
# =========================================================
def main():
    st.set_page_config(page_title="Institutional Portfolio Auditor", layout="wide", page_icon="🏦")
    st.title("🏦 Institutional Portfolio Auditor")
    st.markdown("インタラクティブなリスク評価とAI診断システム")
    st.divider()

    st.sidebar.header("⚙️ 設定 & ポートフォリオ")
    
    st.sidebar.markdown("**🤖 AI診断用 API設定**")
    ai_api_key = st.sidebar.text_input("API Key (現在プレースホルダー)", type="password", help="ここにOpenAI等のAPIキーを入れると本物のAIが動くようになります")
    
    if not os.environ.get("FMP_API_KEY"):
        st.sidebar.warning("⚠️ FMP_API_KEY が設定されていません。株価データの取得に失敗する可能性があります。")
        
    st.sidebar.divider()
    
    region = st.sidebar.selectbox("Market Region", ["US", "Japan"], help="対象市場を選択してください。バックエンドの参照ファイルや無リスク金利が切り替わります。")
    config = MarketConfig.get_config(region)
    
    rebalance_options = {"Monthly (月次)": "M", "Annually (年次)": "Y", "Daily (日次)": "D", "Buy & Hold (放置)": None}
    rebalance_choice = st.sidebar.selectbox("Rebalance Frequency", list(rebalance_options.keys()), help="ポートフォリオの比率を元に戻す頻度。放置すると強い銘柄の比率が勝手に増えます(ドリフト)。")
    rebalance_freq = rebalance_options[rebalance_choice]

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

    if st.sidebar.button("Run Advanced Analysis", type="primary", width="stretch"):
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

            try:
                if not raw_input_data.empty:
                    latest_prices = raw_input_data.ffill().iloc[-1].to_dict()
                    market_caps = {k: (v if v > 0 else 1.0) for k, v in latest_prices.items()}
                else:
                    market_caps = {ticker: 1.0 for ticker in norm_weights.keys()}
            except Exception as e:
                st.warning(f"⚠️ 株価データの取得に失敗しました。一時的に各銘柄を均等として扱います。({e})")
                market_caps = {ticker: 1.0 for ticker in norm_weights.keys()}

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
                n_scenarios=10000, n_years=10
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "1️⃣ 総合診断", 
                "2️⃣ 要因分析", 
                "3️⃣ リスク・相関", 
                "4️⃣ 将来シミュレーション", 
                "5️⃣ 最適化提案"
            ])

            with tab1:
                st.header(f"1. Current Diagnosis ({region} Market)")
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
                
                # 💡【修正】: 幾何平均近似式 (μ - σ^2 / 2) を用いて現実的なCAGRを計算
                if len(returns) > 0:
                    arithmetic_mean = returns.mean() * 252.0
                    variance = returns.var() * 252.0
                    cagr = arithmetic_mean - (variance / 2.0)
                else:
                    cagr = 0.0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Past Annual Return (幾何平均近似・CAGR)", f"{cagr * 100:.2f}%")
                col2.metric("Portfolio Volatility", f"{metrics.get('volatility', 0) * 100:.2f}%")
                col3.metric("Model Fit (Adjusted R²)", f"{r2_score:.1f}%", help="自身のファクター戦略でどれだけ動きを説明できているか")
                
                st.divider()
                st.subheader("📈 Model Fit: Actual vs Predicted")
                if style and style.get('status') == 'success':
                    AuditEngine.plot_actual_vs_predicted(style.get("actual_cumulative"), style.get("predicted_cumulative"))
                else:
                    st.info("💡 ファクターモデルによる予測データが不足しているため、実績値のみを表示します。")
                    st.line_chart(synthetic_portfolio)
                
                st.divider()
                st.subheader("📊 Market Relative Performance (Actual vs Benchmark)")
                AuditEngine.plot_portfolio_vs_benchmark(returns, bm_returns)

            with tab2:
                st.header("2. Factor Attribution")
                
                st.subheader("📉 Cumulative Residuals (Alpha)")
                if style and style.get('status') == 'success':
                    AuditEngine.plot_residuals(style.get("residuals"))
                else:
                    st.info("💡 残差データを表示できません。")

                st.divider()

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

            with tab3:
                st.header("3. Risk & Tracking Analysis")
                if bm_returns.empty:
                    st.info("💡 データ不足のため表示できません（ベンチマークデータの取得に失敗しました）。")
                else:
                    te_actual = metrics.get('tracking_error', 0) * 100
                    
                    st.subheader("🎯 Tracking Error Overview")
                    te_col1, te_col2 = st.columns(2)
                    te_col1.metric("Tracking Error (実績TE)", f"{te_actual:.2f}%")
                    te_col2.metric("Information Ratio (情報レシオ)", f"{metrics.get('info_ratio', 0):.2f}")
                    
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

                if cycle_days:
                    st.divider()
                    st.caption(f"⏳ ボラティリティ周期 (ウェルチ法): 約 {cycle_days} 日")

            with tab4:
                st.header("4. Stress Test & Projection")
                
                st.subheader("👤 Portfolio Projection (10 Years)")
                st.markdown("あなたの現在のポートフォリオに基づいた1万回の将来10年間のシミュレーション結果です。")
                
                if projection:
                    if projection.get("alert_message"):
                        st.warning(projection["alert_message"])

                    final_values = projection["paths"][-1, :]
                    worst_5th = projection['worst_5th']
                    cvar_raw = final_values[final_values <= worst_5th].mean()
                    
                    median_pl = (projection['median'] - 1.0) * 100
                    worst_5th_pl = (worst_5th - 1.0) * 100
                    cvar_pl = (cvar_raw - 1.0) * 100
                    cagr_median = ((projection['median']) ** (1 / 10) - 1.0) * 100
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Median (中央値)", f"{median_pl:+.1f}%", f"年率換算: {cagr_median:+.1f}%", delta_color="normal")
                    p_col2.metric("Worst 5% (下位5%)", f"{worst_5th_pl:+.1f}%")
                    p_col3.metric("CVaR (下位5%平均)", f"{cvar_pl:+.1f}%", help="テールリスク（下位5%の最悪シナリオの平均損益率）")
                    p_col4.metric("Prob of Loss (元本割れ確率)", f"{projection['prob_loss']:.1f}%")
                    
                    AuditEngine.plot_bootstrap_fanchart(projection["paths"])
                    AuditEngine.plot_bootstrap_histogram(final_values)
                else:
                    st.info("💡 シミュレーションに失敗しました。")
                
                st.divider()
                
                st.subheader("⚡ Stress Tests (Crash Replays)")
                st.markdown("過去の主要な金融危機の際、現在のポートフォリオ構成がどの程度の下落を経験したかを追体験します。")
                AuditEngine.plot_crisis_replays(crisis_results)

            with tab5:
                st.header("5. Portfolio Optimization")
                
                st.subheader("⚖️ Efficient Frontier & Weight Optimization")
                st.markdown("現在のポートフォリオから、**シャープレシオ（リスク・リターン比）を最大化**する未来の最適ウェイトを提案します。")
                
                asset_returns = raw_input_data.pct_change().dropna(how='all')
                AuditEngine.optimize_and_plot_frontier(asset_returns, norm_weights, market_caps=market_caps)

if __name__ == "__main__":
    main()
