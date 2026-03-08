"""
app.py
Streamlitを用いたUI構築と、最終結果の可視化・監査を行うメインアプリケーションモジュール。
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# これまでに作成したモジュールのインポート
from config import MarketConfig, FACTOR_TRANSLATION
from data_engine import DataFetcher
from analytics import AdvancedStats, FactorAnalyzer
from simulation import RegimeAnalyzer, HistoryTimeMachine, ProjectionCore

# グラフの見た目をプロ仕様に設定
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# 📊 監査・可視化エンジンクラス (全グラフ復活版)
# =========================================================
class AuditEngine:
    
    @staticmethod
    def plot_correlation_matrix(returns_df):
        """銘柄間の相関関係をヒートマップで可視化（リスク分散の確認用）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = returns_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                    fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title("Asset Correlation Matrix", fontsize=14, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_rolling_correlation(port_returns, bm_returns, window=60):
        """市場との連動性（ローリング相関）の推移を波形グラフで可視化"""
        aligned = pd.concat([port_returns.rename("Portfolio"), bm_returns], axis=1).dropna()
        if len(aligned) < window:
            st.warning("ローリング相関を描画するための十分な期間データがありません。")
            return
            
        rolling_corr = aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rolling_corr.index, rolling_corr.values, color="purple", linewidth=1.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.fill_between(rolling_corr.index, 0, rolling_corr.values, where=(rolling_corr.values > 0), color="red", alpha=0.2)
        ax.fill_between(rolling_corr.index, 0, rolling_corr.values, where=(rolling_corr.values < 0), color="blue", alpha=0.2)
        
        ax.set_title(f"Rolling {window}-Day Market Correlation", fontsize=14, fontweight="bold")
        ax.set_ylabel("Correlation Coefficient")
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_time_machine_chart(crisis_results):
        """過去の危機における最大ドローダウンを棒グラフで警告表示"""
        names = list(crisis_results.keys())
        dd_values = [crisis_results[n]['max_drawdown_pct'] for n in names]
        
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.barplot(x=dd_values, y=names, palette="Reds_r", ax=ax)
        for i, v in enumerate(dd_values):
            ax.text(v - 1, i, f"{v:.1f}%", color='black', va='center', fontweight='bold')
            
        ax.set_title("Historical Stress Tests (Max Drawdown)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Drawdown (%)")
        ax.set_xlim(min(dd_values) * 1.2 if len(dd_values) > 0 else -50, 0)
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_monte_carlo_fanchart(paths):
        """1万回のシミュレーション推移を扇状（ファンチャート）で可視化"""
        fig, ax = plt.subplots(figsize=(10, 5))
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        days = np.arange(paths.shape[0])
        
        ax.fill_between(days, percentiles[0], percentiles[4], color='steelblue', alpha=0.2, label='5th-95th Percentile')
        ax.fill_between(days, percentiles[1], percentiles[3], color='steelblue', alpha=0.4, label='25th-75th Percentile')
        ax.plot(days, percentiles[2], color='darkblue', linewidth=2, label='Median (50th)')
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Break-even (100%)")
        
        ax.set_title("Monte Carlo Projection (1-Year Fan Chart)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value (1.0 = Initial)")
        ax.legend(loc="upper left")
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_monte_carlo_histogram(final_values):
        """Freedman-Diaconis準則を用いた最終資産分布の正確なヒストグラム"""
        fig, ax = plt.subplots(figsize=(10, 4))
        q75, q25 = np.percentile(final_values, [75, 25])
        iqr = q75 - q25
        n = len(final_values)
        
        if iqr > 0 and n > 0:
            bin_width = 2 * iqr * (n ** (-1/3))
            bins = int((final_values.max() - final_values.min()) / bin_width)
            bins = max(10, min(bins, 200))
        else:
            bins = 50

        sns.histplot(final_values, bins=bins, kde=True, ax=ax, color="steelblue", stat="probability")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Break-even (100%)")
        median_val = np.median(final_values)
        ax.axvline(median_val, color="green", linestyle="-", linewidth=2, label=f"Median: {median_val*100:.1f}%")
        
        ax.set_title("Final Value Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Final Portfolio Value (1.0 = Initial)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

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
    st.markdown("Ledoit-Wolf収縮推定と$t$分布GARCHシミュレーションを用いたプロフェッショナルなリスク評価システム")
    st.divider()

    # --- サイドバー (入力部) ---
    st.sidebar.header("📊 Portfolio Configuration")
    region = st.sidebar.selectbox("Market Region", ["US", "Japan"])
    
    uploaded_file = st.sidebar.file_uploader("Upload Weights CSV", type=["csv"])
    manual_input = st.sidebar.text_area("Or Manual Input (Ticker, Weight)", value="AAPL, 50\nMSFT, 50")
    
    weights_dict = {}
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            ticker_col, weight_col = df.columns[0], df.columns[1]
            weights_dict = dict(zip(df[ticker_col], df[weight_col]))
            st.sidebar.success("CSV Loaded Successfully")
        except Exception as e:
            st.sidebar.error(f"CSV Error: {e}")
    else:
        for line in manual_input.strip().split('\n'):
            if ',' in line:
                t, w = line.split(',')
                weights_dict[t.strip()] = float(w.strip())

    # --- 実行ボタン ---
    if st.sidebar.button("Run Advanced Analysis", type="primary"):
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
            
            # 個別銘柄の生データリターン（相関マトリックス用）
            raw_prices = DataFetcher.fetch_market_data(list(norm_weights.keys()))
            raw_returns = raw_prices.pct_change().dropna()
            
            # ベンチマークデータの取得（ローリング相関用）
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
            # 🗂️ 描画レイヤー (タブ化されたダッシュボード)
            # ==========================================
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 概要＆パフォーマンス", 
                "⚠️ リスク＆連動性分析", 
                "🔮 将来シミュレーション", 
                "🔬 ファクター解析"
            ])

            # --- タブ1: 概要 ---
            with tab1:
                st.header("1. Core Risk Metrics (Ledoit-Wolf Estimated)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
                col2.metric("Portfolio Volatility (Shrunk)", f"{metrics.get('volatility', 0) * 100:.2f}%")
                col3.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                col4.metric("Max Drawdown", f"{metrics.get('max_dd', 0) * 100:.2f}%")
                
                st.subheader("Historical Cumulative Growth (Base=100)")
                aligned_growth = pd.concat([synthetic_portfolio.rename("Portfolio"), (1+bm_returns).cumprod()*100], axis=1).dropna()
                st.line_chart(aligned_growth)

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
                    
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric("Median Scenario (中央値)", f"{projection['median'] * 100:.1f}%")
                    p_col2.metric("Worst 5% Scenario (下位5%)", f"{projection['worst_5th'] * 100:.1f}%")
                    p_col3.metric("Probability of Loss (元本割れ確率)", f"{projection['prob_loss']:.1f}%")
                    
                    AuditEngine.plot_monte_carlo_histogram(projection["paths"][-1, :])
                    
                    recovery_rate = AuditEngine.analyze_recovery(projection["paths"])
                    st.info(f"📉 **回復力監査:** ドローダウン発生後、1年以内に元本を回復する確率: **{recovery_rate:.1f}%**")

            # --- タブ4: ファクター解析 ---
            with tab4:
                st.header("4. Market Regime & Factor Exposure")
                st.subheader("Fama-French 3-Factor Model")
                if style:
                    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                    f_col1.metric("Market Beta (市場連動性)", f"{style['beta_market']:.2f}")
                    f_col2.metric("Size (小型株効果)", f"{style['beta_size']:.2f}")
                    f_col3.metric("Value (割安株効果)", f"{style['beta_value']:.2f}")
                    f_col4.metric("Alpha (超過収益)", f"{style['alpha'] * 100:.3f}%")
                else:
                    st.warning("ファクター分析に必要な期間のデータが不足しています。")
                
                st.divider()
                st.subheader("Volatility Cycle Detection")
                if cycle_days:
                    st.info(f"ウェルチ法による現在のボラティリティ周期: **約 {cycle_days} 日**")
                else:
                    st.info("明確な周期性は検出されませんでした。")

            # 免責事項
            st.divider()
            st.caption("⚠️ **Disclaimer**: 本分析はLedoit-Wolf推定、GARCH(1,1)モデル、およびステューデントのt分布に基づく高度なシミュレーションですが、将来の運用成果を保証するものではありません。")

if __name__ == "__main__":
    main()
