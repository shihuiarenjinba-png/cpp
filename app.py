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
# 📊 監査・可視化エンジンクラス
# =========================================================
class AuditEngine:
    @staticmethod
    def plot_monte_carlo_histogram(final_values):
        """
        Freedman-Diaconis準則を用いて、10,000シナリオの最終資産分布を正確に可視化する。
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Freedman-Diaconis準則による最適ビン幅の計算
        q75, q25 = np.percentile(final_values, [75, 25])
        iqr = q75 - q25
        n = len(final_values)
        
        if iqr > 0 and n > 0:
            bin_width = 2 * iqr * (n ** (-1/3))
            bins = int((final_values.max() - final_values.min()) / bin_width)
            bins = max(10, min(bins, 200)) # 極端な値の防止
        else:
            bins = 50

        sns.histplot(final_values, bins=bins, kde=True, ax=ax, color="steelblue", stat="probability")
        
        # 元本割れライン（100%）の描画
        ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Break-even (100%)")
        
        # 中央値の描画
        median_val = np.median(final_values)
        ax.axvline(median_val, color="green", linestyle="-", linewidth=2, label=f"Median: {median_val*100:.1f}%")
        
        ax.set_title("1-Year Projection Distribution (10,000 Scenarios)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Final Portfolio Value (1.0 = 100% Initial)")
        ax.set_ylabel("Probability")
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def analyze_recovery(paths):
        """
        10,000のシナリオパスの中から、一度ドローダウンを起こした後に
        初期投資額（1.0）を回復できたパスの割合を算出する。
        """
        # 初期値 1.0 を下回ったことがあるパスのインデックス
        drawdown_paths_idx = np.where((paths < 1.0).any(axis=0))[0]
        if len(drawdown_paths_idx) == 0:
            return 100.0 # 一度も元本を割らなかった
            
        # ドローダウンを起こしたパスのうち、最終的に1.0以上に回復した数
        recovered_count = np.sum(paths[-1, drawdown_paths_idx] >= 1.0)
        recovery_prob = (recovered_count / len(drawdown_paths_idx)) * 100
        
        return recovery_prob

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
    
    # CSVアップロード機能 (sp500_official_weights.csv 等の読み込みに対応)
    uploaded_file = st.sidebar.file_uploader("Upload Weights CSV", type=["csv"])
    manual_input = st.sidebar.text_area("Or Manual Input (Ticker, Weight)", value="AAPL, 50\nMSFT, 50")
    
    weights_dict = {}
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Ticker と Weight(%) の列名揺れに対応
            ticker_col = df.columns[0]
            weight_col = df.columns[1]
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
            
        with st.spinner("Initializing Quantitative Engine..."):
            
            # 1. データパイプライン
            norm_weights = DataFetcher.normalize_weights(weights_dict)
            synthetic_portfolio = DataFetcher.create_synthetic_portfolio(norm_weights, region=region)
            
            if synthetic_portfolio is None or synthetic_portfolio.empty:
                st.error("データの構築に失敗しました。ティッカー記号を確認してください。")
                return
            
            returns = synthetic_portfolio.pct_change().dropna()

            # 2. リスク・アルファ解析 (Ledoit-Wolf & FF3)
            metrics = AdvancedStats.calculate_metrics(returns, weights_dict=norm_weights, region=region)
            style = FactorAnalyzer.analyze_style(synthetic_portfolio, region=region)
            
            # 3. シミュレーション & タイムマシン
            cycle_days = RegimeAnalyzer.detect_cycle(returns)
            projection = ProjectionCore.run_projection(returns, n_scenarios=10000, n_years=1)
            
            # 危機リプレイ (例としてリーマンショック)
            lehman_replay = HistoryTimeMachine.replay_crisis(norm_weights, "リーマン・ショック (2007-2009)", region)

            # --- 描画レイヤー (ダッシュボード) ---
            st.header("1. Core Risk Metrics (Ledoit-Wolf Estimated)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
            col2.metric("Portfolio Volatility (Shrunk)", f"{metrics.get('volatility', 0) * 100:.2f}%", help="HHIペナルティとLedoit-Wolf収縮推定適用後")
            col3.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
            col4.metric("Max Drawdown", f"{metrics.get('max_dd', 0) * 100:.2f}%")

            st.header("2. Factor Exposure (Fama-French 3-Factor)")
            if style:
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                f_col1.metric("Market Beta (市場連動性)", f"{style['beta_market']:.2f}")
                f_col2.metric("Size (小型株効果)", f"{style['beta_size']:.2f}")
                f_col3.metric("Value (割安株効果)", f"{style['beta_value']:.2f}")
                f_col4.metric("Alpha (超過収益)", f"{style['alpha'] * 100:.3f}%")
            else:
                st.warning("ファクター分析に必要な期間のデータが不足しています。")

            st.header("3. Stress Testing & Regime")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.subheader("Market Regime")
                if cycle_days:
                    st.info(f"現在のボラティリティ周期: **約 {cycle_days} 日**")
                else:
                    st.info("明確な周期性は検出されませんでした。")
                    
            with s_col2:
                st.subheader("History Time Machine")
                if lehman_replay:
                    st.error(f"リーマンショック時 想定最大下落幅: **{lehman_replay['max_drawdown_pct']:.1f}%**")
                else:
                    st.write("危機期間のシミュレーションに必要なデータが合成できませんでした。")

            st.header("4. Stochastic Projection (GARCH & Student's t)")
            if projection:
                AuditEngine.plot_monte_carlo_histogram(projection["paths"][-1, :])
                
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("Median Scenario (中央値)", f"{projection['median'] * 100:.1f}%")
                p_col2.metric("Worst 5% Scenario (下位5%)", f"{projection['worst_5th'] * 100:.1f}%")
                p_col3.metric("Probability of Loss (元本割れ確率)", f"{projection['prob_loss']:.1f}%")
                
                recovery_rate = AuditEngine.analyze_recovery(projection["paths"])
                st.info(f"📉 **回復力監査:** ドローダウン発生後、1年以内に元本を回復する確率: **{recovery_rate:.1f}%**")

            # 免責事項
            st.divider()
            st.caption("⚠️ **Disclaimer**: 本分析はLedoit-Wolf推定、GARCH(1,1)モデル、およびステューデントのt分布に基づく高度なシミュレーションですが、将来の運用成果を保証するものではありません。")

if __name__ == "__main__":
    main()
