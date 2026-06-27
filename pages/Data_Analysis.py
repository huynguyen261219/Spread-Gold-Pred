import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def show_data_analysis():
    """
    Data Analysis Module - Exploratory Data Analysis of Gold Price Spread Dataset
    """

    # =====================================================
    # PAGE CONFIG
    # =====================================================
    st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

    # =====================================================
    # CSS STYLING
    # =====================================================
    st.markdown(
        """
    <style>
    .main .block-container{
        max-width:1400px;
        padding-top:1rem;
    }
    .stApp{
        background-color:#F4F6F9;
    }
    [data-testid="stMetric"]{
        background:white;
        border:1px solid #D9DEE5;
        border-radius:10px;
        padding:15px;
        box-shadow:0px 2px 6px rgba(0,0,0,0.05);
    }
    .section-title{
        color:#0D5A9C;
        font-weight:700;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # HEADER
    # =====================================================
    st.markdown(
        """
    <div style="
    background:#0D5A9C;
    padding:24px;
    border-radius:10px;
    margin-bottom:20px;
    ">
    <h1 style="margin:0;color:white;">Data Analysis Center</h1>
    <p style="margin-top:8px;color:#E5E7EB;">
    Exploratory Data Analysis of Gold Price Spread Dataset
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================
    try:
        df = pd.read_csv("data/dataset.csv")
        print(df.columns.tolist())
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📈 Dataset Overview</h3>", unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Observations", f"{len(df):,}")
    k2.metric("Variables", f"{df.shape[1] - 1}")
    k3.metric("Start Date", str(df["Date"].min().date()))
    k4.metric("End Date", str(df["Date"].max().date()))

    # =====================================================
    # Data Quality Assessment
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>✅ Data Quality Assessment</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Missing Values", int(df.isna().sum().sum()))
    c2.metric("Duplicate Rows", int(df.duplicated().sum()))
    c3.metric("Columns", len(df.columns))
    c4.metric("Rows", len(df))

    # =====================================================
    #  Distribution Analysis
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📊 Distribution Analysis</h3>",
        unsafe_allow_html=True,
    )

    dist_col = st.selectbox(
        "Select Variable",
        [
            "Spread",
            "VND/USD",
            "DXY",
            "TNX",
            "Oil_Price",
            "VNIndex",
            "SP500",
            "Bitcoin",
            "GPR",
        ],
    )

    fig_dist = px.histogram(
        df, x=dist_col, nbins=40, title=f"Distribution of {dist_col}"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # =====================================================
    # OUTLIER ANALYSIS
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📦 Outlier Analysis</h3>", unsafe_allow_html=True
    )

    outlier_col = st.selectbox(
        "Variable for Outlier Detection",
        [
            "Spread",
            "VND/USD",
            "DXY",
            "TNX",
            "Oil_Price",
            "VNIndex",
            "SP500",
            "Bitcoin",
            "GPR",
        ],
        key="outlier",
    )

    fig_box = px.box(df, y=outlier_col, title=f"Boxplot of {outlier_col}")

    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # =====================================================
    # ROLLING MEAN
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📈 Rolling Mean Analysis</h3>",
        unsafe_allow_html=True,
    )

    rolling_df = df.copy()

    rolling_df["MA7"] = rolling_df["Spread"].rolling(7).mean()
    rolling_df["MA30"] = rolling_df["Spread"].rolling(30).mean()
    rolling_df["MA90"] = rolling_df["Spread"].rolling(90).mean()

    fig_ma = go.Figure()

    fig_ma.add_trace(
        go.Scatter(
            x=rolling_df["Date"],
            y=rolling_df["Spread"],
            name="Spread",
            line=dict(color="#1F77B4")
        )
    )

    fig_ma.add_trace(
        go.Scatter(
            x=rolling_df["Date"],
            y=rolling_df["MA7"],
            name="MA7",
            line=dict(color="#FF7F0E")
        )
    )

    fig_ma.add_trace(
        go.Scatter(
            x=rolling_df["Date"],
            y=rolling_df["MA30"],
            name="MA30",
            line=dict(color="#2CA02C")
        )
    )

    fig_ma.add_trace(
        go.Scatter(
            x=rolling_df["Date"],
            y=rolling_df["MA90"],
            name="MA90",
            line=dict(color="#D62728")
        )
    )

    fig_ma.update_layout(title="Spread with Moving Averages", height=550)

    st.plotly_chart(fig_ma, use_container_width=True)

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>🔗 Correlation Heatmap</h3>", unsafe_allow_html=True
    )

    corr_df = df.drop(columns=["Date"])

    corr = corr_df.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()

    st.pyplot(fig)

    # =====================================================
    # BASIC STATISTICS
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📊 Statistical Summary</h3>", unsafe_allow_html=True
    )

    stats_df = (
        df[
            [
                "VND/USD",
                "VNIndex",
                "Oil_Price",
                "TNX",
                "GPR",
                "Bitcoin",
                "DXY",
                "Spread",
            ]
        ]
        .describe()
        .T
    )
    st.dataframe(stats_df, use_container_width=True)

    st.divider()

    # CORELATION WITH SPREAD#

    st.markdown(
        "<h3 class='section-title'>🎯 Correlation with Spread</h3>",
        unsafe_allow_html=True,
    )

    spread_corr = corr["Spread"].drop("Spread").abs().sort_values(ascending=False)

    corr_rank = pd.DataFrame(
        {"Variable": spread_corr.index, "Correlation": spread_corr.values}
    )

    fig_corr = px.bar(
        corr_rank,
        y="Variable",
        x="Correlation",
        title="Feature Correlation with Spread",
        orientation="h"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # =====================================================
    # DATA QUALITY ASSESSMENT
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>✅ Data Quality Summary</h3>", unsafe_allow_html=True
    )

    quality = pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Missing Values",
                "Duplicate Rows",
            ],
            "Value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            ],
        }
    )
    st.dataframe(quality, use_container_width=True)

    # DATA PREVIEW#
    st.markdown(
        "<h3 class='section-title'>🗂️ Dataset Preview</h3>", unsafe_allow_html=True
    )

    preview_rows = st.slider("Rows to Display", 5, 100, 20)

    df_preview = df.copy()
    df_preview["Date"] = df_preview["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(df_preview.tail(preview_rows), hide_index=True,
                 use_container_width=True)

    # =====================================================
    # FOOTER
    # =====================================================
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center;color:#6B7280;font-size:13px;">
    Gold Spread Forecasting System | Data Analysis Module | Master Thesis Demonstration
    </div>
    """,
        unsafe_allow_html=True,
    )
