import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def show_data_analysis():

    # =====================================================
    # PAGE CONFIG
    # =====================================================

    st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

    # =====================================================
    # CSS
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

    <h1 style="
    margin:0;
    color:white;
    ">
    Data Analysis Center
    </h1>

    <p style="
    margin-top:8px;
    color:#E5E7EB;
    ">
    Exploratory Data Analysis of Gold Price Spread Dataset
    </p>

    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_excel("data/dataset.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Dataset Overview</h3>", unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Observations", f"{len(df):,}")

    k2.metric("Variables", f"{df.shape[1]-1}")

    k3.metric("Start Date", str(df["Date"].min().date()))

    k4.metric("End Date", str(df["Date"].max().date()))

    st.divider()

    # =====================================================
    # DISTRIBUTION + PREVIEW
    # =====================================================

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown(
            "<h3 class='section-title'>Spread Distribution</h3>", unsafe_allow_html=True
        )

        fig_hist = px.histogram(df, x="Spread", nbins=40)

        fig_hist.update_layout(height=450, paper_bgcolor="white", plot_bgcolor="white")

        st.plotly_chart(fig_hist, use_container_width=True)

    with right:

        st.markdown(
            "<h3 class='section-title'>Dataset Preview</h3>", unsafe_allow_html=True
        )

        st.dataframe(df.tail(15), height=450, use_container_width=True)

    # =====================================================
    # TIME SERIES
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Spread Time Series</h3>", unsafe_allow_html=True
    )

    fig_ts = px.line(df, x="Date", y="Spread")

    fig_ts.update_traces(line_color="#0D5A9C", line_width=3)

    fig_ts.update_layout(
        height=550, paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified"
    )

    st.plotly_chart(fig_ts, use_container_width=True)

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Correlation Matrix</h3>", unsafe_allow_html=True
    )

    corr_df = df.copy()

    if "Spread_lag1" not in corr_df.columns:
        corr_df["Spread_lag1"] = corr_df["Spread"].shift(1)

    corr_df = corr_df.dropna()

    numeric_cols = [
        "VND/USD",
        "VNIndex",
        "Oil_Price",
        "DXY",
        "TNX",
        "GPR",
        "bitcoin",
        "Spread_lag1",
        "Spread",
    ]

    corr = corr_df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )

    plt.title("Correlation Matrix of Variables", fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")

    plt.yticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    # =====================================================
    # DATA QUALITY
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Data Quality Summary</h3>", unsafe_allow_html=True
    )

    quality = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
            "Value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            ],
        }
    )

    st.dataframe(quality, use_container_width=True)

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
    <div style="
    text-align:center;
    color:#6B7280;
    font-size:13px;
    ">

    Gold Spread Forecasting System |
    Data Analysis Module |
    Master Thesis Demonstration

    </div>
    """,
        unsafe_allow_html=True,
    )
