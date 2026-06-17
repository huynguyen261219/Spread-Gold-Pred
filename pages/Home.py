import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def show_home():

    # =====================================================
    # PAGE CONFIG
    # =====================================================

    st.set_page_config(
        page_title="Gold Spread Forecasting System",
        page_icon="📈",
        layout="wide",
    )

    # =====================================================
    # CUSTOM CSS
    # =====================================================

    st.markdown(
        """
    <style>

    /* =====================================================
    BACKGROUND
    ===================================================== */

    .stApp{
        background:#EAF4F4;
    }

    /* =====================================================
    DATAFRAME HEADER
    ===================================================== */

    thead tr th{
        background-color:#0F4C81 !important;
        color:white !important;
        font-size:15px !important;
        font-weight:700 !important;
        text-align:center !important;
    }

    /* =====================================================
    DATAFRAME BODY
    ===================================================== */

    tbody tr{
        background-color:white !important;
    }

    tbody tr:hover{
        background-color:#E8F6F6 !important;
    }

    /* =====================================================
    DATAFRAME BORDER
    ===================================================== */

    [data-testid="stDataFrame"]{
        border-radius:15px;
        overflow:hidden;
        border:1px solid #BFDCDC;
    }
    /* =====================================================
    MAIN CONTAINER
    ===================================================== */

    .main .block-container{
        max-width:1500px;
        padding-top:1rem;
        padding-bottom:1rem;
    }

    /* =====================================================
    TEXT
    ===================================================== */

    html, body, [class*="css"]{
        font-size:17px;
    }

    h1{
        font-size:38px !important;
    }

    h2{
        font-size:30px !important;
    }

    h3{
        font-size:24px !important;
    }

    .section-title{
        color:#0F4C81;
        font-weight:700;
        margin-bottom:10px;
    }

    /* =====================================================
    METRIC CARD
    ===================================================== */

    [data-testid="stMetric"]{

        background:linear-gradient(
        135deg,
        #DFF6F5,
        #CDEEEE
        );

        border:1px solid #A9D6D5;

        border-radius:16px;

        padding:18px;

        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    }

    [data-testid="stMetricLabel"]{
        justify-content:center;
        font-size:17px;
        font-weight:700;
    }

    [data-testid="stMetricValue"]{
        justify-content:center;
        font-size:28px;
        font-weight:800;
        color:#0A2E4E;
    }

    /* =====================================================
    BUTTON
    ===================================================== */

    div.stButton > button{

        width:100%;
        height:46px;

        border-radius:12px;

        border:none;

        background:#0F4C81;

        color:white;

        font-weight:700;
    }

    div.stButton > button:hover{
        background:#1363A1;
    }

    /* =====================================================
    DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"]{
        border-radius:15px;
        overflow:hidden;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_excel("data/dataset.xlsx")

    metrics = pd.read_csv("models/linear_regression_metrics.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        """
    <div style="
    background:linear-gradient(
    135deg,
    #0F4C81,
    #1B6CA8
    );
    padding:30px;
    border-radius:18px;
    margin-bottom:25px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.15);
    ">

    <h1 style="
    margin:0;
    color:white;
    font-size:40px;
    font-weight:800;
    ">
    Gold Spread Forecasting System
    </h1>

    <p style="
    margin-top:10px;
    font-size:18px;
    color:#DCEEF9;
    ">
    Forecasting Vietnamese Gold Price Spread Using Machine Learning
    </p>

    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Model Performance</h3>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Model", "Linear Regression")

    m2.metric("RMSE", f"{metrics['RMSE'].iloc[0]:,.0f}")

    m3.metric("MAE", f"{metrics['MAE'].iloc[0]:,.0f}")

    m4.metric("R²", f"{metrics['R2'].iloc[0]:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # DATASET INFO
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Dataset Information</h3>",
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    d1.metric("Observations", f"{len(df):,}")

    d2.metric("Variables", "8")

    d3.metric("Period", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")

    st.divider()

    # =====================================================
    # TIME FILTER
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Historical Gold Spread Trend</h3>",
        unsafe_allow_html=True,
    )

    if "period" not in st.session_state:
        st.session_state.period = "1Y"

    b1, b2, b3, b4, b5, b6, b7 = st.columns(7)

    if b1.button("1M"):
        st.session_state.period = "1M"

    if b2.button("3M"):
        st.session_state.period = "3M"

    if b3.button("6M"):
        st.session_state.period = "6M"

    if b4.button("1Y"):
        st.session_state.period = "1Y"

    if b5.button("3Y"):
        st.session_state.period = "3Y"

    if b6.button("5Y"):
        st.session_state.period = "5Y"

    if b7.button("ALL"):
        st.session_state.period = "ALL"

    chart_df = df.copy()

    latest_date = chart_df["Date"].max()

    period = st.session_state.period

    if period == "1M":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(months=1)]

    elif period == "3M":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(months=3)]

    elif period == "6M":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(months=6)]

    elif period == "1Y":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(years=1)]

    elif period == "3Y":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(years=3)]

    elif period == "5Y":
        chart_df = chart_df[chart_df["Date"] >= latest_date - pd.DateOffset(years=5)]

    # =====================================================
    # CHART + MARKET SNAPSHOT
    # =====================================================

    left, right = st.columns([4, 1])

    with left:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_df["Date"],
                y=chart_df["Spread"],
                mode="lines",
                line=dict(color="#0F4C81", width=3),
                fill="tozeroy",
                fillcolor="rgba(15,76,129,0.12)",
            )
        )

        fig.update_layout(
            height=700,
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=15),
        )

        fig.update_xaxes(showgrid=False, rangeslider_visible=False)

        fig.update_yaxes(gridcolor="#E5E7EB")

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown(
            """
        <div style="
        background:#0F4C81;
        padding:12px;
        border-radius:12px;
        text-align:center;
        margin-bottom:10px;
        ">
        <h3 style="color:white;margin:0;">
        Market Snapshot
        </h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.metric("Current Spread", f"{chart_df['Spread'].iloc[-1]:,.0f}")

        st.metric("Average Spread", f"{chart_df['Spread'].mean():,.0f}")

        st.metric("Maximum Spread", f"{chart_df['Spread'].max():,.0f}")

        st.metric("Records", f"{len(chart_df):,}")

        st.metric("Period", period)

    # =====================================================
    # RECENT DATA
    # =====================================================

    st.divider()

    st.markdown(
        "<h3 class='section-title'>Recent Dataset Records</h3>",
        unsafe_allow_html=True,
    )

    st.dataframe(df.tail(20), use_container_width=True, height=400)

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
    <div style="
    text-align:center;
    font-size:14px;
    color:#4B5563;
    padding-bottom:10px;
    ">
    Gold Spread Forecasting System |
    Master Thesis Demonstration |
    University Research Project
    </div>
    """,
        unsafe_allow_html=True,
    )
