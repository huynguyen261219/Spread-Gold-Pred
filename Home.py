import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

/* MAIN */
.main .block-container{
    max-width:1400px;
    padding-top:1rem;
    padding-bottom:1rem;
}

/* BACKGROUND */
.stApp{
    background-color:#F4F6F9;
}

/* KPI CARD */
[data-testid="stMetric"]{
    background:white;
    border:1px solid #D9DEE5;
    border-radius:10px;
    padding:15px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"]{
    justify-content:center;
    font-weight:600;
}

[data-testid="stMetricValue"]{
    justify-content:center;
}

/* SECTION TITLE */
.section-title{
    color:#0D5A9C;
    font-weight:700;
}

/* BUTTON */
div.stButton > button{
    width:100%;
    border-radius:20px;
    border:1px solid #D1D5DB;
    background:white;
    color:#374151;
    font-weight:600;
    height:40px;
}

div.stButton > button:hover{
    border-color:#0D5A9C;
    color:#0D5A9C;
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
background:#0D5A9C;
padding:24px;
border-radius:10px;
margin-bottom:20px;
">

<h1 style="
margin:0;
color:white;
font-size:32px;
">
Gold Spread Forecasting System
</h1>

<p style="
margin-top:8px;
color:#E5E7EB;
font-size:15px;
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

st.markdown("<h3 class='section-title'>Model Performance</h3>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

k1.metric("Selected Model", "Linear Regression")

k2.metric("RMSE", f"{metrics['RMSE'].iloc[0]:,.0f}")

k3.metric("MAE", f"{metrics['MAE'].iloc[0]:,.0f}")

k4.metric("R² Score", f"{metrics['R2'].iloc[0]:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# DATASET INFORMATION
# =====================================================

st.markdown(
    "<h3 class='section-title'>Dataset Information</h3>", unsafe_allow_html=True
)

d1, d2, d3 = st.columns(3)

d1.metric("Observations", f"{len(df):,}")

d2.metric("Input Variables", "8")

d3.metric("Period", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")

st.divider()

# =====================================================
# HISTORICAL GOLD SPREAD TREND
# =====================================================

st.markdown(
    "<h3 class='section-title'>Historical Gold Spread Trend</h3>",
    unsafe_allow_html=True,
)

# =====================================================
# TIME FILTER
# =====================================================

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

# =====================================================
# FILTER DATA
# =====================================================

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
# CHART + SNAPSHOT
# =====================================================

chart_col, info_col = st.columns([4, 1])

with chart_col:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["Spread"],
            mode="lines",
            line=dict(
                color="#0D5A9C",
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(13,90,156,0.12)",
            hovertemplate="<b>Date:</b> %{x}<br>"
            + "<b>Spread:</b> %{y:,.0f}<extra></extra>",
        )
    )

    latest_value = chart_df["Spread"].iloc[-1]

    fig.add_annotation(
        x=chart_df["Date"].iloc[-1],
        y=latest_value,
        text=f"{latest_value:,.0f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="#0D5A9C",
        borderwidth=1,
    )

    fig.update_layout(
        height=700,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
    )

    fig.update_xaxes(
        title="Date",
        showgrid=False,
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        title="Spread (VND/lượng)",
        gridcolor="#ECEFF3",
        zeroline=False,
    )

    st.plotly_chart(fig, use_container_width=True)

with info_col:

    st.markdown("### 📊 Market Snapshot")

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
    "<h3 class='section-title'>Recent Dataset Records</h3>", unsafe_allow_html=True
)

st.dataframe(
    df.tail(20),
    use_container_width=True,
    height=350,
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<div style="
text-align:center;
font-size:13px;
color:#6B7280;
">

Gold Spread Forecasting System |
Master Thesis Demonstration |
University Research Project

</div>
""",
    unsafe_allow_html=True,
)
