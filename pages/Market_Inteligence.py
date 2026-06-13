import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Market Intelligence", page_icon="📊", layout="wide")

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
Market Intelligence Center
</h1>

<p style="
margin-top:8px;
color:#E5E7EB;
">
Market Monitoring and Gold Spread Trend Analysis
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
# CALCULATIONS
# =====================================================

latest_spread = df["Spread"].iloc[-1]

avg_7 = df["Spread"].tail(7).mean()

avg_30 = df["Spread"].tail(30).mean()

max_spread = df["Spread"].max()

min_spread = df["Spread"].min()

volatility = df["Spread"].tail(30).std()

# =====================================================
# MARKET SNAPSHOT
# =====================================================

st.markdown("<h3 class='section-title'>Market Snapshot</h3>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Current Spread", f"{latest_spread:,.0f}")

c2.metric("7-Day Avg", f"{avg_7:,.0f}")

c3.metric("30-Day Avg", f"{avg_30:,.0f}")

c4.metric("Maximum", f"{max_spread:,.0f}")

c5.metric("Minimum", f"{min_spread:,.0f}")

c6.metric("Volatility", f"{volatility:,.0f}")

st.divider()

# =====================================================
# TREND SIGNAL
# =====================================================

st.markdown(
    "<h3 class='section-title'>Market Trend Signal</h3>", unsafe_allow_html=True
)

if avg_7 > avg_30:

    st.success("🟢 Bullish Signal: Short-term average is above long-term average.")

else:

    st.error("🔴 Bearish Signal: Short-term average is below long-term average.")

# =====================================================
# MOVING AVERAGES
# =====================================================

df["MA7"] = df["Spread"].rolling(7).mean()

df["MA30"] = df["Spread"].rolling(30).mean()

st.markdown(
    "<h3 class='section-title'>Moving Average Analysis</h3>", unsafe_allow_html=True
)

fig_ma = go.Figure()

fig_ma.add_trace(
    go.Scatter(x=df["Date"], y=df["Spread"], name="Spread", line=dict(width=2))
)

fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA7"], name="MA7", line=dict(width=2)))

fig_ma.add_trace(
    go.Scatter(x=df["Date"], y=df["MA30"], name="MA30", line=dict(width=2))
)

fig_ma.update_layout(
    height=550, paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified"
)

st.plotly_chart(fig_ma, use_container_width=True)

# =====================================================
# VOLATILITY
# =====================================================

df["Volatility_30"] = df["Spread"].rolling(30).std()

st.markdown(
    "<h3 class='section-title'>30-Day Rolling Volatility</h3>", unsafe_allow_html=True
)

fig_vol = px.line(df, x="Date", y="Volatility_30")

fig_vol.update_layout(height=450, paper_bgcolor="white", plot_bgcolor="white")

st.plotly_chart(fig_vol, use_container_width=True)

# =====================================================
# HIGHEST / LOWEST SPREAD
# =====================================================

left, right = st.columns(2)

with left:

    st.markdown(
        "<h3 class='section-title'>Top 10 Highest Spread</h3>", unsafe_allow_html=True
    )

    top_high = df.nlargest(10, "Spread")[["Date", "Spread"]]

    st.dataframe(top_high, use_container_width=True)

with right:

    st.markdown(
        "<h3 class='section-title'>Top 10 Lowest Spread</h3>", unsafe_allow_html=True
    )

    top_low = df.nsmallest(10, "Spread")[["Date", "Spread"]]

    st.dataframe(top_low, use_container_width=True)

# =====================================================
# MARKET REPORT
# =====================================================

st.divider()

report = pd.DataFrame(
    {
        "Metric": [
            "Current Spread",
            "7-Day Average",
            "30-Day Average",
            "Maximum Spread",
            "Minimum Spread",
            "Volatility",
        ],
        "Value": [latest_spread, avg_7, avg_30, max_spread, min_spread, volatility],
    }
)

st.download_button(
    "📥 Download Market Report",
    report.to_csv(index=False),
    file_name="market_report.csv",
    mime="text/csv",
    use_container_width=True,
)

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
Market Intelligence Module |
Master Thesis Demonstration

</div>
""",
    unsafe_allow_html=True,
)
