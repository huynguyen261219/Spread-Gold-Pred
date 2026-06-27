import streamlit as st
import numpy as np
import plotly.graph_objects as go


# =====================================================
# HOME DASHBOARD
# =====================================================
def show_home():

    # =====================================================
    # STYLE
    # =====================================================
    st.markdown(
        """
    <style>

    .main .block-container{
        max-width:1400px;
        padding-top:1rem;
    }

    .stApp{
        background:#F4F6F9;
    }

    [data-testid="stMetric"]{
        background:linear-gradient(
            135deg,
            #F8FBFF 0%,
            #EEF6FF 100%
        );
        border:1px solid #D7E6F5;
        border-radius:14px;
        padding:18px;
        box-shadow:0 2px 10px rgba(13,90,156,0.08);
        transition:all 0.3s ease;
    }

    [data-testid="stMetric"]:hover{
        transform:translateY(-2px);
        box-shadow:0 4px 12px rgba(13,90,156,0.15);
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
    border-radius:12px;
    margin-bottom:20px;
    ">
    <h1 style="
    margin:0;
    color:white;
    ">
    🏠 Home Dashboard
    </h1>

    <p style="
    margin-top:8px;
    color:#E5E7EB;
    ">
    Vietnamese Gold Spread Forecasting System
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # SIMULATED DATA
    # =====================================================
    np.random.seed(42)

    gold_spread = 15000000 + np.random.normal(0, 200000)
    vndusd = 26000 + np.random.normal(0, 20)
    vnindex = 1500 + np.random.normal(0, 10)
    oil = 70 + np.random.normal(0, 1)

    spread_change = np.random.normal(0, 0.8)

    # =====================================================
    # KPI CARDS
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📊 Market Overview</h3>", unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Gold Spread", f"{gold_spread:,.0f}", f"{spread_change:+.2f}%")

    c2.metric("USD/VND", f"{vndusd:,.0f}", "+0.10%")

    c3.metric("VN-Index", f"{vnindex:,.0f}", "-0.20%")

    c4.metric("Oil Price (USD)", f"{oil:,.2f}", "+0.50%")

    st.divider()

    # # =====================================================
    # # TREND CHART
    # # =====================================================
    # st.markdown(
    #     "<h3 class='section-title'>📈 Gold Spread Trend</h3>", unsafe_allow_html=True
    # )
    #
    # days = 60
    # base = 15000000
    #
    # noise = np.random.normal(0, 15000, days)
    #
    # trend = base + np.cumsum(noise)
    #
    # fig = go.Figure()
    #
    # fig.add_trace(
    #     go.Scatter(
    #         y=trend,
    #         mode="lines",
    #         name="Gold Spread",
    #         line=dict(color="#0D5A9C", width=4),
    #     )
    # )
    #
    # fig.update_layout(
    #     title="Gold Spread Trend Simulation",
    #     height=500,
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    #     hovermode="x unified",
    #     xaxis_title="Time",
    #     yaxis_title="Spread (VND/lượng)",
    #     margin=dict(l=20, r=20, t=50, b=20),
    # )
    #
    # st.plotly_chart(fig, use_container_width=True)
    #
    # st.divider()

    # =====================================================
    # SIGNAL GAUGE
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>🎯 Market Signal Strength</h3>",
        unsafe_allow_html=True,
    )

    signal_score = np.random.randint(40, 90)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=signal_score,
            title={"text": "Signal Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0D5A9C"},
                "steps": [
                    {"range": [0, 40], "color": "#FFEBEE"},
                    {"range": [40, 70], "color": "#FFF3E0"},
                    {"range": [70, 100], "color": "#E8F5E9"},
                ],
            },
        )
    )

    gauge.update_layout(height=350)

    st.plotly_chart(gauge, use_container_width=True)

    st.divider()

    # =====================================================
    # INSIGHTS
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>🧠 Market Insights</h3>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
### Positive Factors

✅ Gold spread remains stable

✅ FX market under control

✅ No abnormal market movement

✅ Forecast reliability remains high
""")

    with col2:

        st.warning("""
### Risk Factors

⚠️ USD fluctuations

⚠️ Global geopolitical tensions

⚠️ Oil market volatility

⚠️ Inflation uncertainty
""")

    st.divider()

    # =====================================================
    # SIGNAL
    # =====================================================
    signal = np.random.choice(["BUY", "HOLD", "SELL"], p=[0.3, 0.5, 0.2])

    st.markdown(
        "<h3 class='section-title'>📌 Market Recommendation</h3>",
        unsafe_allow_html=True,
    )

    if signal == "BUY":

        st.success("🟢 BUY SIGNAL — Market conditions support accumulation.")

    elif signal == "SELL":

        st.error("🔴 SELL SIGNAL — Risk level increasing, caution advised.")

    else:

        st.info("🟡 HOLD SIGNAL — Neutral market conditions.")

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

    Gold Spread Forecasting System<br>

    Master Thesis Demonstration Dashboard

    </div>
    """,
        unsafe_allow_html=True,
    )
