import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def show_market_intelligence():
    """
    Market Intelligence Module - Market Monitoring and Gold Spread Trend Analysis
    """

    # =====================================================
    # PAGE CONFIG
    # =====================================================
    st.set_page_config(page_title="Market Intelligence", page_icon="💹", layout="wide")

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

div[data-baseweb="select"] > div{
    border:3px solid #0D5A9C !important;
    border-radius:16px !important;
    box-shadow:0 4px 12px rgba(13,90,156,0.25) !important;
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
    <h1 style="margin:0;color:white;">Market Intelligence Center</h1>
    <p style="margin-top:8px;color:#E5E7EB;">
    Real-time Market Monitoring and Gold Spread Trend Analysis
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
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # =====================================================
    # ANALYSIS PERIOD
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📅 Analysis Period</h3>",
        unsafe_allow_html=True
    )

    period = st.selectbox(
        "Select Time Range",
        ("All Data","1 Year","6 Months","3 Months","1 Month"),
        placeholder="Select Time Range...",
    )

    df_view = df.copy()

    if period == "1 Year":

        df_view = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=1)]

    elif period == "6 Months":

        df_view = df[
            df["Date"] >=
            df["Date"].max() - pd.DateOffset(months=6)
            ]

    elif period == "3 Months":

        df_view = df[
            df["Date"] >=
            df["Date"].max() - pd.DateOffset(months=3)
            ]

    elif period == "1 Month":

        df_view = df[
            df["Date"] >=
            df["Date"].max() - pd.DateOffset(months=1)
            ]

    # =====================================================
    # KPI
    # =====================================================

    latest_spread = df_view["Spread"].iloc[-1]

    avg_7 = df_view["Spread"].tail(7).mean()

    avg_30 = df_view["Spread"].tail(30).mean()

    max_spread = df_view["Spread"].max()

    min_spread = df_view["Spread"].min()

    volatility = df_view["Spread"].tail(30).std()

    st.markdown(
        "<h3 class='section-title'>🎯 Market Snapshot</h3>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    c4, c5, c6 = st.columns([1, 1, 1])

    previous_spread = df_view["Spread"].iloc[-2]

    delta_pct = (
            (latest_spread - previous_spread)
            / previous_spread
            * 100
    )

    c1.metric(
        "Current Spread",
        f"{latest_spread:,.0f}",
        f"{delta_pct:.2f}% vs previous day"
    )

    c2.metric(
        "7-Day Avg",
        f"{avg_7:,.0f}"
    )

    c3.metric(
        "30-Day Avg",
        f"{avg_30:,.0f}"
    )

    c4.metric(
        "Maximum",
        f"{max_spread:,.0f}"
    )

    c5.metric(
        "Minimum",
        f"{min_spread:,.0f}"
    )

    c6.metric(
        "Volatility",
        f"{volatility:,.0f}"
    )

    st.divider()

    # =====================================================
    # GOLD MARKET OVERVIEW
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>🥇 Gold Market Overview</h3>",
        unsafe_allow_html=True
    )

    fig_gold = go.Figure()

    fig_gold.add_trace(
        go.Scatter(
            x=df_view["Date"],
            y=df_view["SJC"],
            name="Domestic Gold (SJC)",
            line=dict(
                color="#0D5A9C",
                width=3
            )
        )
    )

    fig_gold.add_trace(
        go.Scatter(
            x=df_view["Date"],
            y=df_view["Gold-global(VND)"],
            name="World Gold (Converted)",
            line=dict(
                color="#E53935",
                width=3
            )
        )
    )
    fig_gold.add_trace(
        go.Scatter(
            x=df_view["Date"],
            y=df_view["Spread"],
            name="Spread",
            yaxis="y2",
            line=dict(color="#2E7D32", width=3),
        )
    )

    fig_gold.update_layout(
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified"
    )

    fig_gold.update_layout(yaxis2=dict(overlaying="y", side="right"))

    st.plotly_chart(
        fig_gold,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # TREND SIGNAL
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📡 Market Trend Signal</h3>",
        unsafe_allow_html=True
    )

    if avg_7 > avg_30:

        st.success(
            f"""
    🟢 BULLISH SIGNAL

    7-Day Average:
    {avg_7:,.0f}

    30-Day Average:
    {avg_30:,.0f}

    Short-term momentum is stronger than long-term trend.
    """
        )

    else:

        st.error(
            f"""
    🔴 BEARISH SIGNAL

    7-Day Average:
    {avg_7:,.0f}

    30-Day Average:
    {avg_30:,.0f}

    Short-term momentum is weaker than long-term trend.
    """
        )

    st.divider()

    # =====================================================
    # SPREAD TREND
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📈 Spread Trend Analysis</h3>",
        unsafe_allow_html=True
    )

    fig_spread = px.line(
        df_view,
        x="Date",
        y="Spread",
        title="Historical Gold Spread"
    )

    fig_spread.update_traces(
        line_color="#0D5A9C",
        line_width=3
    )

    fig_spread.update_layout(
        height=550
    )

    st.plotly_chart(
        fig_spread,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # MOVING AVERAGE
    # =====================================================

    # df_view["MA7"] = df_view["Spread"].rolling(7).mean()
    #
    # df_view["MA30"] = df_view["Spread"].rolling(30).mean()
    #
    # df_view["MA90"] = df_view["Spread"].rolling(90).mean()
    #
    # st.markdown(
    #     "<h3 class='section-title'>📊 Moving Average Analysis</h3>",
    #     unsafe_allow_html=True
    # )
    #
    # fig_ma = go.Figure()
    #
    # fig_ma.add_trace(
    #     go.Scatter(
    #         x=df_view["Date"],
    #         y=df_view["Spread"],
    #         name="Spread"
    #     )
    # )
    #
    # fig_ma.add_trace(
    #     go.Scatter(
    #         x=df_view["Date"],
    #         y=df_view["MA7"],
    #         name="MA7"
    #     )
    # )
    #
    # fig_ma.add_trace(
    #     go.Scatter(
    #         x=df_view["Date"],
    #         y=df_view["MA30"],
    #         name="MA30"
    #     )
    # )
    #
    # fig_ma.add_trace(
    #     go.Scatter(
    #         x=df_view["Date"],
    #         y=df_view["MA90"],
    #         name="MA90"
    #     )
    # )
    #
    # fig_ma.update_layout(
    #     height=550,
    #     hovermode="x unified"
    # )
    #
    # st.plotly_chart(
    #     fig_ma,
    #     use_container_width=True
    # )

    # st.divider()

    # =====================================================
    # VOLATILITY
    # =====================================================

    df_view["Volatility30"] = (
        df_view["Spread"]
        .rolling(30)
        .std()
    )

    st.markdown(
        "<h3 class='section-title'>⚡ Market Risk & Volatility</h3>",
        unsafe_allow_html=True
    )

    fig_vol = px.line(
        df_view,
        x="Date",
        y="Volatility30"
    )

    fig_vol.update_traces(
        line_color="#E53935"
    )

    st.plotly_chart(
        fig_vol,
        use_container_width=True
    )

    if volatility < 500000:

        st.success("🟢 Low Risk Market")

    elif volatility < 1500000:

        st.warning("🟡 Medium Risk Market")

    else:

        st.error("🔴 High Risk Market")

    st.divider()

    # =====================================================
    # REGIME ANALYSIS
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>🎯 Spread Regime Analysis</h3>",
        unsafe_allow_html=True
    )

    spread_mean = df_view["Spread"].mean()

    df_view["Regime"] = np.where(
        df_view["Spread"] > spread_mean,
        "High Spread",
        "Low Spread"
    )

    regime = (
        df_view["Regime"]
        .value_counts()
        .reset_index()
    )

    regime.columns = ["Regime", "Count"]

    fig_regime = px.pie(
        regime,
        names="Regime",
        values="Count"
    )

    st.plotly_chart(
        fig_regime,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # MONTHLY TREND
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📅 Monthly Trend Analysis</h3>",
        unsafe_allow_html=True
    )

    monthly = df_view.set_index("Date").resample("ME")["Spread"].mean().reset_index()

    fig_month = px.bar(
        monthly,
        x="Date",
        y="Spread"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # EXTREME VALUES
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📊 Extreme Values Analysis</h3>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Top 10 Highest Spread")

        st.dataframe(
            df_view.nlargest(
                10,
                "Spread"
            )[["Date", "Spread"]],
            use_container_width=True
        )

    with c2:

        st.subheader("Top 10 Lowest Spread")

        st.dataframe(
            df_view.nsmallest(
                10,
                "Spread"
            )[["Date", "Spread"]],
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # MARKET INSIGHT
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>🧠 Market Insight Summary</h3>",
        unsafe_allow_html=True
    )

    trend = (
        "Bullish"
        if avg_7 > avg_30
        else "Bearish"
    )

    st.info(f"""
    Current Trend: {trend}

    Current Spread:
    {latest_spread:,.0f}

    30-Day Average:
    {avg_30:,.0f}

    Volatility:
    {volatility:,.0f}

    Analysis Period:
    {period}

    spread_mean = df_view["Spread"].mean()


    """)

    st.divider()

    # =====================================================
    # MARKET REPORT
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>📋 Export Market Report</h3>", unsafe_allow_html=True
    )

    report = pd.DataFrame(
        {
            "Metric": [
                "Current Spread",
                "7-Day Average",
                "30-Day Average",
                "Maximum Spread",
                "Minimum Spread",
                "Volatility (30-day)",
                "Last Updated",
            ],
            "Value": [
                f"{latest_spread:,.0f}",
                f"{avg_7:,.0f}",
                f"{avg_30:,.0f}",
                f"{max_spread:,.0f}",
                f"{min_spread:,.0f}",
                f"{volatility:,.0f}",
                str(df["Date"].max().date()),
            ],
        }
    )

    st.download_button(
        "📥 Download Market Report (CSV)",
        report.to_csv(index=False),
        file_name="market_intelligence_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # =====================================================
    # FOOTER
    # =====================================================
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center;color:#6B7280;font-size:13px;">
    Gold Spread Forecasting System | Market Intelligence Module | Master Thesis Demonstration
    </div>
    """,
        unsafe_allow_html=True,
    )
