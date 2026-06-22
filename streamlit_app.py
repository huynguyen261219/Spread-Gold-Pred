import streamlit as st
from pages import (
    show_home,
    show_data_analysis,
    show_market_intelligence,
    show_model_results,
    show_prediction_center,
)

st.set_page_config(
    page_title="Gold Spread Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>

    /* =====================================================
    METRIC CARD
    ===================================================== */
    div[data-testid="stMetric"]{

        background: linear-gradient(
            135deg,
            #FBFDFF 0%,
            #F2F7FC 100%
        ) !important;

        border: 1px solid #D7E6F5 !important;

        border-radius: 14px !important;

        padding: 18px !important;

        box-shadow:
            0 2px 8px rgba(13,90,156,0.08) !important;

        transition: all 0.25s ease !important;
    }


    /* =====================================================
    HOVER EFFECT
    ===================================================== */
    div[data-testid="stMetric"]:hover{

        transform: translateY(-3px);

        box-shadow:
            0 6px 18px rgba(13,90,156,0.15) !important;

        border: 1px solid #BFD7F0 !important;
    }


    /* =====================================================
    LABEL
    ===================================================== */
    div[data-testid="stMetricLabel"]{

        color: #0D5A9C !important;

        font-weight: 600 !important;

        font-size: 15px !important;
    }


    /* =====================================================
    VALUE
    ===================================================== */
    div[data-testid="stMetricValue"]{

        color: #1F2937 !important;

        font-weight: 800 !important;

        font-size: 32px !important;
    }


    /* =====================================================
    DELTA
    ===================================================== */
    div[data-testid="stMetricDelta"]{

        font-size: 16px !important;

        font-weight: 600 !important;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )

pages = [
    st.Page(show_home, title="Home", icon="🏠"),
    st.Page(show_data_analysis, title="Data Analysis", icon="📊"),
    st.Page(show_market_intelligence, title="Market Intelligence", icon="📈"),
    st.Page(show_model_results, title="Model Results", icon="🔮"),
    st.Page(show_prediction_center, title="Prediction Center", icon="🤖"),
]


curr_page = st.navigation(pages, position="top")
curr_page.run()
