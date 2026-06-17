import streamlit as st
from pages import (
    show_home,
    show_data_analysis,
    show_market_intelligence,
    show_model_results,
    show_prediction_center,
)

st.set_page_config(initial_sidebar_state="collapsed")


pages = [
    st.Page(show_home, title="Home", icon="🏠"),
    st.Page(show_data_analysis, title="Data Analysis", icon="📊"),
    st.Page(show_market_intelligence, title="Market Intelligence", icon="📈"),
    st.Page(show_model_results, title="Model Results", icon="🔮"),
    st.Page(show_prediction_center, title="Prediction Center", icon="🤖"),
]


curr_page = st.navigation(pages, position="top")
curr_page.run()
