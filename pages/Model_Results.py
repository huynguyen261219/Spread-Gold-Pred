import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_model_results():

    # =====================================================
    # PAGE CONFIG
    # =====================================================

    st.set_page_config(page_title="Model Results", page_icon="🤖", layout="wide")

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
    Model Evaluation Center
    </h1>

    <p style="
    margin-top:8px;
    color:#E5E7EB;
    ">
    Performance Analysis of the Selected Machine Learning Model
    </p>

    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    metrics = pd.read_csv("models/linear_regression_metrics.csv")

    test = pd.read_csv("models/linear_regression_actual_vs_pred_test.csv")

    coef = pd.read_csv("models/linear_regression_coefficients.csv")

    # =====================================================
    # KPI
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Model Performance Summary</h3>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("RMSE", f"{metrics['RMSE'].iloc[0]:,.0f}")

    k2.metric("MAE", f"{metrics['MAE'].iloc[0]:,.0f}")

    if "MAPE" in metrics.columns:
        mape = metrics["MAPE"].iloc[0]
    else:
        mape = 0

    k3.metric("MAPE", f"{mape:.2f}%")

    k4.metric("R² Score", f"{metrics['R2'].iloc[0]:.4f}")

    st.divider()

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Actual vs Predicted Spread</h3>",
        unsafe_allow_html=True,
    )

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(x=test.index, y=test["Actual"], mode="lines", name="Actual")
    )

    fig1.add_trace(
        go.Scatter(x=test.index, y=test["Predicted"], mode="lines", name="Predicted")
    )

    fig1.update_layout(
        height=550,
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Observation",
        yaxis_title="Spread (VND/lượng)",
    )

    st.plotly_chart(fig1, use_container_width=True)

    # =====================================================
    # SCATTER + FEATURE IMPORTANCE
    # =====================================================

    left, right = st.columns(2)

    with left:

        st.markdown(
            "<h3 class='section-title'>Prediction Accuracy</h3>", unsafe_allow_html=True
        )

        fig2 = px.scatter(test, x="Actual", y="Predicted", trendline="ols")

        fig2.update_layout(height=500, paper_bgcolor="white", plot_bgcolor="white")

        st.plotly_chart(fig2, use_container_width=True)

    with right:

        st.markdown(
            "<h3 class='section-title'>Feature Importance</h3>", unsafe_allow_html=True
        )

        coef = coef.sort_values(coef.columns[1], ascending=False)

        fig4 = px.bar(coef, x=coef.columns[0], y=coef.columns[1])

        fig4.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Feature",
            yaxis_title="Coefficient",
        )

        st.plotly_chart(fig4, use_container_width=True)

    # =====================================================
    # RESIDUAL
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Residual Distribution</h3>", unsafe_allow_html=True
    )

    test["Residual"] = test["Actual"] - test["Predicted"]

    fig3 = px.histogram(test, x="Residual", nbins=30)

    fig3.update_layout(height=450, paper_bgcolor="white", plot_bgcolor="white")

    st.plotly_chart(fig3, use_container_width=True)

    # =====================================================
    # METRICS TABLE
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Detailed Performance Metrics</h3>",
        unsafe_allow_html=True,
    )

    st.dataframe(metrics, use_container_width=True)

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>Model Interpretation</h3>", unsafe_allow_html=True
    )

    st.info("""
    The Linear Regression model demonstrates strong predictive
    performance for forecasting the Vietnamese gold price spread.

    The Actual vs Predicted chart indicates that the model is able
    to capture the overall movement of the spread over time.

    The scatter plot shows a strong linear relationship between
    observed and predicted values, while the residual distribution
    suggests that prediction errors are generally centered around zero.

    Feature importance analysis provides insight into the relative
    contribution of each macroeconomic variable to the forecasting process.
    """)
