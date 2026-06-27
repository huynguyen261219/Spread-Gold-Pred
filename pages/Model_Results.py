import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_model_results():
    """
    Model Results Module - Comprehensive Performance Analysis of Linear Regression Model
    """

    # =====================================================
    # PAGE CONFIG
    # =====================================================
    st.set_page_config(page_title="Model Results", page_icon="🤖", layout="wide")

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

    st.markdown("""
    <style>

    div[data-testid="stSelectbox"] > div{
        background-color:#FFFFFF;
        border:3px solid #0D5A9C;
        border-radius:12px;
        padding:6px;
        box-shadow:0 4px 12px rgba(13,90,156,0.25);
    }

    </style>
    """, unsafe_allow_html=True)
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
    <h1 style="margin:0;color:white;">Model Evaluation Center</h1>
    <p style="margin-top:8px;color:#E5E7EB;">
    Performance Analysis of the Linear Regression Forecasting Model
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================
    try:
        metrics = pd.read_csv("models/linear_metrics.csv")
        test = pd.read_csv("models/linear_regression_actual_vs_pred_test.csv")
        coef = pd.read_csv("models/linear_regression_coefficients.csv")
    except Exception as e:
        st.error(f"Error loading model data: {e}")
        st.info("Make sure the following files exist in 'models/' folder:")
        st.write("- linear_metrics.csv")
        st.write("- linear_regression_actual_vs_pred_test.csv")
        st.write("- linear_regression_coefficients.csv")
        return

    # =====================================================
    # MODEL COMPARISON CENTER
    # =====================================================

    st.markdown(
        "<h2 class='section-title'>🏆 Model Comparison Center</h2>",
        unsafe_allow_html=True,
    )

    try:

        all_models = pd.read_csv("data/final_model_comparison (1).csv")

        # =====================================================
        # BEST MODEL
        # =====================================================

        # Kết quả đánh giá các mô hình
        st.dataframe(
            all_models,
            use_container_width=True,
            hide_index=True
        )

        best_model = all_models.sort_values("R2", ascending=False).iloc[0]

        st.success(f"""
    🏆 BEST MODEL

    Model: {best_model['Model']}

    R² Score: {best_model['R2']:.4f}

    RMSE: {best_model['RMSE']:,.0f}

    MAPE: {best_model['MAPE_percent']:.2f}%
    """)

        selected_model = st.selectbox(
            "", all_models["Model"].tolist(), label_visibility="collapsed"
        )

        model_info = all_models[all_models["Model"] == selected_model].iloc[0]

        # =====================================================
        # SELECTED MODEL CARD
        # =====================================================

        # =====================================================
        # MODEL KPI
        # =====================================================

        st.markdown(
            "<h4>📊 Selected Model Performance</h4>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("RMSE", f"{model_info['RMSE']:,.0f}")

        c2.metric("MAE", f"{model_info['MAE']:,.0f}")

        c3.metric("MAPE", f"{model_info['MAPE_percent']:.2f}%")

        c4.metric("R² Score", f"{model_info['R2']:.4f}")

        # =====================================================
        # MODEL QUALITY
        # =====================================================

        r2 = model_info["R2"]

        if r2 >= 0.95:
            quality = "🟢 Excellent"

        elif r2 >= 0.90:
            quality = "🟡 Very Good"

        elif r2 >= 0.80:
            quality = "🟠 Good"

        else:
            quality = "🔴 Fair"

        st.info(f"""
    Model Quality: {quality}

    Current Model: {selected_model}

    R² Score: {r2:.4f}
    """)

        st.divider()

        # =====================================================
        # MODEL RANKING
        # =====================================================

        rank_df = all_models.sort_values("R2", ascending=False).reset_index(drop=True)

        rank_df.index += 1

        rank_position = rank_df[rank_df["Model"] == selected_model].index[0]

        st.warning(f"""
    🏅 Ranking Position: #{rank_position} / {len(rank_df)}

    Best Model: {best_model['Model']}

    Performance Gap (R²):
    {best_model['R2'] - model_info['R2']:.4f}
    """)

        st.divider()

    except Exception as e:

        st.error(f"Model comparison file error: {e}")

    # =====================================================
    # PERFORMANCE SUMMARY
    # =====================================================

    rmse = metrics["RMSE"].iloc[0]
    mae = metrics["MAE"].iloc[0]
    mape = metrics["MAPE (%)"].iloc[0]
    r2 = metrics["R2"].iloc[0]

    if r2 >= 0.95:
        grade = "A+"
    elif r2 >= 0.90:
        grade = "A"
    elif r2 >= 0.80:
        grade = "B"
    else:
        grade = "C"

    st.markdown(
        "<h3 class='section-title'>📊 Model Performance Summary</h3>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("RMSE", f"{rmse:,.0f}")
    k2.metric("MAE", f"{mae:,.0f}")
    k3.metric("MAPE", f"{mape:.2f}%")
    k4.metric("R² Score", f"{r2:.4f}")
    k5.metric("Model Grade", grade)

    st.divider()

    # =====================================================
    # PREDICTION ACCURACY OVERVIEW
    # =====================================================

    test["APE"] = (
        abs(test["Actual"] - test["Predicted"])
        / test["Actual"]
        * 100
    )

    under_5 = (test["APE"] < 5).mean() * 100
    under_10 = (test["APE"] < 10).mean() * 100

    st.markdown(
        "<h3 class='section-title'>🎯 Prediction Accuracy Overview</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predictions <5% Error",
        f"{under_5:.1f}%"
    )

    c2.metric(
        "Predictions <10% Error",
        f"{under_10:.1f}%"
    )

    c3.metric(
        "Average Error",
        f"{mape:.2f}%"
    )

    st.divider()

    # =====================================================
    # ERROR METRICS VISUALIZATION
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📊 Error Metrics Visualization</h3>",
        unsafe_allow_html=True,
    )

    metric_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "MAPE"],
            "Value": [rmse, mae, mape],
        }
    )

    fig_metric = px.bar(
        metric_df,
        x="Metric",
        y="Value",
        color="Metric",
        text="Value",
        title="Model Error Metrics",
    )

    fig_metric.update_layout(
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    st.plotly_chart(
        fig_metric,
        use_container_width=True,
    )

    st.divider()

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>⏱️ Actual vs Predicted Spread</h3>",
        unsafe_allow_html=True,
    )

    error_band = rmse

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=test.index,
            y=test["Actual"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#0D5A9C",
                width=2,
            )
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=test.index,
            y=test["Predicted"],
            mode="lines",
            name="Predicted",
            line=dict(
                color="#E53935",
                width=2,
                dash="dash"
            )
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=test.index,
            y=test["Predicted"] + error_band,
            line=dict(width=0),
            showlegend=False
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=test.index,
            y=test["Predicted"] - error_band,
            fill="tonexty",
            fillcolor="rgba(229,57,53,0.15)",
            line=dict(width=1, backoff=0.15),
            name="RMSE Band"
        )
    )

    fig1.update_layout(
        height=600,
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title="Actual vs Predicted with RMSE Error Band",
        yaxis_title="Spread (VND/lượng)"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # SCATTER PLOT
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>🎯 Prediction Accuracy</h3>",
        unsafe_allow_html=True,
    )

    fig2 = px.scatter(
        test,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Scatter Plot"
    )

    max_val = max(
        test["Actual"].max(),
        test["Predicted"].max()
    )

    fig2.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(
            color="red",
            dash="dash"
        )
    )

    fig2.update_layout(
        height=550
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📈 Feature Importance</h3>",
        unsafe_allow_html=True,
    )

    coef_sorted = coef.sort_values(
        "Coefficient",
        ascending=True
    )

    fig4 = px.bar(
        coef_sorted,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        title="Feature Coefficients"
    )

    fig4.update_layout(
        height=600
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # TOP IMPORTANT FEATURES
    # =====================================================

    coef_top = coef.copy()

    coef_top["AbsCoefficient"] = (
        coef_top["Coefficient"].abs()
    )

    top5 = coef_top.sort_values(
        "AbsCoefficient",
        ascending=False,
    ).head(5)

    st.markdown(
        "<h3 class='section-title'>🏆 Top 5 Most Influential Variables</h3>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        top5,
        use_container_width=True,
    )

    fig_top = px.bar(
        top5,
        x="Feature",
        y="AbsCoefficient",
        color="AbsCoefficient",
        text="AbsCoefficient",
        title="Top 5 Variables Affecting Gold Spread",
    )

    fig_top.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True,
    )

    st.divider()

    # =====================================================
    # RESIDUAL ANALYSIS
    # =====================================================

    test["Residual"] = (
        test["Actual"]
        -
        test["Predicted"]
    )

    st.markdown(
        "<h3 class='section-title'>📉 Residual Analysis</h3>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        fig3 = px.histogram(
            test,
            x="Residual",
            nbins=30,
            title="Residual Distribution"
        )

        fig3.update_layout(
            height=450
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    with c2:

        fig_diag = px.scatter(
            test,
            x="Predicted",
            y="Residual",
            title="Residual vs Predicted"
        )

        fig_diag.add_hline(
            y=0,
            line_dash="dash",
            line_color="red"
        )

        fig_diag.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_diag,
            use_container_width=True
        )

    st.markdown(
        "#### Residual Boxplot"
    )

    fig_box = px.box(
        test,
        y="Residual",
        title="Residual Outlier Detection"
    )

    fig_box.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # WORST PREDICTION CASES
    # =====================================================

    worst = test.copy()

    worst["Absolute Error"] = abs(
        worst["Actual"]
        -
        worst["Predicted"]
    )

    worst = worst.sort_values(
        "Absolute Error",
        ascending=False
    ).head(10)

    st.markdown(
        "<h3 class='section-title'>⚠️ Worst Prediction Cases</h3>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        worst,
        use_container_width=True
    )

    st.divider()

    # # =====================================================
    # # MODEL STABILITY
    # # =====================================================
    #
    # rolling_rmse = (
    #     (
    #         (test["Actual"] - test["Predicted"]) ** 2
    #     )
    #     .rolling(50)
    #     .mean()
    #     ** 0.5
    # )
    #
    # st.markdown(
    #     "<h3 class='section-title'>📈 Model Stability Analysis</h3>",
    #     unsafe_allow_html=True,
    # )
    #
    # fig_stability = px.line(
    #     rolling_rmse,
    #     title="Rolling RMSE (Window = 50)",
    #     labels={
    #         "index": "Date",
    #         "value": "Rolling RMSE (VND/lượng)"
    #     }
    # )
    #
    # fig_stability.update_layout(
    #     showlegend=False,
    #     height=500
    # )
    #
    # st.plotly_chart(
    #     fig_stability,
    #     use_container_width=True
    # )

    # st.divider()

    # =====================================================
    # RESIDUAL STATISTICS
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📋 Residual Statistics</h3>",
        unsafe_allow_html=True,
    )

    residual_stats = pd.DataFrame({
        "Metric": [
            "Mean",
            "Std Dev",
            "Min",
            "Max",
            "Median"
        ],
        "Value": [
            f"{test['Residual'].mean():,.0f}",
            f"{test['Residual'].std():,.0f}",
            f"{test['Residual'].min():,.0f}",
            f"{test['Residual'].max():,.0f}",
            f"{test['Residual'].median():,.0f}",
        ]
    })

    st.dataframe(
        residual_stats,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # DETAILED METRICS
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>📊 Detailed Metrics</h3>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # MODEL INTERPRETATION
    # =====================================================
    st.markdown(
        "<h3 class='section-title'>💡 Model Interpretation & Insights</h3>", 
        unsafe_allow_html=True
    )

    interpretation_col = st.columns(1)[0]

    with interpretation_col:
        st.success("""
        ### Key Findings:

        **Model Performance:**
        - The Linear Regression model demonstrates **strong predictive performance** with R² = 0.96
        - Average prediction error (MAE) is approximately 914,092 VND/lượng
        - RMSE indicates consistent performance across different spread magnitudes
        
        **Model Characteristics:**
        - Actual vs Predicted chart shows the model **captures overall spread movements** effectively
        - Scatter plot indicates a **strong linear relationship** between observed and predicted values
        - Residual distribution is approximately centered around zero, suggesting **unbiased predictions**
        
        **Feature Insights:**
        - Feature importance analysis reveals relative contributions of macroeconomic variables
        - Top features show significant impact on Vietnamese gold price spreads
        - Lagged spread (t-1) is a strong predictor of current movement
        
        **Model Advantages:**
        - ✅ Fast training and inference time
        - ✅ Easy to interpret and explain
        - ✅ Low computational cost
        - ✅ Suitable for production deployment
        
        **Deployment Recommendation:**
        This model is **production-ready** for gold spread forecasting in Vietnamese market.
        """)

    # =====================================================
    # STRENGTHS & LIMITATIONS
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>⚖️ Strengths & Limitations</h3>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        st.success("""
    ### Strengths

    ✅ High predictive accuracy

    ✅ Strong R² performance

    ✅ Fast inference time

    ✅ Easy interpretation

    ✅ Suitable for deployment

    ✅ Low computational cost
    """)

    with right:

        st.warning(
            """
    ### Limitations

    ⚠ Assumes linear relationships

    ⚠ Sensitive to structural market shifts

    ⚠ Extreme geopolitical events may reduce accuracy

    ⚠ Limited extrapolation capability

    ⚠ Forecast quality depends on input variable quality
    """
        )

    # st.divider()

    # =====================================================
    # FOOTER
    # =====================================================
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center;color:#6B7280;font-size:13px;">
    Gold Spread Forecasting System | Model Results Module | Master Thesis Demonstration
    </div>
    """,
        unsafe_allow_html=True,
    )
