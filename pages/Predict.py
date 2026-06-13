import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Gold Spread Prediction",
    page_icon="🔮",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
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
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
background:#0D5A9C;
padding:24px;
border-radius:10px;
margin-bottom:20px;
">

<h1 style="
margin:0;
color:white;
font-size:30px;
">
Gold Spread Prediction Center
</h1>

<p style="
margin-top:8px;
color:#E5E7EB;
">
Real-Time Gold Spread Forecasting Using Linear Regression
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "models/linear_regression.pkl"
)

scaler = joblib.load(
    "models/std_scaler.pkl"
)

FEATURES = [
    "VND/USD",
    "VNIndex",
    "Oil_Price",
    "DXY",
    "TNX",
    "GPR",
    "bitcoin",
    "Spread_lag1"
]

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "🔮 Manual Forecast",
    "📂 Batch Forecast"
])

# =====================================================
# MANUAL FORECAST
# =====================================================

with tab1:

    left_col, right_col = st.columns([2,1])

    with left_col:

        st.markdown(
            "<h3 class='section-title'>Input Parameters</h3>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            vndusd = st.number_input(
                "VND/USD",
                value=26000.0
            )

            vnindex = st.number_input(
                "VNIndex",
                value=1500.0
            )

            oil = st.number_input(
                "Oil Price",
                value=70.0
            )

            dxy = st.number_input(
                "DXY",
                value=100.0
            )

        with c2:

            tnx = st.number_input(
                "TNX",
                value=4.0
            )

            gpr = st.number_input(
                "GPR",
                value=150.0
            )

            bitcoin = st.number_input(
                "Bitcoin",
                value=100000.0
            )

            spread_lag1 = st.number_input(
                "Spread(t-1)",
                value=15000000.0
            )

        predict_btn = st.button(
            "🚀 Forecast Spread",
            use_container_width=True,
            type="primary"
        )

    with right_col:

        st.markdown(
            "<h3 class='section-title'>Forecast Center</h3>",
            unsafe_allow_html=True
        )

        st.info(
            "Enter macroeconomic variables and run forecasting."
        )

    if predict_btn:

        X = pd.DataFrame(
            [[
                vndusd,
                vnindex,
                oil,
                dxy,
                tnx,
                gpr,
                bitcoin,
                spread_lag1
            ]],
            columns=FEATURES
        )

        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]

        diff = pred - spread_lag1

        pct_change = diff / spread_lag1 * 100

        if diff > 0:
            signal = "🟢 INCREASE"
        elif diff < 0:
            signal = "🔴 DECREASE"
        else:
            signal = "🟡 STABLE"

        if abs(pct_change) < 2:
            confidence = "High"

        elif abs(pct_change) < 5:
            confidence = "Medium"

        else:
            confidence = "Low"

        st.divider()

        st.markdown(
            "<h3 class='section-title'>Forecast Result</h3>",
            unsafe_allow_html=True
        )

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Current Spread",
            f"{spread_lag1:,.0f}"
        )

        k2.metric(
            "Forecast Spread",
            f"{pred:,.0f}"
        )

        k3.metric(
            "Difference",
            f"{diff:,.0f}"
        )

        chart_col, info_col = st.columns([2,1])

        with chart_col:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    number={
                        "valueformat": ",.0f"
                    },
                    title={
                        "text":"Predicted Spread"
                    },
                    gauge={
                        "axis":{
                            "range":[0,max(pred*1.3,20000000)]
                        }
                    }
                )
            )

            gauge.update_layout(
                height=380,
                paper_bgcolor="white"
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        with info_col:

            st.metric(
                "Signal",
                signal
            )

            st.metric(
                "Confidence",
                confidence
            )

            st.metric(
                "Change (%)",
                f"{pct_change:.2f}%"
            )

            st.metric(
                "Model",
                "Linear Regression"
            )

        comparison = pd.DataFrame(
            {
                "Type":[
                    "Previous",
                    "Forecast"
                ],
                "Spread":[
                    spread_lag1,
                    pred
                ]
            }
        )

        fig_compare = go.Figure()

        fig_compare.add_bar(
            x=comparison["Type"],
            y=comparison["Spread"]
        )

        fig_compare.update_layout(
            title="Spread Comparison",
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

        st.info(
            f"""
Forecast Summary

Predicted Spread: {pred:,.0f} VND/lượng

Previous Spread: {spread_lag1:,.0f} VND/lượng

Expected Change: {diff:,.0f} VND/lượng

Signal: {signal}

Confidence Level: {confidence}
"""
        )

# =====================================================
# BATCH FORECAST
# =====================================================

with tab2:

    st.markdown(
        "<h3 class='section-title'>Batch Forecast</h3>",
        unsafe_allow_html=True
    )

    template = pd.DataFrame(
        columns=FEATURES
    )

    st.download_button(
        "📥 Download Template",
        template.to_csv(index=False),
        file_name="forecast_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)

        else:
            df_upload = pd.read_excel(uploaded_file)

        st.markdown("### Dataset Preview")

        st.dataframe(
            df_upload.head(),
            use_container_width=True
        )

        if st.button(
            "🚀 Run Batch Forecast",
            use_container_width=True
        ):

            X_scaled = scaler.transform(
                df_upload[FEATURES]
            )

            df_upload["Predicted_Spread"] = model.predict(
                X_scaled
            )

            st.success(
                f"{len(df_upload)} forecasts generated successfully."
            )

            st.dataframe(
                df_upload,
                use_container_width=True
            )

            st.download_button(
                "📥 Download Results",
                df_upload.to_csv(index=False),
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )