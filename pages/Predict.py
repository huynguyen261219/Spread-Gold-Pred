import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


def show_prediction_center():

    st.set_page_config(page_title="Prediction Center", page_icon="🤖", layout="wide")

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
    Prediction Center
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

    model = joblib.load("models/linear_regression.pkl")

    scaler = joblib.load("models/std_scaler.pkl")

    FEATURES = [
        "VND/USD",
        "VNIndex",
        "Oil_Price",
        "DXY",
        "TNX",
        "GPR",
        "bitcoin",
        "Spread_lag1",
    ]

    mode = st.radio(
        "Prediction Mode", ["Single Prediction", "Batch Prediction"], horizontal=True
    )

    if mode == "Single Prediction":

        st.markdown(
            "<h3 class='section-title'>Input Parameters</h3>", unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            vndusd = st.number_input("VND/USD", value=26000.0)

            vnindex = st.number_input("VNIndex", value=1500.0)

            oil = st.number_input("Oil Price", value=70.0)

            dxy = st.number_input("DXY", value=100.0)

        with c2:

            tnx = st.number_input("TNX", value=4.0)

            gpr = st.number_input("GPR", value=150.0)

            bitcoin = st.number_input("Bitcoin", value=100000.0)

            spread_lag1 = st.number_input("Spread Lag 1", value=15000000.0)

        predict_btn = st.button(
            "🚀 Forecast Spread", use_container_width=True, type="primary"
        )

        if predict_btn:

            X = pd.DataFrame(
                [[vndusd, vnindex, oil, dxy, tnx, gpr, bitcoin, spread_lag1]],
                columns=FEATURES,
            )

            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]

            diff = pred - spread_lag1

            pct_change = (diff / spread_lag1) * 100

            if pct_change > 5:
                signal = "🟢 Strong Increase"

            elif pct_change > 0:
                signal = "🟡 Slight Increase"

            elif pct_change > -5:
                signal = "🟠 Slight Decrease"

            else:
                signal = "🔴 Strong Decrease"

            if abs(pct_change) < 2:
                confidence = "High"

            elif abs(pct_change) < 5:
                confidence = "Medium"

            else:
                confidence = "Low"

            st.divider()

            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Current Spread", f"{spread_lag1:,.0f}")

            k2.metric("Forecast Spread", f"{pred:,.0f}")

            k3.metric("Difference", f"{diff:,.0f}")

            k4.metric("Change (%)", f"{pct_change:.2f}%")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    number={"valueformat": ",.0f"},
                    title={"text": "Predicted Spread"},
                    gauge={"axis": {"range": [0, max(pred * 1.3, 20000000)]}},
                )
            )

            gauge.update_layout(height=400)

            st.plotly_chart(gauge, use_container_width=True)

            c1, c2, c3 = st.columns(3)

            c1.metric("Signal", signal)

            c2.metric("Confidence", confidence)

            c3.metric("Model", "Linear Regression")

            comparison = pd.DataFrame(
                {"Type": ["Previous", "Forecast"], "Spread": [spread_lag1, pred]}
            )

            fig_compare = go.Figure()

            fig_compare.add_bar(x=comparison["Type"], y=comparison["Spread"])

            fig_compare.update_layout(title="Spread Comparison", height=350)

            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("### Forecast Interpretation")

            st.success(f"""
    Forecast Spread: {pred:,.0f} VND/lượng

    Previous Spread: {spread_lag1:,.0f} VND/lượng

    Expected Change: {diff:,.0f} VND/lượng

    Percentage Change: {pct_change:.2f}%

    Signal: {signal}

    Confidence Level: {confidence}

    The model indicates the expected future
    difference between domestic gold prices
    and converted world gold prices.
    """)

    else:

        st.markdown(
            "<h3 class='section-title'>Batch Forecast</h3>", unsafe_allow_html=True
        )

        template = pd.DataFrame(columns=FEATURES)

        st.download_button(
            "📥 Download Template",
            template.to_csv(index=False),
            file_name="forecast_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        uploaded = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

        if uploaded:

            if uploaded.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded)
            else:
                df_upload = pd.read_excel(uploaded)

            k1, k2, k3 = st.columns(3)

            k1.metric("Rows", len(df_upload))

            k2.metric("Columns", len(df_upload.columns))

            k3.metric("Missing Values", int(df_upload.isna().sum().sum()))

            st.markdown("### Dataset Preview")

            st.dataframe(df_upload.head(10), use_container_width=True)

            missing = [c for c in FEATURES if c not in df_upload.columns]

            if missing:

                st.error(f"Missing Columns: {', '.join(missing)}")

            else:

                if st.button("🚀 Run Batch Forecast", use_container_width=True):

                    X_scaled = scaler.transform(df_upload[FEATURES])

                    df_upload["Predicted_Spread"] = model.predict(X_scaled)

                    k1, k2, k3, k4 = st.columns(4)

                    k1.metric("Minimum", f"{df_upload['Predicted_Spread'].min():,.0f}")

                    k2.metric("Average", f"{df_upload['Predicted_Spread'].mean():,.0f}")

                    k3.metric("Maximum", f"{df_upload['Predicted_Spread'].max():,.0f}")

                    k4.metric("Records", len(df_upload))

                    fig = px.histogram(
                        df_upload,
                        x="Predicted_Spread",
                        nbins=30,
                        title="Distribution of Predicted Spread",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df_upload, use_container_width=True, height=500)

                    st.info(f"""
    Total Forecasts: {len(df_upload)}

    Average Forecast:
    {df_upload['Predicted_Spread'].mean():,.0f}

    Minimum Forecast:
    {df_upload['Predicted_Spread'].min():,.0f}

    Maximum Forecast:
    {df_upload['Predicted_Spread'].max():,.0f}

    The model has generated spread forecasts
    for all uploaded observations.
    """)

                    st.download_button(
                        "📥 Download Results",
                        df_upload.to_csv(index=False),
                        file_name="prediction_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
