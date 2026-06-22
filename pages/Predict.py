import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go


from tensorflow.keras.models import load_model

# ==========================
# MACHINE LEARNING MODELS
# ==========================

ML_MODELS = {
    "Linear Regression": "linear_pipeline.pkl",
    "Ridge Regression": "ridge_pipeline.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost (1).pkl",
    "LightGBM": "lightgbm.pkl",
    "CatBoost": "catboost.pkl",
}

DL_MODELS = {
    "LSTM": {
        "model": "lstm_model.keras",
        "feature_scaler": "lstm_feature_scaler.pkl",
        "target_scaler": "lstm_target_scaler.pkl",
    },
    "GRU": {
        "model": "gru_model.keras",
        "feature_scaler": "gru_feature_scaler.pkl",
        "target_scaler": "gru_target_scaler.pkl",
    },
    "BiLSTM": {
        "model": "bilstm_model.keras",
        "feature_scaler": "bilstm_feature_scaler.pkl",
        "target_scaler": "bilstm_target_scaler.pkl",
    },
}


ALL_MODELS = list(ML_MODELS.keys()) + list(DL_MODELS.keys())


def show_prediction_center():
    """
    Prediction Center Module - Vietnamese Gold Spread Forecasting System
    with Multi-day Forecasts and Confidence Intervals
    """

    # =====================================================
    # STYLING
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
        border-radius:12px;
        padding:18px;
        box-shadow:0 2px 8px rgba(13,90,156,0.08);
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

    st.markdown(
            """
    <style>

    /* Selectbox */
    div[data-baseweb="select"] > div {
        border: 3px solid #0D5A9C !important;
        border-radius: 16px !important;
        background-color: #F8FAFC !important;

        box-shadow:
            0 0 0 3px rgba(13,90,156,0.10),
            0 4px 12px rgba(13,90,156,0.25) !important;

        min-height: 60px !important;
    }

    /* Hover */
    div[data-baseweb="select"] > div:hover {
        border-color: #1565C0 !important;
    }

    /* Text */
    div[data-baseweb="select"] span {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* Label */
    label[data-testid="stWidgetLabel"] p {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #0D5A9C !important;
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
    <h1 style="margin:0;color:white;">🚀 Prediction Center</h1>
    <p style="margin-top:8px;color:#E5E7EB;">
    Vietnamese Gold Spread Forecasting with Confidence Intervals
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD MODEL & DATA
    # =====================================================

    def load_model_by_name(model_name):

        if model_name in ML_MODELS:

            return {"type": "ml", "model": joblib.load(f"models/{ML_MODELS[model_name]}")}

        cfg = DL_MODELS[model_name]

        return {
            "type": "dl",
            "model": load_model(f"models/{cfg['model']}"),
            "feature_scaler": joblib.load(f"models/{cfg['feature_scaler']}"),
            "target_scaler": joblib.load(f"models/{cfg['target_scaler']}"),
        }

    # =====================================================
    # COMMON PREDICTION FUNCTION
    # =====================================================

    def predict_spread(model_info, X):

        # Machine Learning Models
        if model_info["type"] == "ml":

            pred = model_info["model"].predict(X)

            return float(pred[0])

        # Deep Learning Models
        feature_scaler = model_info["feature_scaler"]
        target_scaler = model_info["target_scaler"]

        X_scaled = feature_scaler.transform(X)

        X_scaled = X_scaled.reshape(
            X_scaled.shape[0],
            1,
            X_scaled.shape[1]
        )

        pred_scaled = model_info["model"].predict(
            X_scaled,
            verbose=0
        )

        pred = target_scaler.inverse_transform(
            pred_scaled
        )

        return float(pred[0][0])

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    st.markdown("### 🤖 Select Forecasting Model")

    selected_model = st.selectbox(
        "Choose Model",
        ALL_MODELS
    )

    model_info = load_model_by_name(selected_model)

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = pd.read_csv("data/dataset.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Spread"] = pd.to_numeric(
        df["Spread"],
        errors="coerce"
    )

    FEATURES = [
        "VND/USD",
        "VNIndex",
        "Oil_Price",
        "DXY",
        "TNX",
        "GPR",
        "Bitcoin",
        "Spread_lag1",
    ]

    # =====================================================
    # PREDICTION MODE SELECTION
    # =====================================================

    st.markdown(
        "<h3 class='section-title'>⚙️ Forecasting Mode</h3>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Select Prediction Mode",
        ["Single Day Forecast", "Multi-Day Forecast", "Batch Forecast"],
        horizontal=True,
    )

    # =====================================================
    # SINGLE DAY PREDICTION
    # =====================================================
    if mode == "Single Day Forecast":

        st.markdown(
            "<h3 class='section-title'>📝 Input Parameters for Tomorrow (t+1)</h3>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            vndusd = st.number_input("VND/USD", value=26000.0, step=100.0)
            vnindex = st.number_input("VNIndex", value=1500.0, step=10.0)
            oil = st.number_input("Oil Price (USD)", value=70.0, step=1.0)
            dxy = st.number_input("DXY (Dollar Index)", value=100.0, step=0.1)

        with c2:
            tnx = st.number_input("TNX (10Y Treasury)", value=4.0, step=0.1)
            gpr = st.number_input("GPR (Geopolitical Risk)", value=150.0, step=5.0)
            Bitcoin = st.number_input(
                "Bitcoin Price (USD)", value=100000.0, step=1000.0
            )
            spread_lag1 = st.number_input(
                "Last Observed Spread (t)", value=15000000.0, step=100000.0
            )

        predict_btn = st.button(
            "🚀 Generate Single Day Forecast", use_container_width=True, type="primary"
        )

        if predict_btn:

            X = pd.DataFrame(
                [[vndusd, vnindex, oil, dxy, tnx, gpr, Bitcoin, spread_lag1]],
                columns=FEATURES,
            )

            pred = predict_spread(model_info, X)

            # =====================================================
            # CALCULATIONS
            # =====================================================
            diff = pred - spread_lag1
            pct_change = (
                diff / spread_lag1) * 100

            # Signal determination
            if pct_change > 5:
                signal = "🟢 Strong Increase"
                signal_color = "green"
            elif pct_change > 0:
                signal = "🟡 Slight Increase"
                signal_color = "yellow"
            elif pct_change > -5:
                signal = "🟠 Slight Decrease"
                signal_color = "orange"
            else:
                signal = "🔴 Strong Decrease"
                signal_color = "red"

            confidence = (
                "High"
                if abs(pct_change) < 2
                else "Medium" if abs(pct_change) < 5 else "Low"
            )

            # Confidence interval (±3%)
            lower = pred * 0.97
            upper = pred * 1.03
            upper_limit = pred * 1.05
            lower_limit = pred * 0.95

            st.divider()

            # =====================================================
            # KEY METRICS
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>📊 Forecast Results (t+1)</h3>",
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Previous Spread (t)",
                f"{spread_lag1:,.0f}",
                help="Most recent observed spread",
            )

            k2.metric(
                "Forecasted Spread (t+1)",
                f"{pred:,.0f}",
                help="Model prediction for tomorrow",
            )

            k3.metric(
                "Expected Change",
                f"{diff:,.0f}",
                delta=diff if diff > 0 else None,
                help="Absolute change vs previous day",
            )

            k4.metric(
                "Change % with(t-1)",
                f"{pct_change:.2f}%",
                help="Percentage change vs previous day",
            )

            st.divider()

            # =====================================================
            # GAUGE CHART
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>📈 Prediction Gauge</h3>",
                unsafe_allow_html=True,
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    delta={"reference": spread_lag1},
                    number={"valueformat": ",.0f"},
                    title={"text": "Forecasted Spread (VND/lượng)"},
                    gauge={
                        "axis": {"range": [spread_lag1 * 0.8, spread_lag1 * 1.2]},
                        "bar": {"color": "#0D5A9C"},
                        "steps": [
                            {
                                "range": [spread_lag1 * 0.8, spread_lag1 * 0.95],
                                "color": "#FFEBEE",
                            },
                            {
                                "range": [spread_lag1 * 0.95, spread_lag1 * 1.05],
                                "color": "#E8F5E9",
                            },
                            {
                                "range": [spread_lag1 * 1.05, spread_lag1 * 1.2],
                                "color": "#FFF3E0",
                            },
                        ],
                    },
                )
            )

            gauge.update_layout(height=450)
            st.plotly_chart(gauge, use_container_width=True)

            st.divider()

            # =====================================================
            # FORECAST CONFIDENCE & RANGE
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>🎯 Forecast Confidence & Range</h3>",
                unsafe_allow_html=True,
            )

            left_col, right_col = st.columns(2)

            with left_col:
                c1, c2, c3 = st.columns(3)
                c1.metric("Signal", signal)
                c2.metric("Confidence", confidence)
                c3.metric("Model", selected_model)

            with right_col:
                st.markdown(f"""
                **Forecast Range Analysis:**
                - 🔵 Lower Bound (±3%): {lower:,.0f}
                - 🟢 Point Estimate: {pred:,.0f}
                - 🔴 Upper Bound (±3%): {upper:,.0f}
                """)

            # =====================================================
            # RANGE VISUALIZATION
            # =====================================================
            st.markdown(
                "<b>Confidence Interval Visualization</b>", unsafe_allow_html=True
            )

            range_data = pd.DataFrame(
                {
                    "Scenario": ["Lower\nBound", "Point\nEstimate", "Upper\nBound"],
                    "Spread": [lower, pred, upper],
                    "Type": ["±3%", "Base", "±3%"],
                }
            )

            fig_range = px.bar(
                range_data,
                x="Scenario",
                y="Spread",
                color="Type",
                color_discrete_map={"±3%": "#E53935", "Base": "#0D5A9C"},
                title="Forecast Range with ±3% Confidence Interval",
                text="Spread",
            )

            fig_range.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_range.update_layout(
                height=400,
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                yaxis_title="Spread (VND/lượng)",
            )

            st.plotly_chart(fig_range, use_container_width=True)

            st.divider()

            # =====================================================
            # INTERPRETATION
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>💡 Forecast Interpretation</h3>",
                unsafe_allow_html=True,
            )

            interpretation_text = f"""
            **Forecast Summary for Tomorrow (t+1):**

            📌 **Predicted Spread:** {pred:,.0f} VND/lượng
            
            📌 **Change vs Today:** {diff:,.0f} VND/lượng ({pct_change:+.2f}%)
            
            📌 **Signal:** {signal}
            
            📌 **Confidence Level:** {confidence}
            
            **Forecast Range (±3% Confidence Interval):**
            - Lower Estimate: {lower:,.0f}
            - Point Forecast: {pred:,.0f}
            - Upper Estimate: {upper:,.0f}

            **Interpretation:**
            The model predicts the gold spread will be around {pred:,.0f} VND/lượng tomorrow.
            The expected change is {diff:,.0f} ({pct_change:+.2f}%) compared to today's spread of {spread_lag1:,.0f}.
            
            With a ±3% confidence interval, the spread should range between {lower:,.0f} and {upper:,.0f}.
            This forecast has {confidence.lower()} confidence based on the magnitude of expected change.
            """

            st.success(interpretation_text)

    # =====================================================
    # MULTI-DAY PREDICTION
    # =====================================================
    elif mode == "Multi-Day Forecast":

        st.markdown(
            "<h3 class='section-title'>📝 Multi-Day Forecast Configuration</h3>",
            unsafe_allow_html=True,
        )

        # Forecast horizon selection
        forecast_days = st.slider("Select Forecast Horizon", 1, 30, 7, step=1)

        st.markdown(
            "<b>Input Current Values (Used for forecasting)</b>", unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            vndusd = st.number_input(
                "VND/USD", value=26000.0, step=100.0, key="md_vndusd"
            )
            vnindex = st.number_input(
                "VNIndex", value=1500.0, step=10.0, key="md_vnindex"
            )
            oil = st.number_input("Oil Price (USD)", value=70.0, step=1.0, key="md_oil")
            dxy = st.number_input(
                "DXY (Dollar Index)", value=100.0, step=0.1, key="md_dxy"
            )

        with c2:
            tnx = st.number_input(
                "TNX (10Y Treasury)", value=4.0, step=0.1, key="md_tnx"
            )
            gpr = st.number_input(
                "GPR (Geopolitical Risk)", value=150.0, step=5.0, key="md_gpr"
            )
            Bitcoin = st.number_input(
                "Bitcoin Price (USD)", value=100000.0, step=1000.0, key="md_Bitcoin"
            )
            latest_spread = float(df["Spread"].iloc[-1])

        spread_current = st.number_input(
            "Current Spread (Today)",
            value=latest_spread,
            step=100000.0,
            key="md_spread",
        )

        forecast_btn = st.button(
            f"🚀 Generate {forecast_days}-Day Forecast",
            use_container_width=True,
            type="primary",
        )

        if forecast_btn:
            print("Generating multi-day forecast...")

            forecast_results = []
            current_spread = spread_current
            current_date = df["Date"].max()

            # Generate multi-day forecast
            for day in range(1, forecast_days + 1):
                X = pd.DataFrame(
                    [[vndusd, vnindex, oil, dxy, tnx, gpr, Bitcoin, current_spread]],
                    columns=FEATURES,
                )

                pred = predict_spread(model_info,X)

                # Calculate metrics
                change = pred - current_spread
                pct_change = (change / current_spread) * 100

                # Confidence interval
                ci_lower = pred * 0.97
                ci_upper = pred * 1.03

                forecast_results.append(
                    {
                        "Day": day,
                        "Date": (current_date + pd.Timedelta(days=day)).strftime(
                            "%Y-%m-%d"
                        ),
                        "Forecast": pred,
                        "Change": change,
                        "Change %": pct_change,
                        "CI Lower": ci_lower,
                        "CI Upper": ci_upper,
                    }
                )

                # Update for next iteration (use predicted spread as lagged value)
                current_spread = pred

            forecast_df = pd.DataFrame(forecast_results)
            print("First Forecast:", forecast_df["Forecast"].iloc[0])

            st.divider()

            # =====================================================
            # FORECAST TABLE
            # =====================================================
            st.markdown(
                f"<h3 class='section-title'>📋 {forecast_days}-Day Forecast Table</h3>",
                unsafe_allow_html=True,
            )

            display_df = forecast_df.copy()
            display_df["Forecast"] = display_df["Forecast"].apply(lambda x: f"{x:,.0f}")
            display_df["Change"] = display_df["Change"].apply(lambda x: f"{x:,.0f}")
            display_df["Change %"] = display_df["Change %"].apply(
                lambda x: f"{x:+.2f}%"
            )
            display_df["CI Lower"] = display_df["CI Lower"].apply(lambda x: f"{x:,.0f}")
            display_df["CI Upper"] = display_df["CI Upper"].apply(lambda x: f"{x:,.0f}")

            st.dataframe(display_df, use_container_width=True)

            st.divider()

            # =====================================================
            # MULTI-DAY CHART WITH CONFIDENCE BANDS
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>📈 Multi-Day Forecast with Confidence Intervals</h3>",
                unsafe_allow_html=True,
            )

            # Prepare data for visualization
            forecast_df["Date_obj"] = pd.to_datetime(forecast_df["Date"])

            fig_multi = go.Figure()

            # Add historical data (last 7 days)
            historical = df.tail(7).copy()
            historical["Date"] = pd.to_datetime(historical["Date"])

            # Lấy điểm cuối của dữ liệu lịch sử
            last_date = historical["Date"].iloc[-1]
            last_spread = historical["Spread"].iloc[-1]

            # Nối điểm cuối lịch sử với chuỗi dự báo
            forecast_x = [last_date] + forecast_df["Date_obj"].tolist()
            forecast_y = [last_spread] + forecast_df["Forecast"].tolist()

            fig_multi.add_trace(
                go.Scatter(
                    x=historical["Date"],
                    y=historical["Spread"],
                    mode="lines",
                    name="Historical Spread",
                    line=dict(color="#0D5A9C", width=3),
                    fillcolor="rgba(13, 90, 156, 0.1)",
                )
            )

            # Add forecast line
            fig_multi.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines+markers",
                    name="Forecasted Spread",
                    line=dict(
                        color="#E53935",
                        width=3,
                        dash="dash"
                    ),
                    marker=dict(size=6),
                )
            )

            # Add confidence interval as shaded area
            fig_multi.add_trace(
                go.Scatter(
                    x=forecast_df["Date_obj"].tolist()
                    + forecast_df["Date_obj"].tolist()[::-1],
                    y=forecast_df["CI Upper"].tolist()
                    + forecast_df["CI Lower"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(229, 57, 53, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="±3% Confidence Interval",
                    hoverinfo="skip",
                )
            )

            fig_multi.update_layout(
                title=f"{forecast_days}-Day Gold Spread Forecast with Confidence Bands",
                height=600,
                paper_bgcolor="white",
                plot_bgcolor="white",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Spread (VND/lượng)",
                legend=dict(x=0.01, y=0.99),
            )

            st.plotly_chart(fig_multi, use_container_width=True)

            st.divider()

            # =====================================================
            # FORECAST STATISTICS
            # =====================================================
            st.markdown(
                "<h3 class='section-title'>📊 Forecast Statistics</h3>",
                unsafe_allow_html=True,
            )

            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            stats_col1.metric("Starting Spread", f"{spread_current:,.0f}")

            stats_col2.metric(
                "Final Forecast",
                f"{forecast_df['Forecast'].iloc[-1]:,.0f}",
                delta=f"{forecast_df['Forecast'].iloc[-1] - spread_current:,.0f}",
            )

            stats_col3.metric(
                "Avg Daily Change", f"{forecast_df['Change'].mean():,.0f}"
            )

            stats_col4.metric(
                "Volatility (Std Dev)", f"{forecast_df['Change'].std():,.0f}"
            )

            st.divider()

            # =====================================================
            # DOWNLOAD FORECAST
            # =====================================================
            st.markdown("<b>📥 Export Forecast Results</b>", unsafe_allow_html=True)

            st.download_button(
                "📥 Download Full Forecast (CSV)",
                forecast_df.to_csv(index=False),
                file_name=f"gold_spread_forecast_{forecast_days}days.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # =====================================================
    # BATCH MODE
    # =====================================================
    else:

        st.markdown(
            "<h3 class='section-title'>📤 Batch Forecast Upload</h3>",
            unsafe_allow_html=True,
        )

        # Template download
        template = pd.DataFrame(columns=FEATURES)
        st.download_button(
            "📥 Download CSV Template",
            template.to_csv(index=False),
            file_name="forecast_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        uploaded = st.file_uploader(
            "Upload CSV or Excel File with Input Features", type=["csv", "xlsx"]
        )

        forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)

        if uploaded:

            df_upload = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith("csv")
                else pd.read_excel(uploaded)
            )

            k1, k2, k3 = st.columns(3)
            k1.metric("Rows", len(df_upload))
            k2.metric("Columns", len(df_upload.columns))
            k3.metric("Missing Values", int(df_upload.isna().sum().sum()))

            st.markdown("### Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)

            missing = [c for c in FEATURES if c not in df_upload.columns]

            if missing:
                st.error(f"❌ Missing Required Columns: {', '.join(missing)}")

            else:

                if st.button(
                    "🚀 Run Batch Forecast",
                    use_container_width=True,
                    type="primary"
                ):

                    all_results = []

                    for idx in range(len(df_upload)):

                        row = df_upload.iloc[idx]

                        current_spread = row["Spread_lag1"]

                        for day in range(1, forecast_days + 1):

                            X = pd.DataFrame(
                                [[
                                    row["VND/USD"],
                                    row["VNIndex"],
                                    row["Oil_Price"],
                                    row["DXY"],
                                    row["TNX"],
                                    row["GPR"],
                                    row["Bitcoin"],
                                    current_spread
                                ]],
                                columns=FEATURES
                            )

                            pred = predict_spread(
                                model_info,
                                X
                            )

                            all_results.append({
                                "Record": idx + 1,
                                "Day": day,
                                "Forecast": pred
                            })

                            current_spread = pred

                    result_df = pd.DataFrame(all_results)

                    fig = px.line(
                        result_df,
                        x="Day",
                        y="Forecast",
                        color="Record",
                        markers=True,
                        title="Batch Multi-Day Forecast"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric(
                        "Min Forecast", f"{df_upload['Predicted_Spread'].min():,.0f}"
                    )
                    k2.metric(
                        "Avg Forecast", f"{df_upload['Predicted_Spread'].mean():,.0f}"
                    )
                    k3.metric(
                        "Max Forecast", f"{df_upload['Predicted_Spread'].max():,.0f}"
                    )
                    k4.metric("Total Records", len(df_upload))

                    st.divider()

                    # Distribution chart
                    fig = px.histogram(
                        df_upload,
                        x="Predicted_Spread",
                        nbins=30,
                        title="Distribution of Batch Forecasts",
                        labels={"Predicted_Spread": "Predicted Spread"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Results table
                    st.markdown("### Batch Forecast Results")
                    st.dataframe(df_upload, use_container_width=True, height=500)

                    # Download results
                    st.download_button(
                        "📥 Download Results (CSV)",
                        df_upload.to_csv(index=False),
                        file_name="batch_prediction_results.csv",
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
    Gold Spread Forecasting System | Prediction Center Module | Master Thesis Demonstration
    </div>
    """,
        unsafe_allow_html=True,
    )
