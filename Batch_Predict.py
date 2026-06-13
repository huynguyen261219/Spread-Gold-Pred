import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Batch Prediction", page_icon="📂", layout="wide")

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
Batch Forecasting Center
</h1>

<p style="
margin-top:8px;
color:#E5E7EB;
">
Bulk Prediction of Vietnamese Gold Price Spread
</p>

</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/linear_regression.pkl")

scaler = joblib.load("models/std_scaler.pkl")

# =====================================================
# TEMPLATE
# =====================================================

st.markdown(
    "<h3 class='section-title'>Prediction Template</h3>", unsafe_allow_html=True
)

template = pd.DataFrame(
    columns=[
        "VND/USD",
        "VNIndex",
        "Oil_Price",
        "DXY",
        "TNX",
        "GPR",
        "bitcoin",
        "Spread_lag1",
    ]
)

csv_template = template.to_csv(index=False)

st.download_button(
    "📥 Download Template",
    csv_template,
    file_name="prediction_template.csv",
    mime="text/csv",
    use_container_width=True,
)

st.info("""
Required Variables:

VND/USD, VNIndex, Oil_Price, DXY,
TNX, GPR, bitcoin, Spread_lag1
""")

st.divider()

# =====================================================
# FILE UPLOAD
# =====================================================

st.markdown("<h3 class='section-title'>Upload Dataset</h3>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

# =====================================================
# PROCESS FILE
# =====================================================

if uploaded:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)

    else:
        df = pd.read_excel(uploaded)

    # ============================================
    # FILE INFO
    # ============================================

    st.markdown(
        "<h3 class='section-title'>File Information</h3>", unsafe_allow_html=True
    )

    k1, k2, k3 = st.columns(3)

    k1.metric("Rows", len(df))

    k2.metric("Columns", len(df.columns))

    k3.metric("Missing Values", int(df.isna().sum().sum()))

    # ============================================
    # PREVIEW
    # ============================================

    st.markdown(
        "<h3 class='section-title'>Dataset Preview</h3>", unsafe_allow_html=True
    )

    st.dataframe(df.head(10), use_container_width=True)

    # ============================================
    # CHECK COLUMNS
    # ============================================

    required_columns = [
        "VND/USD",
        "VNIndex",
        "Oil_Price",
        "DXY",
        "TNX",
        "GPR",
        "bitcoin",
        "Spread_lag1",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:

        st.error(f"Missing Columns: {', '.join(missing)}")

    else:

        st.success("All required columns detected.")

        # ============================================
        # PREDICT
        # ============================================

        if st.button("🚀 Run Batch Forecast", use_container_width=True):

            X_scaled = scaler.transform(df[required_columns])

            df["Prediction"] = model.predict(X_scaled)

            st.success("Batch prediction completed successfully.")

            # ====================================
            # RESULT KPI
            # ====================================

            st.markdown(
                "<h3 class='section-title'>Prediction Summary</h3>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Minimum", f"{df['Prediction'].min():,.0f}")

            c2.metric("Average", f"{df['Prediction'].mean():,.0f}")

            c3.metric("Maximum", f"{df['Prediction'].max():,.0f}")

            c4.metric("Records", len(df))

            # ====================================
            # RESULT TABLE
            # ====================================

            st.markdown(
                "<h3 class='section-title'>Forecast Results</h3>",
                unsafe_allow_html=True,
            )

            st.dataframe(df, use_container_width=True, height=500)

            # ====================================
            # DOWNLOAD
            # ====================================

            csv_result = df.to_csv(index=False)

            st.download_button(
                "📥 Download Forecast Result",
                csv_result,
                file_name="forecast_result.csv",
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
Batch Prediction Module |
Master Thesis Demonstration

</div>
""",
    unsafe_allow_html=True,
)
