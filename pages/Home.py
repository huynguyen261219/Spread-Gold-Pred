import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show_home():
    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_excel("data/dataset.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df_filtered = df.copy()


    performance_df = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Ridge Regression",
                "BiLSTM",
                "LSTM",
                "GRU",
                "Decision Tree",
                "Random Forest",
                "LightGBM",
                "XGBoost",
                "CatBoost",
            ],
            "RMSE": [
                1485999,
                1485999,
                1507311,
                1513489,
                1869837,
                2303996,
                2448672,
                3040627,
                3347495,
                4098987,
            ],
            "MAE": [
                914092,
                914099,
                955210,
                971874,
                1422955,
                1287613,
                1350099,
                2234146,
                2519127,
                2734637,
            ],
            "MAPE": [
                9.66,
                9.66,
                10.88,
                11.39,
                22.81,
                11.05,
                11.08,
                28.26,
                32.14,
                21.69,
            ],
            "R2": [0.96, 0.96, 0.96, 0.96, 0.93, 0.90, 0.89, 0.83, 0.79, 0.68],
        }
    )
    # =====================================================
    # HEADER
    # =====================================================

    st.title("📈 Gold Spread Forecasting System")

    st.write(
        "Forecasting Vietnamese Gold Price Spread Using Statistical, Machine Learning and Deep Learning Models"
    )

    st.divider()

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    selected_model = st.selectbox("Select Forecasting Model", performance_df["Model"])

    selected_row = performance_df[performance_df["Model"] == selected_model].iloc[0]

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "RMSE",
            f"{selected_row['RMSE']:,.0f}"
        )

    with c2:
        st.metric(
            "MAE",
            f"{selected_row['MAE']:,.0f}"
        )

    with c3:
        st.metric(
            "MAPE",
            f"{selected_row['MAPE']:.2f}%"
        )

    with c4:
        st.metric(
            "R²",
            f"{selected_row['R2']:.2f}"
        )
    st.subheader("Model Description")

    if selected_model in [
        "Linear Regression",
        "Ridge Regression"
    ]:

        st.info(
            """
Category: Statistical Model

Scaler: StandardScaler

Characteristics:

• Fast training

• Easy interpretation

• Suitable for deployment
"""
        )

    elif selected_model in [
        "Decision Tree",
        "Random Forest"
    ]:

        st.info(
            """
Category: Machine Learning

Scaler: StandardScaler

Characteristics:

• Nonlinear learning

• Feature interaction

• Strong generalization
"""
        )

    elif selected_model in [
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]:

        st.info(
            """
Category: Boosting Model

Scaler: None

Characteristics:

• Ensemble learning

• Strong predictive power

• Handles complex relationships
"""
        )

    else:

        st.info(
            """
Category: Deep Learning

Scaler: MinMaxScaler

Characteristics:

• Sequential learning

• Time-series forecasting

• Captures nonlinear dynamics
"""
        )
    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("RMSE", f"{selected_row['RMSE']:,.0f}")

    c2.metric("MAE", f"{selected_row['MAE']:,.0f}")

    c3.metric("MAPE", f"{selected_row['MAPE']:.2f}%")

    c4.metric("R²", f"{selected_row['R2']:.2f}")

    st.divider()
    # =====================================================
    # RADAR CHART - MODEL COMPARISON
    # =====================================================

    import numpy as np
    import plotly.graph_objects as go

    metrics_cols = ["RMSE", "MAE", "MAPE", "R2"]

    # chuẩn hóa (min-max để vẽ radar)
    compare_df = performance_df.copy()

    best_model = performance_df.sort_values("RMSE").iloc[0]

    for col in metrics_cols:
        compare_df[col] = (
            (compare_df[col] - compare_df[col].min())
            / (compare_df[col].max() - compare_df[col].min())
        )

    selected_vals = compare_df[
        compare_df["Model"] == selected_model
    ][metrics_cols].values.flatten()

    best_vals = compare_df[
        compare_df["Model"] == best_model["Model"]
    ][metrics_cols].values.flatten()

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=selected_vals,
        theta=metrics_cols,
        fill='toself',
        name='Selected Model',
        line=dict(color="#E53935", width=3)
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=best_vals,
        theta=metrics_cols,
        fill='toself',
        name='Best Model',
        line=dict(color="#0D5A9C", width=3)
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        height=500,
        showlegend=True,
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # =====================================================
    # MODEL INSIGHT
    # =====================================================

    st.markdown("### 📌 Model Insight")

    if selected_model == best_model["Model"]:
        st.success(
            "Model này đang là mô hình tốt nhất theo RMSE. " "Phù hợp để deploy production."
        )
    else:
        gap = selected_row["RMSE"] - best_model["RMSE"]

        st.warning(f"""
    Model này chưa phải tốt nhất.

    Chênh lệch RMSE với best model: {gap:,.0f}

    → Có thể cân nhắc dùng ensemble hoặc tuning hyperparameter.
    """)

    # =====================================================
    # SCALING INFO
    # =====================================================

    st.markdown("### ⚙️ Preprocessing Method")

    if selected_model in ["Linear Regression", "Ridge Regression"]:
        st.info("StandardScaler được áp dụng cho dữ liệu đầu vào")

    elif selected_model in ["LSTM", "GRU", "BiLSTM"]:
        st.info("MinMaxScaler (0-1 normalization) được sử dụng cho chuỗi thời gian")

    elif selected_model in ["XGBoost", "LightGBM", "CatBoost"]:
        st.info("Không cần scaling - tree-based models")

    else:
        st.info("StandardScaler được sử dụng mặc định")
        # =====================================================
    # DATASET INFORMATION
    # =====================================================

    d1, d2, d3 = st.columns(3)

    d1.metric("Observations", f"{len(df):,}")

    d2.metric("Variables", len(df.columns) - 1)

    d3.metric("Period", f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}")

    st.divider()

    # best_model = performance_df.sort_values(
    #     "RMSE"
    # ).iloc[0]

    st.success(
        f"""
    🏆 Best Performing Model

    Model: {best_model['Model']}

    RMSE: {best_model['RMSE']:,.0f}

    MAE: {best_model['MAE']:,.0f}

    MAPE: {best_model['MAPE']:.2f}%

    R²: {best_model['R2']:.2f}
    """
        )
    # =====================================================
    # RMSE COMPARISON
    # =====================================================

    st.subheader("Model RMSE Comparison")
    colors = [
        "#E53935" if m == selected_model else "#0D5A9C" for m in performance_df["Model"]
    ]
    fig_rmse = go.Figure()

    fig_rmse.add_bar(
        x=performance_df["Model"],
        y=performance_df["RMSE"],
        marker_color=colors,
        text=[
            f"{v:,.0f}"
            for v in performance_df["RMSE"]
        ]
    )

    fig_rmse.update_traces(
        textposition="outside"
    )

    fig_rmse.update_layout(
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Model",
        yaxis_title="RMSE",
        showlegend=False
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    # =====================================================
    # MODEL RANKING
    # =====================================================

    st.subheader("Model Ranking")
    ranking_df = performance_df.sort_values(
    "RMSE"
    ).reset_index(drop=True)

    ranking_df.index += 1

    ranking_df.index.name = "Rank"
    st.dataframe(ranking_df, use_container_width=True, height=450)

    # =====================================================
    # RECENT DATA
    # =====================================================

    st.subheader("Recent Dataset Records")

    st.dataframe(
        df.tail(20),
        use_container_width=True,
        height=350
    )

    latest_date = df["Date"].max()
    time_option = st.selectbox("Select Time Range", ["7D", "1M", "3M", "6M", "All"])
    
    if time_option == "7D":
        df_filtered = df[df["Date"] >= latest_date - pd.Timedelta(days=7)]
    elif time_option == "1M":
        df_filtered = df[df["Date"] >= latest_date - pd.Timedelta(days=30)]
    elif time_option == "3M":
        df_filtered = df[df["Date"] >= latest_date - pd.Timedelta(days=90)]
    elif time_option == "6M":
        df_filtered = df[df["Date"] >= latest_date - pd.Timedelta(days=180)]
    else:
        df_filtered = df.copy()

    # ====================================================
    # =====================================================
    # SPREAD TREND
    # =====================================================
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["Spread"],
            mode="lines",
            name="Spread",
            line=dict(color="#0D5A9C", width=3)
        )
    )

    fig.update_layout(
        title=f"Historical Gold Spread Trend ({time_option})",
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Market Snapshot")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Current Spread",
        f"{df_filtered['Spread'].iloc[-1]:,.0f}",
        delta=None
    )

    k2.metric(
        "Average Spread",
        f"{df_filtered['Spread'].mean():,.0f}"
    )

    k3.metric(
        "Max Spread",
        f"{df_filtered['Spread'].max():,.0f}"
    )

    k4.metric(
        "Records",
        f"{len(df_filtered):,}"
    )
    st.markdown(
    f"""
    <div style="
        background-color:#0D5A9C;
        padding:10px;
        border-radius:10px;
        color:white;
        font-weight:bold;
        text-align:center;
    ">
        Period: {df_filtered['Date'].min().date()} → {df_filtered['Date'].max().date()}
    </div>
    """,
    unsafe_allow_html=True
)
    # =====================================================
    # RMSE COMPARISON
    # =====================================================

    rmse_sorted = performance_df.sort_values("RMSE")

    colors = [
        "#E53935"
        if m == selected_model
        else "#0D5A9C"
        for m in rmse_sorted["Model"]
    ]

    fig_rmse = go.Figure()

    fig_rmse.add_bar(
    x=rmse_sorted["Model"],
    y=rmse_sorted["RMSE"],
    marker_color=colors,
    text=[
        f"{v:,.0f}"
        for v in rmse_sorted["RMSE"]
    ],
    textposition="outside"
)

    fig_rmse.update_layout(title="RMSE Comparison Across Models", height=500)

    st.plotly_chart(fig_rmse, use_container_width=True)

    # =====================================================
    # RANKING
    # =====================================================

    st.subheader("Model Ranking")

    ranking_df = performance_df.sort_values(
    "RMSE"
    ).reset_index(drop=True)

    ranking_df.index += 1

    ranking_df.index.name = "Rank"

    st.dataframe(ranking_df, use_container_width=True)

    # =====================================================
    # RECENT DATA
    # =====================================================

    st.subheader("Recent Dataset")

    st.dataframe(df.tail(20), use_container_width=True)


show_home()
