# Gold Price Prediction Dashboard

A Streamlit-based web application for predicting gold prices using trained machine learning models. The dashboard provides prediction results, model evaluation metrics, and interactive visualizations for performance analysis.

## Live Demo

The application is available on Streamlit Community Cloud:

**https://your-app-name.streamlit.app**

> Replace the URL above with your deployed Streamlit application.

## Features

* Predict gold prices using trained machine learning models.
* Compare actual and predicted values.
* Display evaluation metrics (RMSE, MAE, MAPE, and R² Score).
* Visualize model performance with interactive Plotly charts.
* Analyze feature coefficients and prediction results.

## Project Structure

```text
.
├── data/                   # Input datasets
├── models/                 # Trained model files
├── utils/                  # Utility functions
├── assets/                 # Static resources
├── streamlit_app.py        # Main Streamlit application
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

Install **uv** if you haven't already:

```bash
pip install uv
```

or

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install all project dependencies:

```bash
uv sync
```

## Run the Application

Start the Streamlit application:

```bash
uv run streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Deployment

The application can be deployed using **Streamlit Community Cloud**.

Documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud

## Technologies

* Python
* uv
* Streamlit
* Plotly
* Pandas
* NumPy
* Scikit-learn
* Joblib

## License

This project is intended for research and educational purposes.
