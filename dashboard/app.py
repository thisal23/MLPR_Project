import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Sri Lankan Paddy Production Forecasting",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Sri Lankan Paddy Production Forecasting")
st.caption("Historical analysis and machine learning-based production forecasting")

# EXPECTED PROJECT FILES (same structure as your original code)
HISTORICAL_PATH = "data/processed/paddy_cleaned.csv"
WALK_FORWARD_PATH = "experiments/walk_forward_model_comparison.csv"
FIXED_ORIGIN_PATH = "experiments/fixed_origin_model_comparison.csv"
FORECAST_PATH = "experiments/future_forecasts_all_models.csv"

PREDICTION_FILES = {
    "Walk-forward": {
        "Random Forest": (
            "experiments/random_forest/random_forest_predictions.csv",
            ["Predicted Production", "Predicted"],
        ),
        "XGBoost": (
            "experiments/xgboost/walk_forward_predictions.csv",
            ["Predicted", "Predicted Production"],
        ),
        "Hybrid ARIMA + RF": (
            "experiments/hybrid/walk_forward_predictions.csv",
            ["Hybrid_Predicted", "Predicted"],
        ),
    },
    "Fixed-origin": {
        "ARIMA": (
            "experiments/arima/fixed_origin_predictions.csv",
            ["Predicted", "Predicted Production"],
        ),
        "XGBoost": (
            "experiments/xgboost/fixed_origin_predictions.csv",
            ["Predicted", "Predicted Production"],
        ),
        "Hybrid ARIMA + RF": (
            "experiments/hybrid/fixed_origin_predictions.csv",
            ["Hybrid_Predicted", "Predicted"],
        ),
    },
}

IMPORTANCE_FILES = {
    "XGBoost": "experiments/xgboost/feature_importance.csv",
    "Random Forest": "experiments/random_forest/feature_importance.csv",
}

INDIVIDUAL_FORECAST_FILES = {
    "ARIMA": "experiments/arima/future_forecast.csv",
    "Random Forest": "experiments/random_forest/random_forest_future_forecast.csv",
}

# DATA LOADING
@st.cache_data(show_spinner=False)
def read_local_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def normalize_forecast(data, model_name=None):
    """Convert supported individual forecast formats to dashboard format."""
    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()
    result.columns = [str(c).strip() for c in result.columns]

    rename = {}
    if "Forecasted Production" in result.columns:
        rename["Forecasted Production"] = "Predicted_Production"
    if "Predicted Production" in result.columns:
        rename["Predicted Production"] = "Predicted_Production"
    if "Predicted" in result.columns and "Predicted_Production" not in result.columns:
        rename["Predicted"] = "Predicted_Production"
    if "Hybrid_Predicted" in result.columns and "Predicted_Production" not in result.columns:
        rename["Hybrid_Predicted"] = "Predicted_Production"

    result = result.rename(columns=rename)

    if model_name and "Model" not in result.columns:
        result["Model"] = model_name

    if "Forecast_Horizon" not in result.columns:
        result["Forecast_Horizon"] = np.arange(1, len(result) + 1)

    return result


def find_column(df, candidates):
    if df is None or df.empty:
        return None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def get_first_existing_csv(candidates):
    for path in candidates:
        data = read_local_csv(path)
        if not data.empty:
            return data, path
    return pd.DataFrame(), None


# Sidebar navigation. Data is loaded directly from the project folders.
st.sidebar.header("🌾 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Navigate",
    [
        "Project Overview",
        "Historical Data Analysis",
        "Model Performance",
        "Prediction Analysis",
        "Future Forecasting",
        "Model Insights",
    ],
)


def load_project_csv(path):
    """Load a CSV directly from its expected local project path."""
    return read_local_csv(path)


historical = load_project_csv(HISTORICAL_PATH)
walk_forward = load_project_csv(WALK_FORWARD_PATH)
fixed_origin = load_project_csv(FIXED_ORIGIN_PATH)
future_forecasts = load_project_csv(FORECAST_PATH)

# Combine the consolidated forecast file with the two individual forecasts,
# following the approach in the supplied source code.
forecast_parts = []
if not future_forecasts.empty:
    forecast_parts.append(normalize_forecast(future_forecasts))

for model_name, path in INDIVIDUAL_FORECAST_FILES.items():
    individual = load_project_csv(path)
    if not individual.empty:
        forecast_parts.append(normalize_forecast(individual, model_name))

if forecast_parts:
    future_forecasts = pd.concat(forecast_parts, ignore_index=True, sort=False)
    # Remove exact duplicate rows caused by loading the same file locally and by upload.
    future_forecasts = future_forecasts.drop_duplicates()

# Standardize historical numeric columns.
if not historical.empty:
    historical.columns = [str(c).strip() for c in historical.columns]
    if "Year" in historical.columns:
        historical["Year"] = pd.to_numeric(historical["Year"], errors="coerce")
    if "Production 000 Mt." in historical.columns:
        historical["Production 000 Mt."] = pd.to_numeric(
            historical["Production 000 Mt."], errors="coerce"
        )
    historical = historical.dropna(subset=["Year"]).sort_values("Year")

# PAGE 1: PROJECT OVERVIEW
if page == "Project Overview":
    st.header("Project Overview")
    st.write(
        """
        This project investigates machine learning and statistical approaches
        for forecasting annual paddy production in Sri Lanka.

        The project compares ARIMA, Random Forest, XGBoost, and a Hybrid
        ARIMA + Random Forest model.
        """
    )

    if not historical.empty:
        first_year = int(historical["Year"].min())
        last_year = int(historical["Year"].max())
        record_count = len(historical)

        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Records", f"{record_count:,}")
        col2.metric("Starting Year", first_year)
        col3.metric("Ending Year", last_year)

        st.subheader("Project Dataset")
        st.dataframe(historical, use_container_width=True, hide_index=True)

        production_col = find_column(historical, ["Production 000 Mt.", "Production"])
        if production_col:
            st.subheader("Historical Production Preview")
            preview_fig = px.line(
                historical,
                x="Year",
                y=production_col,
                markers=True,
                title="Annual Paddy Production in Sri Lanka",
            )
            preview_fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Production (000 Mt.)",
            )
            st.plotly_chart(preview_fig, use_container_width=True)
    else:
        st.warning("Historical dataset not found.")
        st.info(
            "Expected file: data/processed/paddy_cleaned.csv. "
            "Use the sidebar uploader to upload that CSV or a ZIP containing it."
        )

    st.subheader("Forecasting Models")
    st.markdown(
        """
        1. ARIMA
        2. Random Forest Regression
        3. XGBoost Regression
        4. Hybrid ARIMA + Random Forest
        """
    )

# PAGE 2: HISTORICAL DATA ANALYSIS
elif page == "Historical Data Analysis":
    st.header("Historical Paddy Production")

    production_col = find_column(historical, ["Production 000 Mt."])
    if historical.empty or "Year" not in historical.columns or production_col is None:
        st.warning("Historical dataset not found or required columns are missing.")
        st.caption("Required columns: Year and Production 000 Mt.")
    else:
        data = historical.dropna(subset=["Year", production_col]).copy()
        data[production_col] = pd.to_numeric(data[production_col], errors="coerce")
        data = data.dropna(subset=[production_col]).sort_values("Year")

        min_year = int(data["Year"].min())
        max_year = int(data["Year"].max())
        selected_years = st.slider(
            "Select year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )
        data = data[data["Year"].between(selected_years[0], selected_years[1])]

        fig = px.line(
            data,
            x="Year",
            y=production_col,
            markers=True,
            title="Annual Paddy Production in Sri Lanka",
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Production (000 Mt.)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Production", f"{data[production_col].mean():,.2f}")
        col2.metric("Maximum Production", f"{data[production_col].max():,.2f}")
        col3.metric("Minimum Production", f"{data[production_col].min():,.2f}")

        st.subheader("Production Distribution")
        histogram = px.histogram(
            data,
            x=production_col,
            nbins=15,
            title="Distribution of Paddy Production",
        )
        st.plotly_chart(histogram, use_container_width=True)

        historical_chart_data = data.copy()
        historical_chart_data["Production_Change"] = historical_chart_data[
            production_col
        ].diff()

        change_chart = px.bar(
            historical_chart_data.dropna(subset=["Production_Change"]),
            x="Year",
            y="Production_Change",
            title="Year-to-Year Production Change",
        )
        st.plotly_chart(change_chart, use_container_width=True)

        with st.expander("View historical data"):
            st.dataframe(data, use_container_width=True, hide_index=True)

        st.download_button(
            "Download historical data",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="historical_paddy_data.csv",
            mime="text/csv",
        )

# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    experiment = st.radio(
        "Select evaluation experiment",
        ["Walk-forward", "Fixed-origin"],
        horizontal=True,
    )

    comparison = walk_forward if experiment == "Walk-forward" else fixed_origin

    if comparison.empty:
        expected_path = WALK_FORWARD_PATH if experiment == "Walk-forward" else FIXED_ORIGIN_PATH
        st.warning(
            "Model comparison file is missing. "
            "Run the model evaluation script first or upload the CSV."
        )
        st.caption(f"Expected file: {expected_path}")
    else:
        st.subheader(f"{experiment} Results")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        metric = st.selectbox("Select evaluation metric", ["MAE", "RMSE", "MAPE"])
        model_col = find_column(comparison, ["Model"])
        metric_col = find_column(comparison, [metric])

        if model_col and metric_col:
            chart_data = comparison.copy()
            chart_data[metric_col] = pd.to_numeric(chart_data[metric_col], errors="coerce")
            chart_data = chart_data.dropna(subset=[metric_col])

            if not chart_data.empty:
                chart = px.bar(
                    chart_data,
                    x=model_col,
                    y=metric_col,
                    color=model_col,
                    title=f"Model Comparison - {metric}",
                    text_auto=".3f",
                )
                chart.update_layout(yaxis_title=metric)
                st.plotly_chart(chart, use_container_width=True)
                st.caption("Lower MAE, RMSE, and MAPE values indicate lower prediction error.")
            else:
                st.warning(f"No numeric values found in {metric}.")
        else:
            st.warning("The comparison file must contain Model and metric columns.")

        st.download_button(
            "Download comparison results",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name=f"{experiment.lower()}_model_comparison.csv",
            mime="text/csv",
        )

# PAGE 4: PREDICTION ANALYSIS
elif page == "Prediction Analysis":
    st.header("Actual vs Predicted Production")

    experiment = st.selectbox(
        "Select experiment",
        ["Walk-forward", "Fixed-origin"],
    )

    model_files = PREDICTION_FILES[experiment]
    fig = go.Figure()
    actual_added = False
    available_models = []

    for model_name, (path, prediction_candidates) in model_files.items():
        data = load_project_csv(path)

        if data.empty:
            st.warning(f"{model_name} prediction file not found: {path}")
            continue

        data.columns = [str(c).strip() for c in data.columns]
        actual_column = find_column(data, ["Actual", "Actual Production"])
        prediction_column = find_column(data, prediction_candidates)

        if "Year" not in data.columns:
            st.warning(f"Year column missing in {model_name}: {path}")
            continue

        if actual_column is None:
            st.warning(f"Actual production column missing in {model_name}.")
            continue

        if prediction_column is None:
            st.warning(
                f"Prediction column not found in {model_name}. "
                f"Expected one of: {', '.join(prediction_candidates)}"
            )
            continue

        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        data[actual_column] = pd.to_numeric(data[actual_column], errors="coerce")
        data[prediction_column] = pd.to_numeric(data[prediction_column], errors="coerce")
        data = data.dropna(subset=["Year"]).sort_values("Year")

        if not actual_added:
            fig.add_trace(
                go.Scatter(
                    x=data["Year"],
                    y=data[actual_column],
                    mode="lines+markers",
                    name="Actual Production",
                    line=dict(width=3),
                )
            )
            actual_added = True

        fig.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data[prediction_column],
                mode="lines+markers",
                name=model_name,
            )
        )

        available_models.append((model_name, data, actual_column, prediction_column))

    if available_models:
        fig.update_layout(
            title=f"{experiment}: Actual vs Predicted",
            xaxis_title="Year",
            yaxis_title="Production (000 Mt.)",
            hovermode="x unified",
            legend_title="Forecasting Models",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Available Models")
        st.write(", ".join([item[0] for item in available_models]))

        metric_rows = []
        for model_name, data, actual_column, prediction_column in available_models:
            actual = data[actual_column]
            predicted = data[prediction_column]
            valid = actual.notna() & predicted.notna()
            if valid.any():
                a = actual[valid].to_numpy()
                p = predicted[valid].to_numpy()
                mae = float(np.mean(np.abs(a - p)))
                rmse = float(np.sqrt(np.mean((a - p) ** 2)))
                nonzero = np.abs(a) > 1e-12
                mape = (
                    float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
                    if nonzero.any()
                    else np.nan
                )
                metric_rows.append(
                    {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}
                )

        if metric_rows:
            st.subheader("Metrics calculated from the displayed predictions")
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No prediction files are available for this experiment. "
            "Check the file paths or upload the prediction CSVs in the sidebar."
        )

# PAGE 5: FUTURE FORECASTING
elif page == "Future Forecasting":
    st.header("Future Paddy Production Forecasting")

    if future_forecasts.empty:
        st.warning(
            "Future forecast files not found. "
            "Generate the model forecasts first or upload forecast CSV files."
        )
        st.caption(
            f"Expected consolidated file: {FORECAST_PATH}. "
            "Individual ARIMA and Random Forest forecast files are also supported."
        )
    else:
        future_forecasts.columns = [str(c).strip() for c in future_forecasts.columns]
        model_col = find_column(future_forecasts, ["Model"])
        year_col = find_column(future_forecasts, ["Year"])
        prediction_col = find_column(
            future_forecasts,
            ["Predicted_Production", "Forecasted Production", "Predicted"],
        )
        horizon_col = find_column(future_forecasts, ["Forecast_Horizon"])

        if model_col is None or year_col is None or prediction_col is None:
            st.error(
                "Forecast data needs Model, Year, and Predicted_Production "
                "(or Forecasted Production) columns."
            )
            st.dataframe(future_forecasts, use_container_width=True)
        else:
            future_forecasts[year_col] = pd.to_numeric(
                future_forecasts[year_col], errors="coerce"
            )
            future_forecasts[prediction_col] = pd.to_numeric(
                future_forecasts[prediction_col], errors="coerce"
            )
            future_forecasts = future_forecasts.dropna(
                subset=[year_col, prediction_col, model_col]
            )

            model_names = sorted(future_forecasts[model_col].dropna().unique())
            selected_model = st.selectbox("Select forecasting model", model_names)

            model_data = future_forecasts[
                future_forecasts[model_col] == selected_model
            ].copy()

            if horizon_col:
                model_data[horizon_col] = pd.to_numeric(
                    model_data[horizon_col], errors="coerce"
                )
                max_horizon = int(model_data[horizon_col].max())
                options = [v for v in [1, 3, 5, 10] if v <= max_horizon]
                if not options:
                    options = [max_horizon]
                horizon = st.selectbox("Forecast horizon", options)
                model_data = model_data[model_data[horizon_col] <= horizon]
            else:
                st.caption(
                    "Forecast_Horizon column not present; showing all available forecast years."
                )

            model_data = model_data.sort_values(year_col)

            if not model_data.empty:
                fig = px.line(
                    model_data,
                    x=year_col,
                    y=prediction_col,
                    markers=True,
                    title=f"{selected_model}: Future Forecast",
                )
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Production (000 Mt.)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Forecast Values")
                st.dataframe(model_data, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download forecast values",
                    data=model_data.to_csv(index=False).encode("utf-8"),
                    file_name=f"{str(selected_model).lower().replace(' ', '_')}_forecast.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No forecast values available for this horizon.")

# PAGE 6: MODEL INSIGHTS
elif page == "Model Insights":
    st.header("Model Insights")

    selected_model = st.selectbox(
        "Select model",
        list(IMPORTANCE_FILES.keys()),
    )

    path = IMPORTANCE_FILES[selected_model]
    importance = load_project_csv(path)

    if not importance.empty:
        feature_col = find_column(importance, ["Feature"])
        importance_col = find_column(importance, ["Importance"])

        if feature_col is None or importance_col is None:
            st.error("Feature importance file must contain Feature and Importance columns.")
            st.dataframe(importance, use_container_width=True)
        else:
            importance[importance_col] = pd.to_numeric(
                importance[importance_col], errors="coerce"
            )
            importance = importance.dropna(subset=[importance_col]).sort_values(
                importance_col, ascending=True
            )

            fig = px.bar(
                importance,
                x=importance_col,
                y=feature_col,
                orientation="h",
                title=f"{selected_model} Feature Importance",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Importance Values")
            st.dataframe(importance.sort_values(importance_col, ascending=False),
                         use_container_width=True, hide_index=True)

            st.download_button(
                "Download feature importance",
                data=importance.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_model.lower().replace(' ', '_')}_feature_importance.csv",
                mime="text/csv",
            )
    else:
        st.info("Feature importance file is not available yet.")
        st.caption(f"Expected file: {path}")

    st.markdown(
        """
        **Interpretation note:** Feature importance describes how a
        tree-based model used its input features. It does not prove that
        a feature causes changes in paddy production.
        """
    )

# DATA STATUS FOOTER - EXPLICITLY SHOWS WHY A PAGE MAY BE EMPTY
with st.sidebar.expander("Data file status", expanded=False):
    status = {
        "Historical data": not historical.empty,
        "Walk-forward comparison": not walk_forward.empty,
        "Fixed-origin comparison": not fixed_origin.empty,
        "Future forecasts": not future_forecasts.empty,
    }
    for label, found in status.items():
        st.write(f"{'✅' if found else '❌'} {label}")
    st.caption("Prediction and feature-importance files are checked when their pages are opened.")
