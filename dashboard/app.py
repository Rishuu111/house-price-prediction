"""Streamlit dashboard for the saved House Price Prediction model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_prices

MODEL_PATH = PROJECT_ROOT / "model" / "final_model.joblib"
METADATA_PATH = PROJECT_ROOT / "model" / "final_model_metadata.json"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .estimate { background: #e8f3f5; border-left: 5px solid #176b87; padding: 1rem 1.2rem; border-radius: 6px; }
    .muted { color: #5f6b72; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_metadata() -> dict[str, object]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def feature_schema(model: object) -> tuple[list[str], list[str], dict[str, float]]:
    """Read feature groups and numeric defaults from the fitted preprocessor."""
    preprocessor = model.named_steps["preprocessor"]
    numeric: list[str] = []
    categorical: list[str] = []
    numeric_defaults: dict[str, float] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric = list(columns)
            imputer = transformer
            for column, value in zip(numeric, imputer.statistics_):
                if pd.notna(value):
                    numeric_defaults[column] = float(value)
        elif name == "categorical":
            categorical = list(columns)
    return numeric, categorical, numeric_defaults


@st.cache_resource
def get_model() -> object:
    return load_model(MODEL_PATH)


st.title("House Price Prediction")
st.markdown(
    "Estimate a property's market price from the features used by the finalized model.",
)

with st.sidebar:
    st.header("Project Information")
    metadata = read_metadata()
    st.write("A prepared housing dataset is processed by the saved model pipeline.")
    if metadata:
        st.metric("Selected model", str(metadata.get("model_name", "Unavailable")))
        st.caption(f"Training rows: {metadata.get('training_rows', 'Unavailable')}")
        validation = metadata.get("validation_metrics", {})
        if isinstance(validation, dict):
            st.caption(f"Validation RMSE: {validation.get('rmse', 'Unavailable')}")
    st.divider()
    st.markdown(
        "**Important:** The result is an estimate, not a guaranteed sale price. "
        "Actual prices depend on market conditions, property details, and information not represented in the dataset."
    )

if not MODEL_PATH.exists():
    st.warning("The finalized model is not available yet. Run the training, comparison, and finalization steps first.")
    st.stop()

try:
    model = get_model()
    numeric_features, categorical_features, numeric_defaults = feature_schema(model)
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

st.subheader("Property Features")
st.caption("Enter the details available for the property. Unused extra columns are not required.")

with st.form("prediction_form"):
    values: dict[str, object] = {}
    columns = st.columns(2)
    for position, feature in enumerate(numeric_features):
        with columns[position % 2]:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=numeric_defaults.get(feature, 0.0),
                format="%.2f",
            )
    for position, feature in enumerate(categorical_features):
        with columns[position % 2]:
            values[feature] = st.text_input(feature.replace("_", " ").title())
    submitted = st.form_submit_button("Estimate House Price", type="primary", use_container_width=True)

if submitted:
    try:
        result = predict_prices(values, MODEL_PATH)
        price = float(result.iloc[0]["predicted_price"])
        st.markdown(
            f'<div class="estimate"><strong>Estimated house price</strong><br><span style="font-size: 2rem;">${price:,.2f}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("This estimate uses the same saved preprocessing and trained model used by the project pipeline.")
    except Exception as error:
        st.error(f"Prediction could not be generated: {error}")
