"""Generate house-price predictions with the saved final pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import pandas as pd


def as_feature_frame(
    features: Mapping[str, object] | Sequence[Mapping[str, object]] | pd.DataFrame,
) -> pd.DataFrame:
    """Convert supported feature input into a validated, non-empty DataFrame."""
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    else:
        frame = pd.DataFrame(list(features))
    if frame.empty:
        raise ValueError("At least one housing feature row is required.")
    if frame.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")
    return frame


def load_model(model_path: str | Path = "model/final_model.joblib") -> object:
    """Load the saved pipeline containing preprocessing and the trained model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)


def required_features(model: object) -> list[str]:
    """Read the feature schema learned by the saved preprocessing step."""
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    names = list(getattr(preprocessor, "feature_names_in_", []))
    if not names:
        raise ValueError("Saved model does not contain a training feature schema.")
    return names


def validate_features(features: pd.DataFrame, model: object) -> pd.DataFrame:
    """Check required columns and return them in the training order."""
    required = required_features(model)
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {', '.join(missing)}")
    return features.loc[:, required].copy()


def predict_prices(
    features: Mapping[str, object] | Sequence[Mapping[str, object]] | pd.DataFrame,
    model_path: str | Path = "model/final_model.joblib",
) -> pd.DataFrame:
    """Validate housing features and return predicted prices."""
    model = load_model(model_path)
    frame = as_feature_frame(features)
    validated = validate_features(frame, model)
    predictions = model.predict(validated)
    return pd.DataFrame({"predicted_price": predictions}, index=frame.index)


def format_predictions(predictions: pd.DataFrame) -> str:
    """Return readable currency-formatted prediction text."""
    return "\n".join(
        f"Row {index}: predicted house price = ${price:,.2f}"
        for index, price in predictions["predicted_price"].items()
    )
