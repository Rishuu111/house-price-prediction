"""Generate house-price predictions from a saved model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(
    model_path: str | Path, input_path: str | Path, output_path: str | Path
) -> pd.DataFrame:
    """Predict prices for feature rows and save the results."""
    model = joblib.load(model_path)
    features = pd.read_csv(input_path)
    predictions = features.copy()
    predictions["predicted_price"] = model.predict(features)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/final_model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/predictions.csv")
    arguments = parser.parse_args()
    predict(arguments.model, arguments.input, arguments.output)
