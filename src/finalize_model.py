"""Select and retrain the measured best house-price model on all prepared data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .train_model import (
    RANDOM_STATE,
    build_models,
    build_pipeline,
    resolve_target_column,
)

SELECTION_RULE = "Lowest RMSE, then lowest MAE, then highest R2"


def select_model(comparison_path: str | Path) -> tuple[str, dict[str, object]]:
    """Select the comparison result with the best documented metric ranking."""
    source = Path(comparison_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {source}. Run src.train_model first."
        )
    comparison = json.loads(source.read_text(encoding="utf-8"))
    models = comparison.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Comparison results contain no model metrics: {source}")
    ranked_models = sorted(
        models,
        key=lambda result: (
            float(result["rmse"]),
            float(result["mae"]),
            -float(result["r2"]),
        ),
    )
    selected = ranked_models[0]
    model_name = str(selected["model"])
    if model_name not in build_models():
        raise ValueError(f"Unknown model in comparison results: {model_name}")
    return model_name, selected


def finalize_model(
    input_path: str | Path = "data/cleaned/housing_cleaned.csv",
    target_column: str = "SalePrice",
    comparison_path: str | Path = "reports/model_comparison.json",
    model_directory: str | Path = "model",
    metadata_path: str | Path = "model/final_model_metadata.json",
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Retrain the measured winner on all prepared rows and save reusable artifacts."""
    data = pd.read_csv(input_path)
    resolved_target = resolve_target_column(data, target_column)
    features = data.drop(columns=[resolved_target])
    target = data[resolved_target]
    model_name, validation_result = select_model(comparison_path)

    estimators = build_models(random_state)
    pipeline = build_pipeline(features, estimators[model_name])
    pipeline.fit(features, target)

    destination = Path(model_directory)
    destination.mkdir(parents=True, exist_ok=True)
    final_model_path = destination / "final_model.joblib"
    preprocessor_path = destination / "final_preprocessor.joblib"
    joblib.dump(pipeline, final_model_path)
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)

    metadata = {
        "model_name": model_name,
        "selection_rule": SELECTION_RULE,
        "validation_metrics": validation_result,
        "training_data": str(input_path),
        "target_column": resolved_target,
        "training_rows": int(len(data)),
        "feature_count": int(features.shape[1]),
        "random_state": random_state,
        "final_model_path": str(final_model_path),
        "preprocessor_path": str(preprocessor_path),
        "note": "Validation metrics come from the shared held-out comparison split; the final model was refit on all prepared rows.",
    }
    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned/housing_cleaned.csv")
    parser.add_argument("--target", default="SalePrice")
    parser.add_argument("--comparison", default="reports/model_comparison.json")
    parser.add_argument("--model-directory", default="model")
    parser.add_argument("--metadata", default="model/final_model_metadata.json")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    arguments = parser.parse_args()
    print(
        json.dumps(
            finalize_model(
                arguments.input,
                arguments.target,
                arguments.comparison,
                arguments.model_directory,
                arguments.metadata,
                arguments.random_state,
            ),
            indent=2,
        )
    )
