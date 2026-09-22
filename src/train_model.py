"""Train and compare reproducible regression models for house prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2


def resolve_target_column(data: pd.DataFrame, target_column: str) -> str:
    """Resolve a target name across the project's normalized column convention."""
    if target_column in data.columns:
        return target_column
    normalized_target = "_".join(target_column.strip().casefold().split())
    matches = {
        column: "_".join(str(column).strip().casefold().split())
        for column in data.columns
    }
    for column, normalized_column in matches.items():
        if normalized_column == normalized_target:
            return column
    raise KeyError(f"Target column not found: {target_column}")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build one preprocessing definition shared by every candidate model."""
    numeric_features = features.select_dtypes(include="number").columns
    categorical_features = features.select_dtypes(exclude="number").columns
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Return the regression candidates with fixed seeds where supported."""
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }


def build_pipeline(
    features: pd.DataFrame,
    estimator: object,
) -> Pipeline:
    """Combine shared preprocessing with one selected regression estimator."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("regressor", estimator),
        ]
    )


def calculate_metrics(actual: pd.Series, predicted: object) -> dict[str, float]:
    """Calculate the requested regression metrics for held-out predictions."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def save_comparison_table(results: pd.DataFrame, output_directory: Path) -> None:
    """Save the ranked comparison as CSV and a report-friendly Markdown table."""
    output_directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_directory / "model_comparison.csv", index=False)
    (output_directory / "model_comparison.md").write_text(
        "# Regression Model Comparison\n\n"
        + results.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def train(
    input_path: str | Path,
    target_column: str,
    output_path: str | Path = "model",
    results_path: str | Path = "reports/model_comparison.json",
    predictions_path: str | Path = "reports/model_predictions",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Train, compare, and save all candidate models on one shared split."""
    data = pd.read_csv(input_path)
    target_column = resolve_target_column(data, target_column)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(data) < 2:
        raise ValueError("At least two rows are required for a train/test split.")

    features = data.drop(columns=[target_column])
    target = data[target_column]

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    model_directory = Path(output_path)
    model_directory.mkdir(parents=True, exist_ok=True)
    prediction_directory = Path(predictions_path)
    prediction_directory.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []
    model_paths: dict[str, str] = {}

    for model_name, estimator in build_models(random_state).items():
        pipeline = build_pipeline(features_train, estimator)
        pipeline.fit(features_train, target_train)
        predictions = pipeline.predict(features_test)
        metrics = calculate_metrics(target_test, predictions)
        model_path = model_directory / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        pd.DataFrame(
            {"actual_price": target_test, "predicted_price": predictions},
            index=target_test.index,
        ).to_csv(prediction_directory / f"{model_name}.csv", index=False)
        comparison_rows.append({"model": model_name, **metrics})
        model_paths[model_name] = str(model_path)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["rmse", "mae", "r2"], ascending=[True, True, False]
    ).reset_index(drop=True)
    comparison.insert(0, "rank", range(1, len(comparison) + 1))
    results_destination = Path(results_path)
    save_comparison_table(comparison, results_destination.parent)

    results = {
        "selection_rule": "Lowest RMSE, then lowest MAE, then highest R2",
        "target_column": target_column,
        "random_state": random_state,
        "test_size": test_size,
        "total_rows": int(len(data)),
        "training_rows": int(len(features_train)),
        "test_rows": int(len(features_test)),
        "feature_count": int(features.shape[1]),
        "models": comparison.to_dict(orient="records"),
        "best_model": str(comparison.iloc[0]["model"]),
        "model_paths": model_paths,
    }
    results_destination.parent.mkdir(parents=True, exist_ok=True)
    results_destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned/housing_cleaned.csv")
    parser.add_argument("--target", default="SalePrice")
    parser.add_argument("--output", default="model")
    parser.add_argument("--results", default="reports/model_comparison.json")
    parser.add_argument("--predictions", default="reports/model_predictions")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    arguments = parser.parse_args()
    metrics = train(
        arguments.input,
        arguments.target,
        arguments.output,
        results_path=arguments.results,
        predictions_path=arguments.predictions,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
    )
    print(json.dumps(metrics, indent=2))
