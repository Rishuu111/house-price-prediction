"""Evaluate a saved house-price model and create diagnostic report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_EXPLANATIONS = {
    "mae": "Average absolute prediction error in the same price units as the target. Lower is better.",
    "mse": "Average squared prediction error. Larger errors receive more weight. Lower is better.",
    "rmse": "Square root of MSE, in the same price units as the target. Lower is better.",
    "r2": "Proportion of target variance explained relative to predicting the test-set mean. Higher is better; it can be negative.",
}


def resolve_target_column(data: pd.DataFrame, target_column: str) -> str:
    """Resolve original or normalized target names."""
    if target_column in data.columns:
        return target_column
    normalized_target = "_".join(target_column.strip().casefold().split())
    for column in data.columns:
        normalized_column = "_".join(str(column).strip().casefold().split())
        if normalized_column == normalized_target:
            return column
    raise KeyError(f"Target column not found: {target_column}")


def calculate_metrics(actual: pd.Series, predicted: object) -> dict[str, float]:
    """Calculate the four requested regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def save_diagnostic_charts(
    actual: pd.Series,
    predicted: object,
    output_directory: str | Path,
) -> list[Path]:
    """Save actual-vs-predicted and residual diagnostic charts."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    predicted_series = pd.Series(predicted, index=actual.index, name="predicted_price")
    residuals = actual - predicted_series
    sns.set_theme(style="whitegrid", context="notebook")
    charts: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=actual, y=predicted_series, alpha=0.65, s=38, ax=axis, color="#176b87")
    minimum = min(actual.min(), predicted_series.min())
    maximum = max(actual.max(), predicted_series.max())
    axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#8b1e3f", label="Perfect prediction")
    axis.set_title("Actual vs Predicted House Prices")
    axis.set_xlabel("Actual price")
    axis.set_ylabel("Predicted price")
    axis.legend()
    figure.tight_layout()
    destination = output_path / "actual_vs_predicted_prices.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    charts.append(destination)

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x=predicted_series, y=residuals, alpha=0.65, s=38, ax=axis, color="#d97745")
    axis.axhline(0, linestyle="--", color="#176b87")
    axis.set_title("Residuals vs Predicted Prices")
    axis.set_xlabel("Predicted price")
    axis.set_ylabel("Residual (actual - predicted)")
    figure.tight_layout()
    destination = output_path / "residuals_vs_predicted_prices.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    charts.append(destination)

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.histplot(residuals, kde=True, ax=axis, color="#6f9eaa")
    axis.axvline(0, linestyle="--", color="#8b1e3f")
    axis.set_title("Residual Distribution")
    axis.set_xlabel("Residual (actual - predicted)")
    axis.set_ylabel("Homes")
    figure.tight_layout()
    destination = output_path / "residual_distribution.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    charts.append(destination)
    return charts


def save_report(
    metrics: dict[str, float],
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save metrics as JSON and a concise Markdown interpretation."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": str(model_path),
        "evaluation_data": str(input_path),
        "metrics": metrics,
        "metric_explanations": METRIC_EXPLANATIONS,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    lines = [
        "# Model Evaluation",
        "",
        f"- Model: `{model_path}`",
        f"- Evaluation data: `{input_path}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| {metric.upper()} | {value:,.4f} | {METRIC_EXPLANATIONS[metric]} |"
        for metric, value in metrics.items()
    )
    lines.extend(
        [
            "",
            "These metrics describe performance on the supplied evaluation data. They do not prove that the model will perform equally well on future homes or on a different population.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    input_path: str | Path,
    target_column: str,
    model_path: str | Path,
    chart_directory: str | Path = "reports/charts",
    report_path: str | Path = "reports/model_evaluation.json",
    predictions_path: str | Path = "reports/evaluation_predictions.csv",
) -> dict[str, float]:
    """Evaluate a saved model and persist metrics, predictions, and charts."""
    data = pd.read_csv(input_path)
    resolved_target = resolve_target_column(data, target_column)
    actual = pd.to_numeric(data[resolved_target], errors="raise")
    features = data.drop(columns=[resolved_target])
    model = joblib.load(model_path)
    predicted = model.predict(features)
    metrics = calculate_metrics(actual, predicted)

    predictions_destination = Path(predictions_path)
    predictions_destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "actual_price": actual,
            "predicted_price": predicted,
            "residual": actual - predicted,
        }
    ).to_csv(predictions_destination, index=False)
    save_diagnostic_charts(actual, predicted, chart_directory)
    save_report(metrics, model_path, input_path, report_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned/housing_cleaned.csv")
    parser.add_argument("--target", default="SalePrice")
    parser.add_argument("--model", default="model/random_forest.joblib")
    parser.add_argument("--charts", default="reports/charts")
    parser.add_argument("--report", default="reports/model_evaluation.json")
    parser.add_argument("--predictions", default="reports/evaluation_predictions.csv")
    arguments = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                arguments.input,
                arguments.target,
                arguments.model,
                arguments.charts,
                arguments.report,
                arguments.predictions,
            ),
            indent=2,
        )
    )
