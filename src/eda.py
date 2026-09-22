"""Create report-ready exploratory data analysis charts for housing data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_cleaning import load_data

DEFAULT_TARGET_COLUMN = "saleprice"
DEFAULT_OUTPUT_DIRECTORY = Path("reports/charts")
MAX_CATEGORICAL_LEVELS = 15
MAX_NUMERICAL_FEATURES = 12


def _normalise_name(name: str) -> str:
    """Match the column naming convention used by the cleaning pipeline."""
    return "_".join(str(name).strip().casefold().split())


def _safe_filename(value: str) -> str:
    """Convert a column name into a portable chart filename."""
    return re.sub(r"[^a-z0-9_-]+", "_", value.casefold()).strip("_")


def _save_figure(figure: plt.Figure, output_directory: Path, filename: str) -> Path:
    """Apply consistent report styling and save a closed figure."""
    figure.tight_layout()
    destination = output_directory / filename
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def _numeric_columns(data: pd.DataFrame, target_column: str) -> list[str]:
    return data.select_dtypes(include="number").columns.difference(
        [target_column], sort=False
    ).tolist()


def plot_numeric_distributions(
    data: pd.DataFrame, output_directory: Path
) -> Path | None:
    """Save a grid of distributions for every numerical variable."""
    columns = data.select_dtypes(include="number").columns.tolist()
    if not columns:
        return None
    columns_per_row = 3
    rows = (len(columns) + columns_per_row - 1) // columns_per_row
    figure, axes = plt.subplots(
        rows,
        columns_per_row,
        figsize=(15, max(4, rows * 3.5)),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis, column in zip(axes_flat, columns):
        sns.histplot(data[column].dropna(), kde=True, ax=axis, color="#176b87")
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("")
    for axis in axes_flat[len(columns) :]:
        axis.set_visible(False)
    figure.suptitle("Numerical Variable Distributions", fontsize=16, y=1.02)
    return _save_figure(figure, output_directory, "numerical_distributions.png")


def plot_categorical_frequencies(
    data: pd.DataFrame, output_directory: Path
) -> list[Path]:
    """Save a readable top-category frequency chart for each categorical column."""
    destinations: list[Path] = []
    columns = data.select_dtypes(exclude="number").columns
    for column in columns:
        frequencies = data[column].fillna("missing").value_counts().head(MAX_CATEGORICAL_LEVELS)
        if frequencies.empty:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x=frequencies.values,
            y=frequencies.index.astype(str),
            hue=frequencies.index.astype(str),
            legend=False,
            ax=axis,
            palette="crest",
        )
        axis.set_title(f"Most Frequent Values: {column.replace('_', ' ').title()}")
        axis.set_xlabel("Record count")
        axis.set_ylabel("")
        destinations.append(
            _save_figure(
                figure,
                output_directory,
                f"categorical_{_safe_filename(column)}_frequency.png",
            )
        )
    return destinations


def plot_price_distribution(
    data: pd.DataFrame, target_column: str, output_directory: Path
) -> Path:
    """Save the target-price distribution with median and mean markers."""
    target = pd.to_numeric(data[target_column], errors="coerce").dropna()
    if target.empty:
        raise ValueError(f"Target column has no numeric values: {target_column}")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(target, kde=True, ax=axis, color="#d97745")
    axis.axvline(target.mean(), color="#8b1e3f", linestyle="--", label="Mean")
    axis.axvline(target.median(), color="#176b87", linestyle="-.", label="Median")
    axis.set_title("House Price Distribution")
    axis.set_xlabel(target_column.replace("_", " ").title())
    axis.set_ylabel("Homes")
    axis.legend()
    return _save_figure(figure, output_directory, "house_price_distribution.png")


def plot_feature_price_relationships(
    data: pd.DataFrame, target_column: str, output_directory: Path
) -> Path | None:
    """Save scatterplots for the numerical features most correlated with price."""
    numeric = data.select_dtypes(include="number")
    if target_column not in numeric:
        return None
    correlations = numeric.corr(numeric_only=True)[target_column].drop(target_column).abs()
    features = correlations.nlargest(MAX_NUMERICAL_FEATURES).index.tolist()
    if not features:
        return None
    columns_per_row = 3
    rows = (len(features) + columns_per_row - 1) // columns_per_row
    figure, axes = plt.subplots(
        rows, columns_per_row, figsize=(15, max(4, rows * 4)), squeeze=False
    )
    for axis, feature in zip(axes.ravel(), features):
        sns.scatterplot(data=data, x=feature, y=target_column, alpha=0.55, s=28, ax=axis)
        axis.set_title(f"{feature.replace('_', ' ').title()} vs Price")
        axis.set_xlabel(feature.replace("_", " ").title())
        axis.set_ylabel("Price")
    for axis in axes.ravel()[len(features) :]:
        axis.set_visible(False)
    figure.suptitle("Important Numerical Features and House Price", fontsize=16, y=1.02)
    return _save_figure(figure, output_directory, "feature_price_relationships.png")


def _outlier_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate IQR-based outlier counts for numerical columns."""
    rows = []
    for column in data.select_dtypes(include="number"):
        values = data[column].dropna()
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        count = int(((values < lower_bound) | (values > upper_bound)).sum())
        rows.append(
            {
                "column": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": count,
                "outlier_percent": count / len(values) * 100 if len(values) else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_outliers(data: pd.DataFrame, output_directory: Path) -> tuple[Path | None, Path]:
    """Save numerical boxplots and a machine-readable IQR summary."""
    numeric_columns = data.select_dtypes(include="number").columns[:MAX_NUMERICAL_FEATURES]
    summary = _outlier_summary(data)
    summary_path = output_directory / "outlier_summary.csv"
    summary.to_csv(summary_path, index=False)
    if len(numeric_columns) == 0:
        return None, summary_path
    figure, axis = plt.subplots(figsize=(12, max(5, len(numeric_columns) * 0.4)))
    sns.boxplot(data=data[numeric_columns], orient="h", ax=axis, color="#6f9eaa")
    axis.set_title("Numerical Variable Outlier Scan (IQR Rule)")
    axis.set_xlabel("Value")
    axis.set_ylabel("")
    return _save_figure(figure, output_directory, "numerical_outliers.png"), summary_path


def plot_correlation_heatmap(
    data: pd.DataFrame, output_directory: Path
) -> Path | None:
    """Save a heatmap of correlations among numerical variables."""
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    figure, axis = plt.subplots(figsize=(max(8, numeric.shape[1] * 0.8), 7))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="vlag", center=0, annot=False, ax=axis)
    axis.set_title("Numerical Feature Correlations")
    return _save_figure(figure, output_directory, "numerical_correlation_heatmap.png")


def run_eda(
    data: pd.DataFrame, target_column: str, output_directory: str | Path
) -> list[Path]:
    """Generate all EDA artifacts and return their output paths."""
    target = _normalise_name(target_column)
    if target not in data.columns:
        raise KeyError(f"Target column not found: {target}")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    artifacts: list[Path] = []
    for artifact in (
        plot_numeric_distributions(data, output_path),
        plot_price_distribution(data, target, output_path),
        plot_feature_price_relationships(data, target, output_path),
        plot_correlation_heatmap(data, output_path),
    ):
        if artifact is not None:
            artifacts.append(artifact)
    artifacts.extend(plot_categorical_frequencies(data, output_path))
    outlier_chart, outlier_summary = plot_outliers(data, output_path)
    if outlier_chart is not None:
        artifacts.append(outlier_chart)
    artifacts.append(outlier_summary)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned/housing_cleaned.csv")
    parser.add_argument("--target", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIRECTORY))
    arguments = parser.parse_args()
    input_path = Path(arguments.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}. Run the cleaning pipeline first."
        )
    artifacts = run_eda(load_data(input_path), arguments.target, arguments.output)
    print(f"Created {len(artifacts)} EDA artifact(s) in {arguments.output}")


if __name__ == "__main__":
    main()