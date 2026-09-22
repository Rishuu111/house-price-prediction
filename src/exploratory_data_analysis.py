"""Reusable exploratory summaries and visualizations for housing data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact schema and missing-value summary."""
    return pd.DataFrame(
        {
            "dtype": data.dtypes.astype(str),
            "missing_values": data.isna().sum(),
            "unique_values": data.nunique(dropna=False),
        }
    )


def save_numeric_distributions(
    data: pd.DataFrame, output_directory: str | Path
) -> None:
    """Save histograms for numeric columns."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    numeric_columns = data.select_dtypes(include="number").columns
    for column in numeric_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data[column].dropna(), kde=True, ax=axis)
        axis.set_title(f"Distribution of {column}")
        figure.tight_layout()
        figure.savefig(output_path / f"{column}_distribution.png", dpi=150)
        plt.close(figure)
