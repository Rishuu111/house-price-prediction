"""Verify a housing dataset and write a read-only Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def find_dataset(data_directory: str | Path) -> Path | None:
    """Return the first supported dataset file, excluding hidden files."""
    candidates = sorted(
        path
        for path in Path(data_directory).rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return candidates[0] if candidates else None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a supported dataset without modifying the source file."""
    import pandas as pd

    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
    }
    return readers[path.suffix.lower()](path)


def suspicious_values(data: pd.DataFrame) -> list[str]:
    """Identify simple schema and value issues for human review."""
    findings: list[str] = []
    for column in data.select_dtypes(include="number"):
        negative_count = int((data[column] < 0).sum())
        if negative_count:
            findings.append(f"`{column}` contains {negative_count} negative value(s).")
        infinite_count = int(data[column].isin([float("inf"), float("-inf")]).sum())
        if infinite_count:
            findings.append(f"`{column}` contains {infinite_count} infinite value(s).")
    for column in data.select_dtypes(include=["object", "string"]):
        blank_count = int(data[column].fillna("").astype(str).str.strip().eq("").sum())
        if blank_count:
            findings.append(f"`{column}` contains {blank_count} blank string value(s).")
    return findings


def build_report(data_path: Path | None, data: pd.DataFrame | None) -> str:
    """Build a concise Markdown verification report."""
    lines = ["# Dataset Summary", "", "## Verification Result", ""]
    if data_path is None or data is None:
        lines.extend(
            [
                "**Status:** No supported dataset file was found.",
                "",
                "Searched `data/` recursively for `.csv`, `.xlsx`, `.xls`, and `.parquet` files.",
                "The original dataset was not modified. Add a dataset under `data/` and rerun:",
                "",
                "```bash",
                "python -m src.verify_dataset",
                "```",
                "",
                "The checks below are pending because there is no dataset to inspect.",
            ]
        )
        return "\n".join(lines) + "\n"

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    missing = data.isna().sum()
    missing = missing[missing > 0]
    duplicate_count = int(data.duplicated().sum())
    lines.extend(
        [
            f"**Status:** Dataset verified: `{data_path.as_posix()}`",
            "",
            "## Shape",
            "",
            f"- Rows: **{len(data):,}**",
            f"- Columns: **{len(data.columns):,}**",
            "",
            "## Columns",
            "",
            "- Names: " + ", ".join(f"`{column}`" for column in data.columns),
            "- Numerical: " + (", ".join(f"`{column}`" for column in numeric_columns) or "None"),
            "- Categorical: " + (", ".join(f"`{column}`" for column in categorical_columns) or "None"),
            "",
            "## Data Quality",
            "",
            f"- Missing values: {missing.to_dict() if not missing.empty else 'None'}",
            f"- Duplicate records: **{duplicate_count:,}**",
            "",
            "### Data Types",
            "",
            "| Column | Data type |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| `{column}` | `{dtype}` |" for column, dtype in data.dtypes.items())
    findings = suspicious_values(data)
    lines.extend(["", "### Suspicious or Inconsistent Values", ""])
    lines.extend(f"- {finding}" for finding in findings) if findings else lines.append("- None detected by the automated checks.")
    lines.extend(
        [
            "",
            "## Verification Notes",
            "",
            "- The source file was loaded read-only and was not rewritten.",
            "- Suspicious values are screening results and require domain review.",
        ]
    )
    return "\n".join(lines) + "\n"


def verify_dataset(data_directory: str | Path, report_path: str | Path) -> None:
    """Discover, inspect, and report on the available dataset."""
    dataset_path = find_dataset(data_directory)
    data = load_dataset(dataset_path) if dataset_path else None
    report = build_report(dataset_path, data)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", default="data")
    parser.add_argument("--report", default="dataset_summary.md")
    arguments = parser.parse_args()
    verify_dataset(arguments.data_directory, arguments.report)