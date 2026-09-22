"""Reusable, non-destructive cleaning utilities for housing data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "?"}
DEFAULT_UNNECESSARY_COLUMNS = {"id", "index", "pid"}
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a supported dataset without changing the source file."""
    source = Path(path)
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
    }
    try:
        reader = readers[source.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported dataset format. Use: {supported}") from error
    return reader(source)


def _normalise_column_name(name: object) -> str:
    """Make column names stable for downstream code and CLI options."""
    return "_".join(str(name).strip().casefold().split())


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.columns = [_normalise_column_name(column) for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(f"Column names become duplicated after normalisation: {duplicates}")
    return cleaned


def _normalise_category(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    normalised = " ".join(str(value).strip().casefold().split())
    return pd.NA if normalised in MISSING_TOKENS else normalised


def _format_numeric_series(series: pd.Series) -> pd.Series:
    """Convert common currency, thousands-separator, and parenthesis formats."""
    formatted = series.astype("string").str.strip()
    negative = formatted.str.startswith("(") & formatted.str.endswith(")")
    formatted = (
        formatted.str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    converted = pd.to_numeric(formatted, errors="coerce")
    return converted.where(~negative, -converted)


def _coerce_numeric_columns(data: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    cleaned = data.copy()
    for column in cleaned.columns:
        if column in required_columns or not pd.api.types.is_object_dtype(cleaned[column]):
            continue
        candidate = _format_numeric_series(cleaned[column])
        non_empty = cleaned[column].notna().sum()
        if non_empty and candidate.notna().sum() / non_empty >= 0.8:
            cleaned[column] = candidate
    for column in required_columns:
        if column in cleaned:
            cleaned[column] = _format_numeric_series(cleaned[column])
    return cleaned


def _clean_categories(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    for column in cleaned.select_dtypes(include=["object", "string", "category"]):
        cleaned[column] = cleaned[column].map(_normalise_category)
    return cleaned


def _impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median = cleaned[column].median()
            if pd.notna(median):
                cleaned[column] = cleaned[column].fillna(median)
        else:
            modes = cleaned[column].mode(dropna=True)
            cleaned[column] = cleaned[column].fillna(modes.iloc[0] if not modes.empty else "unknown")
    return cleaned


def clean_data(
    data: pd.DataFrame,
    target_column: str,
    drop_columns: Iterable[str] | None = None,
    non_negative_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned copy while leaving the input DataFrame untouched.

    The target is required to be numeric and strictly positive. Numeric feature
    columns are median-imputed; categorical feature columns use their mode or
    ``unknown`` when no mode exists. Pass domain-specific columns through
    ``non_negative_columns`` when negative values are invalid for those fields.
    """
    cleaned = _normalise_columns(data).drop_duplicates()
    cleaned = cleaned.dropna(axis=1, how="all")
    unnecessary = set(drop_columns or ()) | DEFAULT_UNNECESSARY_COLUMNS
    unnecessary |= {
        column for column in cleaned.columns if column.startswith("unnamed:")
    }
    cleaned = cleaned.drop(columns=unnecessary & set(cleaned.columns))
    target = _normalise_column_name(target_column)
    if target not in cleaned.columns:
        raise KeyError(f"Target column not found: {target}")

    cleaned = _coerce_numeric_columns(cleaned, [target])
    cleaned[target] = _format_numeric_series(cleaned[target])
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.dropna(subset=[target])
    cleaned = cleaned[cleaned[target] > 0]

    non_negative = {
        _normalise_column_name(column) for column in (non_negative_columns or ())
    }
    for column in non_negative & set(cleaned.columns):
        cleaned = cleaned[cleaned[column].isna() | (cleaned[column] >= 0)]

    cleaned = _clean_categories(cleaned)
    cleaned = _impute_missing_values(cleaned)
    return cleaned


def save_data(data: pd.DataFrame, path: str | Path) -> None:
    """Save a cleaned dataset separately, creating its parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def clean_file(
    input_path: str | Path,
    output_path: str | Path,
    target_column: str,
    drop_columns: Iterable[str] | None = None,
    non_negative_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load, clean, and save a dataset to a different path."""
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Input and output paths must be different.")
    cleaned = clean_data(
        load_data(source),
        target_column=target_column,
        drop_columns=drop_columns,
        non_negative_columns=non_negative_columns,
    )
    save_data(cleaned, destination)
    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--drop-column", action="append", default=[])
    parser.add_argument("--non-negative-column", action="append", default=[])
    arguments = parser.parse_args()
    clean_file(
        arguments.input,
        arguments.output,
        arguments.target,
        drop_columns=arguments.drop_column,
        non_negative_columns=arguments.non_negative_column,
    )
