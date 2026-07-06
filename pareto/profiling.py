"""Deterministic column profiling.

Raw data is never sent to an LLM. We only build a per-column summary payload
with dtype, missingness, examples, and candidate roles.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import ParserError


def load_raw_file(source: str | Path | Any) -> pd.DataFrame:
    """Load CSV / Excel / Stata (.dta) data into a dataframe.

    `source` may be a local path or a file-like object such as Streamlit's
    UploadedFile. The latter has a `.name` but is not saved to the working
    directory, so we must pass the object itself to pandas.
    """
    is_path = isinstance(source, str | Path)
    suffix = Path(source if is_path else source.name).suffix.lower()
    file_obj = Path(source) if is_path else source

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if suffix in (".csv", ".tsv"):
        return _read_delimited(file_obj, suffix=suffix)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_obj)
    if suffix == ".dta":
        return pd.read_stata(file_obj)
    raise ValueError(f"Unsupported file extension: {suffix} (csv/tsv/xlsx/dta)")


def _read_delimited(file_obj: Any, *, suffix: str) -> pd.DataFrame:
    """Read delimited text with delimiter sniffing and fail-loud diagnostics."""
    sample = _read_text_sample(file_obj)
    separators = ["\t"] if suffix == ".tsv" else _candidate_separators(file_obj)
    attempts: list[str] = []
    best_single_col: pd.DataFrame | None = None
    best_table: pd.DataFrame | None = None

    for sep in separators:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        try:
            df = pd.read_csv(
                file_obj,
                sep=sep,
                engine="python",
                skiprows=_detect_header_skiprows(sample, sep),
            )
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            attempts.append(f"{sep!r}: {exc}")
            continue

        if df.shape[1] > 1 and (best_table is None or df.shape[1] > best_table.shape[1]):
            best_table = df
        best_single_col = df

    if best_table is not None:
        return best_table
    if best_single_col is not None:
        return best_single_col

    detail = " | ".join(attempts[:4])
    raise ValueError(
        "Delimited file could not be parsed consistently. "
        "Check whether the file has metadata rows before the header, mixed delimiters, "
        f"or an unsupported encoding. Parser attempts: {detail}"
    )


def _candidate_separators(file_obj: Any) -> list[str]:
    candidates = [",", ";", "\t", "|"]
    sample = _read_text_sample(file_obj)
    if not sample:
        return candidates

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return candidates

    sniffed = dialect.delimiter
    return [sniffed, *[sep for sep in candidates if sep != sniffed]]


def _detect_header_skiprows(sample: str, sep: str) -> int:
    """Skip metadata rows before the widest delimited header row."""
    rows = list(csv.reader(sample.splitlines(), delimiter=sep))
    scored: list[tuple[int, int]] = []
    for idx, row in enumerate(rows[:25]):
        non_empty = [cell for cell in row if cell.strip()]
        if len(non_empty) > 1:
            scored.append((len(non_empty), idx))
    if not scored:
        return 0
    _, header_idx = max(scored, key=lambda item: (item[0], -item[1]))
    return header_idx


def _read_text_sample(file_obj: Any, size: int = 8192) -> str:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if isinstance(file_obj, Path):
        raw = file_obj.read_bytes()[:size]
    else:
        raw = file_obj.read(size)

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8-sig", errors="replace")


def _guess_join_keys(df: pd.DataFrame) -> list[str]:
    """Guess merge-key candidates from names and medium cardinality."""
    candidates: list[str] = []
    n = len(df)
    for col in df.columns:
        name = str(col).lower()
        looks_like_id = any(tok in name for tok in ("id", "code", "kod", "key", "fips", "no"))
        nunique = df[col].nunique(dropna=True)
        reasonable_cardinality = 1 < nunique < n
        if looks_like_id or reasonable_cardinality:
            candidates.append(str(col))
    return candidates


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Build the structured summary profile used by the cleaning agent."""
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": {},
        "potential_join_keys": _guess_join_keys(df),
        "duplicate_row_count": int(df.duplicated().sum()),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "pct_missing": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info["stats"] = {
                "min": float(desc.get("min", float("nan"))),
                "max": float(desc.get("max", float("nan"))),
                "mean": float(desc.get("mean", float("nan"))),
                "std": float(desc.get("std", float("nan"))),
            }
        else:
            top_values = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            sample = series.dropna().astype(str).head(20).tolist()
            col_info["looks_like_date"] = any(
                any(sep in v for sep in ("-", "/", ".")) and any(ch.isdigit() for ch in v)
                for v in sample
            )
        profile["columns"][str(col)] = col_info

    return profile
