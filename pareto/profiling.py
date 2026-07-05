"""Deterministik kolon profilleme.

Prototipteki `p1_cleaning/ingestion.py`'den migre edildi. Doğru kurulmuş çekirdek
ilke korunuyor: ham dataframe'i LLM'e ASLA göndermeyiz — yalnız kolon başına
özet payload (dtype/eksik/örnek/aday-rol). Bu hem gizlilik (yüzey-min)
hem token maliyeti içindir.

`load_raw_file` .dta desteğini `pyreadstat` best-effort ile korur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_file(path: str | Path) -> pd.DataFrame:
    """CSV / Excel / Stata (.dta) dosyasını dataframe olarak yükler."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)  # openpyxl backend
    if suffix == ".dta":
        return pd.read_stata(path)  # pyreadstat/pandas best-effort
    raise ValueError(f"Desteklenmeyen dosya uzantısı: {path.suffix} (csv/tsv/xlsx/dta)")


def _guess_join_keys(df: pd.DataFrame) -> list[str]:
    """id/kod benzeri isimli ya da tekrarlı-orta-kardinaliteli kolonları merge-key adayı sayar."""
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
    """Karar Defteri Agent'ının ihtiyaç duyduğu yapılandırılmış özet profili üretir.

    LLM'e giden payload budur; ham satır asla gitmez.
    """
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
