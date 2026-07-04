"""Data Ingestion & Profiling Modülü.

CSV veya .dta dosyalarını pandas dataframe'ine çevirir; kolon istatistiklerini,
formatlarını, eksik verileri ve potansiyel join-anahtarlarını çıkarır.
Bu profil, Karar Defteri Agent'ına (decision_ledger.py) girdi olarak gider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_file(path: str | Path) -> pd.DataFrame:
    """CSV veya Stata (.dta) dosyasını dataframe olarak yükler."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"Desteklenmeyen dosya uzantısı: {path.suffix} (sadece .csv / .dta)")


def _guess_join_keys(df: pd.DataFrame) -> list[str]:
    """Yüksek kardinaliteli, tekrar eden, id/kod benzeri isimli kolonları join-anahtarı adayı sayar."""
    candidates = []
    for col in df.columns:
        name = col.lower()
        looks_like_id = any(token in name for token in ("id", "code", "kod", "key", "no"))
        n = len(df)
        nunique = df[col].nunique(dropna=True)
        reasonable_cardinality = 1 < nunique < n  # tam benzersiz olmayan ama tekrarlı bir yapı
        if looks_like_id or reasonable_cardinality:
            candidates.append(col)
    return candidates


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Karar Defteri Agent'ının ihtiyaç duyduğu yapılandırılmış profili üretir.

    Bilinçli olarak insan-okunur + LLM-okunur bir sözlük döndürüyoruz; ham
    dataframe'i LLM'e asla doğrudan göndermiyoruz (gizlilik + token maliyeti).
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
            # Tarih olabilecek string kolonları işaretle (LLM için ipucu)
            sample = series.dropna().astype(str).head(20).tolist()
            col_info["looks_like_date"] = any(
                any(sep in v for sep in ("-", "/", ".")) and any(ch.isdigit() for ch in v)
                for v in sample
            )
        profile["columns"][col] = col_info

    return profile
