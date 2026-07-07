"""Kolon profilleme (deterministik).

Ham veri hiçbir zaman LLM'e gönderilmez. Sadece kolon bazında özet bilgi
(dtype, eksiklik, örnekler ve aday roller) üretilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd

# Streamlit'in UploadedFile nesnesi tam olarak IO[bytes] değildir ama
# .seek() / .name / .read() ile duck-type olarak uyumludur.
FileLike = str | Path | IO[bytes] | Any


def load_raw_file(source: FileLike) -> pd.DataFrame:
    """CSV / Excel / Stata (.dta) dosyasını dataframe olarak yükler."""

    if isinstance(source, (str, Path)):
        # mypy tip daraltmasını yalnızca doğrudan isinstance() bloğu
        # içinde yapar; bunu bir bool değişkende saklayıp sonra
        # kullanmak (örn. `is_path = isinstance(...)`) daraltmayı
        # kaybettirir. Bu yüzden dallanmayı burada, doğrudan yapıyoruz.
        path = Path(source)
        suffix = path.suffix.lower()
        file_obj: FileLike = path
    else:
        # Streamlit UploadedFile gibi dosya benzeri nesneler için
        # isim özniteliği güvenli biçimde alınır.
        name = getattr(source, "name", "")
        suffix = Path(name).suffix.lower()
        file_obj = source

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    if suffix in (".csv", ".tsv"):
        return _read_delimited(file_obj, suffix=suffix)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_obj)

    if suffix == ".dta":
        return pd.read_stata(file_obj)

    raise ValueError(f"Desteklenmeyen dosya uzantısı: {suffix}")


def _read_delimited(file_obj: FileLike, *, suffix: str) -> pd.DataFrame:
    """CSV / TSV dosyasını deterministik olarak okur.

    Not: Ayraç (delimiter) sniffing veya başlık (header) satırı otomatik
    algılama kasıtlı olarak uygulanmaz — bu, ayrı bir görev kapsamındadır.
    Dosya uzantısına göre sabit ayraç kullanılır (.csv -> ',', .tsv -> '\\t').
    """

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    sep = "\t" if suffix == ".tsv" else ","

    try:
        return pd.read_csv(
            file_obj,
            sep=sep,
        )
    except Exception as exc:
        raise ValueError(f"Dosya okunamadı: {exc}") from exc


def _guess_join_keys(df: pd.DataFrame) -> list[str]:
    """Birleştirme (join) anahtarı olabilecek kolonları tahmin eder.

    Not: Yalnızca genel amaçlı, alan-bağımsız (domain-agnostic) anahtar
    kelimeler kullanılır (id, code, key, vb.). "fon kodu", "getiri" gibi
    finans alanına özgü hardcode string'ler burada kasıtlı olarak yer
    almaz; kolon rolü tahmini domain'e özel varsayımlar barındırmamalıdır.
    """
    candidates: list[str] = []
    n = len(df)

    for col in df.columns:
        name = str(col).lower()

        looks_like_id = any(
            token in name
            for token in (
                "id",
                "code",
                "kod",
                "key",
                "fips",
                "no",
            )
        )

        nunique = df[col].nunique(dropna=True)
        reasonable_cardinality = 1 < nunique < n

        if looks_like_id or reasonable_cardinality:
            candidates.append(str(col))

    return candidates


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Temizleme ajanı için deterministik kolon profili üretir."""

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

            col_info["top_values"] = {
                str(key): int(value)
                for key, value in top_values.items()
            }

            sample = (
                series.dropna()
                .astype(str)
                .head(20)
                .tolist()
            )

            col_info["looks_like_date"] = any(
                any(sep in value for sep in ("-", "/", "."))
                and any(ch.isdigit() for ch in value)
                for value in sample
            )

        profile["columns"][str(col)] = col_info

    return profile