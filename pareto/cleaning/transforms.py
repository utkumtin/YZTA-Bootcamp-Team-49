"""Kısıtlı/vetted transform kütüphanesi — L3 (prompt-injection savunması).

review'in KRİTİK bulgusu: prototipte LLM keyfi pandas kodu üretip `exec` ile aynı
süreçte çalıştırıyordu (`__builtins__` açık → `__import__` ile FS/network → RCE).
Bu KALDIRILDI. Yeni model:
  - LLM SADECE bu kapalı sözlükten transform adı + tipli parametre seçer (kod ÜRETMEZ).
  - Her transform hem `apply` (deterministik uygulama) hem `render` (yeniden üretilebilir
    kod ÜRETİMİ değil; sabit şablondan RENDER) sağlar → denetim izi = metot bölümü.
  - Kapsam şunlarla sınırlı: type-coercion / leading-zero / tarih /
    NA-marker / kolon-adı / duplicate. Imputation/outlier/keyfi-reshape KAPSAM DIŞI.

`codegen.py` bu render'ları birleştirip audit-trail script'i üretir. Keyfi kod yoktur.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Transform:
    name: str
    apply: Callable[..., pd.DataFrame]  # (df, **params) -> df
    render: Callable[..., str]  # (**params) -> reproducible code line(s)
    doc: str


# --------------------------------------------------------------------------- #
# Vetted transform implementasyonları (saf, yan-etkisiz)
# --------------------------------------------------------------------------- #
def _rename_column(df: pd.DataFrame, *, old: str, new: str) -> pd.DataFrame:
    return df.rename(columns={old: new})


def _coerce_numeric(df: pd.DataFrame, *, col: str) -> pd.DataFrame:
    out = df.copy()
    cleaned = out[col].astype(str).str.replace(",", "", regex=False).str.strip()
    out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


def _preserve_leading_zeros(df: pd.DataFrame, *, col: str, width: int) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).str.strip().str.zfill(width)
    return out


def _parse_date(df: pd.DataFrame, *, col: str, fmt: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], format=fmt, errors="coerce")
    return out


def _standardize_na(df: pd.DataFrame, *, col: str, markers: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].replace(markers, pd.NA)
    return out


def _drop_duplicates(df: pd.DataFrame, *, subset: list[str] | None = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)


# --------------------------------------------------------------------------- #
# Render şablonları (SABİT string; LLM parametreleri doldurur, kod yazmaz)
# --------------------------------------------------------------------------- #
def _r_rename(*, old: str, new: str) -> str:
    return f"df = df.rename(columns={{{old!r}: {new!r}}})"


def _r_coerce_numeric(*, col: str) -> str:
    return (
        f"df[{col!r}] = pd.to_numeric("
        f"df[{col!r}].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')"
    )


def _r_leading_zeros(*, col: str, width: int) -> str:
    return f"df[{col!r}] = df[{col!r}].astype(str).str.strip().str.zfill({width})"


def _r_parse_date(*, col: str, fmt: str | None = None) -> str:
    return f"df[{col!r}] = pd.to_datetime(df[{col!r}], format={fmt!r}, errors='coerce')"


def _r_standardize_na(*, col: str, markers: list[str]) -> str:
    return f"df[{col!r}] = df[{col!r}].replace({markers!r}, pd.NA)"


def _r_drop_duplicates(*, subset: list[str] | None = None) -> str:
    return f"df = df.drop_duplicates(subset={subset!r})"


REGISTRY: dict[str, Transform] = {
    t.name: t
    for t in (
        Transform("rename_column", _rename_column, _r_rename, "Kolon adını değiştir."),
        Transform(
            "coerce_numeric",
            _coerce_numeric,
            _r_coerce_numeric,
            "Binlik ayraç/whitespace temizle, sayıya çevir (parse edilemeyen → NA).",
        ),
        Transform(
            "preserve_leading_zeros",
            _preserve_leading_zeros,
            _r_leading_zeros,
            "Öndeki sıfırları koru (örn. FIPS kodu) — string + zfill.",
        ),
        Transform("parse_date", _parse_date, _r_parse_date, "Tarih kolonunu datetime'a çevir."),
        Transform(
            "standardize_na",
            _standardize_na,
            _r_standardize_na,
            "NA-marker'ları (örn. 'N/A', '-999') gerçek NA'ya çevir.",
        ),
        Transform(
            "drop_duplicates",
            _drop_duplicates,
            _r_drop_duplicates,
            "Tekrar eden satırları at (opsiyonel anahtar kümesiyle).",
        ),
    )
}

ALLOWED_TRANSFORMS = tuple(REGISTRY)


def get_transform(name: str) -> Transform:
    if name not in REGISTRY:
        raise ValueError(
            f"İzin verilmeyen transform: {name!r}. Kapalı taksonomi: {ALLOWED_TRANSFORMS}. "
            "Keyfi kod render edilmez (L3)."
        )
    return REGISTRY[name]


def apply_transform(df: pd.DataFrame, name: str, params: dict[str, Any]) -> pd.DataFrame:
    """Kapalı sözlükten seçilmiş transform'u tipli parametrelerle uygular."""
    return get_transform(name).apply(df, **params)
