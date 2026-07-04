"""Asenkron Çalıştırıcı.

N adet spesifikasyonu concurrent.futures ile paralel koşar. Her spesifikasyon
gerçek bir estimator çağrısıdır (statsmodels OLS veya linearmodels PanelOLS/TWFE) --
biz estimator'ı sıfırdan yazmıyoruz, sadece besliyoruz.

Her koşu şu şemayı döndürür:
    {"spec_dict": {...}, "katsayi": float, "p_value": float,
     "standart_hata": float, "hata_mesaji": None | str}
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from ..config import SETTINGS
from .multiverse import Specification


@dataclass
class RunResult:
    spec_dict: dict
    katsayi: float | None
    p_value: float | None
    standart_hata: float | None
    hata_mesaji: str | None


def _run_ols(df: pd.DataFrame, spec: Specification) -> RunResult:
    cols = [spec.treatment, *spec.controls]
    sub = df.dropna(subset=[spec.outcome, *cols])
    X = sm.add_constant(sub[cols])
    y = sub[spec.outcome]
    cluster_groups = sub[spec.cluster_by]

    model = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_groups}
    )
    return RunResult(
        spec_dict=spec.as_dict(),
        katsayi=float(model.params[spec.treatment]),
        p_value=float(model.pvalues[spec.treatment]),
        standart_hata=float(model.bse[spec.treatment]),
        hata_mesaji=None,
    )


def _run_twfe(df: pd.DataFrame, spec: Specification) -> RunResult:
    sub = df.dropna(subset=[spec.outcome, spec.treatment, *spec.controls])
    panel = sub.set_index([spec.unit_fe, spec.time_fe])

    exog_cols = [spec.treatment, *spec.controls]
    exog = sm.add_constant(panel[exog_cols])

    model = PanelOLS(
        panel[spec.outcome], exog, entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)

    return RunResult(
        spec_dict=spec.as_dict(),
        katsayi=float(model.params[spec.treatment]),
        p_value=float(model.pvalues[spec.treatment]),
        standart_hata=float(model.std_errors[spec.treatment]),
        hata_mesaji=None,
    )


def _run_single_spec(df: pd.DataFrame, spec: Specification) -> RunResult:
    try:
        if spec.estimator == "OLS":
            return _run_ols(df, spec)
        if spec.estimator == "TWFE":
            return _run_twfe(df, spec)
        raise ValueError(f"Bilinmeyen estimator: {spec.estimator}")
    except Exception as exc:  # noqa: BLE001 -- hatayı yut, sonuç listesine yaz
        return RunResult(
            spec_dict=spec.as_dict(),
            katsayi=None,
            p_value=None,
            standart_hata=None,
            hata_mesaji=str(exc),
        )


def run_multiverse(
    df: pd.DataFrame,
    specs: list[Specification],
    *,
    max_workers: int = 4,
) -> list[RunResult]:
    """Tüm spesifikasyonları paralel çalıştırır ve sonuç listesini döndürür."""
    results: list[RunResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_spec, df, spec): spec for spec in specs
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def persist_results(results: list[RunResult], run_id: str) -> Path:
    out_dir = Path(SETTINGS.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_results.json"
    payload: list[dict[str, Any]] = [asdict(r) for r in results]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = out_dir / "latest_results.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
