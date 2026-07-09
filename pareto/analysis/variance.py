"""Varyans muhakemesi — deterministik çekirdek.

Ürünün tezi: varyans = spesifikasyon çokluevreni (savunulabilir seçimler), LLM
gürültüsü DEĞİL. Bu modül DETERMİNİSTİK olan kısmı yapar: özet istatistikler +
3-bant robust/fragile kuralı (panelde açıkça yazılı, "betimleyici, formal joint
test değil"). Kural eşikleri başlangıç değeri; spike'ta kalibre edilir.

Matched-pair (ceteris-paribus) + ANOVA partial-R² atıf teşhisi ve LLM narrative
ayrı dikişlerdir (aşağıda; Sprint-2). LLM narrative açıklama YAPAR, ölçmez.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

from ..contracts import EstimationResult


class Band(StrEnum):
    ROBUST = "robust"  # işaret ≥%95 VE anlamlılık ≥%70
    MIXED = "mixed"  # arada
    FRAGILE = "fragile"  # işaret-uyumu <%90


# Başlangıç eşikleri (spike'ta kalibre edilir)
SIGN_ROBUST = 0.95
SIG_ROBUST = 0.70
SIGN_FRAGILE = 0.90


class VarianceSummary(dict):
    """Panelde gösterilen 3 sayı + bant + sayımlar. dict tabanlı → JSON-serializable."""


def summarize(results: list[EstimationResult]) -> VarianceSummary:
    """İşaret-uyumu %, anlamlılık %, nokta aralığı, 3-bant etiketi (deterministik)."""
    ok = [r for r in results if r.status == "ok" and r.coefficient is not None]
    n_total = len(results)
    n_ok = len(ok)
    n_failed = n_total - n_ok

    if n_ok == 0:
        return VarianceSummary(
            n_total=n_total,
            n_ok=0,
            n_failed=n_failed,
            sign_agreement=None,
            significance_rate=None,
            point_min=None,
            point_max=None,
            band=None,
        )

    coefs = [r.coefficient for r in ok if r.coefficient is not None]
    n_pos = sum(1 for c in coefs if c > 0)
    n_neg = n_ok - n_pos
    modal_sign = 1 if n_pos >= n_neg else -1
    sign_agreement = max(n_pos, n_neg) / n_ok

    # anlamlılık: modal işarette CI 0'ı dışlıyor mu
    n_sig_modal = sum(
        1
        for r in ok
        if r.coefficient is not None
        and (r.coefficient > 0) == (modal_sign > 0)
        and r.ci_low is not None
        and r.ci_high is not None
        and (r.ci_low > 0 or r.ci_high < 0)
    )
    significance_rate = n_sig_modal / n_ok

    if sign_agreement < SIGN_FRAGILE:
        band = Band.FRAGILE
    elif sign_agreement >= SIGN_ROBUST and significance_rate >= SIG_ROBUST:
        band = Band.ROBUST
    else:
        band = Band.MIXED

    return VarianceSummary(
        n_total=n_total,
        n_ok=n_ok,
        n_failed=n_failed,
        sign_agreement=round(sign_agreement, 3),
        significance_rate=round(significance_rate, 3),
        point_min=round(min(coefs), 6),
        point_max=round(max(coefs), 6),
        modal_sign=modal_sign,
        band=band.value,
    )


ROBUST_RULE_TEXT = (
    "Robust: işaret-uyumu ≥%95 VE anlamlılık ≥%70 · Kırılgan: işaret-uyumu <%90 · "
    "arası Karışık. Bu betimleyici bir kuraldır, formal joint test değildir."
)


AXES: tuple[str, ...] = (
    "control_set",
    "sample",
    "pre_period",
    "clustering",
    "never_treated",
    "estimator",
    "weighting",
)

_RSS_EPS = 1e-12


def _axis_values(spec: Any) -> dict[str, Any]:
    return {
        "control_set": tuple(getattr(spec, "controls", ()) or ()),
        "sample": getattr(spec, "sample_filter", None),
        "pre_period": getattr(spec, "pre_period_window", None),
        "clustering": getattr(spec, "cluster_by", None),
        "never_treated": getattr(spec, "include_never_treated", None),
        "estimator": getattr(spec, "estimator", None),
        "weighting": getattr(spec, "weight_col", None),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _level_key(value: Any) -> str:
    return repr(_jsonable(value))


def _is_significant(result: EstimationResult) -> bool | None:
    if result.ci_low is not None and result.ci_high is not None:
        return result.ci_low > 0 or result.ci_high < 0
    if result.p_value is not None:
        return result.p_value < 0.05
    return None


def _sign(coefficient: float) -> int | None:
    if coefficient > 0:
        return 1
    if coefficient < 0:
        return -1
    return None


def _same_except(left: dict[str, Any], right: dict[str, Any], axis: str) -> bool:
    return all(left[name] == right[name] for name in AXES if name != axis)


def _matched_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        deltas: list[float] = []
        sign_flips = 0
        sign_comparable = 0
        sig_flips = 0
        sig_comparable = 0

        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                if left["axes"][axis] == right["axes"][axis]:
                    continue
                if not _same_except(left["axes"], right["axes"], axis):
                    continue

                left_coef = left["coefficient"]
                right_coef = right["coefficient"]
                delta = right_coef - left_coef
                deltas.append(abs(delta))

                left_sign = _sign(left_coef)
                right_sign = _sign(right_coef)
                if left_sign is not None and right_sign is not None:
                    sign_comparable += 1
                    if left_sign != right_sign:
                        sign_flips += 1

                left_sig = _is_significant(left["result"])
                right_sig = _is_significant(right["result"])
                if left_sig is not None and right_sig is not None:
                    sig_comparable += 1
                    if left_sig != right_sig:
                        sig_flips += 1

        n_pairs = len(deltas)
        axis_out: dict[str, Any] = {
            "n_pairs": n_pairs,
            "mean_abs_delta": round(float(np.mean(deltas)), 6) if deltas else None,
            "max_abs_delta": round(float(max(deltas)), 6) if deltas else None,
            "sign_comparable_pairs": sign_comparable,
            "sign_flip_count": sign_flips,
            "sign_flip_rate": round(sign_flips / sign_comparable, 6)
            if sign_comparable
            else None,
        }
        if sig_comparable:
            axis_out["significance_flip_count"] = sig_flips
            axis_out["significance_flip_rate"] = round(sig_flips / sig_comparable, 6)
        else:
            axis_out["significance_flip_count"] = None
            axis_out["significance_flip_rate"] = None
        out[axis] = axis_out
    return out


def _design_matrix(rows: list[dict[str, Any]], axes: tuple[str, ...]) -> np.ndarray | None:
    columns: list[np.ndarray] = [np.ones(len(rows))]
    for axis in axes:
        levels = sorted({_level_key(row["axes"][axis]) for row in rows})
        if len(levels) <= 1:
            continue
        for level in levels[1:]:
            columns.append(
                np.array(
                    [1.0 if _level_key(row["axes"][axis]) == level else 0.0 for row in rows],
                    dtype=float,
                )
            )
    if not columns:
        return None
    return np.column_stack(columns)


def _rss(y: np.ndarray, x: np.ndarray) -> float | None:
    if len(y) <= x.shape[1]:
        return None
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residuals = y - x @ beta
    return float(np.sum(residuals**2))


def _anova_partial_r2(rows: list[dict[str, Any]]) -> tuple[dict[str, float | None], list[str]]:
    warnings: list[str] = []
    out: dict[str, float | None] = {}

    if len(rows) < 3:
        return {axis: None for axis in AXES}, ["ANOVA partial-R² için yeterli gözlem yok."]

    y = np.array([row["coefficient"] for row in rows], dtype=float)
    full_x = _design_matrix(rows, AXES)
    if full_x is None:
        return {axis: None for axis in AXES}, ["ANOVA partial-R² için eksen varyasyonu yok."]

    full_rss = _rss(y, full_x)
    if full_rss is None:
        return {axis: None for axis in AXES}, ["ANOVA partial-R² için model derecesi yetersiz."]

    for axis in AXES:
        levels = {row["axes"][axis] for row in rows}
        if len(levels) <= 1:
            out[axis] = None
            warnings.append(f"{axis}: tek seviyeli eksen; partial-R² hesaplanmadı.")
            continue

        reduced_axes = tuple(name for name in AXES if name != axis)
        reduced_x = _design_matrix(rows, reduced_axes)
        if reduced_x is None:
            out[axis] = None
            warnings.append(f"{axis}: reduced model kurulamadı.")
            continue

        reduced_rss = _rss(y, reduced_x)
        if reduced_rss is None or abs(reduced_rss) <= _RSS_EPS:
            out[axis] = None
            warnings.append(f"{axis}: yetersiz gözlem veya sıfıra yakın RSS.")
            continue

        partial = (reduced_rss - full_rss) / reduced_rss
        out[axis] = round(float(max(0.0, partial)), 6)

    return out, warnings


def diagnose_axes(results: list[EstimationResult], specs) -> dict:  # noqa: ANN001
    """Matched-pair (ceteris-paribus) + ANOVA partial-R² atıf teşhisi.

    SPRINT-2: hangi eksenin işaret/anlamlılık dönüşünü sürüklediğini korelasyonla
    DEĞİL, faktöriyel içinde tek-eksen-değişen çiftlerle atfeder. Bkz docs/scrum.
    """
    warnings: list[str] = []
    specs_by_id = {getattr(spec, "spec_id", None): spec for spec in specs}
    rows: list[dict[str, Any]] = []
    n_failed = 0
    n_missing_coefficient = 0
    n_missing_spec = 0

    for result in results:
        if result.status != "ok":
            n_failed += 1
            continue
        if result.coefficient is None:
            n_missing_coefficient += 1
            continue

        spec = specs_by_id.get(result.spec_id)
        if spec is None:
            n_missing_spec += 1
            continue

        axes = _axis_values(spec)
        rows.append(
            {
                "spec_id": result.spec_id,
                "coefficient": float(result.coefficient),
                "result": result,
                "axes": axes,
            }
        )

    if n_failed:
        warnings.append(f"{n_failed} başarısız sonuç axis teşhisinden dışlandı.")
    if n_missing_coefficient:
        warnings.append(f"{n_missing_coefficient} coefficient=None sonucu dışlandı.")
    if n_missing_spec:
        warnings.append(f"{n_missing_spec} sonuç için eşleşen spec_id bulunamadı.")
    if len(rows) < 2:
        warnings.append("Matched-pair teşhisi için yeterli başarılı/eşleşmiş sonuç yok.")

    matched_pairs = _matched_pairs(rows)
    anova_partial_r2, anova_warnings = _anova_partial_r2(rows)
    warnings.extend(anova_warnings)

    return {
        "n_results": len(results),
        "n_specs": len(specs),
        "n_used": len(rows),
        "n_excluded": {
            "failed": n_failed,
            "missing_coefficient": n_missing_coefficient,
            "missing_spec": n_missing_spec,
        },
        "axes": list(AXES),
        "matched_pairs": matched_pairs,
        "anova_partial_r2": anova_partial_r2,
        "warnings": warnings,
    }
