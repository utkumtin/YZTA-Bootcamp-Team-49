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


def diagnose_axes(results: list[EstimationResult], specs) -> dict:  # noqa: ANN001
    """Matched-pair (ceteris-paribus) + ANOVA partial-R² atıf teşhisi.

    SPRINT-2: hangi eksenin işaret/anlamlılık dönüşünü sürüklediğini korelasyonla
    DEĞİL, faktöriyel içinde tek-eksen-değişen çiftlerle atfeder. Bkz docs/scrum.
    """
    raise NotImplementedError("Matched-pair + ANOVA teşhisi Sprint-2 kapsamında.")
