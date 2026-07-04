"""Multiverse Genişletici.

Spec Generator'ın önerdiği serbestlik derecelerinin (kontroller x clustering x
estimator) Kartezyen çarpımını alıp N adet çalıştırılabilir spesifikasyon
objesi üreten DETERMİNİSTİK bir fonksiyon. Burada hiç LLM çağrısı yok --
kasıtlı: rastgelelik değil, tam numaralandırma istiyoruz.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..config import SETTINGS
from .spec_generator import SpecMenu

# Şimdilik desteklenen estimator havuzu (staggered / Callaway-Sant'Anna stretch goal).
SUPPORTED_ESTIMATORS = ("OLS", "TWFE")


@dataclass(frozen=True)
class Specification:
    spec_id: str
    outcome: str
    treatment: str
    controls: list[str]
    unit_fe: str | None
    time_fe: str | None
    pre_period_window: int
    cluster_by: str
    estimator: str
    sample_filter: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "controls": self.controls,
            "unit_fe": self.unit_fe,
            "time_fe": self.time_fe,
            "pre_period_window": self.pre_period_window,
            "cluster_by": self.cluster_by,
            "estimator": self.estimator,
            "sample_filter": self.sample_filter,
        }


def build_multiverse(
    *,
    outcome: str,
    treatment: str,
    unit_col: str,
    time_col: str,
    spec_menu: SpecMenu,
    estimators: tuple[str, ...] = SUPPORTED_ESTIMATORS,
) -> list[Specification]:
    """Kontroller x pre-period x clustering x estimator kartezyen çarpımını üretir."""
    for est in estimators:
        if est not in SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"Desteklenmeyen estimator: {est}. Staggered/Callaway-Sant'Anna stretch goal."
            )

    combos = itertools.product(
        spec_menu.control_sets,
        spec_menu.pre_period_windows,
        spec_menu.clustering_levels,
        estimators,
    )

    specs: list[Specification] = []
    for i, (controls, window, cluster_col, estimator) in enumerate(combos):
        unit_fe = unit_col if estimator == "TWFE" else None
        time_fe = time_col if estimator == "TWFE" else None
        specs.append(
            Specification(
                spec_id=f"spec_{i:04d}",
                outcome=outcome,
                treatment=treatment,
                controls=list(controls),
                unit_fe=unit_fe,
                time_fe=time_fe,
                pre_period_window=window,
                cluster_by=cluster_col,
                estimator=estimator,
            )
        )

    if len(specs) > SETTINGS.max_specifications:
        raise ValueError(
            f"{len(specs)} spesifikasyon üretildi, güvenlik sınırı "
            f"{SETTINGS.max_specifications}. Serbestlik derecelerini daraltın."
        )
    return specs
