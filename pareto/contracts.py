"""Veri kontratı: CleanPanel.

Temizleme → analiz dikişi. Tipli `CleanPanel` = long df + manifest; fail-loud
`validate_contract()`. Manifest'e göre tiered tasarım-tespiti yapar:
  - Tier1 (panel + treatment + never-treated) → tam DiD
  - Tier2 (yapı-yok / kesitsel)               → OLS-multiverse degrade + "ilişkisel" banner
  - Tier3 (kullanılamaz)                        → nazik ret

Deterministik; yeni sistem değil, dikişin uzantısı. `EstimationResult` (nullable
event-study/group-time alanları) staggered fast-follow için baştan rezerve edilir.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field


class Tier(StrEnum):
    TIER1_PANEL_DID = "tier1_panel_did"  # tam DiD mümkün
    TIER2_CROSS_OLS = "tier2_cross_ols"  # yalnız OLS-multiverse, "ilişkisel, nedensel değil"
    TIER3_UNUSABLE = "tier3_unusable"  # nazik ret


class PanelManifest(BaseModel):
    """CleanPanel'in kendini tanımlayan meta'sı — analiz katmanı bunu okur."""

    unit_col: str | None = None  # panel birim (örn. fips)
    time_col: str | None = None  # panel zaman (örn. year)
    treatment_col: str | None = None  # tedavi göstergesi / kohort
    treatment_cohort_col: str | None = None  # staggered kohort (fast-follow); nullable
    never_treated_col: str | None = None  # never-treated maskesi; nullable
    outcome_cols: tuple[str, ...] = ()
    covariate_cols: tuple[str, ...] = ()
    weight_col: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)  # kaynak/atıf izi


class CleanPanel(BaseModel):
    """Temizlenmiş long-format veri + manifest. df Pydantic dışında tutulur."""

    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame
    manifest: PanelManifest

    def tier(self) -> Tier:
        m = self.manifest
        if not m.outcome_cols:
            return Tier.TIER3_UNUSABLE
        has_panel = m.unit_col is not None and m.time_col is not None
        has_treatment = m.treatment_col is not None or m.treatment_cohort_col is not None
        if has_panel and has_treatment:
            return Tier.TIER1_PANEL_DID
        return Tier.TIER2_CROSS_OLS

    def validate_contract(self) -> Tier:
        """Fail-loud kontrat kontrolü. Kolon eksikse ANINDA patlar.

        `validate` adı kullanılamaz: pydantic `BaseModel.validate` ile çakışır (mypy override).
        """
        m = self.manifest
        cols = set(self.df.columns)

        declared = [
            m.unit_col,
            m.time_col,
            m.treatment_col,
            m.treatment_cohort_col,
            m.never_treated_col,
            m.weight_col,
            *m.outcome_cols,
            *m.covariate_cols,
        ]
        missing = [c for c in declared if c is not None and c not in cols]
        if missing:
            raise ValueError(
                f"CleanPanel manifest'i df'te olmayan kolonlara işaret ediyor: {missing}"
            )

        tier = self.tier()
        if tier is Tier.TIER3_UNUSABLE:
            raise ValueError("CleanPanel kullanılamaz (Tier3): en az bir outcome kolonu gerekli.")
        return tier


class EstimationResult(BaseModel):
    """Estimator çıktısının estimator-agnostik şeması (runner/variance/panel bunu tüketir).

    Event-study / group-time alanları staggered (Callaway-Sant'Anna) fast-follow için
    baştan nullable rezerve edilir — soyutlama-çizgisi committed'a sızmaz."""

    spec_id: str
    estimator: str
    coefficient: float | None = None
    std_error: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_obs: int | None = None
    status: str = "ok"  # "ok" | "failed"
    error: str | None = None

    # --- staggered fast-follow için rezerve (committed'da None) ---
    event_study: dict[str, float] | None = None
    group_time_atts: dict[str, float] | None = None
