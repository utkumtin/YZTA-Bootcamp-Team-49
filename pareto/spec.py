"""Atom birimi: Specification.

`spec = {outcome, regressors, fixed_effects, clustering, sample, estimator}`
OLS, TWFE-DiD ve staggered bu uzayda farklı noktalar; estimator yalnız bir eksen.
Tek mimari → "iki ayrı sistem (fork)" riskini öldürür.

Prototipteki `Specification` dataclass'ından migre edildi ve Pydantic'e taşındı;
`sample_filter` (örneklem ekseni) ve `include_never_treated` (never-treated ekseni)
eksenleri eklendi.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, model_validator

Estimator = Literal["OLS", "TWFE"]  # committed çekirdek; staggered {CS,SA,BJS} = fast-follow
SUPPORTED_ESTIMATORS: tuple[Estimator, ...] = ("OLS", "TWFE")


class Specification(BaseModel):
    """Tek bir çalıştırılabilir analiz tarifi. Immutable (frozen) — hash'lenebilir."""

    model_config = {"frozen": True}

    spec_id: str
    outcome: str
    treatment: str
    controls: tuple[str, ...] = ()
    unit_fe: str | None = None
    time_fe: str | None = None
    pre_period_window: int | None = None
    cluster_by: str
    estimator: Estimator = "OLS"

    # Yeni eksenler (prototipte eksikti):
    sample_filter: str | None = None  # örneklem ekseni: pandas query (validate edilir)
    include_never_treated: bool = True  # never-treated dahil/hariç ekseni
    weight_col: str | None = None  # opsiyonel ağırlıklandırma ekseni

    @model_validator(mode="after")
    def _twfe_needs_fixed_effects(self) -> Specification:
        if self.estimator == "TWFE" and (self.unit_fe is None or self.time_fe is None):
            raise ValueError("TWFE estimator unit_fe ve time_fe gerektirir.")
        return self

    def content_hash(self) -> str:
        """spec_id hariç içeriğin deterministik hash'i (dedup + reprodüksiyon için)."""
        payload = self.model_dump(exclude={"spec_id"})
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
