"""Context & Hipotez — estimand-first.

Hipotez = tipli TEORİK ESTIMAND nesnesi (LLM serbest metni değil — review sorun:
prototipte `--research-context` serbest string'di). Onayda DONAR (immutable + hash;
spec-menüsünden ÖNCE) → pre-registration artifact, anchoring'i öldürür.

Tipli model + dondurma bir Sprint-1 dikişidir (aşağıda, gerçek). Hibrit onay akışı
(Socratic: kullanıcı yön/işareti agent'ı görmeden beyan eder · TAC: teknik
formalizasyonu agent önerir + ima-edilen-sonuç geri-çevirisi) Sprint-2 LLM işidir.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel


class Estimand(BaseModel):
    """Teorik estimand — verinin kanıtlayamayacağı kimlik varsayımı dahil."""

    model_config = {"frozen": True}

    estimand_type: Literal["ATT", "ATE", "LATE"] = "ATT"
    treatment: str  # kavramsal tedavi + kodlama
    treatment_coding: str
    outcome: str  # sonuç + birim/ölçek
    outcome_unit: str
    population: str  # popülasyon
    time_scope: str  # zaman
    expected_sign: Literal["positive", "negative", "ambiguous"]  # Socratic beyan
    identification_assumption: str = "parallel_trends"
    h0: str
    h1: str

    def freeze_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def elicit_estimand(research_story: str, available_columns: list[str]) -> Estimand:
    """Hibrit onay (Socratic + TAC) ile estimand'ı kur.

    SPRINT-2: yön/işaret + kavramsal tedavi/sonuç kullanıcıdan (agent'ı görmeden);
    teknik formalizasyon agent'tan + ima-edilen-sonuç geri-çevirisi + hedefli teyit;
    agent emin değilse sorar. Validator Specification'a temiz eşlemeyi fail-loud kontrol eder.
    """
    raise NotImplementedError("Estimand elicitation (Socratic/TAC) Sprint-2 kapsamında.")
