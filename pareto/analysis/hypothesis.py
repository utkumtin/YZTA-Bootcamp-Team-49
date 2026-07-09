"""Bağlam ve Hipotez - estimand-first akışı."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel

from ..config import ModelRole
from ..llm.guardrails import prompt_json
from ..llm.router import build_agent
from ..spec import Specification

logger = logging.getLogger(__name__)


class Estimand(BaseModel):
    """Tanımlayıcı varsayımı da içeren teorik estimand."""

    model_config = {"frozen": True}

    estimand_type: Literal["ATT", "ATE", "LATE"] = "ATT"
    treatment: str
    treatment_coding: str
    outcome: str
    outcome_unit: str
    population: str
    time_scope: str
    expected_sign: Literal["positive", "negative", "ambiguous"]
    identification_assumption: str = "parallel_trends"
    h0: str
    h1: str

    def freeze_hash(self) -> str:
        blob = json.dumps(
            self.model_dump(),
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class SocraticDeclaration(BaseModel):
    """Kullanıcının ajan öncesi kavramsal beyanı."""

    conceptual_treatment: str
    conceptual_outcome: str
    expected_sign: Literal["positive", "negative", "ambiguous"]


class TACProposal(BaseModel):
    """Teknik resmileştirme ve sade dilde geri çeviri."""

    estimand_type: Literal["ATT", "ATE", "LATE"] = "ATT"
    treatment: str
    treatment_coding: str
    outcome: str
    outcome_unit: str
    population: str
    time_scope: str
    expected_sign: Literal["positive", "negative", "ambiguous"]
    identification_assumption: str = "parallel_trends"
    h0: str
    h1: str

    implied_result_translation: str
    confirmation_question: str

    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_estimand(self) -> Estimand:
        return Estimand(
            estimand_type=self.estimand_type,
            treatment=self.treatment,
            treatment_coding=self.treatment_coding,
            outcome=self.outcome,
            outcome_unit=self.outcome_unit,
            population=self.population,
            time_scope=self.time_scope,
            expected_sign=self.expected_sign,
            identification_assumption=self.identification_assumption,
            h0=self.h0,
            h1=self.h1,
        )


class FrozenEstimand(BaseModel):
    """Onaylanmış estimand artifaktı."""

    model_config = {"frozen": True}

    estimand: Estimand
    freeze_hash: str


def draft_tac_proposal(
    *,
    research_story: str,
    available_columns: list[str],
    declaration: SocraticDeclaration,
) -> TACProposal:
    """Sokratik beyan sonrası TAC önerisini hazırlar."""

    if not available_columns:
        raise ValueError(
            "TAC önerisi için en az bir mevcut kolon gereklidir."
        )

    prompt = (
        "Araştırma hikayesi:\n"
        f"{prompt_json(research_story)}\n\n"

        "Kullanıcının Sokratik beyanı:\n"
        f"{prompt_json(declaration.model_dump())}\n\n"

        "Veri setinde bulunan kullanılabilir kolonlar:\n"
        f"<available_columns>{prompt_json(available_columns)}</available_columns>\n\n"

        "Kurallar:\n"
        "- Treatment yalnızca yukarıdaki kolonlardan biri olabilir.\n"
        "- Outcome yalnızca yukarıdaki kolonlardan biri olabilir.\n"
        "- Listede olmayan kolon isimleri üretme.\n"
        "- Eğer eşleme yapılamıyorsa needs_clarification=True döndür.\n"
        "- clarification_question alanını doldur.\n"
        "- expected_sign değerini değiştirme.\n"
        "- Kullanıcının teorik beyanını koru.\n"
    )

    agent = build_agent(
        ModelRole.JUDGE,
        system_prompt=(
            "You formalize causal estimands.\n"
            "Treatment and outcome MUST be selected only from "
            "<available_columns>.\n"
            "Never invent unavailable columns.\n"
            "If you cannot confidently map the conceptual variables "
            "to existing columns, return "
            "needs_clarification=True.\n"
            "Preserve the user's expected_sign."
        ),
        output_type=TACProposal,
    )

    return agent.run_sync(prompt).output


def freeze_estimand(
    proposal: TACProposal,
    *,
    approved: bool,
) -> FrozenEstimand:
    """Estimand'ı kullanıcı onayı sonrası dondurur."""

    if proposal.needs_clarification:
        question = (
            proposal.clarification_question
            or proposal.confirmation_question
        )
        raise ValueError(
            f"Estimand dondurulamıyor: {question}"
        )

    if not approved:
        raise ValueError(
            "Estimand'ın dondurulması için kullanıcı onayı gereklidir."
        )

    estimand = proposal.to_estimand()

    return FrozenEstimand(
        estimand=estimand,
        freeze_hash=estimand.freeze_hash(),
    )


def validate_estimand_spec_mapping(
    frozen: FrozenEstimand,
    spec: Specification,
    *,
    available_columns: list[str] | None = None,
) -> list[str]:
    """Estimand ile Specification eşleşmesini doğrular."""

    estimand = frozen.estimand

    errors: list[str] = []
    warnings: list[str] = []

    if spec.treatment != estimand.treatment_coding:
        errors.append("Treatment mismatch.")

    if spec.outcome != estimand.outcome:
        errors.append("Outcome mismatch.")

    if available_columns is not None:
        cols: set[str] = set(map(str, available_columns))

        if (
            spec.cluster_by
            and spec.cluster_by not in cols
        ):
            errors.append(
                f"Kümeleme kolonu '{spec.cluster_by}' veri setinde bulunamadı."
            )

        for control in spec.controls:
            if control not in cols:
                errors.append(
                    f"Kontrol kolonu '{control}' veri setinde bulunamadı."
                )

        if spec.estimator == "TWFE":
            if (
                not spec.unit_fe
                or spec.unit_fe not in cols
            ):
                errors.append(
                    "TWFE için geçerli bir unit_fe kolonu gereklidir."
                )

            if (
                not spec.time_fe
                or spec.time_fe not in cols
            ):
                errors.append(
                    "TWFE için geçerli bir time_fe kolonu gereklidir."
                )

    if errors:
        raise ValueError(
            "Estimand → Specification doğrulaması başarısız: "
            + "; ".join(errors)
        )

    if (
        estimand.identification_assumption == "parallel_trends"
        and spec.estimator == "OLS"
    ):
        warning = (
            f"Hash {frozen.freeze_hash}: "
            "parallel_trends varsayımı OLS ile eşleştirildi. "
            "Bu durum hata değildir ancak önerilen yöntem değildir."
        )

        logger.warning(warning)
        warnings.append(warning)

    return warnings