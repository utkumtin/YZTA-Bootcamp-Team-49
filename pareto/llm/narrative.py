"""Varyans anlatısı (JUDGE): açıklar, ölçmez.

Deterministik çekirdeğin (`summarize` + `diagnose_axes`) çıktısını okuyup panelde
gösterilecek tipli bir anlatıya çevirir. Tüm sayılar deterministik çekirdekten
gelir; LLM sayı üretmez, yalnızca verilen ölçümleri açıklar. Anlatının andığı
her eksen teşhis çıktısındaki eksen listesinde olmalıdır, aksi halde fail-loud.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..config import ModelRole
from .guardrails import prompt_json
from .router import build_agent


class AxisComment(BaseModel):
    """Tek eksen için atıf yorumu: eksen adı + Türkçe açıklama."""

    axis: str
    yorum: str


class VarianceNarrative(BaseModel):
    """JUDGE çıktısı: genel özet + işaret/anlamlılık dönüşünü süren eksen yorumları."""

    ozet: str
    eksen_yorumlari: list[AxisComment] = []


_SYSTEM_PROMPT = (
    "You explain specification-curve variance results for an econometrics panel. "
    "Every number comes from the deterministic core output you are given; you never "
    "compute, estimate, or invent numbers, thresholds, or axis names. "
    "Explain which axes drive sign or significance changes using only the matched-pair "
    "and ANOVA partial-R2 figures provided, and refer to axes strictly by the names "
    "listed under 'axes'. "
    "Write ozet and yorum in Turkish."
)


def _build_narrative_prompt(summary: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    return (
        "Deterministic variance summary (summarize output):\n"
        f"{prompt_json(summary)}\n\n"
        "Deterministic axis diagnosis (diagnose_axes output):\n"
        f"{prompt_json(diagnosis)}\n\n"
        "Return a VarianceNarrative. ozet: one or two sentences stating the overall "
        "picture, for example how many runs share the modal sign and which axis drives "
        "the flips. eksen_yorumlari: one comment per axis that materially drives sign "
        "or significance changes; skip axes with no signal."
    )


def _validate_axes(narrative: VarianceNarrative, diagnosis: dict[str, Any]) -> None:
    """JUDGE'ın andığı her eksen teşhiste olmalı; uydurma eksen fail-loud."""
    known = set(diagnosis.get("axes", []))
    unknown = [c.axis for c in narrative.eksen_yorumlari if c.axis not in known]
    if unknown:
        raise ValueError(f"JUDGE teşhiste olmayan eksen andı: {unknown}")


def generate_narrative(
    summary: dict[str, Any], diagnosis: dict[str, Any]
) -> VarianceNarrative:
    """JUDGE: deterministik özet + eksen teşhisi -> tipli varyans anlatısı.

    Ölçüm `summarize` ve `diagnose_axes`'ten gelir; LLM yalnızca açıklama yazar.
    Panel sayıları deterministik çıktıdan gösterir, anlatı onların yorumudur.
    """
    if not summary.get("n_ok"):
        raise ValueError("Anlatı için başarılı sonuç yok; önce multiverse koşulmalı.")

    agent = build_agent(
        ModelRole.JUDGE,
        system_prompt=_SYSTEM_PROMPT,
        output_type=VarianceNarrative,
    )
    narrative = agent.run_sync(_build_narrative_prompt(summary, diagnosis)).output
    _validate_axes(narrative, diagnosis)
    return narrative
