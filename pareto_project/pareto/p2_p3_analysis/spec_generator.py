"""Spec Generator (LLM).

Kullanıcının araştırma hikayesini (H0/H1 + baseline spesifikasyon) alır,
mantıklı "serbestlik dereceleri" önerir:
  - 5-6 farklı kontrol seti kombinasyonu
  - 2 farklı pre-period penceresi
  - 2 farklı clustering seviyesi

Bu çıktı, multiverse.py tarafından kartezyen çarpıma sokulur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm_client import call_llm_json

_SYSTEM_PROMPT = """\
Sen bir ekonometri danışmanısın. Araştırmacının hikayesini (hipotez, baseline
spesifikasyon, veri kolonları) okuyup, savunulabilir bir "serbestlik dereceleri
menüsü" öneriyorsun. Amaç: tek bir "doğru" regresyon yerine, makul varyasyonların
uzayını tanımlamak.

Çıktı JSON şeması:
{
  "control_sets": [["kolon1", "kolon2"], ["kolon1"], ...],   // 5-6 farklı kombinasyon
  "pre_period_windows": [int, int],                            // 2 farklı pencere (yıl/dönem sayısı)
  "clustering_levels": ["kolon_a", "kolon_b"],                 // 2 farklı clustering kolonu
  "gerekce": "Her önerinin neden makul bir araştırmacı seçimi olduğuna dair kısa açıklama"
}

Sadece baseline'da mevcut olan veya mantıken türetilebilecek kolon adlarını kullan.
"""


@dataclass
class SpecMenu:
    control_sets: list[list[str]]
    pre_period_windows: list[int]
    clustering_levels: list[str]
    gerekce: str


def generate_spec_menu(
    research_context: str,
    baseline_spec: dict,
    available_columns: list[str],
) -> SpecMenu:
    user_prompt = (
        f"Araştırma bağlamı (H0/H1):\n{research_context}\n\n"
        f"Baseline spesifikasyon:\n{json.dumps(baseline_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Kullanılabilir kolonlar: {available_columns}"
    )
    raw = call_llm_json(_SYSTEM_PROMPT, user_prompt)
    return SpecMenu(
        control_sets=raw["control_sets"],
        pre_period_windows=raw["pre_period_windows"],
        clustering_levels=raw["clustering_levels"],
        gerekce=raw.get("gerekce", ""),
    )
