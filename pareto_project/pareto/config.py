"""Pareto genel ayarları.

Tüm modüller ayarları buradan okur; böylece model adı, eşik değerleri vb.
tek bir yerden yönetilir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoSettings:
    # LLM ayarları
    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.0  # Karar defteri deterministik olmalı

    # Gatekeeper eşikleri
    missing_value_hard_threshold: float = 0.5  # %50'den fazla eksikse -> her zaman insana sor
    auto_approve_low_risk: bool = True

    # Çalışma dizinleri
    audit_trail_dir: str = "runs/audit_trail"
    results_dir: str = "runs"

    # Multiverse sınırları (kaza ile 10.000 spec üretmeyi engeller)
    max_specifications: int = 500


SETTINGS = ParetoSettings()


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY tanımlı değil. `export ANTHROPIC_API_KEY=...` ile ayarlayın."
        )
    return key
