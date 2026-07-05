"""Pareto genel ayarları — tek kaynak.

Model-router rolleri, privacy modu, sert spec tavanı,
determinizm pinleri. Sırlar buraya YAZILMAZ; yalnız env/`st.secrets` üzerinden okunur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum


class PrivacyMode(StrEnum):
    """İki açık mod. Private modda yalnız no-train/ZDR uçlar kullanılır."""

    PUBLIC = "public"  # public veri + canned/free model → gizlilik sorunu yok
    PRIVATE = "private"  # özel veri → no-train zorlanır, free-train uçlar YASAK


class ModelRole(StrEnum):
    """Model-router eksenleri: mekanik iş ucuza, yargı güçlü+pinli modele."""

    MECHANICAL = "mechanical"  # narration, format, sınıflandırma → ucuz+hızlı
    JUDGE = "judge"  # estimand/menü/teşhis yargısı → pinli güçlü model


@dataclass(frozen=True)
class ParetoSettings:
    # --- Model router. Gerçek model adları providers.py'de. ---
    judge_model: str = "gemini-3.5-flash"  # pinli yargı modeli (thinking ON)
    mechanical_model: str = "gemini-flash-lite"  # ucuz mekanik default
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.0  # deterministik → reprodüksiyon + cache

    # --- Privacy ---
    privacy_mode: PrivacyMode = PrivacyMode.PUBLIC

    # --- Gatekeeper eşikleri (temizleme human-in-the-loop) ---
    missing_value_hard_threshold: float = 0.5  # %50+ eksik → her zaman insana sor
    auto_approve_low_risk: bool = True

    # --- Multiverse sınırları (sert tavan 24) ---
    max_specifications: int = 24  # aşımda uyar + logla, sessiz kırpma yok

    # --- Determinizm (seed + BLAS/hash pin) ---
    seed: int = 20260704
    deterministic_env: dict[str, str] = field(
        default_factory=lambda: {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    )

    # --- Çalışma dizinleri (lokal artifacts, phone-home yok) ---
    runs_dir: str = "runs"
    store_dir: str = "runs/store"
    audit_trail_dir: str = "runs/audit_trail"


SETTINGS = ParetoSettings()


def get_api_key(provider_env: str) -> str:
    """BYOK: sağlayıcı anahtarını env'den okur (örn. 'GEMINI_API_KEY').

    Anahtar sadece env / `st.secrets` üzerinden gelir; repo'ya asla girmez.
    """
    key = os.environ.get(provider_env)
    if not key:
        raise OSError(
            f"{provider_env} tanımlı değil. `export {provider_env}=...` ile ayarlayın "
            "(veya Streamlit'te BYOK alanına girin)."
        )
    return key
