"""Sağlayıcı/model kayıtları — declarative, privacy-aware.

review sorun #3: prototip ham Anthropic SDK'ya (tek sağlayıcı, paid) sabitlenmişti.
Burada model-router'ın besleneceği declarative config var: rol → sağlayıcı zinciri.
Yargı modeli PİNLİ (Gemini free-tier, thinking ON); mekanikte failover zinciri.

Privacy modu: PRIVATE modda yalnız `no_train=True` uçlar seçilir —
free-train uçlar (Gemini free) YASAK. Anahtarlar env/BYOK; burada asla saklanmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelRole, PrivacyMode


@dataclass(frozen=True)
class ProviderModel:
    provider: str  # pydantic-ai provider prefix (örn. "google-gla", "groq")
    model_id: str
    api_key_env: str  # BYOK env değişkeni
    no_train: bool  # PRIVATE modda yalnız True seçilebilir
    thinking: bool = False


# Yargı: pinli, thinking ON. Mekanik: failover zinciri (ucuz → hızlı → geniş).
_JUDGE_CHAIN: tuple[ProviderModel, ...] = (
    ProviderModel(
        "google-gla", "gemini-3.5-flash", "GEMINI_API_KEY", no_train=False, thinking=True
    ),
)
_MECHANICAL_CHAIN: tuple[ProviderModel, ...] = (
    ProviderModel("google-gla", "gemini-flash-lite", "GEMINI_API_KEY", no_train=False),
    ProviderModel("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY", no_train=True),
    ProviderModel("openrouter", "deepseek/deepseek-r1:free", "OPENROUTER_API_KEY", no_train=False),
)

# PRIVATE mod: no-train/ZDR uçlar (paid Gemini/Claude no-train+DPA · Groq no-retention).
_PRIVATE_JUDGE_CHAIN: tuple[ProviderModel, ...] = (
    ProviderModel(
        "google-gla", "gemini-3.1-pro", "GEMINI_PAID_API_KEY", no_train=True, thinking=True
    ),
)
_PRIVATE_MECHANICAL_CHAIN: tuple[ProviderModel, ...] = (
    ProviderModel("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY", no_train=True),
)


def chain_for(role: ModelRole, privacy: PrivacyMode) -> tuple[ProviderModel, ...]:
    """Rol + privacy moduna göre failover zincirini döndürür (fail-loud on private)."""
    if privacy is PrivacyMode.PRIVATE:
        chain = _PRIVATE_JUDGE_CHAIN if role is ModelRole.JUDGE else _PRIVATE_MECHANICAL_CHAIN
        if any(not m.no_train for m in chain):  # emniyet: private'da free-train sızmasın
            raise RuntimeError("PRIVATE modda no-train olmayan uç seçilemez.")
        return chain
    return _JUDGE_CHAIN if role is ModelRole.JUDGE else _MECHANICAL_CHAIN
