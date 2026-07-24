"""Sağlayıcı/model kayıtları — declarative, privacy-aware.

review sorun #3: prototip ham Anthropic SDK'ya (tek sağlayıcı, paid) sabitlenmişti.
Burada model-router'ın besleneceği declarative config var: rol → sağlayıcı zinciri.
Yargı modeli PİNLİ (tek üyeli zincir, failover yok); mekanikte failover zinciri.

Model ID'leri `.env`/`st.secrets`'tan çözülür (slot başına bir değişken), böylece
yeni model çıktığında kod değişmez. Çözüm sırası: UI seçimi → env → secrets → default.
Slotun `provider` / `api_key_env` / `no_train` alanları KODDA pinli kalır — bunlar
gizlilik ve kimlik-doğrulama garantileri, serbest ayar değil.

Privacy modu: PRIVATE modda yalnız `no_train=True` uçlar seçilir —
free-train uçlar (Gemini free) YASAK. Anahtarlar env/BYOK; burada asla saklanmaz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import ModelRole, PrivacyMode, resolve_setting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderModel:
    provider: str  # pydantic-ai provider prefix (örn. "google", "groq")
    model_id: str
    api_key_env: str  # BYOK env değişkeni
    no_train: bool  # PRIVATE modda yalnız True seçilebilir
    thinking: bool = False


@dataclass(frozen=True)
class ModelSlot:
    """Zincirdeki bir konum: sabit kimlik + model ID'sinin çözüm kuralı.

    `options` boşsa slot UI'da gösterilmez (yalnız `.env`'den ayarlanır).
    """

    key: str  # session/UI anahtarı
    provider: str
    api_key_env: str
    no_train: bool
    model_env: str  # .env değişkeni
    default_model: str
    options: tuple[str, ...] = ()  # UI selectbox seçenekleri (küratörlü)
    thinking: bool = False


# Küratörlü UI listesi: yalnız performansından emin olduğumuz modeller.
# Yeni model eklemek = bu tuple'a bir satır. İlk eleman .env defaultudur.
_JUDGE_OPTIONS: tuple[str, ...] = ("gemini-3.5-flash",)

JUDGE_SLOT = ModelSlot(
    key="judge",
    provider="google",
    api_key_env="GEMINI_API_KEY",
    no_train=False,
    model_env="GEMINI_JUDGE_MODEL",
    default_model=_JUDGE_OPTIONS[0],
    options=_JUDGE_OPTIONS,
    thinking=True,
)
JUDGE_PRIVATE_SLOT = ModelSlot(
    key="judge_private",
    provider="google",
    api_key_env="GEMINI_PAID_API_KEY",
    no_train=True,
    model_env="GEMINI_JUDGE_PRIVATE_MODEL",
    default_model="gemini-3.1-pro",
    thinking=True,
)
MECH_GEMINI_SLOT = ModelSlot(
    key="mech_gemini",
    provider="google",
    api_key_env="GEMINI_API_KEY",
    no_train=False,
    model_env="GEMINI_MECHANICAL_MODEL",
    default_model="gemini-flash-lite",
)
MECH_GROQ_SLOT = ModelSlot(
    key="mech_groq",
    provider="groq",
    api_key_env="GROQ_API_KEY",
    no_train=True,
    model_env="GROQ_MECHANICAL_MODEL",
    default_model="llama-3.3-70b-versatile",
)
MECH_OPENROUTER_SLOT = ModelSlot(
    key="mech_openrouter",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    no_train=False,
    model_env="OPENROUTER_MECHANICAL_MODEL",
    default_model="deepseek/deepseek-r1:free",
)

# Yargı: pinli tek üye. Mekanik: failover zinciri (ucuz → hızlı → geniş).
# PRIVATE mod: no-train/ZDR uçlar (paid Gemini no-train+DPA · Groq no-retention).
_JUDGE_SLOTS: tuple[ModelSlot, ...] = (JUDGE_SLOT,)
_MECHANICAL_SLOTS: tuple[ModelSlot, ...] = (
    MECH_GEMINI_SLOT,
    MECH_GROQ_SLOT,
    MECH_OPENROUTER_SLOT,
)
_PRIVATE_JUDGE_SLOTS: tuple[ModelSlot, ...] = (JUDGE_PRIVATE_SLOT,)
_PRIVATE_MECHANICAL_SLOTS: tuple[ModelSlot, ...] = (MECH_GROQ_SLOT,)


def _session_choice(slot: ModelSlot) -> str:
    """UI'dan (BYOK paneli) seçilen model ID'si; yoksa boş string.

    Streamlit yoksa veya script bağlamı dışındaysak sessizce boş döner.
    Listede olmayan değer yok sayılır: seçenek listesi daraltıldığında eski
    oturum değeri uygulamayı çökertmesin.
    """
    try:
        import streamlit as st

        choice = str(st.session_state.get(f"model_choice_{slot.key}", "")).strip()
    except Exception:
        return ""
    if not choice:
        return ""
    if choice not in slot.options:
        logger.warning(
            "Oturumdaki model seçimi listede yok, yok sayıldı: %s=%s", slot.key, choice
        )
        return ""
    return choice


def _resolve(slot: ModelSlot, *, allow_session: bool) -> ProviderModel:
    """Slotu somut bir uca indirger. Yalnız `model_id` ayarlanabilir."""
    model_id = (_session_choice(slot) if allow_session else "") or resolve_setting(
        slot.model_env, slot.default_model
    )
    return ProviderModel(
        provider=slot.provider,
        model_id=model_id,
        api_key_env=slot.api_key_env,
        no_train=slot.no_train,
        thinking=slot.thinking,
    )


def chain_for(role: ModelRole, privacy: PrivacyMode) -> tuple[ProviderModel, ...]:
    """Rol + privacy moduna göre failover zincirini döndürür (fail-loud on private).

    Zincir çağrı anında kurulur: `.env` yükleme sırası ve oturum-içi UI seçimi
    ancak böyle yansır. UI seçimi yalnız PUBLIC modda dinlenir — private uçlar
    (no-train garantisi + paid anahtar) deploy sahibinin kontrolünde kalır.
    """
    if privacy is PrivacyMode.PRIVATE:
        slots = _PRIVATE_JUDGE_SLOTS if role is ModelRole.JUDGE else _PRIVATE_MECHANICAL_SLOTS
        chain = tuple(_resolve(s, allow_session=False) for s in slots)
        if any(not m.no_train for m in chain):  # emniyet: private'da free-train sızmasın
            raise RuntimeError("PRIVATE modda no-train olmayan uç seçilemez.")
        return chain
    slots = _JUDGE_SLOTS if role is ModelRole.JUDGE else _MECHANICAL_SLOTS
    return tuple(_resolve(s, allow_session=True) for s in slots)
