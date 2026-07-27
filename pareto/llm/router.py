"""Model Router — PydanticAI tabanlı, tipli I/O, test edilebilir.

review sorun #3'ün çözümü. Mekanik iş ucuz modele, yargı pinli güçlü modele
(providers.py zincirleri). Tipli çıktı: `output_type` bir Pydantic modeli olduğunda
PydanticAI şema-zorlaması + retry yapar → prototipteki regex-JSON ayıklama gitti.

Test: `use_test_model(...)` ile PydanticAI `TestModel`/`FunctionModel` enjekte edilir —
API yakmadan (test stratejisinin tamamı buna dayanıyor). Reprodüksiyon
dondurmadan gelir (menu.freeze), model stabilitesinden değil.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from ..config import SETTINGS, ModelRole, PrivacyMode, get_api_key, resolve_api_key
from .providers import ProviderModel, chain_for

logger = logging.getLogger(__name__)

_TEST_MODEL: Any | None = None  # test enjeksiyonu (TestModel/FunctionModel)


@contextmanager
def use_test_model(model: Any):
    """Test kapsamı: gerçek sağlayıcı yerine PydanticAI test modelini kullan."""
    global _TEST_MODEL
    previous = _TEST_MODEL
    _TEST_MODEL = model
    try:
        yield
    finally:
        _TEST_MODEL = previous


def _model_from_provider(pm: ProviderModel) -> Any:
    """Zincir girdisinden PydanticAI model nesnesi kurar (BYOK api_key ile)."""
    if pm.provider == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            pm.model_id,
            provider=GoogleProvider(api_key=get_api_key(pm.api_key_env)),
        )
    if pm.provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        # get_api_key() gerçek bir anahtar yoksa dummy key döner (S3-05, canned
        # mode) — OpenRouterProvider'a her zaman geçerli bir string gider.
        # get_api_key() artık hiç OSError fırlatmaz (bkz. config.get_api_key),
        # dolayısıyla burada yakalanacak bir "eksik anahtar" hatası riski yok.
        return OpenRouterModel(
            pm.model_id,
            provider=OpenRouterProvider(api_key=get_api_key(pm.api_key_env)),
        )
    if pm.provider == "groq":
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        return GroqModel(
            pm.model_id,
            provider=GroqProvider(api_key=get_api_key(pm.api_key_env)),
        )
    raise RuntimeError(f"Bilinmeyen sağlayıcı: {pm.provider!r}")


def _get_effective_privacy_mode() -> PrivacyMode:
    """UI seçimi varsa kullan, yoksa varsayılan ayara dön."""
    try:
        import streamlit as st
    except ImportError:
        return SETTINGS.privacy_mode
    raw = st.session_state.get("privacy_mode", SETTINGS.privacy_mode.value)
    return PrivacyMode.PRIVATE if str(raw) == PrivacyMode.PRIVATE.value else PrivacyMode.PUBLIC


def _chain_model(
    chain: tuple[ProviderModel, ...], *, fail_on_missing_keys: bool = False
) -> tuple[Any, bool]:
    """Zinciri tek modele indirger: tek üye → kendisi, çok üye → FallbackModel.

    S3-05 sonrası `config.get_api_key` artık hiç OSError fırlatmıyor (gerçek anahtar
    yoksa dummy key ile canned moda düşüyor) — dolayısıyla burada eskiden var olan
    "anahtarı eksik üye uyarıyla atlanır" davranışı artık hiç tetiklenmiyordu (ölü
    kod), kaldırıldı. Her üye — gerçek ya da dummy bir anahtarla — zincire girer.

    İkinci dönüş değeri (`canned_mode`), zincirdeki HİÇBİR üyenin gerçek bir anahtar
    bulamadığını (byok/env/secrets'ın hiçbirinde eşleşme olmadığını, hepsinin dummy
    key ile kurulduğunu) işaret eder. Bunu `_resolve_model` cache katmanına taşır:
    `CachedModel` cache-miss'te bu bayrak açıkken sahte bir ağ isteği atıp çirkin bir
    401/auth hatasına düşmek yerine açık bir hata fırlatır (bkz. cache.py, S3-05
    review Sorun B).
    """
    models: list[Any] = []
    any_real_key = False
    missing_key_envs: list[str] = []
    for pm in chain:
        _key, source = resolve_api_key(pm.api_key_env)
        if source != "none":
            any_real_key = True
        else:
            missing_key_envs.append(pm.api_key_env)
        models.append(_model_from_provider(pm))

    if fail_on_missing_keys and missing_key_envs:
        eksik = ", ".join(sorted(set(missing_key_envs)))
        raise OSError(f"eksik anahtarlar: {eksik}")

    canned_mode = not any_real_key
    if len(models) == 1:
        return models[0], canned_mode

    from pydantic_ai.models.fallback import FallbackModel

    return FallbackModel(*models), canned_mode


def _resolve_model(role: ModelRole) -> tuple[Any, dict[str, Any]]:
    """Test modeli varsa onu; yoksa cache'li failover zincirini + ekstra model_settings'i döndürür.

    JUDGE zinciri tek üyelidir, dolayısıyla pinli kalır (failover yalnız mekanikte).
    İkinci eleman (`extra_model_settings`), zincirin tek üyeli olduğu durumda o üyenin
    `ProviderModel.extra_model_settings`'i (örn. OpenRouter ZDR zorlaması) — çok üyeli
    zincirlerde (yalnız MECHANICAL) hiçbir slot bu alanı kullanmadığı için her zaman
    boş; ileride çok üyeli bir slot bu alanı kullanırsa yanlış üyeye uygulanmasın diye
    burada bilinçli olarak atlanır. Aynı gerekçeyle `ProviderModel.thinking` de yalnız
    tek üyeli zincirde okunur ve (`"off"` hariç) buraya eklenir — pydantic-ai'nin
    cross-provider `ModelSettings.thinking` alanına `build_agent()` üzerinden taşınır;
    "off" hiç key eklemez, model kendi varsayılanını kullanır (bkz. ADR 0004, 2026-07-24
    notu #2).
    """
    if _TEST_MODEL is not None:
        return _TEST_MODEL, {}
    from .cache import wrap_with_cache

    chain = chain_for(role, _get_effective_privacy_mode())
    extra_model_settings: dict[str, Any] = {}
    if len(chain) == 1:
        pm = chain[0]
        extra_model_settings = dict(pm.extra_model_settings or {})
        if pm.thinking != "off":
            extra_model_settings["thinking"] = pm.thinking

    model, canned_mode = _chain_model(
        chain, fail_on_missing_keys=_get_effective_privacy_mode() is PrivacyMode.PRIVATE
    )
    return wrap_with_cache(model, canned_mode=canned_mode), extra_model_settings


def build_agent(role: ModelRole, *, system_prompt: str, output_type: Any | None = None):
    """Rol için tipli bir PydanticAI Agent kurar. output_type verilirse şema zorlanır."""
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # fail-loud, sessiz düşme yok
        raise RuntimeError(
            "pydantic-ai kurulu değil. `uv sync` / `pip install pydantic-ai` gerekli."
        ) from exc

    model, extra_model_settings = _resolve_model(role)
    kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        # Determinizm pini: temp=0 → tekrarlanabilir yanıt + cache isabeti.
        # extra_model_settings genelde boş; yalnız OpenRouter ZDR gibi slota özgü
        # ayarlar taşıyan zincirlerde dolu (bkz. providers.py: ProviderModel).
        "model_settings": {"temperature": SETTINGS.llm_temperature, **extra_model_settings},
    }
    if output_type is not None:
        kwargs["output_type"] = output_type
    return Agent(model, **kwargs)
