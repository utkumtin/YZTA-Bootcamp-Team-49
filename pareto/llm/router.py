"""Model Router — PydanticAI tabanlı, tipli I/O, test edilebilir.

Mekanik iş ucuz modele, yargı pinli güçlü modele (providers.py zincirleri).
Tipli çıktı: `output_type` bir Pydantic modeli olduğunda PydanticAI
şema-zorlaması + retry yapar → prototipteki regex-JSON ayıklama gitti.

Test: `use_test_model(...)` ile PydanticAI `TestModel`/`FunctionModel` enjekte
edilir — API yakmadan (test stratejisinin tamamı buna dayanıyor). Reprodüksiyon
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

# NVIDIA NIM'in OpenAI-uyumlu ucu. pydantic-ai'de ayrı bir `nvidia` provider modülü
# yok (bkz. pydantic_ai.providers listesi), o yüzden OpenAI istemcisi base_url ile
# yönlendiriliyor. Sabit burada çünkü providers.py deklaratif config tutar; hangi
# pydantic-ai sınıfının hangi adrese bağlandığı bu dosyanın işi.
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


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
    """Zincir girdisinden PydanticAI model nesnesi kurar (BYOK api_key ile).

    `allow_canned=True` ile çağrılır: gerçek anahtar yoksa constructor dummy
    key ile kurulur. Bu güvenlidir çünkü hangi üyelerin zincire gireceğine
    (gerçek anahtarı olanlar, ya da hiç yoksa hepsi) `_chain_model` karar
    verir; burada yalnız kurulum yapılır. Dummy key ile kurulmuş bir modele
    gerçek bir ağ isteği gitmesi, `canned_mode` bayrağı `CachedModel`'e
    taşındığı için ayrıca engellenir (bkz. cache.py).
    """
    if pm.provider == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            pm.model_id,
            provider=GoogleProvider(api_key=get_api_key(pm.api_key_env, allow_canned=True)),
        )
    if pm.provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(
            pm.model_id,
            provider=OpenRouterProvider(api_key=get_api_key(pm.api_key_env, allow_canned=True)),
        )
    if pm.provider == "groq":
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        return GroqModel(
            pm.model_id,
            provider=GroqProvider(api_key=get_api_key(pm.api_key_env, allow_canned=True)),
        )
    if pm.provider == "nvidia":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        # get_api_key() ÖNCE çağrılır: openrouter dalındaki gerekçenin aynısı —
        # eksik anahtar _chain_model'in yakaladığı OSError olarak yükselsin,
        # sağlayıcının kendi UserError'ına düşmesin.
        return OpenAIChatModel(
            pm.model_id,
            provider=OpenAIProvider(
                api_key=get_api_key(pm.api_key_env),
                base_url=NVIDIA_NIM_BASE_URL,
            ),
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


def is_canned_mode(role: ModelRole = ModelRole.MECHANICAL) -> bool:
    """Verilen rol için zincirin canned modda olup olmadığını hesaplar.

    `_chain_model` ile aynı mantığı (zincirdeki hiçbir üyenin gerçek anahtarı
    yoksa canned) paylaşır, ama modelleri gerçekten kurmadan yalnız anahtar
    çözümü yapar — UI banner'ının tek doğru kaynağı burasıdır (O1: banner artık
    tek bir sağlayıcıya değil, aktif rolün tüm zincirine bakar).
    """
    chain = chain_for(role, _get_effective_privacy_mode())
    return not any(resolve_api_key(pm.api_key_env)[1] != "none" for pm in chain)


def _chain_model(chain: tuple[ProviderModel, ...]) -> tuple[Any, bool]:
    """Zinciri tek modele indirger: tek üye → kendisi, çok üye → FallbackModel.

    Gerçek anahtarı olmayan üyeler zincire hiç girmez (kısmi BYOK'ta, örn.
    yalnız OPENROUTER_API_KEY girilmişse, anahtarsız Gemini/Groq üyelerine
    dummy key ile gerçek bir ağ isteği atılmasını engeller — hem gizlilik hem
    performans). Zincirdeki HİÇBİR üyenin gerçek anahtarı yoksa (tam canned
    senaryo), tüm üyeler yine de dummy key ile kurulur ki failover mekaniği ve
    golden-path cache replay'i çalışabilsin.

    İkinci dönüş değeri (`canned_mode`) zincirdeki hiçbir üyenin gerçek anahtar
    bulamadığını işaret eder ve `_resolve_model` tarafından cache katmanına
    taşınır: `CachedModel` cache-miss'te bu bayrak açıkken sahte bir ağ isteği
    atıp çirkin bir 401/auth hatasına düşmek yerine açık bir hata fırlatır
    (bkz. cache.py: `CannedModeCacheMissError`).
    """
    models: list[Any] = [
        _model_from_provider(pm) for pm in chain if resolve_api_key(pm.api_key_env)[1] != "none"
    ]

    canned_mode = not models
    if canned_mode:
        # Hiçbir üyenin gerçek anahtarı yok: zincir yine de dummy key'lerle kurulur.
        models = [_model_from_provider(pm) for pm in chain]

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

    effective_privacy_mode = _get_effective_privacy_mode()
    chain = chain_for(role, effective_privacy_mode)
    extra_model_settings: dict[str, Any] = {}
    if len(chain) == 1:
        pm = chain[0]
        extra_model_settings = dict(pm.extra_model_settings or {})
        if pm.thinking != "off":
            extra_model_settings["thinking"] = pm.thinking

    if effective_privacy_mode is PrivacyMode.PRIVATE:
        # Private modda no-train garantisi zincirdeki HERHANGİ bir slot için
        # geçerli olmalı (Gemini paid, Groq private, OpenRouter private, ...),
        # tek bir slota hardcoded değil. Hiçbir üye gerçek anahtar bulamazsa
        # kullanıcı "private" kilidine bakarken donmuş bir kaydı izliyor
        # olurdu — bu yüzden burada açıkça ve yönlendirici bir hata fırlatılır.
        if not any(resolve_api_key(pm.api_key_env)[1] != "none" for pm in chain):
            env_names = ", ".join(sorted({pm.api_key_env for pm in chain}))
            raise OSError(
                f"eksik anahtarlar: {env_names}. Private mod için zincirdeki en az bir "
                "sağlayıcıya gerçek bir API anahtarı gerekir. Ayarlar sekmesinden BYOK "
                f"anahtarınızı girin, `.env` dosyasına ekleyin, ya da `export {env_names}=...` "
                "ile ortam değişkeni olarak tanımlayın."
            )

    model, canned_mode = _chain_model(chain)
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