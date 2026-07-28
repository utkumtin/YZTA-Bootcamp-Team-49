"""Sağlayıcı/model kayıtları — declarative, privacy-aware.

review sorun #3: prototip ham Anthropic SDK'ya (tek sağlayıcı, paid) sabitlenmişti.
Burada model-router'ın besleneceği declarative config var: rol → sağlayıcı zinciri.
Yargı modeli PİNLİ (tek üyeli zincir, failover yok); mekanikte failover zinciri.

Model ID'leri `.env`/`st.secrets`'tan çözülür (slot başına bir değişken), böylece
yeni model çıktığında kod değişmez. Çözüm sırası: UI seçimi → env → secrets → default.
Slotun `provider` / `api_key_env` / `no_train` alanları KODDA pinli kalır — bunlar
gizlilik ve kimlik-doğrulama garantileri, serbest ayar değil.

JUDGE için sağlayıcı (Gemini/Groq/OpenRouter/NVIDIA) VE model seçimi artık hem PUBLIC
hem PRIVATE modda UI'dan yapılabilir — kullanıcı yalnız küratörlü slot kümesi içinden
seçer, kimlik alanları asla serbest değildir. MECHANICAL bu seçimin dışında, davranışı
değişmedi (failover zinciri, UI'da hiç gösterilmez). NVIDIA yalnız PUBLIC katmanda var:
ücretsiz NIM ucu no-train garantisi taşımıyor.

Privacy modu: PRIVATE modda yalnız `no_train=True` uçlar seçilir —
free-train uçlar (Gemini free) YASAK. Anahtarlar env/BYOK; burada asla saklanmaz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from ..config import ModelRole, PrivacyMode, resolve_setting

logger = logging.getLogger(__name__)

# Küratörlü thinking/reasoning derinliği — pydantic-ai'nin tam skalası
# ('minimal'..'xhigh') değil, mevcut "serbest metin yok, küratörlü liste" ile
# tutarlı bir alt küme. router.py:_resolve_model bunu pydantic-ai'nin cross-
# provider `ModelSettings.thinking` alanına taşır; "off" hiç key eklemez.
ThinkingChoice = Literal["off", "low", "medium", "high"]


@dataclass(frozen=True)
class ProviderModel:
    provider: str  # pydantic-ai provider prefix (örn. "google", "groq")
    model_id: str
    api_key_env: str  # BYOK env değişkeni
    no_train: bool  # PRIVATE modda yalnız True seçilebilir
    thinking: ThinkingChoice = "off"
    # Sağlayıcıya özgü ekstra model_settings (örn. OpenRouter ZDR zorlaması).
    # Yalnız ihtiyaç duyan slotlarda dolu; build_agent() temperature ile birleştirir.
    extra_model_settings: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelOption:
    """Küratörlü bir model seçeneği: kimlik + ekipçe doldurulacak performans/maliyet notu.

    `performance_note`/`*_cost_note` yalnız görüntüleme alanı — otomatik maliyet
    hesaplama/toplama burada yapılmaz (ayrı bir altyapı işi, bilinçli olarak ertelendi).
    """

    model_id: str
    performance_note: str = ""
    input_cost_note: str = ""
    output_cost_note: str = ""


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
    options: tuple[ModelOption, ...] = ()  # UI selectbox seçenekleri (küratörlü)
    default_thinking: ThinkingChoice = "off"
    thinking_options: tuple[ThinkingChoice, ...] = ()  # boşsa UI'da gösterilmez
    extra_model_settings: dict[str, Any] | None = None


# Küratörlü UI listeleri: yalnız performansından emin olduğumuz modeller.
# Yeni model eklemek = ilgili tuple'a bir satır. İlk eleman .env defaultudur.
# TODO(ekip): aşağıdaki Groq/OpenRouter listeleri yer tutucudur — ekibin kendi
# testlerinden geçirdiği gerçek model ID'leri + performance_note/*_cost_note ile
# değiştirilmeli (bkz. ADR 0004, 2026-07-24 notu).
_JUDGE_GEMINI_OPTIONS: tuple[ModelOption, ...] = (ModelOption(model_id="gemini-3.5-flash"),)
_JUDGE_GEMINI_PRIVATE_OPTIONS: tuple[ModelOption, ...] = (ModelOption(model_id="gemini-3.1-pro"),)
_JUDGE_GROQ_OPTIONS: tuple[ModelOption, ...] = (ModelOption(model_id="llama-3.3-70b-versatile"),)
_JUDGE_OPENROUTER_OPTIONS: tuple[ModelOption, ...] = (
    ModelOption(model_id="deepseek/deepseek-r1:free"),
)
# Private: ZDR zorunlu, ":free" uçları hariç tutulur (bkz. JUDGE_OPENROUTER_PRIVATE_SLOT).
_JUDGE_OPENROUTER_PRIVATE_OPTIONS: tuple[ModelOption, ...] = (
    ModelOption(model_id="deepseek/deepseek-r1"),
)
# Tüm JUDGE slotlarında aynı küratörlü thinking seçenekleri (bkz. ADR 0004,
# 2026-07-24 notu #2) — hangi sağlayıcı seçilirse seçilsin aynı seçenekler sunulur.
_JUDGE_THINKING_OPTIONS: tuple[ThinkingChoice, ...] = ("off", "low", "medium", "high")

JUDGE_SLOT = ModelSlot(
    key="judge",
    provider="google",
    api_key_env="GEMINI_API_KEY",
    no_train=False,
    model_env="GEMINI_JUDGE_MODEL",
    default_model=_JUDGE_GEMINI_OPTIONS[0].model_id,
    options=_JUDGE_GEMINI_OPTIONS,
    default_thinking="medium",
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
JUDGE_PRIVATE_SLOT = ModelSlot(
    key="judge_private",
    provider="google",
    api_key_env="GEMINI_PAID_API_KEY",
    no_train=True,
    model_env="GEMINI_JUDGE_PRIVATE_MODEL",
    default_model=_JUDGE_GEMINI_PRIVATE_OPTIONS[0].model_id,
    options=_JUDGE_GEMINI_PRIVATE_OPTIONS,
    default_thinking="medium",
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
JUDGE_GROQ_SLOT = ModelSlot(
    key="judge_groq",
    provider="groq",
    api_key_env="GROQ_API_KEY",
    no_train=True,
    model_env="GROQ_JUDGE_MODEL",
    default_model=_JUDGE_GROQ_OPTIONS[0].model_id,
    options=_JUDGE_GROQ_OPTIONS,
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
JUDGE_GROQ_PRIVATE_SLOT = ModelSlot(
    key="judge_groq_private",
    provider="groq",
    api_key_env="GROQ_API_KEY",
    no_train=True,  # Groq hesap-seviyesinde ZDR (bkz. .env.example) — ops doğrulaması gerekir
    model_env="GROQ_JUDGE_PRIVATE_MODEL",
    default_model=_JUDGE_GROQ_OPTIONS[0].model_id,
    options=_JUDGE_GROQ_OPTIONS,
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
JUDGE_OPENROUTER_SLOT = ModelSlot(
    key="judge_openrouter",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    no_train=False,
    model_env="OPENROUTER_JUDGE_MODEL",
    default_model=_JUDGE_OPENROUTER_OPTIONS[0].model_id,
    options=_JUDGE_OPENROUTER_OPTIONS,
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
JUDGE_NVIDIA_SLOT = ModelSlot(
    key="judge_nvidia",
    provider="nvidia",
    api_key_env="NVIDIA_API_KEY",
    # NIM'in ücretsiz ucu ZDR/no-train garantisi vermiyor → bu slot PUBLIC'e özgüdür,
    # _PRIVATE_JUDGE_SLOTS'a ASLA eklenmemeli (tests/test_privacy_routing.py bekçisi).
    no_train=False,
    model_env="NVIDIA_JUDGE_MODEL",
    default_model="nvidia/nemotron-3-super-120b-a12b",
    # options bilerek boş: küratörlü liste JUDGE benchmark'ı sonuçlanınca doldurulacak
    # (bkz. benchmarks/README.md). Boşken slot yalnız `.env`'den ayarlanır — ModelSlot
    # docstring'indeki sözleşme. UI'da sağlayıcı görünür, model kutusu `.env` pinini gösterir.
    options=(),
    thinking_options=_JUDGE_THINKING_OPTIONS,
)
# L7 detective scanner (Prompt Guard): UI'da gösterilmez; merkezi model/env
# sözleşmesine dahil edilir ki çağrı yolu diğer LLM katmanlarıyla uyumlu olsun.
PROMPT_GUARD_SLOT = ModelSlot(
    key="prompt_guard",
    provider="groq",
    api_key_env="GROQ_API_KEY",
    no_train=True,
    model_env="PARETO_L7_PROMPT_GUARD_MODEL",
    default_model="meta-llama/llama-prompt-guard-2-86m",
)
JUDGE_OPENROUTER_PRIVATE_SLOT = ModelSlot(
    key="judge_openrouter_private",
    provider="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    no_train=True,  # yalnız bu slotta True: aşağıdaki zdr zorlamasıyla birlikte anlamlı
    model_env="OPENROUTER_JUDGE_PRIVATE_MODEL",
    default_model=_JUDGE_OPENROUTER_PRIVATE_OPTIONS[0].model_id,
    options=_JUDGE_OPENROUTER_PRIVATE_OPTIONS,
    thinking_options=_JUDGE_THINKING_OPTIONS,
    # İstek-bazlı ZDR zorlaması (hesap-seviyesi değil) — router.py: _model_from_provider
    # bunu OpenRouterModel'in model_settings'ine taşır. Bkz. openrouter.ai/docs/features/
    # provider-routing#zero-data-retention-enforcement
    extra_model_settings={"openrouter_provider": {"zdr": True}},
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

# Yargı: her (sağlayıcı × privacy) için ayrı pinli tek-üyeli slot — hangi slotun
# aktif olduğu artık UI'dan seçilebilir (bkz. _session_provider_choice/chain_for).
# Mekanik: failover zinciri (ucuz → hızlı → geniş), UI'da hiç gösterilmez.
# PRIVATE mod: no-train/ZDR uçlar (paid Gemini no-train+DPA · Groq no-retention ·
# OpenRouter istek-bazlı ZDR zorlaması).
_JUDGE_SLOTS: tuple[ModelSlot, ...] = (
    JUDGE_SLOT,
    JUDGE_GROQ_SLOT,
    JUDGE_OPENROUTER_SLOT,
    JUDGE_NVIDIA_SLOT,
)
_MECHANICAL_SLOTS: tuple[ModelSlot, ...] = (
    MECH_GEMINI_SLOT,
    MECH_GROQ_SLOT,
    MECH_OPENROUTER_SLOT,
)
_PRIVATE_JUDGE_SLOTS: tuple[ModelSlot, ...] = (
    JUDGE_PRIVATE_SLOT,
    JUDGE_GROQ_PRIVATE_SLOT,
    JUDGE_OPENROUTER_PRIVATE_SLOT,
)
_PRIVATE_MECHANICAL_SLOTS: tuple[ModelSlot, ...] = (MECH_GROQ_SLOT,)

# Sağlayıcı adı -> slot eşlemesi (JUDGE'ın iki seviyeli UI seçicisi için).
_JUDGE_SLOTS_BY_PROVIDER: dict[str, ModelSlot] = {s.provider: s for s in _JUDGE_SLOTS}
_PRIVATE_JUDGE_SLOTS_BY_PROVIDER: dict[str, ModelSlot] = {
    s.provider: s for s in _PRIVATE_JUDGE_SLOTS
}
JUDGE_PROVIDER_CHOICES: tuple[str, ...] = tuple(_JUDGE_SLOTS_BY_PROVIDER)  # UI sırası


def judge_slots_for(privacy: PrivacyMode) -> dict[str, ModelSlot]:
    """UI için: privacy moduna göre sağlayıcı adı -> slot eşlemesi."""
    if privacy is PrivacyMode.PRIVATE:
        return dict(_PRIVATE_JUDGE_SLOTS_BY_PROVIDER)
    return dict(_JUDGE_SLOTS_BY_PROVIDER)


def option_ids(slot: ModelSlot) -> tuple[str, ...]:
    """Slotun küratörlü model ID'leri (UI/oturum doğrulaması için düz liste)."""
    return tuple(o.model_id for o in slot.options)


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
    if choice not in option_ids(slot):
        logger.warning("Oturumdaki model seçimi listede yok, yok sayıldı: %s=%s", slot.key, choice)
        return ""
    return choice


def _session_thinking_choice(slot: ModelSlot) -> ThinkingChoice | None:
    """UI'dan seçilen thinking seviyesi; yoksa None.

    `_session_choice` ile aynı desen: Streamlit yoksa veya listede olmayan bir
    değer varsa sessizce (uyarıyla) yok sayılır, uygulama çökmez.

    Dönüş tipi `str` değil `ThinkingChoice | None`: seçim zaten
    `slot.thinking_options` üyeliğiyle doğrulanıyor, yani daraltılmış tip
    çağıranın gördüğü gerçeği anlatıyor. Üyelik kontrolü `in` yerine döngüyle
    yapılıyor çünkü `in` tip daraltmaz — eşleşen seçeneği döndürmek `cast`
    ihtiyacını da ortadan kaldırıyor.
    """
    try:
        import streamlit as st

        choice = str(st.session_state.get(f"thinking_choice_{slot.key}", "")).strip()
    except Exception:
        return None
    if not choice:
        return None
    for option in slot.thinking_options:
        if option == choice:
            return option
    logger.warning("Oturumdaki thinking seçimi listede yok, yok sayıldı: %s=%s", slot.key, choice)
    return None


def _session_provider_choice(*, privacy: PrivacyMode) -> str:
    """Seçilen JUDGE sağlayıcısı: UI → `PARETO_JUDGE_PROVIDER` → boş string.

    `_session_choice` ile aynı desen: private<->public geçişinde eski oturum
    değeri o katmanda geçerli değilse sessizce yok sayılır (uygulama çökmez).

    Env halkası model ID'siyle simetri için var. `.env.example` çözüm sırasını
    zaten "uygulama içi seçim → bu dosya → st.secrets → koddaki default" diye
    belgeliyor ve `_resolve` model ID'sini böyle çözüyordu; sağlayıcı bu sıranın
    dışında kalmıştı. Streamlit olmayan bağlamlarda (scriptler, ör.
    `scripts/run_model_benchmark.py`) sağlayıcı başka türlü hedeflenemiyordu —
    zincir her zaman koddaki default'a düşüyordu.

    Doğrulama env yolunda da aynı: `judge_slots_for(privacy)` üyeliği aranır,
    yani PRIVATE modda yalnız o katmanın slotları seçilebilir ve
    `provider`/`api_key_env`/`no_train` her hâlükârda kodda pinli kalır.
    """
    choice = ""
    try:
        import streamlit as st

        choice = str(st.session_state.get("judge_provider_choice", "")).strip()
    except Exception:
        choice = ""
    source = "Oturumdaki"
    if not choice:
        choice = resolve_setting("PARETO_JUDGE_PROVIDER", "").strip()
        source = "PARETO_JUDGE_PROVIDER'daki"
    if not choice:
        return ""
    if choice not in judge_slots_for(privacy):
        logger.warning("%s judge sağlayıcı seçimi listede yok, yok sayıldı: %s", source, choice)
        return ""
    return choice


def _resolve(slot: ModelSlot, *, allow_session: bool) -> ProviderModel:
    """Slotu somut bir uca indirger. `model_id` ve `thinking` oturumdan seçilebilir;
    kimlik alanları (`provider`/`api_key_env`/`no_train`) hep koddan gelir."""
    model_id = (_session_choice(slot) if allow_session else "") or resolve_setting(
        slot.model_env, slot.default_model
    )
    thinking = (_session_thinking_choice(slot) if allow_session else None) or slot.default_thinking
    return ProviderModel(
        provider=slot.provider,
        model_id=model_id,
        api_key_env=slot.api_key_env,
        no_train=slot.no_train,
        thinking=thinking,
        extra_model_settings=slot.extra_model_settings,
    )


def chain_for(role: ModelRole, privacy: PrivacyMode) -> tuple[ProviderModel, ...]:
    """Rol + privacy moduna göre failover zincirini döndürür (fail-loud on private).

    Zincir çağrı anında kurulur: `.env` yükleme sırası ve oturum-içi UI seçimi
    ancak böyle yansır.

    JUDGE: sağlayıcı + model seçimi hem PUBLIC hem PRIVATE modda UI'dan okunur.
    Kullanıcı yalnız küratörlü slot kümesi içinden seçer — `provider`/`api_key_env`/
    `no_train` her zaman kodda pinli kalır, seçim asla bunları değiştiremez. Zincir
    JUDGE için her zaman tek üyelidir (pinli, failover yok) — seçilebilen şey hangi
    slotun pinli olduğudur, failover'a girip girmeyeceği değil.

    MECHANICAL: UI'da hiç gösterilmez. PUBLIC modda 3 sağlayıcılı failover zinciri;
    PRIVATE modda tek uca (Groq) sabittir. Anahtarı JUDGE'ın Groq slotuyla ORTAKTIR
    (`GROQ_API_KEY`): kullanıcı BYOK ile kendi Groq anahtarını verdiyse mekanik çağrı
    da onun hesabından gider — uç seçime kapalı, anahtar değil (bkz. PRIVACY.md).
    """
    if role is ModelRole.JUDGE:
        by_provider = judge_slots_for(privacy)
        default_provider = (
            JUDGE_PRIVATE_SLOT.provider if privacy is PrivacyMode.PRIVATE else JUDGE_SLOT.provider
        )
        provider_choice = _session_provider_choice(privacy=privacy) or default_provider
        slot = by_provider[provider_choice]
        # Açık anotasyon: ilk atama tek elemanlı olduğu için mypy `chain`i
        # `tuple[ProviderModel]` çıkarıyor, aşağıdaki değişken uzunluklu atama
        # da o dar tiple çakışıyordu.
        chain: tuple[ProviderModel, ...] = (_resolve(slot, allow_session=True),)
        if privacy is PrivacyMode.PRIVATE and any(not m.no_train for m in chain):
            raise RuntimeError("PRIVATE modda no-train olmayan uç seçilemez.")
        return chain

    if privacy is PrivacyMode.PRIVATE:
        chain = tuple(_resolve(s, allow_session=False) for s in _PRIVATE_MECHANICAL_SLOTS)
        if any(not m.no_train for m in chain):  # emniyet: private'da free-train sızmasın
            raise RuntimeError("PRIVATE modda no-train olmayan uç seçilemez.")
        return chain
    return tuple(_resolve(s, allow_session=True) for s in _MECHANICAL_SLOTS)
