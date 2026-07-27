"""Router smoke testleri: cache, failover ve canlı sağlayıcı çağrıları.

Mekanik testler API yakmadan cache ve failover davranışını doğrular.
`live` işaretli testler yalnız ilgili BYOK anahtarı ortamda tanımlıysa koşar;
CI'da anahtar olmadığından otomatik atlanır. Anahtar varken sağlayıcı kotası
tükenirse (HTTP 429) test kırmızıya düşmez, atlanır.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from pareto.config import ModelRole, PrivacyMode, load_dotenv_file
from pareto.llm import cache as cache_module
from pareto.llm.cache import CachedModel, CannedModeCacheMissError, cache_enabled
from pareto.llm.providers import (
    _JUDGE_SLOTS,
    _MECHANICAL_SLOTS,
    _PRIVATE_JUDGE_SLOTS,
    _PRIVATE_MECHANICAL_SLOTS,
    JUDGE_GROQ_PRIVATE_SLOT,
    JUDGE_GROQ_SLOT,
    JUDGE_OPENROUTER_PRIVATE_SLOT,
    JUDGE_OPENROUTER_SLOT,
    JUDGE_SLOT,
    _resolve,
    chain_for,
)
from pareto.llm.router import _chain_model, _model_from_provider, _resolve_model

# BYOK anahtarları .env'de durabilir; canlı testlerin skip kararı öncesi yükle.
load_dotenv_file()

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def _counting_model(counter: dict[str, int]) -> FunctionModel:
    """Kaç kez çağrıldığını sayan sahte model."""

    def call(messages, info):  # noqa: ANN001 - pydantic-ai imzası
        counter["n"] += 1
        return ModelResponse(parts=[TextPart(content="sabit yanıt")])

    return FunctionModel(call)


# ---------------------------------------------------------------------------
# Cache — temp=0 yanıtları diskten döner, API'ye tekrar gidilmez
# ---------------------------------------------------------------------------


def test_cache_ayni_istegi_tek_api_cagrisina_indirger(tmp_path):
    counter = {"n": 0}
    agent = Agent(
        CachedModel(_counting_model(counter), tmp_path),
        model_settings={"temperature": 0.0},
    )

    first = agent.run_sync("merhaba")
    second = agent.run_sync("merhaba")

    assert counter["n"] == 1, "ikinci istek cache'ten dönmeliydi"
    assert first.output == second.output
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_farkli_promptlar_ayri_girdi_olur(tmp_path):
    counter = {"n": 0}
    agent = Agent(
        CachedModel(_counting_model(counter), tmp_path),
        model_settings={"temperature": 0.0},
    )

    agent.run_sync("birinci soru")
    agent.run_sync("ikinci soru")

    assert counter["n"] == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cache_farkli_thinking_ayri_girdi_olur(tmp_path):
    """thinking artık model_settings'in bir parçası; cache anahtarı bunu da içermeli —
    aksi halde bir kullanıcı thinking seviyesini değiştirdiğinde eski seviyenin
    cache'lenmiş yanıtı sessizce geri döner (bkz. ADR 0004, 2026-07-24 notu #2)."""
    counter = {"n": 0}
    agent = Agent(CachedModel(_counting_model(counter), tmp_path))

    agent.run_sync("merhaba", model_settings={"temperature": 0.0, "thinking": "low"})
    agent.run_sync("merhaba", model_settings={"temperature": 0.0, "thinking": "high"})
    agent.run_sync("merhaba", model_settings={"temperature": 0.0, "thinking": "low"})

    assert counter["n"] == 2, "farklı thinking seviyeleri ayrı model çağrısı üretmeli"
    assert len(list(tmp_path.glob("*.json"))) == 2, "thinking cache anahtarının parçası olmalı"


def test_cache_deterministik_olmayan_istegi_atlar(tmp_path):
    counter = {"n": 0}
    agent = Agent(
        CachedModel(_counting_model(counter), tmp_path),
        model_settings={"temperature": 0.7},
    )

    agent.run_sync("merhaba")
    agent.run_sync("merhaba")

    assert counter["n"] == 2, "temp != 0 iken her istek modele gitmeli"
    assert list(tmp_path.glob("*.json")) == []


def test_cache_bozuk_dosyayi_yeniden_uretir(tmp_path):
    counter = {"n": 0}
    agent = Agent(
        CachedModel(_counting_model(counter), tmp_path),
        model_settings={"temperature": 0.0},
    )

    agent.run_sync("merhaba")
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text("bozuk json {{{")

    agent.run_sync("merhaba")

    assert counter["n"] == 2, "bozuk cache sessizce yutulmamalı, yanıt yeniden üretilmeli"


def test_cache_env_bayragiyla_kapanir(monkeypatch):
    monkeypatch.setenv("PARETO_LLM_CACHE", "0")
    assert not cache_enabled()
    monkeypatch.setenv("PARETO_LLM_CACHE", "1")
    assert cache_enabled()


# ---------------------------------------------------------------------------
# Canned mode + cache miss — S3-05 review Sorun B
# ---------------------------------------------------------------------------


def test_cache_canned_modda_cache_miss_acik_hata_verir(tmp_path):
    """Canned mode'da (gerçek anahtar yok) cache miss olursa sarmalanan modele hiç
    gidilmemeli — aksi halde dummy key ile gerçek bir ağ isteği denenir ve kullanıcı
    anlamsız bir 401/auth hatası görür."""
    counter = {"n": 0}
    agent = Agent(
        CachedModel(_counting_model(counter), tmp_path, canned_mode=True),
        model_settings={"temperature": 0.0},
    )

    with pytest.raises(CannedModeCacheMissError, match="Canned mode"):
        agent.run_sync("merhaba")

    assert counter["n"] == 0, "canned modda sarmalanan modele hiç gidilmemeli"
    assert list(tmp_path.glob("*.json")) == [], "başarısız istek cache'e yazılmamalı"


def test_cache_canned_modda_cache_hit_hatasiz_doner(tmp_path):
    """Canned mode'da bile cache'te karşılığı olan bir istek normal şekilde dönmeli —
    frozen replay tam olarak bunun için var."""
    counter = {"n": 0}
    warm_agent = Agent(
        CachedModel(_counting_model(counter), tmp_path, canned_mode=False),
        model_settings={"temperature": 0.0},
    )
    warm_agent.run_sync("merhaba")
    assert counter["n"] == 1

    canned_agent = Agent(
        CachedModel(_counting_model(counter), tmp_path, canned_mode=True),
        model_settings={"temperature": 0.0},
    )
    result = canned_agent.run_sync("merhaba")

    assert result.output == "sabit yanıt"
    assert counter["n"] == 1, "cache hit'te gerçek modele hiç gidilmemeli"


def test_resolve_model_canned_modda_cache_miss_uctan_uca_hata_verir(monkeypatch, tmp_path):
    """`_resolve_model` → `CachedModel` zincirinin bütünü: hiç anahtar yokken ve
    cache'te karşılık yokken kullanıcı ham bir sağlayıcı hatası yerine açıklayıcı bir
    hata görmeli."""
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(
        cache_module, "SETTINGS", replace(cache_module.SETTINGS, llm_cache_dir=str(tmp_path))
    )
    monkeypatch.setattr("streamlit.session_state", {"privacy_mode": "public"})

    model, _extra = _resolve_model(ModelRole.JUDGE)
    agent = Agent(model, model_settings={"temperature": 0.0})

    with pytest.raises(CannedModeCacheMissError):
        agent.run_sync("merhaba")


# ---------------------------------------------------------------------------
# Failover — mekanik doğrulama (JUDGE pinli kalır)
# ---------------------------------------------------------------------------


def test_failover_birincil_dusunce_yedege_gecer():
    def patlayan(messages, info):  # noqa: ANN001
        raise ModelAPIError("birincil-model", "sağlayıcı erişilemez")

    fallback = FallbackModel(FunctionModel(patlayan), TestModel(custom_output_text="yedek yanıt"))
    result = Agent(fallback).run_sync("ping")

    assert result.output == "yedek yanıt"


def test_judge_zinciri_tek_uyeli_ve_pinli():
    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)
    assert len(chain) == 1, "JUDGE failover'a girmez, pinli tek model olmalı"
    assert chain[0].model_id == JUDGE_SLOT.default_model


# ---------------------------------------------------------------------------
# Model seçimi — .env / st.secrets / UI override'ları
# ---------------------------------------------------------------------------


def test_env_model_idyi_override_eder_kimligi_etmez(monkeypatch):
    """Yeni model çıkınca kod değil .env değişsin; ama sağlayıcı/anahtar/no-train
    env'e AÇILMAZ — bunlar gizlilik ve kimlik-doğrulama garantileri."""
    monkeypatch.setenv(JUDGE_SLOT.model_env, "gemini-test-9")

    uc = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)[0]

    assert uc.model_id == "gemini-test-9"
    assert uc.provider == JUDGE_SLOT.provider
    assert uc.api_key_env == JUDGE_SLOT.api_key_env
    assert uc.no_train is JUDGE_SLOT.no_train


def test_bos_env_koddaki_defaulta_duser(monkeypatch):
    monkeypatch.setenv(JUDGE_SLOT.model_env, "   ")

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].model_id == JUDGE_SLOT.default_model


def test_ui_secimi_env_pinini_gecer(monkeypatch):
    monkeypatch.setenv(JUDGE_SLOT.model_env, "env-pinli-model")
    monkeypatch.setattr(
        "streamlit.session_state",
        {f"model_choice_{JUDGE_SLOT.key}": JUDGE_SLOT.options[0].model_id},
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].model_id == JUDGE_SLOT.options[0].model_id


def test_listede_olmayan_oturum_secimi_yok_sayilir(monkeypatch):
    """Seçenek listesi daraltıldığında eski oturum değeri uygulamayı çökertmemeli."""
    monkeypatch.setattr(
        "streamlit.session_state", {f"model_choice_{JUDGE_SLOT.key}": "uydurma-model-id"}
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].model_id == JUDGE_SLOT.default_model


def test_resolve_allow_session_false_hala_gormezden_gelir(monkeypatch):
    """`_resolve(..., allow_session=False)` mekanizması değişmedi — session'ı görmezden gelir.

    Bu, MECHANICAL private zincirinin (ve JUDGE'ın PUBLIC'te kapalı, private'ta artık
    True ile çağrıldığı — bkz. `test_judge_private_modda_oturum_secimi_artik_okunur`)
    dayandığı temel davranış; `allow_session` parametresinin kendisi değişmedi, yalnız
    JUDGE artık private modda `True` ile çağrılıyor.
    """
    monkeypatch.setenv(JUDGE_SLOT.model_env, "env-pinli-model")
    monkeypatch.setattr(
        "streamlit.session_state",
        {f"model_choice_{JUDGE_SLOT.key}": JUDGE_SLOT.options[0].model_id},
    )

    assert _resolve(JUDGE_SLOT, allow_session=True).model_id == JUDGE_SLOT.options[0].model_id
    assert _resolve(JUDGE_SLOT, allow_session=False).model_id == "env-pinli-model"


def test_judge_private_modda_oturum_secimi_artik_okunur(monkeypatch):
    """JUDGE private modda artık UI seçimini okur (bilinçli gevşetme, bkz. ADR 0004 2026-07-24)."""
    monkeypatch.setattr(
        "streamlit.session_state",
        {
            "judge_provider_choice": "groq",
            f"model_choice_{JUDGE_GROQ_PRIVATE_SLOT.key}": (
                JUDGE_GROQ_PRIVATE_SLOT.options[0].model_id
            ),
        },
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PRIVATE)

    assert chain[0].provider == "groq"
    assert chain[0].model_id == JUDGE_GROQ_PRIVATE_SLOT.options[0].model_id
    assert chain[0].no_train is True


def test_private_mekanik_slotlar_ui_secimine_kapali():
    """Private MECHANICAL uçları kullanıcı seçimine açılmaz: no-train garantisi ve paid
    anahtar deploy sahibinin kontrolünde kalır. Bu kapsamda JUDGE'ın davranışı değişti
    (bkz. yukarısı); MECHANICAL değişmedi."""
    for slot in _PRIVATE_MECHANICAL_SLOTS:
        assert slot.options == (), f"{slot.key} private mekanik zincirde ama UI'da seçilebilir"


def test_private_judge_slotlari_kuratorlu_secenek_tasir():
    """JUDGE private slotları artık UI'da seçilebilir olmalı — boş `options` bir regresyon olur."""
    for slot in _PRIVATE_JUDGE_SLOTS:
        assert slot.options != (), f"{slot.key} private judge ama küratörlü liste boş"


def test_judge_provider_secimi_farkli_slota_yonlendirir(monkeypatch):
    monkeypatch.setattr("streamlit.session_state", {"judge_provider_choice": "openrouter"})

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].provider == "openrouter"
    assert chain[0].api_key_env == JUDGE_OPENROUTER_SLOT.api_key_env


def test_listede_olmayan_provider_secimi_yok_sayilir(monkeypatch):
    """Seçenek listesinde olmayan sağlayıcı seçimi varsayılan sağlayıcıya düşer."""
    monkeypatch.setattr("streamlit.session_state", {"judge_provider_choice": "uydurma-saglayici"})

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].provider == JUDGE_SLOT.provider


def test_provider_secimi_kimlik_alanlarini_asamaz():
    """Sağlayıcı seçilebilir olsa da provider/api_key_env/no_train hep koddaki slottan gelir."""
    for slot in (JUDGE_SLOT, JUDGE_GROQ_SLOT, JUDGE_OPENROUTER_SLOT):
        uc = _resolve(slot, allow_session=False)
        assert uc.provider == slot.provider
        assert uc.api_key_env == slot.api_key_env
        assert uc.no_train == slot.no_train


def test_judge_private_her_saglayicida_no_train_true(monkeypatch):
    for provider in ("google", "groq", "openrouter"):
        monkeypatch.setattr("streamlit.session_state", {"judge_provider_choice": provider})

        chain = chain_for(ModelRole.JUDGE, PrivacyMode.PRIVATE)

        assert chain[0].no_train is True, f"{provider}: private judge no_train=False olamaz"


def test_openrouter_private_judge_zdr_deklare_edilir():
    """ZDR deklarasyonu slotta tanımlı olmalı ve `_resolve` sonrasında da hayatta kalmalı."""
    zdr = {"openrouter_provider": {"zdr": True}}
    assert JUDGE_OPENROUTER_PRIVATE_SLOT.extra_model_settings == zdr

    uc = _resolve(JUDGE_OPENROUTER_PRIVATE_SLOT, allow_session=False)

    assert uc.extra_model_settings == zdr


def test_model_from_provider_openrouter_dogru_sinifi_kurar(monkeypatch):
    """`_model_from_provider` artık OpenRouter için genel string yerine tipli OpenRouterModel kurar
    (network çağrısı yok, yalnız construction — ZDR'ın taşınabilmesi buna dayanıyor)."""
    from pydantic_ai.models.openrouter import OpenRouterModel

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-anahtar")

    pm = _resolve(JUDGE_OPENROUTER_SLOT, allow_session=False)
    model = _model_from_provider(pm)

    assert isinstance(model, OpenRouterModel)


def test_resolve_model_extra_settings_openrouter_private_icin_dolu(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-anahtar")
    monkeypatch.setattr(
        "streamlit.session_state",
        {"privacy_mode": "private", "judge_provider_choice": "openrouter"},
    )

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {"openrouter_provider": {"zdr": True}}


def test_resolve_model_extra_settings_diger_slotlarda_bos(monkeypatch):
    """Groq judge'ın `default_thinking="off"` olması + extra_model_settings'i olmaması
    nedeniyle özel bir ayarı yok — bu, Gemini judge'ın thinking taşıdığı yeni
    davranıştan (bkz. aşağıdaki thinking testleri) ayrı, kasıtlı bir karşılaştırma."""
    monkeypatch.setenv("GROQ_API_KEY", "test-anahtar")
    monkeypatch.setattr(
        "streamlit.session_state",
        {"privacy_mode": "public", "judge_provider_choice": "groq"},
    )

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {}


def test_resolve_model_thinking_gemini_judge_icin_medium_akar(monkeypatch):
    """2a fix: JUDGE_SLOT.default_thinking="medium" artık extra_model_settings'e taşınıyor."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.setattr("streamlit.session_state", {"privacy_mode": "public"})

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {"thinking": "medium"}


def test_resolve_model_thinking_gemini_private_judge_icin_medium_akar(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_API_KEY", "test-anahtar")
    monkeypatch.setattr("streamlit.session_state", {"privacy_mode": "private"})

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {"thinking": "medium"}


def test_resolve_model_thinking_session_secimi_override_eder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.setattr(
        "streamlit.session_state",
        {"privacy_mode": "public", f"thinking_choice_{JUDGE_SLOT.key}": "high"},
    )

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {"thinking": "high"}


def test_resolve_model_thinking_off_secilirse_key_eklenmez(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.setattr(
        "streamlit.session_state",
        {"privacy_mode": "public", f"thinking_choice_{JUDGE_SLOT.key}": "off"},
    )

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == {}


def test_thinking_session_secimi_gecersizse_yok_sayilir(monkeypatch):
    """Seçenek listesinde olmayan bir thinking değeri uygulamayı çökertmemeli."""
    monkeypatch.setattr(
        "streamlit.session_state", {f"thinking_choice_{JUDGE_SLOT.key}": "uydurma-seviye"}
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].thinking == JUDGE_SLOT.default_thinking


def test_judge_slotlari_thinking_secenegi_tasir():
    for slot in _JUDGE_SLOTS + _PRIVATE_JUDGE_SLOTS:
        assert slot.thinking_options != (), f"{slot.key} JUDGE ama thinking seçeneği boş"


def test_mekanik_slotlarda_thinking_secimi_kapali():
    for slot in _MECHANICAL_SLOTS + _PRIVATE_MECHANICAL_SLOTS:
        assert slot.thinking_options == (), f"{slot.key} MECHANICAL ama thinking UI'da açık"


def test_env_override_private_no_train_garantisini_bozmaz(monkeypatch):
    for slot in _PRIVATE_JUDGE_SLOTS + _PRIVATE_MECHANICAL_SLOTS:
        monkeypatch.setenv(slot.model_env, "baska-bir-model")

    for role in (ModelRole.JUDGE, ModelRole.MECHANICAL):
        chain = chain_for(role, PrivacyMode.PRIVATE)
        assert chain, "private zincir boş olamaz"
        assert all(uc.no_train for uc in chain)


def test_mekanik_zincir_fallback_modele_indirgenir(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.setenv("GROQ_API_KEY", "test-anahtar")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-anahtar")

    model, canned_mode = _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))

    assert isinstance(model, FallbackModel)
    assert canned_mode is False, "en az bir üye gerçek anahtar buldu, canned mode olmamalı"


def test_zincirde_anahtari_eksik_uyeler_canned_key_ile_kurulur(monkeypatch):
    """S3-05: Eksik anahtarlar OSError fırlatmak yerine dummy key alarak cache'e hit
    edecekleri beklentisiyle başlatılır. Bu üye grubunda GEMINI_API_KEY gerçek olduğu
    için `canned_mode` False kalmalı — yalnız o üyenin görmediği bir gerçek anahtar
    yokluğu, zincirin tamamını canned moda düşürmez."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    model, canned_mode = _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))

    # Tüm modeller dummy key ile de olsa başlatıldığı için zincir korunur (FallbackModel)
    from pydantic_ai.models.fallback import FallbackModel

    assert isinstance(model, FallbackModel), (
        "eksik üyeler atlanmaz, dummy key ile zincire dahil edilir"
    )
    assert canned_mode is False, "GEMINI_API_KEY gerçek: zincir tamamen canned değil"


def test_zincirde_hic_anahtar_yoksa_canned_moda_duser(monkeypatch):
    """S3-05: Önceden OSError atan bu durum, artık modeli Canned Dummy Key ile başlatır
    ve akışı kesmez; hiçbir üye gerçek anahtar bulamadığı için `canned_mode` True olmalı."""
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)

    model, canned_mode = _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))

    assert model is not None, "Hiç anahtar yoksa bile Canned Mod için zincir kurulmalı"
    assert canned_mode is True, "Hiçbir üye gerçek anahtar bulamadı, canned mode açık olmalı"


def test_ozel_modda_ucretsiz_gemini_anahtariyla_istek_kurulmaz(monkeypatch):
    """Private modda yalnız free-tier anahtar varken sessiz fallback olmamalı."""
    monkeypatch.setenv("GEMINI_API_KEY", "ucretsiz-anahtar")
    monkeypatch.delenv("GEMINI_PAID_API_KEY", raising=False)
    monkeypatch.setattr("streamlit.session_state", {"privacy_mode": "private"})

    with pytest.raises(OSError, match="eksik anahtarlar"):
        _resolve_model(ModelRole.JUDGE)


# ---------------------------------------------------------------------------
# Canlı smoke — anahtar ortamda yoksa atlanır
# ---------------------------------------------------------------------------

_gemini_key_var = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
_groq_key_var = bool(os.environ.get("GROQ_API_KEY"))


def _run_or_skip(agent: Agent, prompt: str):
    """Kota tükenmesi sağlayıcı sınırıdır, kod hatası değil: yalnız 429 atlanır.

    Diğer tüm HTTP hataları (ör. thinking/structured-output uyumsuzluğunun döndürdüğü
    400) yükselmeye devam eder; testin koruduğu şey budur.
    """
    try:
        return agent.run_sync(prompt)
    except ModelHTTPError as exc:
        if exc.status_code == 429:
            pytest.skip(f"sağlayıcı kotası tükendi (HTTP {exc.status_code})")
        raise


@pytest.mark.live
@pytest.mark.skipif(not _gemini_key_var, reason="GEMINI_API_KEY tanımlı değil")
def test_gemini_canli_smoke_deterministik(tmp_path):
    pm = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)[0]
    agent = Agent(
        CachedModel(_model_from_provider(pm), tmp_path),
        system_prompt="Sana verilen kelimeyi aynen tekrar et, başka hiçbir şey yazma.",
        model_settings={"temperature": 0.0},
    )

    first = _run_or_skip(agent, "Şu kelimeyi aynen tekrar et: PARETO")
    second = _run_or_skip(agent, "Şu kelimeyi aynen tekrar et: PARETO")

    assert "PARETO" in first.output.upper()
    assert first.output == second.output, "temp=0 + cache → birebir aynı yanıt"
    assert len(list(tmp_path.glob("*.json"))) == 1, "ikinci çağrı cache'ten dönmeli"


@pytest.mark.live
@pytest.mark.skipif(not _gemini_key_var, reason="GEMINI_API_KEY tanımlı değil")
def test_gemini_canli_thinking_ile_yapili_cikti_uretir():
    """2a doğrulaması: JUDGE'ın gerçek kullanım şekli — `output_type` + `thinking`
    birlikte. pydantic-ai'nin bilinen thinking/structured-output uyumsuzluk
    raporlarına karşı gerçek bir güvence (bkz. GH pydantic/pydantic-ai#793, #2293)."""

    class _Cevap(BaseModel):
        kelime: str

    pm = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)[0]
    agent = Agent(
        _model_from_provider(pm),
        system_prompt="Kullanıcının verdiği kelimeyi `kelime` alanına aynen yaz.",
        model_settings={"temperature": 0.0, "thinking": "medium"},
        output_type=_Cevap,
    )

    result = _run_or_skip(agent, "PARETO")

    assert result.output.kelime.strip().upper() == "PARETO"


@pytest.mark.live
@pytest.mark.skipif(not _groq_key_var, reason="GROQ_API_KEY tanımlı değil")
def test_groq_canli_smoke_deterministik():
    # Private mekanik zincirin tek üyesi Groq ucudur; BYOK çözümü de böyle test edilir.
    pm = chain_for(ModelRole.MECHANICAL, PrivacyMode.PRIVATE)[0]
    agent = Agent(
        _model_from_provider(pm),
        system_prompt="Sana verilen kelimeyi aynen tekrar et, başka hiçbir şey yazma.",
        model_settings={"temperature": 0.0},
    )

    result = _run_or_skip(agent, "Şu kelimeyi aynen tekrar et: MULTIVERSE")

    assert "MULTIVERSE" in result.output.upper()
