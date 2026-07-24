"""Router smoke testleri: cache, failover ve canlı sağlayıcı çağrıları.

Mekanik testler API yakmadan cache ve failover davranışını doğrular.
`live` işaretli testler yalnız ilgili BYOK anahtarı ortamda tanımlıysa koşar;
CI'da anahtar olmadığından otomatik atlanır.
"""

from __future__ import annotations

import os

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from pareto.config import ModelRole, PrivacyMode, load_dotenv_file
from pareto.llm.cache import CachedModel, cache_enabled
from pareto.llm.providers import (
    _PRIVATE_JUDGE_SLOTS,
    _PRIVATE_MECHANICAL_SLOTS,
    JUDGE_SLOT,
    _resolve,
    chain_for,
)
from pareto.llm.router import _chain_model, _model_from_provider

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
        "streamlit.session_state", {f"model_choice_{JUDGE_SLOT.key}": JUDGE_SLOT.options[0]}
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].model_id == JUDGE_SLOT.options[0]


def test_listede_olmayan_oturum_secimi_yok_sayilir(monkeypatch):
    """Seçenek listesi daraltıldığında eski oturum değeri uygulamayı çökertmemeli."""
    monkeypatch.setattr(
        "streamlit.session_state", {f"model_choice_{JUDGE_SLOT.key}": "uydurma-model-id"}
    )

    chain = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)

    assert chain[0].model_id == JUDGE_SLOT.default_model


def test_oturum_secimi_yalnizca_izin_verilen_cagride_okunur(monkeypatch):
    """`allow_session=False` (private mod) oturum seçimini görmezden gelmeli."""
    monkeypatch.setenv(JUDGE_SLOT.model_env, "env-pinli-model")
    monkeypatch.setattr(
        "streamlit.session_state", {f"model_choice_{JUDGE_SLOT.key}": JUDGE_SLOT.options[0]}
    )

    assert _resolve(JUDGE_SLOT, allow_session=True).model_id == JUDGE_SLOT.options[0]
    assert _resolve(JUDGE_SLOT, allow_session=False).model_id == "env-pinli-model"


def test_private_zincirdeki_slotlar_ui_secimine_kapali():
    """Private uçlar kullanıcı seçimine açılmaz: no-train garantisi ve paid anahtar
    deploy sahibinin kontrolünde kalır. (Groq slotu her iki zincirde de kullanılıyor.)"""
    for slot in _PRIVATE_JUDGE_SLOTS + _PRIVATE_MECHANICAL_SLOTS:
        assert slot.options == (), f"{slot.key} private zincirde ama UI'da seçilebilir"


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

    model = _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))

    assert isinstance(model, FallbackModel)


def test_zincirde_anahtari_eksik_uye_atlanir(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-anahtar")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    model = _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))

    assert not isinstance(model, FallbackModel), "tek kullanılabilir üye kaldıysa sarmalanmaz"


def test_zincirde_hic_anahtar_yoksa_fail_loud(monkeypatch):
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)

    with pytest.raises(OSError, match="eksik anahtarlar"):
        _chain_model(chain_for(ModelRole.MECHANICAL, PrivacyMode.PUBLIC))


# ---------------------------------------------------------------------------
# Canlı smoke — anahtar ortamda yoksa atlanır
# ---------------------------------------------------------------------------

_gemini_key_var = "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ
_groq_key_var = "GROQ_API_KEY" in os.environ


@pytest.mark.live
@pytest.mark.skipif(not _gemini_key_var, reason="GEMINI_API_KEY tanımlı değil")
def test_gemini_canli_smoke_deterministik(tmp_path):
    pm = chain_for(ModelRole.JUDGE, PrivacyMode.PUBLIC)[0]
    agent = Agent(
        CachedModel(_model_from_provider(pm), tmp_path),
        system_prompt="Sana verilen kelimeyi aynen tekrar et, başka hiçbir şey yazma.",
        model_settings={"temperature": 0.0},
    )

    first = agent.run_sync("Şu kelimeyi aynen tekrar et: PARETO")
    second = agent.run_sync("Şu kelimeyi aynen tekrar et: PARETO")

    assert "PARETO" in first.output.upper()
    assert first.output == second.output, "temp=0 + cache → birebir aynı yanıt"
    assert len(list(tmp_path.glob("*.json"))) == 1, "ikinci çağrı cache'ten dönmeli"


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

    result = agent.run_sync("Şu kelimeyi aynen tekrar et: MULTIVERSE")

    assert "MULTIVERSE" in result.output.upper()
