"""Model Router — PydanticAI tabanlı, tipli I/O, test edilebilir.

review sorun #3'ün çözümü. Mekanik iş ucuz modele, yargı pinli güçlü modele
(providers.py zincirleri). Tipli çıktı: `output_type` bir Pydantic modeli olduğunda
PydanticAI şema-zorlaması + retry yapar → prototipteki regex-JSON ayıklama gitti.

Test: `use_test_model(...)` ile PydanticAI `TestModel`/`FunctionModel` enjekte edilir —
API yakmadan (test stratejisinin tamamı buna dayanıyor). Reprodüksiyon
dondurmadan gelir (menu.freeze), model stabilitesinden değil.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from ..config import ModelRole, PrivacyMode, SETTINGS, get_api_key
from .providers import ProviderModel, chain_for

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
    # Google dışı sağlayıcılarda da BYOK/.env anahtarını ortama pinle.
    os.environ[pm.api_key_env] = get_api_key(pm.api_key_env)
    # Diğer sağlayıcılar: "<provider>:<model_id>" (groq vb. optional extra gerekir)
    return f"{pm.provider}:{pm.model_id}"


def _get_effective_privacy_mode() -> PrivacyMode:
    """UI seçimi varsa kullan, yoksa varsayılan ayara dön."""
    try:
        import streamlit as st
    except ImportError:
        return SETTINGS.privacy_mode
    raw = st.session_state.get("privacy_mode", SETTINGS.privacy_mode.value)
    return PrivacyMode.PRIVATE if str(raw) == PrivacyMode.PRIVATE.value else PrivacyMode.PUBLIC


def _resolve_model(role: ModelRole) -> Any:
    """Test modeli varsa onu; yoksa zincirin ilk (birincil) modelini döndürür."""
    if _TEST_MODEL is not None:
        return _TEST_MODEL
    primary = chain_for(role, _get_effective_privacy_mode())[0]
    return _model_from_provider(primary)


def build_agent(role: ModelRole, *, system_prompt: str, output_type: Any | None = None):
    """Rol için tipli bir PydanticAI Agent kurar. output_type verilirse şema zorlanır."""
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # fail-loud, sessiz düşme yok
        raise RuntimeError(
            "pydantic-ai kurulu değil. `uv sync` / `pip install pydantic-ai` gerekli."
        ) from exc

    model = _resolve_model(role)
    kwargs: dict[str, Any] = {"system_prompt": system_prompt}
    if output_type is not None:
        kwargs["output_type"] = output_type
    return Agent(model, **kwargs)
