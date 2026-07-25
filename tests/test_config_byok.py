"""`resolve_api_key`in BYOK-switch farkındalığı: sağlayıcı başına açma/kapama.

Sidebar/Ayarlar redesign'ının çekirdek backend kuralı — BYOK artık `os.environ`'a
yazılmıyor, yalnız `st.session_state["byok_keys"]` + `byok_enabled_{env_name}`
switch'i üzerinden `resolve_api_key` içinde çözülüyor. `st.secrets` yolu bu
PR'da değişmedi, burada test edilmiyor.
"""

from __future__ import annotations

from pareto.config import resolve_api_key


def test_byok_switch_acikken_env_i_gecersiz_kilar(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-anahtari")
    monkeypatch.setattr(
        "streamlit.session_state",
        {
            "byok_keys": {"GEMINI_API_KEY": "byok-anahtari"},
            "byok_enabled_GEMINI_API_KEY": True,
        },
    )

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("byok-anahtari", "byok")


def test_byok_switch_kapaliyken_env_e_duser(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-anahtari")
    monkeypatch.setattr(
        "streamlit.session_state",
        {
            "byok_keys": {"GEMINI_API_KEY": "byok-anahtari"},
            "byok_enabled_GEMINI_API_KEY": False,
        },
    )

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("env-anahtari", "env")


def test_byok_switch_anahtari_hic_yoksa_varsayilan_acik(monkeypatch):
    """İlk kayıttan önce switch widget'ı hiç render olmamışsa session'da anahtar
    hiç yoktur — geriye dönük uyumluluk için varsayılan davranış AÇIK olmalı."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-anahtari")
    monkeypatch.setattr(
        "streamlit.session_state",
        {"byok_keys": {"GEMINI_API_KEY": "byok-anahtari"}},
    )

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("byok-anahtari", "byok")


def test_byok_kayitli_degilken_dogrudan_env_e_duser(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-anahtari")
    monkeypatch.setattr("streamlit.session_state", {})

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("env-anahtari", "env")


def test_hicbiri_tanimli_degilken_none_doner(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("streamlit.session_state", {})

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("", "none")


def test_gemini_yoksa_google_api_key_aliasina_duser(monkeypatch):
    """GEMINI_API_KEY tanımsızken _API_KEY_ALIASES üzerinden GOOGLE_API_KEY'e
    düşülmeli — config.py'deki alias zincirinin pozitif yolu daha önce test
    edilmiyordu (yalnız 'hiçbiri yokken' negatif durumu vardı)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-anahtari")
    monkeypatch.setattr("streamlit.session_state", {})

    key, source = resolve_api_key("GEMINI_API_KEY")

    assert (key, source) == ("google-anahtari", "env")
