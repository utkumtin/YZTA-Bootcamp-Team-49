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


def _load_dotenv() -> None:
    """Repo kökündeki `.env` dosyasını yükle (Streamlit cwd'den bağımsız)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))


_load_dotenv()

_API_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "GEMINI_API_KEY": ("GOOGLE_API_KEY",),
}

BYOK_GEMINI_WIDGET_KEY = "byok_gemini_input"


def store_byok_key(env_name: str, value: str) -> None:
    """Sidebar BYOK anahtarını session_state'e yazar."""
    try:
        import streamlit as st
    except ImportError:
        return
    st.session_state.setdefault("byok_keys", {})[env_name] = value.strip()


def _from_streamlit_session(candidates: tuple[str, ...]) -> str:
    """BYOK oturum anahtarı (widget + byok_keys)."""
    try:
        import streamlit as st
    except ImportError:
        return ""

    byok = st.session_state.get("byok_keys")
    if isinstance(byok, dict):
        for name in candidates:
            key = str(byok.get(name, "")).strip()
            if key:
                return key

    widget_val = str(st.session_state.get(BYOK_GEMINI_WIDGET_KEY, "")).strip()
    if widget_val and "GEMINI_API_KEY" in candidates:
        return widget_val

    return ""


def _from_env(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


def _from_secrets(candidates: tuple[str, ...]) -> str:
    try:
        import streamlit as st
    except ImportError:
        return ""
    try:
        for name in candidates:
            if name in st.secrets:
                key = str(st.secrets[name]).strip()
                if key:
                    return key
    except Exception:
        pass
    return ""


def get_api_key_source(provider_env: str) -> str:
    """Anahtarın nereden geldiğini döndürür: byok | env | secrets."""
    candidates = (provider_env, *_API_KEY_ALIASES.get(provider_env, ()))
    if _from_streamlit_session(candidates):
        return "byok"
    if _from_env(candidates):
        return "env"
    if _from_secrets(candidates):
        return "secrets"
    return "none"


def get_api_key(provider_env: str) -> str:
    """BYOK oturum öncelikli; yoksa `.env`; son çare `st.secrets`."""
    candidates = (provider_env, *_API_KEY_ALIASES.get(provider_env, ()))

    key = _from_streamlit_session(candidates)
    if key:
        return key

    key = _from_env(candidates)
    if key:
        return key

    key = _from_secrets(candidates)
    if key:
        return key

    raise OSError(
        f"{provider_env} tanımlı değil. Şunlardan biriyle ayarlayın:\n"
        f"  • ana sayfa: **Anahtarı kaydet** (BYOK, oturum boyunca)\n"
        f"  • proje kökünde `.env`: {provider_env}=...\n"
        f"  • terminal: `export {provider_env}=...` (Streamlit'i yeniden başlat)"
    )
