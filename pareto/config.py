"""Pareto genel ayarları — tek kaynak.

Model-router rolleri, privacy modu, sert spec tavanı,
determinizm pinleri. Sırlar buraya YAZILMAZ; yalnız oturum BYOK'u/env/
`st.secrets` üzerinden okunur.
"""

from __future__ import annotations

import logging
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
    # --- Model router. Model adları/slotları providers.py'de (tek kaynak). ---
    llm_temperature: float = 0.0  # deterministik → reprodüksiyon + cache

    # --- Privacy ---
    privacy_mode: PrivacyMode = PrivacyMode.PUBLIC

    # --- Gatekeeper eşikleri (temizleme human-in-the-loop) ---
    missing_value_hard_threshold: float = 0.5  # %50+ eksik → her zaman insana sor
    auto_approve_low_risk: bool = True

    # --- Multiverse sınırları (sert tavan 24) ---
    max_specifications: int = 24  # aşımda uyar + logla, sessiz kırpma yok
    default_weight_col: str = "population"

    # --- Determinizm (seed + BLAS/hash pin) ---
    seed: int = 20260704
    deterministic_env: dict[str, str] = field(
        default_factory=lambda: {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    )

    # --- Çalışma dizinleri (lokal artifacts, phone-home yok) ---
    runs_dir: str = "runs"
    store_dir: str = "runs/store"
    audit_trail_dir: str = "runs/audit_trail"
    llm_cache_dir: str = "runs/llm_cache"  # temp=0 yanıt cache'i (free-tier RPM azaltır)


SETTINGS = ParetoSettings()
logger = logging.getLogger(__name__)


def load_dotenv_file() -> None:
    """Repo kökündeki `.env` dosyasını yükle (Streamlit cwd'den bağımsız)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv bulunamadı, .env yüklenmedi.")
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))


_API_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "GEMINI_API_KEY": ("GOOGLE_API_KEY",),
}


def _from_env(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


_secrets_warned = False


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
    except Exception as exc:
        # Süreç başına bir kez uyar: her anahtar/model slotu için tekrarlanırsa
        # (zincir kurulumu başına 5+) log okunmaz hale gelir.
        global _secrets_warned
        if not _secrets_warned:
            logger.warning("st.secrets okunamadı: %s", exc)
            _secrets_warned = True
    return ""


def _from_session_byok(candidates: tuple[str, ...]) -> str:
    """Oturumdaki BYOK anahtarını oku — yalnız ilgili switch açıksa (veya hiç
    ayarlanmamışsa, geriye dönük uyumluluk için varsayılan AÇIK).

    `_from_secrets` ile aynı desen: Streamlit yoksa veya session_state'e
    erişim başarısızsa (headless/test bağlamı) sessizce boş döner —
    `resolve_api_key` env/secrets'a düşsün.
    """
    try:
        import streamlit as st
    except ImportError:
        return ""
    try:
        byok_keys = st.session_state.get("byok_keys", {})
        if not isinstance(byok_keys, dict):
            return ""
        for name in candidates:
            value = str(byok_keys.get(name, "")).strip()
            if value and st.session_state.get(f"byok_enabled_{name}", True):
                return value
    except Exception:
        return ""
    return ""


def resolve_api_key(provider_env: str) -> tuple[str, str]:
    """API anahtarını çöz ve kaynağı döndür: byok | env | secrets | none."""
    candidates = (provider_env, *_API_KEY_ALIASES.get(provider_env, ()))
    byok_key = _from_session_byok(candidates)
    if byok_key:
        return byok_key, "byok"
    env_key = _from_env(candidates)
    if env_key:
        return env_key, "env"
    secret_key = _from_secrets(candidates)
    if secret_key:
        return secret_key, "secrets"
    return "", "none"


def get_api_key(provider_env: str) -> str:
    """API anahtarını döndür.

    Gerçek anahtar yoksa constructor'ın kurulabilmesi için dummy key döndürülür.
    Çağrının gerçekten canned mode'da olup olmadığı resolve_api_key() üzerinden
    takip edilir.
    """
    key, _source = resolve_api_key(provider_env)

    if key:
        return key

    # S3-05: constructor'ın kurulabilmesi için dummy key.
    # Gerçek ağ isteği CachedModel tarafından canned_mode'da engellenir.
    return "PARETO_CANNED_MODE_DUMMY_KEY"


def resolve_setting(env_name: str, default: str) -> str:
    """Sır olmayan bir ayarı çöz: env → `st.secrets` → kod defaultu.

    `resolve_api_key`'in kardeşi; aynı arama sırasını kullanır ama eksiklik
    hata değildir — model ID'si gibi ayarlarda defaulta düşmek doğru davranış.
    Boş/whitespace değer "tanımsız" sayılır.
    """
    candidates = (env_name,)
    return _from_env(candidates) or _from_secrets(candidates) or default
