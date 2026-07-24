"""Streamlit ortak UI — BYOK (yalnız ana sayfa) + kompakt sidebar + oturum durumu."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from .config import get_api_key, get_api_key_source, resolve_setting
from .llm.providers import JUDGE_PRIVATE_SLOT, JUDGE_SLOT

BYOK_WIDGET_KEYS: dict[str, str] = {
    "GEMINI_API_KEY": "byok_gemini_input",
    "GROQ_API_KEY": "byok_groq_input",
    "OPENROUTER_API_KEY": "byok_openrouter_input",
}


def render_compact_sidebar() -> str:
    """Diğer sayfalarda: gizlilik modu + oturum özeti (BYOK formu yok)."""
    st.header("Oturum")
    mode = st.radio(
        "Gizlilik modu",
        options=["public", "private"],
        key="privacy_mode",
        help="public = free model + canned demo. private = yalnız no-train uçlar.",
    )

    _render_session_pills()
    _render_api_key_status()

    st.caption(f"Aktif mod: **{mode}**")
    return mode


def render_byok_panel() -> None:
    """Ana sayfada bir kez: sağlayıcı BYOK girişleri (session boyunca kalır)."""
    st.subheader("API anahtarı (BYOK)")
    st.caption(
        "Anahtarı burada bir kez kaydedin; diğer sayfalarda tekrar girmeniz gerekmez. "
        "Hem `.env` hem BYOK doluysa **BYOK önceliklidir**."
    )

    with st.form("byok_form", clear_on_submit=False):
        st.text_input("Gemini API key", type="password", key=BYOK_WIDGET_KEYS["GEMINI_API_KEY"])
        st.text_input("Groq API key", type="password", key=BYOK_WIDGET_KEYS["GROQ_API_KEY"])
        st.text_input(
            "OpenRouter API key", type="password", key=BYOK_WIDGET_KEYS["OPENROUTER_API_KEY"]
        )
        saved = st.form_submit_button("Anahtarları kaydet", type="primary")

    if saved:
        byok_keys = st.session_state.setdefault("byok_keys", {})
        saved_any = False
        for env_name, widget_key in BYOK_WIDGET_KEYS.items():
            raw = str(st.session_state.get(widget_key, "")).strip()
            if raw:
                byok_keys[env_name] = raw
                os.environ[env_name] = raw
                saved_any = True
        if saved_any:
            st.success("Anahtarlar oturuma kaydedildi — tüm sayfalarda geçerli.")
        else:
            st.warning("Kaydedilecek anahtar bulunamadı.")

    if st.session_state.get("byok_keys") and st.button(
        "Tüm BYOK anahtarlarını temizle", key="byok_clear_all"
    ):
        for env_name in BYOK_WIDGET_KEYS:
            os.environ.pop(env_name, None)
        st.session_state["byok_keys"] = {}
        for widget_key in BYOK_WIDGET_KEYS.values():
            st.session_state.pop(widget_key, None)
        st.rerun()

    _render_model_choice()
    _render_api_key_status(detailed=True)


def _render_model_choice() -> None:
    """Yargı modeli seçimi — küratörlü liste (serbest metin yok).

    Seçim `os.environ`'a YAZILMAZ (anahtar akışının aksine): `GROQ_MECHANICAL_MODEL`
    gibi değişkenler private zinciri de besliyor; env'e yazmak kullanıcı seçimini
    private uçlara sızdırırdı. Widget değeri oturumda kalır, zincir kurulurken
    yalnız public modda okunur (`llm/providers.py: chain_for`).
    """
    pinned = resolve_setting(JUDGE_SLOT.model_env, JUDGE_SLOT.default_model)
    options = list(JUDGE_SLOT.options)
    if pinned not in options:  # operatörün .env pini listede yoksa da görünsün
        options.insert(0, pinned)

    st.subheader("Model seçimi")
    st.selectbox(
        "Yargı modeli",
        options=options,
        index=options.index(pinned),
        key=f"model_choice_{JUDGE_SLOT.key}",
        help=(
            "Estimand, spec menüsü, temizleme önerisi ve varyans anlatısı bu modelle üretilir. "
            "Yalnız **public** modu etkiler; private mod modeli deploy sahibinin "
            f"`{JUDGE_PRIVATE_SLOT.model_env}` ayarına bağlıdır."
        ),
    )


def render_session_overview() -> None:
    """Ana sayfada: yüklü veri / estimand / spec özeti."""
    st.subheader("Oturum durumu")
    cols = st.columns(3)
    with cols[0]:
        df = st.session_state.get("clean_df")
        if df is not None:
            st.success(f"Veri: **{df.shape[0]}** satır × **{df.shape[1]}** kolon")
        else:
            st.info("Veri: henüz yüklenmedi")
    with cols[1]:
        frozen = st.session_state.get("frozen_estimand")
        if frozen is not None:
            st.success(f"Estimand: donduruldu (`{frozen.freeze_hash[:8]}…`)")
        else:
            st.info("Estimand: yok")
    with cols[2]:
        specs = st.session_state.get("analysis_specs")
        if specs:
            st.success(f"Spesifikasyon: **{len(specs)}** adet")
        else:
            st.info("Spesifikasyon: yok")


def _render_session_pills() -> None:
    if st.session_state.get("clean_df") is not None:
        df = st.session_state["clean_df"]
        st.caption(f"Veri: {df.shape[0]}×{df.shape[1]}")
    if st.session_state.get("frozen_estimand") is not None:
        h = st.session_state["frozen_estimand"].freeze_hash[:8]
        st.caption(f"Estimand: `{h}…`")
    specs = st.session_state.get("analysis_specs")
    if specs:
        st.caption(f"Specs: {len(specs)}")


def _render_api_key_status(*, detailed: bool = False) -> None:
    byok_keys = st.session_state.get("byok_keys", {})
    providers = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")
    for provider in providers:
        byok_value = str(byok_keys.get(provider, "")).strip() if isinstance(byok_keys, dict) else ""
        if byok_value:
            st.caption(
                f"{provider}: **algılandı** ✓ — kaynak: **BYOK (oturum)** ({len(byok_value)} kar.)"
            )
            continue
        try:
            key = get_api_key(provider)
            source = get_api_key_source(provider)
            label = {"env": ".env", "secrets": "st.secrets", "none": "yok"}[source]
            st.caption(f"{provider}: **algılandı** ✓ — kaynak: **{label}** ({len(key)} kar.)")
        except OSError:
            msg = f"{provider}: **yok** — ana sayfada BYOK kaydet veya `.env` kullan"
            if detailed:
                st.warning(msg)
            else:
                st.caption(msg)


def render_clean_panel(df_before: pd.DataFrame, df_after: pd.DataFrame, script: str) -> None:
    """Uygulanan temizlik özeti: satır/kolon/eksik değişimi + üretilen script."""
    st.subheader("CleanPanel — uygulanan temizlik")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Satır", df_after.shape[0], delta=df_after.shape[0] - df_before.shape[0])
    with c2:
        st.metric("Kolon", df_after.shape[1], delta=df_after.shape[1] - df_before.shape[1])
    with c3:
        missing_before = int(df_before.isna().sum().sum())
        missing_after = int(df_after.isna().sum().sum())
        st.metric("Toplam eksik", missing_after, delta=missing_after - missing_before)
    with st.expander("Üretilen script (reprodüksiyon)", expanded=False):
        st.code(script, language="python")
