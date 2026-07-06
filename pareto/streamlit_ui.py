"""Streamlit ortak UI — BYOK (yalnız ana sayfa) + kompakt sidebar + oturum durumu."""

from __future__ import annotations

import streamlit as st

from .config import BYOK_GEMINI_WIDGET_KEY, get_api_key, get_api_key_source, store_byok_key


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
    """Ana sayfada bir kez: Gemini BYOK girişi (session boyunca kalır)."""
    st.subheader("API anahtarı (BYOK)")
    st.caption(
        "Anahtarı burada bir kez kaydedin; diğer sayfalarda tekrar girmeniz gerekmez. "
        "Hem `.env` hem BYOK doluysa **BYOK önceliklidir**."
    )

    with st.form("byok_gemini_form", clear_on_submit=False):
        st.text_input(
            "Gemini API key",
            type="password",
            key=BYOK_GEMINI_WIDGET_KEY,
            help="Yapıştır → **Anahtarı kaydet**.",
        )
        saved = st.form_submit_button("Anahtarı kaydet", type="primary")

    if saved:
        raw = str(st.session_state.get(BYOK_GEMINI_WIDGET_KEY, "")).strip()
        if raw:
            store_byok_key("GEMINI_API_KEY", raw)
            st.success("Anahtar oturuma kaydedildi — tüm sayfalarda geçerli.")
        else:
            st.warning("Anahtar boş.")

    if st.session_state.get("byok_keys", {}).get("GEMINI_API_KEY") and st.button(
        "Anahtarı temizle", key="byok_clear_gemini"
    ):
        st.session_state.get("byok_keys", {}).pop("GEMINI_API_KEY", None)
        st.session_state.pop(BYOK_GEMINI_WIDGET_KEY, None)
        st.rerun()

    _render_api_key_status(detailed=True)


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
    try:
        key = get_api_key("GEMINI_API_KEY")
        source = get_api_key_source("GEMINI_API_KEY")
        label = {"byok": "BYOK (oturum)", "env": ".env", "secrets": "st.secrets"}[source]
        st.caption(f"Gemini: **algılandı** ✓ — kaynak: **{label}** ({len(key)} kar.)")
    except OSError:
        msg = "Gemini: **yok** — ana sayfada BYOK kaydet veya `.env` kullan"
        if detailed:
            st.warning(msg)
        else:
            st.caption(msg)
