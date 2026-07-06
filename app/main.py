"""Pareto — Streamlit girişi (Streamlit + Community Cloud).

Ana sayfa: BYOK (bir kez) + oturum durumu. Diğer sayfalar: kompakt sidebar.
Oturum verisi `st.session_state` ile sayfalar arası kalır.

Çalıştırma:  streamlit run app/main.py
"""

from __future__ import annotations

import os

# Windows: Turkce/Unicode LLM prompt'lari icin UTF-8 modu
os.environ.setdefault("PYTHONUTF8", "1")

import streamlit as st

from pareto.streamlit_ui import render_byok_panel, render_compact_sidebar, render_session_overview

st.set_page_config(page_title="Pareto", page_icon="📊", layout="wide")

with st.sidebar:
    render_compact_sidebar()

st.title("📊 Pareto")
st.caption(
    "Tek bir kesin cevap değil; savunulabilir seçimler üzerinde bir dağılım. "
    "Sözümüz: **savunulabilir sonuç.**"
)

render_byok_panel()
render_session_overview()

st.subheader("Akış")
st.markdown(
    "1. **Temizleme** — profil → karar defteri + üretilen kod (human-in-the-loop)\n"
    "2. **Analiz** — estimand/H0-H1 → savunulabilir spec menüsü (dondurulur)\n"
    "3. **Varyans Paneli** — çokluevren sonuçları: spec curve + 3-bant kırılganlık teşhisi"
)

st.info(
    "Veri, estimand ve spec çıktıları oturum boyunca saklanır — sayfa değiştirince kaybolmaz."
)
