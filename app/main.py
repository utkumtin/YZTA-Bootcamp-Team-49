"""Pareto — Streamlit girişi (Streamlit + Community Cloud).

Ana sayfa: Ayarlar (Genel Bakış + Ayarlar sekmeleri). Diğer sayfalar: kompakt
sidebar (marka + gizlilik modu + oturum özeti).
Oturum verisi `st.session_state` ile sayfalar arası kalır.

Çalıştırma:  streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

from pareto.config import load_dotenv_file
from pareto.streamlit_ui import (
    render_compact_sidebar,
    render_main_nav_style,
    render_session_overview,
    render_settings_panel,
)

load_dotenv_file()

st.set_page_config(page_title="Pareto", page_icon="📊", layout="wide")

# S3-05: Canned Mode bildirimi artık burada DEĞİL — O4: `render_compact_sidebar()`
# içinde (`pareto/streamlit_ui.py::_render_canned_mode_banner`) her sayfada
# gösterilir, yalnız bu landing sayfasında değil.

with st.sidebar:
    render_compact_sidebar()

_TAB_LABELS = [":material/bar_chart: Genel Bakış", ":material/settings: Ayarlar"]
tab_overview, tab_settings = st.tabs(_TAB_LABELS, key="main_tab", on_change="rerun")
render_main_nav_style()

with tab_overview:
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

with tab_settings:
    render_settings_panel()