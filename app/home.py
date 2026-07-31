"""Ana sayfa: Genel Bakış + Ayarlar sekmeleri.

Oturum verisi `st.session_state` ile sayfalar arası kalır.
"""

from __future__ import annotations

import streamlit as st

from pareto.streamlit_ui import (
    render_compact_sidebar,
    render_main_nav_style,
    render_session_overview,
    render_settings_panel,
)

# S3-05: Canned Mode bildirimi burada DEĞİL — `render_compact_sidebar()` içinde, her sayfada
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
