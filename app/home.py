"""Ana sayfa: Genel Bakış + Ayarlar sekmeleri.

Oturum verisi `st.session_state` ile sayfalar arası kalır.
"""

from __future__ import annotations

import streamlit as st

from app.demo import DEMO_DATASET_PATH
from pareto.config import PrivacyMode
from pareto.profiling import load_raw_file, profile_dataframe
from pareto.streamlit_ui import (
    render_compact_sidebar,
    render_main_nav_style,
    render_session_overview,
    render_settings_panel,
)

# JUDGE_DEMO_SONNET_5_SLOT (bkz. pareto/llm/providers.py) — demo committed cache'i
# yalnız bu sağlayıcı için var. `.env`deki PARETO_JUDGE_PROVIDER'a güvenmiyoruz:
# boş/yanlış ayarlanırsa (ör. bu makinede olduğu gibi) zincir koddaki Gemini
# defaultuna düşer ve gerçek bir anahtar varsa canlı, başarısız bir çağrı yapar —
# demoyu sessizce kırar. Oturum içi seçim `_session_provider_choice`de env'den
# önce geldiği için burada zorlamak env'in durumundan bağımsız hep işe yarar.
_DEMO_JUDGE_PROVIDER = "demo_sonnet_5"

# `privacy_mode`/`judge_provider_choice` birer widget key'i: aşağıdaki sidebar/Ayarlar
# render'ından SONRA doğrudan atanırlarsa Streamlit `StreamlitAPIException` fırlatır
# ("cannot be modified after the widget... is instantiated"). Bu yüzden buton tıklamasında
# yalnız bir bayrak bırakılır, gerçek atama bir sonraki run'ın en başında (herhangi bir
# widget render olmadan önce) yapılır.
if st.session_state.pop("_demo_apply_provider_override", False):
    st.session_state["privacy_mode"] = PrivacyMode.PUBLIC.value
    st.session_state["judge_provider_choice"] = _DEMO_JUDGE_PROVIDER

# S3-05: Canned Mode bildirimi burada DEĞİL — `render_compact_sidebar()` içinde, her sayfada
# gösterilir, yalnız bu landing sayfasında değil.

with st.sidebar:
    render_compact_sidebar()

# Demo modu, oturumu committed medicaid panelinden başlatır — bir ziyaretçinin
# kendi verisini yüklemesi ya da API anahtarı girmesi gerekmez (bkz. app/demo.py,
# JUDGE_DEMO_SONNET_5_SLOT). Bilinçli olarak burada (app/), `pareto/`de değil —
# çekirdek hiçbir dataset adı/kolonu bilmemeli (bkz. tests/test_core_data_agnostic.py).
_DEMO_RESET_KEYS = (
    "clean_df",
    "clean_profile",
    "clean_df_raw",
    "ledger",
    "resolutions",
    "run_id",
    "last_script",
    "last_repro_dir",
    "last_audit_path",
    "last_ledger_path",
    "cleaning_uploaded_file_id",
    "cleaning_file_uploader",
    "cleaning_failed_file_id",
    "cleaning_failed_file_error",
    "socratic_submitted",
    "estimand_draft",
    "frozen_estimand",
    "last_research_story",
    "last_declaration",
    "declaration_draft",
    "research_story_draft",
    "analysis_state",
    "_spec_estimand_hash",
    "analysis_specs",
    "analysis_frozen_menu",
    "menu_proposal",
    "menu_proposal_model",
    "menu_approved_axes",
    "frozen_spec_menu",
    "multiverse_run_id",
    "multiverse_handle",
    "multiverse_results_path",
    "judge_provider_choice",
)

_DEMO_UPLOADED_FILE_ID = "demo-medicaid-panel"


def _render_demo_mode_entry() -> None:
    """Tek tıkla committed medicaid demo'suna gir/çık.

    Jüri gibi anahtarsız/veri yüklemeden gelen bir ziyaretçi için — Temizleme
    sayfasının dosya yükleyicisini atlar, `clean_df`'i doğrudan committed
    `DEMO_DATASET_PATH`'ten doldurur. Analiz sayfasındaki Sokratik form da
    `demo_mode` bayrağını görünce `app.demo` sabitleriyle önceden dolar.
    """
    if st.session_state.get("demo_mode"):
        st.success("Demo modu aktif — Medicaid genişleme veri seti yüklü.", icon=":material/movie:")
        if st.button("Demo modundan çık"):
            for key in _DEMO_RESET_KEYS:
                st.session_state.pop(key, None)
            st.session_state["demo_mode"] = False
            st.rerun()
        return

    st.info(
        "Kendi verinizi yüklemek ya da API anahtarı girmek istemiyorsanız, "
        "committed Medicaid genişleme veri setiyle hazır bir tanıtım turu yapın.",
        icon=":material/movie:",
    )
    if st.button("Demo moduna gir", type="primary"):
        for key in _DEMO_RESET_KEYS:
            st.session_state.pop(key, None)
        df = load_raw_file(DEMO_DATASET_PATH)
        st.session_state["clean_df"] = df
        st.session_state["clean_profile"] = profile_dataframe(df)
        st.session_state["cleaning_uploaded_file_id"] = _DEMO_UPLOADED_FILE_ID
        st.session_state["_demo_apply_provider_override"] = True
        st.session_state["demo_mode"] = True
        st.rerun()


_TAB_LABELS = [":material/bar_chart: Genel Bakış", ":material/settings: Ayarlar"]
tab_overview, tab_settings = st.tabs(_TAB_LABELS, key="main_tab", on_change="rerun")
render_main_nav_style()

with tab_overview:
    _render_demo_mode_entry()

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
