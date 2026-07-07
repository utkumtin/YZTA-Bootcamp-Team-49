"""2 - Analiz: estimand-first akışı (YENİ MİMARİ)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pareto.streamlit_ui import render_compact_sidebar
from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    freeze_estimand,
    validate_estimand_spec_mapping,
    draft_tac_proposal,
)
from pareto.analysis.menu import (
    build_deterministic_menu,
    generate_spec_menu,
    freeze_spec_menu,
    expand_to_specs,
)


st.title("2 - Analysis (v2)")


# -----------------------------
# DATA LOADING
# -----------------------------

# -----------------------------
# DATA LOADING
# -----------------------------

def _df_from_state() -> pd.DataFrame | None:
    df = st.session_state.get("clean_df")
    return df if isinstance(df, pd.DataFrame) else None


def _guess(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    for c in columns:
        if any(k in c.lower() for k in keywords):
            return c
    return None


df = _df_from_state()

if df is None:
    st.warning("Veriseti bulunamadı. Lütfen önce Temizleme (Cleaning) adımını tamamlayın.")
    manual = st.text_area("Kolonları manuel olarak yapıştırın")
    columns = [c.strip() for c in manual.split(",") if c.strip()]
else:
    columns = [str(c) for c in df.columns]
    st.success(f"Yüklenen veriseti: {df.shape}")

if not columns:
    st.stop()




# -----------------------------
# ADIM 1: SOKRATİK BEYAN
# -----------------------------
# -----------------------------
# ADIM 1: SOKRATİK BEYAN
# -----------------------------

if "socratic_submitted" not in st.session_state:
    st.session_state.socratic_submitted = False

if not st.session_state.socratic_submitted:

    with st.form("socratic_form"):

        st.subheader("Adım 1: Sokratik Beyan")

        st.caption(
            "Bu aşamada veri setindeki kolonları düşünmeden yalnızca "
            "araştırma probleminizi kavramsal olarak tanımlayın."
        )

        research_story = st.text_area(
            "Araştırma Hikayesi",
            help="Araştırmanın bağlamını ve hipotezini açıklayın.",
        )

        conceptual_treatment = st.text_input(
            "Kavramsal Müdahale (Treatment)",
            value="",
            placeholder="Örn: Asgari ücret artışı",
        )

        conceptual_outcome = st.text_input(
            "Kavramsal Çıktı (Outcome)",
            value="",
            placeholder="Örn: İşsizlik oranı",
        )

        sign_map = {
            "positive": "Pozitif",
            "negative": "Negatif",
            "ambiguous": "Belirsiz",
        }

        expected_sign = st.selectbox(
            "Beklenen Etki Yönü",
            options=["positive", "negative", "ambiguous"],
            format_func=lambda x: sign_map[x],
        )

        submitted_socratic = st.form_submit_button(
            "Teknik Eşlemeye Geç"
        )

    if submitted_socratic:

        if (
            not research_story.strip()
            or not conceptual_treatment.strip()
            or not conceptual_outcome.strip()
        ):
            st.error(
                "Lütfen araştırma hikayesini, müdahaleyi ve çıktıyı doldurun."
            )
            st.stop()

        st.session_state["declaration_draft"] = SocraticDeclaration(
            conceptual_treatment=conceptual_treatment.strip(),
            conceptual_outcome=conceptual_outcome.strip(),
            expected_sign=expected_sign,
        )

        st.session_state["research_story_draft"] = research_story.strip()

        st.session_state.socratic_submitted = True

        st.rerun()

else:

    st.success("✓ Sokratik beyan tamamlandı.")

    with st.expander(
        "Sokratik Beyanı Görüntüle / Düzenle",
        expanded=False,
    ):

        decl = st.session_state["declaration_draft"]

        st.write(f"**Araştırma Hikayesi:** {st.session_state['research_story_draft']}")
        st.write(f"**Müdahale:** {decl.conceptual_treatment}")
        st.write(f"**Çıktı:** {decl.conceptual_outcome}")
        st.write(f"**Beklenen Yön:** {decl.expected_sign}")

        if st.button("Sokratik Beyanı Düzenle"):

            st.session_state.socratic_submitted = False
            st.session_state.pop("estimand_draft", None)

            st.rerun()
# -----------------------------
# ADIM 3: TEYİT EKRANI & DONDURMA
# -----------------------------

draft_proposal = st.session_state.get("estimand_draft")
frozen = st.session_state.get("frozen_estimand")

if draft_proposal and not frozen:
    if getattr(draft_proposal, "needs_clarification", False):
        st.warning("⚠️ Modelin eksik bilgiler için netleşmeye ihtiyacı var:")
        st.info(draft_proposal.clarification_question)
        
        with st.form("clarification_form"):
            clarity_answer = st.text_area("Yanıtınız / Ek Açıklama:")
            if st.form_submit_button("Açıklamayı Gönder ve Tekrar Öneri Al"):
                updated_story = st.session_state.get("last_research_story", "") + "\n\nKullanıcı Açıklaması: " + clarity_answer
                
                with st.spinner("Ek açıklamanızla yeniden değerlendiriliyor..."):
                    try:
                        new_proposal = draft_tac_proposal(
                            research_story=updated_story,
                            available_columns=columns,
                            declaration=st.session_state["last_declaration"],
                        )
                        st.session_state["estimand_draft"] = new_proposal
                        st.session_state["last_research_story"] = updated_story
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hata: {e}")
    else:
        st.info("Aşağıdaki TAC Proposal önerisini inceleyin ve onaylayın.")
        st.json(draft_proposal.model_dump())
        
        if st.button("Öneriyi Onayla ve Dondur"):
            frozen = freeze_estimand(draft_proposal, approved=True)
            st.session_state["frozen_estimand"] = frozen
            st.session_state["analysis_state"] = st.session_state["analysis_state_draft"]
            st.success("Estimand başarıyla donduruldu!")
            st.rerun()


# -----------------------------
# LOAD STATE
# -----------------------------

frozen = st.session_state.get("frozen_estimand")
state = st.session_state.get("analysis_state")

if frozen is None or state is None:
    st.stop()


st.subheader("Dondurulmuş Estimand")
st.json(frozen.estimand.model_dump())
st.code(f"hash={frozen.freeze_hash}")

# -----------------------------
# MENU BUILD (NEW FLOW)
# -----------------------------

unit_col = state["unit_col"]
time_col = state["time_col"]
cluster_by = state["cluster_by"]
controls = state["controls"]


# estimator selection
estimators = ["OLS"]
if unit_col and time_col:
    estimators.append("TWFE")


# -----------------------------
# OPTION 1: DETERMINISTIC MENU
# -----------------------------

menu = build_deterministic_menu(
    controls=controls,
    cluster_by=cluster_by,
    estimators=estimators,
    available_columns=columns,
)


# -----------------------------
# OPTION 2 (optional): LLM MENU
# -----------------------------
# Uncomment if you want LLM-based robustness menu
#
# proposal_menu = generate_spec_menu(
#     frozen=frozen,
#     available_columns=columns,
# )
#
# menu = freeze_spec_menu(
#     proposal_menu,
#     available_columns=columns,
#     approved=True,
# ).menu


# -----------------------------
# FREEZE MENU (deterministic hash)
# -----------------------------

frozen_menu = menu.freeze()
st.code(f"menu_hash={frozen_menu.menu_hash}")



# -----------------------------
# EXPAND SPECIFICATIONS
# -----------------------------

try:
    specs = expand_to_specs(
        frozen_menu,
        outcome=frozen.estimand.outcome,
        treatment=frozen.estimand.treatment_coding,
        unit_col=unit_col or cluster_by,
        time_col=time_col or cluster_by,
    )

    # validation against estimand
    found_warnings = []
    for spec in specs:
        spec_warnings = validate_estimand_spec_mapping(
            frozen,
            spec,
            available_columns=columns,
        )
        if spec_warnings:
            found_warnings.extend(spec_warnings)

    if found_warnings:
        with st.expander("⚠️ Spesifikasyon Eşleme Uyarıları (Yumuşatılmış Kurallar)", expanded=False):
            for w in set(found_warnings):
                st.warning(w)

except ValueError as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# OUTPUT
# -----------------------------

st.subheader("Spesifikasyon Seti")

df_specs = pd.DataFrame(
    [
        {
            **s.model_dump(),
            "hash": s.content_hash(),
        }
        for s in specs
    ]
)

st.dataframe(df_specs)


st.success(f"{len(specs)} spesifikasyon üretildi")