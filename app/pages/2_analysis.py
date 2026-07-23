"""2 - Analiz: estimand-first akışı (YENİ MİMARİ)."""

from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

from pareto.analysis.runner import launch_multiverse
from pareto.memory.store import ProjectStore
from pareto.streamlit_ui import render_compact_sidebar

from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    draft_tac_proposal,
    freeze_estimand,
    validate_estimand_spec_mapping,
)

from pareto.analysis.menu import (
    ALL_AXES,
    SpecMenu,
    build_deterministic_menu,
    evaluate_menu_defensibility,
    expand_to_specs,
    freeze_spec_menu,
    generate_spec_menu,
    validate_spec_menu_to_specs,
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    render_compact_sidebar()


st.title("2 - Analiz (v2)")


def _persist_frozen_menu(*, frozen_estimand, frozen_menu, specs, run_id: str | None = None) -> None:
    store = ProjectStore(project_id="analysis")
    store.save(
        "frozen_menu",
        {
            "estimand_hash": frozen_estimand.freeze_hash,
            "menu_hash": frozen_menu.menu_hash,
            "spec_count": len(specs),
            "run_id": run_id,
            "estimand": frozen_estimand.estimand.model_dump(),
            "menu": frozen_menu.model_dump(),
        },
    )


def _preview_menu_status(proposal, *, available_columns, outcome, treatment, unit_col, time_col) -> tuple[bool, list[str], int]:
    return evaluate_menu_defensibility(
        proposal,
        available_columns=available_columns,
        outcome=outcome,
        treatment=treatment,
        unit_col=unit_col,
        time_col=time_col,
    )


@st.fragment
def _render_multiverse_progress(handle) -> None:
    progress = handle.read_progress()
    total = max(int(progress.get("total", 0) or 0), 1)
    done = int(progress.get("done", 0) or 0)

    with st.status("Multiverse çalışıyor…", expanded=True):
        st.write(f"Run: {handle.run_dir.name}")
        st.progress(done / total)
        st.caption(f"{done}/{total} spesifikasyon işlendi")

        if handle.is_done():
            st.success("Multiverse tamamlandı.")
            results_path = str(handle.results_path)
            st.caption(f"Sonuçlar: {results_path}")
            st.page_link("app/pages/3_variance_panel.py", label="Varyans panelini aç")
        else:
            time.sleep(1)
            st.rerun()


# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------

def _df_from_state() -> pd.DataFrame | None:
    df = st.session_state.get("clean_df")
    return df if isinstance(df, pd.DataFrame) else None


df = _df_from_state()


if df is None:
    st.warning(
        "Veriseti bulunamadı. "
        "Lütfen önce Temizleme (Cleaning) adımını tamamlayın."
    )

    manual = st.text_area(
        "Kolonları manuel olarak girin (virgülle ayırın)"
    )

    columns = [
        c.strip()
        for c in manual.split(",")
        if c.strip()
    ]

else:
    columns = [
        str(c)
        for c in df.columns
    ]

    st.success(
        f"Yüklenen veriseti: {df.shape}"
    )


if not columns:
    st.stop()



# -------------------------------------------------
# SESSION DEFAULTS
# -------------------------------------------------

defaults = {
    "socratic_submitted": False,
    "estimand_draft": None,
    "frozen_estimand": None,
    "last_research_story": "",
    "last_declaration": None,
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value



# -------------------------------------------------
# ADIM 1
# SOCRATIC DECLARATION
# -------------------------------------------------

if not st.session_state.socratic_submitted:

    with st.form(
        "socratic_form"
    ):

        st.subheader(
            "Adım 1: Sokratik Beyan"
        )

        st.caption(
            "Bu aşamada veri kolonlarını düşünmeden "
            "araştırma problemini kavramsal olarak tanımlayın."
        )


        research_story = st.text_area(
            "Araştırma Hikayesi",
            help=(
                "Araştırmanın bağlamını, problemini "
                "ve beklenen mekanizmayı açıklayın."
            ),
        )


        conceptual_treatment = st.text_input(
            "Kavramsal Müdahale (Treatment)",
            placeholder=(
                "Örn: Eğitim programına katılım"
            ),
        )


        conceptual_outcome = st.text_input(
            "Kavramsal Çıktı (Outcome)",
            placeholder=(
                "Örn: Gelir seviyesi"
            ),
        )


        expected_sign = st.selectbox(
            "Beklenen Etki Yönü",
            options=[
                "positive",
                "negative",
                "ambiguous",
            ],
            format_func=lambda x: {
                "positive": "Pozitif",
                "negative": "Negatif",
                "ambiguous": "Belirsiz",
            }[x],
        )


        submitted = st.form_submit_button(
            "Teknik Eşlemeye Geç"
        )


    if submitted:

        if (
            not research_story.strip()
            or not conceptual_treatment.strip()
            or not conceptual_outcome.strip()
        ):

            st.error(
                "Araştırma hikayesi, müdahale ve çıktı zorunludur."
            )

            st.stop()



        declaration = SocraticDeclaration(
            conceptual_treatment=(
                conceptual_treatment.strip()
            ),
            conceptual_outcome=(
                conceptual_outcome.strip()
            ),
            expected_sign=expected_sign,
        )


        st.session_state[
            "declaration_draft"
        ] = declaration


        st.session_state[
            "research_story_draft"
        ] = research_story.strip()


        st.session_state[
            "last_research_story"
        ] = research_story.strip()


        st.session_state[
            "last_declaration"
        ] = declaration


        st.session_state.socratic_submitted = True


        st.rerun()



# -------------------------------------------------
# SOCRATIC SUMMARY
# -------------------------------------------------

else:

    st.success(
        "✓ Sokratik beyan tamamlandı."
    )


    with st.expander(
        "Sokratik Beyanı Görüntüle / Düzenle"
    ):

        declaration = (
            st.session_state["declaration_draft"]
        )


        st.write(
            "**Araştırma Hikayesi:** "
            f"{st.session_state['research_story_draft']}"
        )

        st.write(
            "**Müdahale:** "
            f"{declaration.conceptual_treatment}"
        )

        st.write(
            "**Çıktı:** "
            f"{declaration.conceptual_outcome}"
        )

        st.write(
            "**Beklenen yön:** "
            f"{declaration.expected_sign}"
        )


        if st.button(
            "Sokratik Beyanı Düzenle"
        ):

            st.session_state.socratic_submitted = False

            st.session_state.pop(
                "estimand_draft",
                None,
            )

            st.session_state.pop(
                "frozen_estimand",
                None,
            )

            st.rerun()



# -------------------------------------------------
# ADIM 2
# TAC PROPOSAL
# -------------------------------------------------

draft_proposal = st.session_state.get(
    "estimand_draft"
)

frozen_estimand = st.session_state.get(
    "frozen_estimand"
)



if (
    st.session_state.socratic_submitted
    and draft_proposal is None
    and frozen_estimand is None
):

    st.subheader(
        "Adım 2: Teknik Eşleme Önerisi"
    )


    st.caption(
        "Sistem, kavramsal beyanı veri kolonları ile "
        "eşleyerek bir estimand önerisi oluşturur."
    )


    if st.button(
        "TAC Proposal Oluştur",
        type="primary",
    ):

        try:

            with st.spinner(
                "Teknik eşleme hazırlanıyor..."
            ):

                proposal = draft_tac_proposal(
                    research_story=(
                        st.session_state[
                            "last_research_story"
                        ]
                    ),
                    available_columns=columns,
                    declaration=(
                        st.session_state[
                            "last_declaration"
                        ]
                    ),
                )


                st.session_state[
                    "estimand_draft"
                ] = proposal


                st.rerun()


        except Exception as exc:

            st.error(
                f"TAC proposal oluşturulamadı: {exc}"
            )



# -------------------------------------------------
# TAC CLARIFICATION
# -------------------------------------------------

draft_proposal = st.session_state.get(
    "estimand_draft"
)


if (
    draft_proposal is not None
    and not frozen_estimand
):

    if getattr(draft_proposal, "needs_clarification", False):

        st.warning(
            "⚠️ Teknik eşleme için ek açıklama gerekiyor."
        )


        st.info(
            draft_proposal.clarification_question
        )


        with st.form(
            "clarification_form"
        ):

            answer = st.text_area(
                "Ek açıklama"
            )


            submit_answer = st.form_submit_button(
                "Tekrar değerlendir"
            )


        if submit_answer:

            updated_story = (
                st.session_state[
                    "last_research_story"
                ]
                + "\n\nKullanıcı açıklaması:\n"
                + answer
            )


            try:

                proposal = draft_tac_proposal(
                    research_story=updated_story,
                    available_columns=columns,
                    declaration=(
                        st.session_state[
                            "last_declaration"
                        ]
                    ),
                )


                st.session_state[
                    "estimand_draft"
                ] = proposal


                st.session_state[
                    "last_research_story"
                ] = updated_story


                st.rerun()


            except Exception as exc:
                st.error(f"Tekrar değerlendirme başarısız: {exc}")
    else:
        st.subheader("Adım 3: Estimand Teyidi")
        st.info(
            "Aşağıdaki öneriyi inceleyin. "
            "Onaydan sonra estimand dondurulur."
        )
        st.json(draft_proposal.model_dump())

        if st.button(
            "Öneriyi Onayla ve Dondur",
            type="primary",
        ):
            frozen = freeze_estimand(
                draft_proposal,
                approved=True,
            )
            st.session_state["frozen_estimand"] = frozen
            st.success("Estimand başarıyla donduruldu.")
            st.rerun()

# -------------------------------------------------
# ANALYSIS STATE
# -------------------------------------------------

frozen_estimand = st.session_state.get(
    "frozen_estimand"
)


if frozen_estimand is None:

    if st.session_state.get(
        "socratic_submitted"
    ):

        st.info(
            "Estimand henüz dondurulmadı. "
            "Önce teknik eşleme önerisini onaylayın."
        )

    st.stop()



state = st.session_state.get("analysis_state")

if state is None:
    guessed_unit = next((c for c in columns if "id" in c.lower()), columns[0])
    guessed_time = next((c for c in columns if "year" in c.lower() or "date" in c.lower()), columns[0])
    guessed_cluster = guessed_unit

    st.subheader("Analiz Yapılandırması")
    with st.form("analysis_state_form"):
        unit_col_input = st.selectbox("Birim kolonu (unit)", options=columns, index=columns.index(guessed_unit))
        time_col_input = st.selectbox("Zaman kolonu (time)", options=columns, index=columns.index(guessed_time))
        cluster_options = [None] + columns
        cluster_by_input = st.selectbox(
            "Kümeleme kolonu (cluster)",
            options=cluster_options,
            index=cluster_options.index(guessed_cluster),
            format_func=lambda c: (
                c
                if c is not None
                else "Kümeleme yok (heteroskedastisiteye dayanıklı SE)"
            ),
        )
        controls_input = st.multiselect(
            "Kontrol kolonları",
            options=columns,
            default=[],
            help="Treatment ve outcome kolonlarını kontrol olarak seçmeyin.",
        )
        saved_state = st.form_submit_button("Yapılandırmayı kaydet", type="primary")

    if not saved_state:
        st.info("Devam etmek için analiz yapılandırmasını kaydedin.")
        st.stop()

    state = {
        "unit_col": unit_col_input,
        "time_col": time_col_input,
        "cluster_by": cluster_by_input,
        "controls": controls_input,
    }
    st.session_state["analysis_state"] = state
    st.rerun()



# -------------------------------------------------
# HASH INVALIDATION
# -------------------------------------------------

if (
    st.session_state.get(
        "_spec_estimand_hash"
    )
    != frozen_estimand.freeze_hash
):

    st.session_state.pop(
        "analysis_specs",
        None,
    )

    st.session_state.pop(
        "analysis_frozen_menu",
        None,
    )


    st.session_state[
        "_spec_estimand_hash"
    ] = frozen_estimand.freeze_hash



# -------------------------------------------------
# FROZEN ESTIMAND VIEW
# -------------------------------------------------

st.subheader(
    "Dondurulmuş Estimand"
)


st.json(
    frozen_estimand.estimand.model_dump()
)


st.code(
    f"hash={frozen_estimand.freeze_hash}"
)



# -------------------------------------------------
# ANALYSIS CONFIG
# -------------------------------------------------

unit_col = state.get(
    "unit_col"
)

time_col = state.get(
    "time_col"
)

cluster_by = state.get(
    "cluster_by"
)

controls = state.get(
    "controls",
    [],
)



# -------------------------------------------------
# ESTIMATOR OPTIONS
# -------------------------------------------------

estimators = [
    "OLS"
]


if unit_col and time_col:

    estimators.append(
        "TWFE"
    )



# -------------------------------------------------
# SPEC MENU
# -------------------------------------------------

st.subheader(
    "Spesifikasyon Menüsü"
)

menu_status_placeholder = st.empty()

menu_source = st.radio(
    "Menü kaynağı",
    options=[
        "deterministic",
        "llm",
    ],
    format_func=lambda x:
        "Deterministik"
        if x == "deterministic"
        else "LLM (JUDGE)",
    horizontal=True,
)



menu: SpecMenu | None = None



# -------------------------------------------------
# DETERMINISTIC MENU
# -------------------------------------------------

if menu_source == "deterministic":

    menu = build_deterministic_menu(
        controls=controls,
        cluster_by=cluster_by,
        estimators=estimators,
        available_columns=columns,
    )

    try:
        preview_specs = expand_to_specs(
            menu.freeze(),
            outcome=frozen_estimand.estimand.outcome,
            treatment=frozen_estimand.estimand.treatment_coding,
            unit_col=unit_col or cluster_by,
            time_col=time_col or cluster_by,
        )
        menu_status_placeholder.success(
            f"Canlı spec sayacı: {len(preview_specs)} spesifikasyon üretilebilir"
        )
    except ValueError as exc:
        menu_status_placeholder.warning(
            f"Canlı spec sayacı: kapıdan geçemiyor — {exc}"
        )



# -------------------------------------------------
# LLM MENU
# -------------------------------------------------

else:


    if st.button(
        "JUDGE menü önerisi oluştur",
        type="primary",
    ):

        try:

            with st.spinner(
                "JUDGE dayanıklılık menüsü hazırlıyor..."
            ):

                menu_proposal = generate_spec_menu(
                    frozen=frozen_estimand,
                    available_columns=columns,
                )


                st.session_state[
                    "menu_proposal"
                ] = menu_proposal


                st.session_state.pop(
                    "frozen_spec_menu",
                    None,
                )


        except Exception as exc:

            st.error(
                f"Menü oluşturulamadı: {exc}"
            )


            st.stop()



    menu_proposal = st.session_state.get(
        "menu_proposal"
    )


    if menu_proposal is None:

        st.info(
            "Önce JUDGE menü önerisi oluşturun."
        )

        st.stop()



    st.caption(
        menu_proposal.overall_rationale
    )

    preview_ok, preview_reasons, preview_count = _preview_menu_status(
        menu_proposal,
        available_columns=columns,
        outcome=frozen_estimand.estimand.outcome,
        treatment=frozen_estimand.estimand.treatment_coding,
        unit_col=unit_col or cluster_by,
        time_col=time_col or cluster_by,
    )

    if preview_ok:
        menu_status_placeholder.success(f"Canlı spec sayacı: {preview_count} spesifikasyon üretilebilir")
    else:
        menu_status_placeholder.warning("Canlı spec sayacı: kapıdan geçemiyor")
        for reason in preview_reasons:
            menu_status_placeholder.caption(f"• {reason}")

    approved_axes = set(st.session_state.get("menu_approved_axes", []))

    for axis in menu_proposal.axes:
        with st.expander(
            f"{axis.axis_name} (baseline: {axis.baseline_level})"
        ):
            st.write(axis.rationale)

            cols = st.columns(4)
            with cols[0]:
                if st.button("Onayla", key=f"{axis.axis_name}_approve", use_container_width=True):
                    approved_axes.add(axis.axis_name)
                    st.session_state["menu_approved_axes"] = approved_axes
                    st.rerun()
            with cols[1]:
                if st.button("Düzenle", key=f"{axis.axis_name}_edit", use_container_width=True):
                    st.session_state[f"{axis.axis_name}_editing"] = True
                    st.rerun()
            with cols[2]:
                if axis.candidate_levels:
                    to_remove = st.selectbox(
                        "Silinecek seviye",
                        options=axis.candidate_levels,
                        key=f"{axis.axis_name}_remove_select",
                    )
                    if st.button("Çıkar", key=f"{axis.axis_name}_remove", use_container_width=True):
                        axis.candidate_levels = [level for level in axis.candidate_levels if level != to_remove]
                        st.session_state["menu_proposal"] = menu_proposal
                        st.rerun()
            with cols[3]:
                candidate_level = st.text_input(
                    "Yeni seviye",
                    key=f"{axis.axis_name}_new_level",
                    label_visibility="collapsed",
                )
                if st.button("Ekle", key=f"{axis.axis_name}_add", use_container_width=True):
                    if candidate_level.strip():
                        axis.candidate_levels = [*axis.candidate_levels, candidate_level.strip()]
                        st.session_state["menu_proposal"] = menu_proposal
                        st.rerun()

            if axis.axis_name in approved_axes:
                st.caption("✓ Onaylandı")

            if st.session_state.get(f"{axis.axis_name}_editing", False):
                baseline_value = st.text_input(
                    "Baseline",
                    value=axis.baseline_level,
                    key=f"{axis.axis_name}_baseline",
                )
                candidate_text = st.text_area(
                    "Aday seviyeler (virgülle)",
                    value=", ".join(axis.candidate_levels),
                    key=f"{axis.axis_name}_candidates",
                )
                if st.button("Kaydet", key=f"{axis.axis_name}_save"):
                    axis.baseline_level = baseline_value.strip()
                    axis.candidate_levels = [
                        item.strip()
                        for item in candidate_text.split(",")
                        if item.strip()
                    ]
                    st.session_state["menu_proposal"] = menu_proposal
                    st.session_state[f"{axis.axis_name}_editing"] = False
                    st.rerun()

            if axis.candidate_levels:
                st.write("Aday seviyeler:")
                st.write(axis.candidate_levels)

    if menu_proposal.needs_clarification:

        st.warning(
            menu_proposal.clarification_question
            or
            "Menü için ek açıklama gerekiyor."
        )

        st.stop()



    defensibility_ok, reasons, spec_count = evaluate_menu_defensibility(
        menu_proposal,
        available_columns=columns,
        outcome=frozen_estimand.estimand.outcome,
        treatment=frozen_estimand.estimand.treatment_coding,
        unit_col=unit_col or cluster_by,
        time_col=time_col or cluster_by,
    )

    if defensibility_ok:
        st.success(f"Savunulabilirlik kapısı geçti: {spec_count} spesifikasyon üretilebilir.")
    else:
        st.warning("Savunulabilirlik kapısı: bu menü dondurulamaz.")
        for reason in reasons:
            st.caption(f"• {reason}")

    axes_needing_approval = {
        axis.axis_name for axis in menu_proposal.axes if axis.candidate_levels
    }
    all_approved = axes_needing_approval.issubset(approved_axes)

    if not all_approved:
        missing = axes_needing_approval - approved_axes
        st.warning(
            "Dondurmadan önce şu eksenleri onaylayın: "
            + ", ".join(sorted(missing))
        )

    proposed_active_axes = st.multiselect(
        "Dondurmadan önce aktif eksenler",
        options=list(ALL_AXES),
        default=list(ALL_AXES),
        help="Seçilen eksenler dondurulan menünün parçası olur.",
    )

    if defensibility_ok and all_approved and st.button("Spesifikasyon menüsünü dondur"):

        try:
            frozen_menu_obj = freeze_spec_menu(
                menu_proposal,
                available_columns=columns,
                approved=True,
                active_axes=tuple(proposed_active_axes),
            )
            st.session_state["frozen_spec_menu"] = frozen_menu_obj
            st.success("Spesifikasyon menüsü donduruldu.")
            st.rerun()

        except ValueError as exc:
            st.error(str(exc))



    frozen_menu_obj = st.session_state.get(
        "frozen_spec_menu"
    )


    if frozen_menu_obj is None:

        st.info(
            "Menüyü onaylayarak devam edin."
        )

        st.stop()



    menu = frozen_menu_obj.menu



# -------------------------------------------------
# ACTIVE AXES
# -------------------------------------------------

if menu is None:

    st.stop()



if menu_source == "deterministic":
    active_axes = st.multiselect(
        "Aktif eksenler",
        options=list(ALL_AXES),
        default=list(menu.active_axes),
        help="Seçilen eksenler faktöriyel genişlemeye dahil edilir.",
    )
    if active_axes:
        menu = menu.model_copy(update={"active_axes": tuple(active_axes)})
else:
    st.caption("LLM menüsünde aktif eksenler dondurma anında sabitlenir.")



# -------------------------------------------------
# FREEZE MENU HASH
# -------------------------------------------------

frozen_menu = menu.freeze()


st.code(
    f"menu_hash={frozen_menu.menu_hash}"
)



# -------------------------------------------------
# EXPAND SPECIFICATIONS
# -------------------------------------------------

try:

    specs = expand_to_specs(
        frozen_menu,

        outcome=(
            frozen_estimand
            .estimand
            .outcome
        ),

        treatment=(
            frozen_estimand
            .estimand
            .treatment_coding
        ),

        unit_col=(
            unit_col
            or cluster_by
        ),

        time_col=(
            time_col
            or cluster_by
        ),
    )

    validate_spec_menu_to_specs(frozen_menu, specs)



    warnings = []


    for spec in specs:

        result = validate_estimand_spec_mapping(
            frozen_estimand,
            spec,
            available_columns=columns,
        )


        if result:

            warnings.extend(
                result
            )



    if warnings:

        with st.expander(
            "⚠️ Spesifikasyon eşleme uyarıları",
            expanded=False,
        ):

            for warning in sorted(
                set(warnings)
            ):

                st.warning(
                    warning
                )



except ValueError as exc:

    st.error(
        str(exc)
    )

    st.stop()



# -------------------------------------------------
# OUTPUT
# -------------------------------------------------

st.session_state[
    "analysis_specs"
] = specs


st.session_state[
    "analysis_frozen_menu"
] = frozen_menu


persisted_run_id = None
if frozen_estimand and frozen_menu:
    persisted_run_id = f"{frozen_estimand.freeze_hash[:8]}-{frozen_menu.menu_hash[:8]}-{uuid.uuid4().hex[:6]}"

_persist_frozen_menu(
    frozen_estimand=frozen_estimand,
    frozen_menu=frozen_menu,
    specs=specs,
    run_id=persisted_run_id,
)

st.subheader(
    "Spesifikasyon Seti"
)



df_specs = pd.DataFrame(
    [
        {
            **spec.model_dump(),
            "hash": spec.content_hash(),
        }

        for spec in specs
    ]
)



st.dataframe(
    df_specs,
    use_container_width=True,
)



st.success(
    f"{len(specs)} spesifikasyon üretildi."
)

st.subheader("Multiverse")

if st.button("Multiverse başlat", type="primary"):
    run_id = f"{frozen_estimand.freeze_hash[:8]}-{frozen_menu.menu_hash[:8]}-{uuid.uuid4().hex[:6]}"
    handle = launch_multiverse(df, specs, run_id)
    st.session_state["multiverse_handle"] = handle
    st.session_state["multiverse_run_id"] = run_id
    st.rerun()

if st.session_state.get("multiverse_handle") is not None:
    _render_multiverse_progress(st.session_state["multiverse_handle"])
