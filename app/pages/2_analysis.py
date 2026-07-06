"""2 - Analysis: estimand-first workflow (NEW ARCHITECTURE)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pareto.streamlit_ui import render_compact_sidebar
from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    TACProposal,
    freeze_estimand,
    validate_estimand_spec_mapping,
)
from pareto.analysis.menu import (
    ALL_AXES,
    SpecMenu,
    build_deterministic_menu,
    expand_to_specs,
    freeze_spec_menu,
    generate_spec_menu,
)


with st.sidebar:
    render_compact_sidebar()

st.title("2 - Analysis")


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
    st.warning("No dataframe found. Please complete Cleaning step first.")
    manual = st.text_area("Paste columns manually")
    columns = [c.strip() for c in manual.split(",") if c.strip()]
else:
    columns = [str(c) for c in df.columns]
    st.success(f"Loaded dataset: {df.shape}")

if not columns:
    st.stop()


# -----------------------------
# GUESSES
# -----------------------------

treatment_guess = _guess(columns, ("treat", "policy", "kod"))
outcome_guess = _guess(columns, ("outcome", "rate", "return"))
unit_guess = _guess(columns, ("id", "unit", "state"))
time_guess = _guess(columns, ("year", "date", "time"))


# -----------------------------
# SOCrATIC FORM
# -----------------------------

with st.form("estimand_form"):

    st.subheader("Socratic declaration")

    conceptual_treatment = st.text_input(
        "Conceptual treatment",
        value=treatment_guess or columns[0],
    )

    conceptual_outcome = st.text_input(
        "Conceptual outcome",
        value=outcome_guess or columns[-1],
    )

    expected_sign = st.selectbox(
        "Expected sign",
        ["positive", "negative", "ambiguous"],
    )

    st.subheader("Technical mapping")

    treatment_col = st.selectbox("Treatment column", columns)
    outcome_col = st.selectbox("Outcome column", columns)

    unit_options = ["<none>", *columns]
    time_options = ["<none>", *columns]

    unit_col = st.selectbox("Unit column", unit_options)
    time_col = st.selectbox("Time column", time_options)

    cluster_by = st.selectbox("Cluster by", columns)

    controls = st.multiselect(
        "Controls",
        [c for c in columns if c not in {treatment_col, outcome_col}],
    )

    submitted = st.form_submit_button("Freeze estimand")


if submitted:

    declaration = SocraticDeclaration(
        conceptual_treatment=conceptual_treatment,
        conceptual_outcome=conceptual_outcome,
        expected_sign=expected_sign,
    )

    proposal = TACProposal(
        estimand_type="ATT",
        treatment=conceptual_treatment,
        treatment_coding=treatment_col,
        outcome=outcome_col,
        outcome_unit="unit",
        population="dataset",
        time_scope="all",
        expected_sign=expected_sign,
        identification_assumption="associational",
        h0="0",
        h1="!=0",
        implied_result_translation="",
        confirmation_question="Confirm estimand?",
    )

    frozen = freeze_estimand(proposal, approved=True)

    st.session_state["frozen_estimand"] = frozen
    st.session_state["analysis_state"] = {
        "proposal": proposal,
        "unit_col": None if unit_col == "<none>" else unit_col,
        "time_col": None if time_col == "<none>" else time_col,
        "cluster_by": cluster_by,
        "controls": controls,
    }


# -----------------------------
# LOAD STATE
# -----------------------------

frozen = st.session_state.get("frozen_estimand")
state = st.session_state.get("analysis_state")

if frozen is None or state is None:
    st.info("Estimand henüz dondurulmadı — aşağıdaki formu doldurun.")
    st.stop()

# Estimand değişmediyse önceki spec çıktısını koru
if st.session_state.get("_specs_estimand_hash") != frozen.freeze_hash:
    st.session_state.pop("analysis_specs", None)
    st.session_state.pop("analysis_frozen_menu", None)
    st.session_state["_specs_estimand_hash"] = frozen.freeze_hash


st.subheader("Frozen estimand")
st.json(frozen.estimand.model_dump())
st.code(f"hash={frozen.freeze_hash}")

# -----------------------------
# MENU BUILD
# -----------------------------

unit_col = state["unit_col"]
time_col = state["time_col"]
cluster_by = state["cluster_by"]
controls = state["controls"]

estimators = ["OLS"]
if unit_col and time_col:
    estimators.append("TWFE")

# Estimand değişince menü önbelleğini sıfırla
if st.session_state.get("_menu_estimand_hash") != frozen.freeze_hash:
    st.session_state.pop("menu_proposal", None)
    st.session_state.pop("frozen_spec_menu", None)
    st.session_state["_menu_estimand_hash"] = frozen.freeze_hash

st.subheader("Spec menu")

menu_source = st.radio(
    "Menu source",
    options=["deterministic", "llm"],
    format_func=lambda x: "Deterministic" if x == "deterministic" else "LLM (JUDGE)",
    horizontal=True,
    key="menu_source",
)

menu: SpecMenu | None = None

if menu_source == "deterministic":
    menu = build_deterministic_menu(
        controls=controls,
        cluster_by=cluster_by,
        estimators=estimators,
        available_columns=columns,
    )
else:
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        generate_clicked = st.button("Generate menu proposal", type="primary")

    if generate_clicked:
        try:
            with st.spinner("JUDGE is designing the robustness menu..."):
                st.session_state["menu_proposal"] = generate_spec_menu(
                    frozen=frozen,
                    available_columns=columns,
                )
                st.session_state.pop("frozen_spec_menu", None)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Menu generation failed: {exc}")
            st.caption(
                "Ana sayfada BYOK kaydet (veya `.env`). Hem BYOK hem `.env` doluysa BYOK kullanılır."
            )

    proposal = st.session_state.get("menu_proposal")
    if proposal is None:
        st.info("Click **Generate menu proposal** to get JUDGE recommendations per axis.")
        st.stop()

    st.caption(proposal.overall_rationale)
    for axis in proposal.axes:
        with st.expander(f"{axis.axis_name} — baseline: `{axis.baseline_level}`"):
            st.markdown(axis.rationale)
            if axis.candidate_levels:
                st.write("Candidates:", ", ".join(f"`{c}`" for c in axis.candidate_levels))

    if proposal.needs_clarification:
        st.warning(proposal.clarification_question or "Clarification required before freezing.")
        st.stop()

    if st.button("Freeze spec menu"):
        try:
            st.session_state["frozen_spec_menu"] = freeze_spec_menu(
                proposal,
                available_columns=columns,
                approved=True,
            )
            st.success("Spec menu frozen.")
        except ValueError as exc:
            st.error(str(exc))

    frozen_from_llm = st.session_state.get("frozen_spec_menu")
    if frozen_from_llm is None:
        st.info("Review the proposal above, then click **Freeze spec menu**.")
        st.stop()

    menu = frozen_from_llm.menu

active_axes = st.multiselect(
    "Active axes (empty = auto-detect levels with >1 option)",
    options=list(ALL_AXES),
    help="Only selected axes enter the factorial expansion; others stay pinned to baseline.",
)
if active_axes:
    menu = menu.model_copy(update={"active_axes": tuple(active_axes)})

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
    for spec in specs:
        validate_estimand_spec_mapping(
            frozen,
            spec,
            available_columns=columns,
        )

    st.session_state["analysis_specs"] = specs
    st.session_state["analysis_frozen_menu"] = frozen_menu

except ValueError as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# OUTPUT
# -----------------------------

st.subheader("Specification set")

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


st.success(f"{len(specs)} specifications generated")