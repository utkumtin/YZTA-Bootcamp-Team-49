"""2 - Analysis: estimand-first workflow (NEW ARCHITECTURE)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    TACProposal,
    freeze_estimand,
    validate_estimand_spec_mapping,
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
    st.stop()


st.subheader("Frozen estimand")
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