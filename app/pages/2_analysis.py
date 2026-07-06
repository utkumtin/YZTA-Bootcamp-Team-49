"""2 - Analysis: estimand-first workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pareto.analysis.hypothesis import (
    SocraticDeclaration,
    TACProposal,
    freeze_estimand,
    validate_estimand_spec_mapping,
)
from pareto.analysis.menu import SpecMenu, expand_to_specs

st.title("2 - Analysis")


def _df_from_state() -> pd.DataFrame | None:
    df = st.session_state.get("clean_df")
    return df if isinstance(df, pd.DataFrame) else None


def _looks_like(columns: list[str], needles: tuple[str, ...]) -> str | None:
    for col in columns:
        name = col.lower()
        if any(needle in name for needle in needles):
            return col
    return None


def _default_index(options: list[str], value: str | None) -> int:
    return options.index(value) if value in options else 0


def _build_proposal(
    *,
    declaration: SocraticDeclaration,
    estimand_type: str,
    treatment_col: str,
    outcome_col: str,
    outcome_unit: str,
    population: str,
    time_scope: str,
    identification_assumption: str,
) -> TACProposal:
    sign_text = {
        "positive": "increase",
        "negative": "decrease",
        "ambiguous": "change",
    }[declaration.expected_sign]
    h1_op = {"positive": ">", "negative": "<", "ambiguous": "!="}[declaration.expected_sign]
    return TACProposal(
        estimand_type=estimand_type,  # type: ignore[arg-type]
        treatment=declaration.conceptual_treatment,
        treatment_coding=treatment_col,
        outcome=outcome_col,
        outcome_unit=outcome_unit,
        population=population,
        time_scope=time_scope,
        expected_sign=declaration.expected_sign,
        identification_assumption=identification_assumption,
        h0=f"{estimand_type} = 0",
        h1=f"{estimand_type} {h1_op} 0",
        implied_result_translation=(
            f"If the estimate matches your prior, {declaration.conceptual_treatment} "
            f"would {sign_text} {declaration.conceptual_outcome} for {population}."
        ),
        confirmation_question=(
            f"Freeze {estimand_type}: {treatment_col} -> {outcome_col} "
            f"with expected sign '{declaration.expected_sign}'?"
        ),
    )


df = _df_from_state()
if df is None:
    st.warning("No dataframe found from Cleaning. Upload/profile data on page 1 first.")
    manual_columns = st.text_area(
        "Or paste column names manually",
        placeholder="state, year, treatment, outcome, control_1",
    )
    columns = [c.strip() for c in manual_columns.replace("\n", ",").split(",") if c.strip()]
else:
    columns = [str(c) for c in df.columns]
    st.success(f"Using dataframe from Cleaning: {len(df)} rows x {df.shape[1]} columns")
    with st.expander("Columns", expanded=False):
        st.write(columns)

if not columns:
    st.stop()

treatment_guess = _looks_like(columns, ("treat", "expanded", "policy", "fon kodu", "kod"))
outcome_guess = _looks_like(columns, ("outcome", "rate", "return", "getiri", "1 yıl", "yıl"))
unit_guess = _looks_like(columns, ("state", "unit", "id", "fon kodu", "kod"))
time_guess = _looks_like(columns, ("year", "date", "tarih", "time"))

with st.form("estimand_form"):
    st.subheader("Socratic declaration")
    research_story = st.text_area(
        "Research question",
        value="Estimate whether the selected treatment is associated with the selected outcome.",
    )
    conceptual_treatment = st.text_input(
        "Conceptual treatment",
        value=treatment_guess or columns[0],
    )
    conceptual_outcome = st.text_input(
        "Conceptual outcome",
        value=outcome_guess or columns[min(1, len(columns) - 1)],
    )
    expected_sign = st.selectbox("Expected sign", ["negative", "positive", "ambiguous"])

    st.subheader("TAC technical mapping")
    left, right = st.columns(2)
    with left:
        estimand_type = st.selectbox("Estimand", ["ATT", "ATE", "LATE"])
        treatment_col = st.selectbox(
            "Treatment column",
            columns,
            index=_default_index(columns, treatment_guess),
        )
        outcome_col = st.selectbox(
            "Outcome column",
            columns,
            index=_default_index(columns, outcome_guess),
        )
        outcome_unit = st.text_input("Outcome unit", value="original data unit")
    with right:
        unit_options = ["<none>", *columns]
        time_options = ["<none>", *columns]
        unit_col = st.selectbox(
            "Unit column",
            unit_options,
            index=_default_index(unit_options, unit_guess),
        )
        time_col = st.selectbox(
            "Time column",
            time_options,
            index=_default_index(time_options, time_guess),
        )
        cluster_default = unit_col if unit_col != "<none>" else columns[0]
        cluster_by = st.selectbox(
            "Cluster by",
            columns,
            index=_default_index(columns, cluster_default),
        )

    population = st.text_input("Population", value="uploaded dataset")
    time_scope = st.text_input("Time scope", value="available rows")
    identification_assumption = st.selectbox(
        "Identification assumption",
        ["associational", "parallel_trends"],
        help="Use associational when the dataset is not a panel DiD setup.",
    )
    controls = st.multiselect(
        "Control columns",
        [c for c in columns if c not in {treatment_col, outcome_col, unit_col, time_col}],
    )

    submitted = st.form_submit_button("Create and freeze estimand")

if submitted:
    declaration = SocraticDeclaration(
        conceptual_treatment=conceptual_treatment,
        conceptual_outcome=conceptual_outcome,
        expected_sign=expected_sign,  # type: ignore[arg-type]
    )
    proposal = _build_proposal(
        declaration=declaration,
        estimand_type=estimand_type,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        outcome_unit=outcome_unit,
        population=population,
        time_scope=time_scope,
        identification_assumption=identification_assumption,
    )
    frozen = freeze_estimand(proposal, approved=True)
    st.session_state["frozen_estimand"] = frozen
    st.session_state["analysis_form"] = {
        "research_story": research_story,
        "proposal": proposal,
        "unit_col": None if unit_col == "<none>" else unit_col,
        "time_col": None if time_col == "<none>" else time_col,
        "cluster_by": cluster_by,
        "controls": controls,
    }

frozen = st.session_state.get("frozen_estimand")
form_state = st.session_state.get("analysis_form")
if frozen is None or form_state is None:
    st.stop()

proposal = form_state["proposal"]
st.subheader("Frozen estimand")
st.caption(proposal.implied_result_translation)
st.code(f"freeze_hash = {frozen.freeze_hash}", language="text")
st.json(frozen.estimand.model_dump())

unit_col = form_state["unit_col"]
time_col = form_state["time_col"]
estimators = ["OLS"] if frozen.estimand.identification_assumption != "parallel_trends" else []
if unit_col and time_col:
    estimators.append("TWFE")
if not estimators:
    st.error("parallel_trends needs both a unit column and a time column for TWFE.")
    st.stop()

menu = SpecMenu(
    control_sets=[[], form_state["controls"]] if form_state["controls"] else [[]],
    clustering_levels=[form_state["cluster_by"]],
    estimators=estimators,  # type: ignore[arg-type]
    active_axes=["control_set", "estimator"] if len(estimators) > 1 else ["control_set"],
    rationale="Deterministic baseline menu from the confirmed estimand and selected columns.",
)
frozen_menu = menu.freeze()

st.subheader("Frozen spec menu")
st.code(f"menu_hash = {frozen_menu.menu_hash}", language="text")
st.json(frozen_menu.menu.model_dump())

try:
    specs = expand_to_specs(
        frozen_menu,
        outcome=frozen.estimand.outcome,
        treatment=frozen.estimand.treatment_coding,
        unit_col=unit_col or form_state["cluster_by"],
        time_col=time_col or form_state["cluster_by"],
    )
    for spec in specs:
        validate_estimand_spec_mapping(frozen, spec, available_columns=columns)
except ValueError as exc:
    st.error(str(exc))
else:
    st.subheader("Specification candidates")
    st.dataframe(pd.DataFrame([s.model_dump() | {"content_hash": s.content_hash()} for s in specs]))
