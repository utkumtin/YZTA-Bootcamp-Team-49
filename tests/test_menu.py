import pytest
from pydantic_ai.models.test import TestModel

from pareto.analysis.hypothesis import Estimand, FrozenEstimand, TACProposal, freeze_estimand
from pareto.analysis.menu import (
    SpecMenu,
    SpecMenuAxis,
    SpecMenuProposal,
    expand_to_specs,
    freeze_spec_menu,
    generate_spec_menu,
    validate_spec_menu_to_specs,
)
from pareto.llm.router import use_test_model
from pareto.spec import Specification

_MENU_COLUMNS = ["state", "year", "expanded", "uninsured_rate", "population", "unemployment_rate"]


def _frozen(**kw):
    base: dict = {
        "control_sets": [["x1"], ["x1", "x2"]],
        "clustering_levels": ("g",),
        "estimators": ("OLS",),
    }
    base.update(kw)
    return SpecMenu(**base).freeze()


def _fake_frozen_estimand() -> FrozenEstimand:
    proposal = TACProposal(
        estimand_type="ATT",
        treatment="Medicaid expansion adoption",
        treatment_coding="expanded",
        outcome="uninsured_rate",
        outcome_unit="percentage points",
        population="US states",
        time_scope="2010-2020",
        expected_sign="negative",
        identification_assumption="parallel_trends",
        h0="ATT = 0",
        h1="ATT < 0",
        implied_result_translation="Lower uninsured rates in expansion states.",
        confirmation_question="Freeze?",
    )
    return freeze_estimand(proposal, approved=True)


def _menu_proposal_args() -> dict[str, object]:
    return {
        "axes": [
            {
                "axis_name": "control_set",
                "baseline_level": "none",
                "candidate_levels": ["unemployment_rate"],
                "rationale": "No controls baseline; unemployment as defensible covariate.",
            },
            {
                "axis_name": "sample",
                "baseline_level": "none",
                "candidate_levels": [],
                "rationale": "Full sample is the committed baseline.",
            },
            {
                "axis_name": "pre_period",
                "baseline_level": "none",
                "candidate_levels": ["3"],
                "rationale": "3-year pre-trend window as robustness.",
            },
            {
                "axis_name": "clustering",
                "baseline_level": "state",
                "candidate_levels": [],
                "rationale": "State-level clustering matches the panel unit.",
            },
            {
                "axis_name": "never_treated",
                "baseline_level": "true",
                "candidate_levels": ["false"],
                "rationale": "Never-treated control is the committed design.",
            },
            {
                "axis_name": "estimator",
                "baseline_level": "TWFE",
                "candidate_levels": [],
                "rationale": "TWFE matches parallel-trends DiD.",
            },
            {
                "axis_name": "weighting",
                "baseline_level": "population",
                "candidate_levels": ["none"],
                "rationale": "Population-weighted is default for state aggregates.",
            },
        ],
        "overall_rationale": "Conservative Medicaid expansion robustness menu.",
        "needs_clarification": False,
        "clarification_question": None,
    }


def test_freeze_is_deterministic_16char_hash():
    # NEDEN: reprodüksiyon garantisi dondurmadan gelir; aynı menü → aynı hash.
    assert _frozen().menu_hash == _frozen().menu_hash
    assert len(_frozen().menu_hash) == 16


def test_silent_axis_pinned_to_baseline():
    # NEDEN: aktif olmayan eksen baseline'a (ilk seviye) pinlenir → okunaklı + tekrarlanabilir.
    frozen = _frozen(estimators=("OLS",), active_axes=("control_set",))
    specs = expand_to_specs(frozen, outcome="y", treatment="d", unit_col="u", time_col="t")
    assert len(specs) == 2  # yalnız kontrol seti ekseni açık
    assert {s.estimator for s in specs} == {"OLS"}


def test_hard_cap_24_fails_loud():
    # NEDEN: sert tavan 24. Aşımda sessiz kırpma YOK — patlar.
    frozen = SpecMenu(
        control_sets=[[f"x{i}"] for i in range(30)],
        clustering_levels=("g",),
        active_axes=("control_set",),
    ).freeze()
    with pytest.raises(ValueError, match="sert tavan"):
        expand_to_specs(frozen, outcome="y", treatment="d", unit_col="u", time_col="t")


def test_weighting_axis_expands_with_population_default():
    frozen = SpecMenu(
        control_sets=[[]],
        clustering_levels=("state",),
        estimators=("OLS",),
        weighting_levels=("population", None),
        active_axes=("weighting",),
    ).freeze()
    specs = expand_to_specs(
        frozen, outcome="y", treatment="d", unit_col="state", time_col="year"
    )
    assert len(specs) == 2
    assert {s.weight_col for s in specs} == {"population", None}


def test_validate_spec_menu_to_specs_accepts_clean_mapping():
    frozen = _frozen()
    specs = expand_to_specs(frozen, outcome="y", treatment="d", unit_col="u", time_col="t")
    validate_spec_menu_to_specs(frozen, specs)


def test_validate_spec_menu_to_specs_fails_loud_on_dirty_spec():
    frozen = _frozen()
    dirty = Specification(
        spec_id="bad",
        outcome="y",
        treatment="d",
        controls=("x99",),
        cluster_by="g",
        estimator="OLS",
    )
    with pytest.raises(ValueError, match="Spec validation failed"):
        validate_spec_menu_to_specs(frozen, [dirty])


def test_testmodel_proposes_expected_axes_and_levels():
    frozen_estimand = _fake_frozen_estimand()
    columns = ["state", "year", "expanded", "uninsured_rate", "population", "unemployment_rate"]

    with use_test_model(TestModel(custom_output_args=_menu_proposal_args())):
        proposal = generate_spec_menu(frozen=frozen_estimand, available_columns=columns)

    assert not proposal.needs_clarification
    axis_names = {a.axis_name for a in proposal.axes}
    assert axis_names == {
        "control_set",
        "sample",
        "pre_period",
        "clustering",
        "never_treated",
        "estimator",
        "weighting",
    }

    weighting = next(a for a in proposal.axes if a.axis_name == "weighting")
    assert weighting.baseline_level == "population"
    assert "none" in weighting.candidate_levels

    frozen_menu = freeze_spec_menu(proposal, available_columns=columns, approved=True)
    assert frozen_menu.menu.weighting_levels[0] == "population"
    assert frozen_menu.menu.estimators[0] == "TWFE"


def test_menu_hash_stable_after_optional_clustering():
    # NEDEN: clustering ekseninin None kabul etmesi mevcut donmuş menü hash'lerini KIRMAMALI.
    # Altın değer, clustering_levels'ın `tuple[str, ...]` olduğu sürümde üretildi.
    assert _frozen().menu_hash == "f6e3c64072312e2e"


def test_clustering_none_level_freezes_instead_of_failing():
    # NEDEN: JUDGE "none" önerdiğinde menü dondurulamıyor, LLM menü akışı fiilen
    # kapanıyordu. "none" artık kümeleme-yok seviyesine normalize olmalı.
    args = _menu_proposal_args()
    clustering = next(a for a in args["axes"] if a["axis_name"] == "clustering")  # type: ignore[index]
    clustering["baseline_level"] = "none"

    proposal = SpecMenuProposal(**args)
    frozen_menu = freeze_spec_menu(
        proposal,
        available_columns=_MENU_COLUMNS,
        approved=True,
    )
    assert frozen_menu.menu.clustering_levels == (None,)


def test_clustering_axis_expands_none_and_column_as_two_specs():
    # NEDEN: kümeleme seçimi bir varyans ekseni; "none" ile kolon aynı spec'e çökerse
    # panel bu ekseni hiç ölçemez.
    args = _menu_proposal_args()
    clustering = next(a for a in args["axes"] if a["axis_name"] == "clustering")  # type: ignore[index]
    clustering["candidate_levels"] = ["none"]

    frozen_menu = freeze_spec_menu(
        SpecMenuProposal(**args),
        available_columns=_MENU_COLUMNS,
        approved=True,
        active_axes=("clustering",),
    )
    specs = expand_to_specs(
        frozen_menu,
        outcome="uninsured_rate",
        treatment="expanded",
        unit_col="state",
        time_col="year",
    )
    assert {s.cluster_by for s in specs} == {"state", None}
    validate_spec_menu_to_specs(frozen_menu, specs)


def test_unknown_clustering_column_still_fails_loud():
    # NEDEN: "none" gevşemesi uydurma kolonlara kapı açmamalı; halüsinasyon hâlâ patlatır.
    args = _menu_proposal_args()
    clustering = next(a for a in args["axes"] if a["axis_name"] == "clustering")  # type: ignore[index]
    clustering["baseline_level"] = "hayali_kolon"

    with pytest.raises(ValueError, match="Invalid clustering column"):
        freeze_spec_menu(
            SpecMenuProposal(**args),
            available_columns=_MENU_COLUMNS,
            approved=True,
        )


def test_freeze_spec_menu_rejects_unapproved():
    proposal = SpecMenuProposal(**_menu_proposal_args())
    with pytest.raises(ValueError, match="User approval required"):
        freeze_spec_menu(
            proposal,
            available_columns=["state", "year", "expanded", "uninsured_rate", "population"],
            approved=False,
        )


def test_freeze_spec_menu_rejects_clarification_needed():
    args = _menu_proposal_args()
    args["needs_clarification"] = True
    args["clarification_question"] = "Which clustering level?"
    proposal = SpecMenuProposal(**args)
    with pytest.raises(ValueError, match="Which clustering level"):
        freeze_spec_menu(
            proposal,
            available_columns=["state"],
            approved=True,
        )
