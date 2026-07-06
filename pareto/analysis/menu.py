"""Spec menu — axis-based robustness menu generation and freezing.

İki parça:
  1. `FrozenSpecMenu` (immutable + hash) — reprodüksiyon garantisinin kaynağı.
  2. `expand_to_specs()` — aktif eksenlerin faktöriyel çarpımı → Specification listesi.

LLM menü üretimi: JUDGE her eksende savunulabilir seviyeler + baseline + gerekçe önerir
(kapalı liste değil). Kullanıcı onayladıktan sonra menü dondurulur; runtime tekrar LLM'e sormaz.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from typing import Literal

from pydantic import BaseModel

from ..config import SETTINGS, ModelRole
from ..llm.guardrails import prompt_json
from ..llm.router import build_agent
from ..spec import SUPPORTED_ESTIMATORS, Specification
from .hypothesis import FrozenEstimand

logger = logging.getLogger(__name__)

HARD_CAP = SETTINGS.max_specifications  # sert tavan 24

AxisName = Literal[
    "control_set",
    "sample",
    "pre_period",
    "clustering",
    "never_treated",
    "estimator",
    "weighting",
]

ALL_AXES: tuple[AxisName, ...] = (
    "control_set",
    "sample",
    "pre_period",
    "clustering",
    "never_treated",
    "estimator",
    "weighting",
)

DEFAULT_WEIGHT_COL = "population"


# -----------------------------
# LLM PROPOSAL STRUCTURES
# -----------------------------


class SpecMenuAxis(BaseModel):
    """Tek eksen: baseline + savunulabilir aday seviyeler + gerekçe (kapalı liste değil)."""

    axis_name: AxisName
    baseline_level: str
    candidate_levels: list[str] = []
    rationale: str


class SpecMenuProposal(BaseModel):
    axes: list[SpecMenuAxis]
    overall_rationale: str
    needs_clarification: bool = False
    clarification_question: str | None = None


# -----------------------------
# FROZEN MENU (runtime artifact)
# -----------------------------


class SpecMenu(BaseModel):
    """Savunulabilir seviyeler menüsü. Sessiz eksenler baseline'a (ilk seviye) pinlenir."""

    control_sets: list[list[str]]
    sample_filters: tuple[str | None, ...] = (None,)
    pre_period_windows: tuple[int | None, ...] = (None,)
    clustering_levels: tuple[str, ...]
    never_treated_levels: tuple[bool, ...] = (True,)
    estimators: tuple[str, ...] = ("OLS", "TWFE")
    weighting_levels: tuple[str | None, ...] = (None,)
    active_axes: tuple[str, ...] = ()  # boş → _default_active
    rationale: str = ""

    def freeze(self) -> FrozenSpecMenu:
        return FrozenSpecMenu(menu=self, menu_hash=_menu_hash(self))


class FrozenSpecMenu(BaseModel):
    model_config = {"frozen": True}

    menu: SpecMenu
    menu_hash: str


# -----------------------------
# AXIS HELPERS
# -----------------------------


def _axis_levels(proposal: SpecMenuProposal, name: str) -> list[str]:
    for axis in proposal.axes:
        if axis.axis_name == name:
            return [axis.baseline_level, *axis.candidate_levels]
    raise ValueError(f"Missing axis: {name}")


def _parse_control_set(level: str) -> list[str]:
    if level in ("none", "[]", ""):
        return []
    return [c.strip() for c in level.split("+") if c.strip()]


def _menu_hash(menu: SpecMenu) -> str:
    blob = json.dumps(menu.model_dump(), sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _default_active(menu: SpecMenu) -> set[str]:
    """Birden çok seviyesi olan her ekseni default aktif say."""
    active: set[str] = set()
    if len(menu.control_sets) > 1:
        active.add("control_set")
    if len(menu.sample_filters) > 1:
        active.add("sample")
    if len(menu.pre_period_windows) > 1:
        active.add("pre_period")
    if len(menu.clustering_levels) > 1:
        active.add("clustering")
    if len(menu.never_treated_levels) > 1:
        active.add("never_treated")
    if len(menu.estimators) > 1:
        active.add("estimator")
    if len(menu.weighting_levels) > 1:
        active.add("weighting")
    return active


def _resolve_weighting_levels(
    *,
    available_columns: list[str] | None,
    weight_col: str | None,
) -> tuple[str | None, ...]:
    """Default nüfus-ağırlıklı; kolon yoksa yalnızca ağırlıksız."""
    cols = set(available_columns or [])
    preferred = weight_col or DEFAULT_WEIGHT_COL
    if preferred in cols:
        return (preferred, None)  # baseline: nüfus-ağırlıklı; aday: ağırlıksız
    return (None,)


# -----------------------------
# DETERMINISTIC BASELINE (LLM'siz)
# -----------------------------


def build_deterministic_menu(
    *,
    controls: list[str] | None,
    cluster_by: str,
    estimators: list[str],
    available_columns: list[str] | None = None,
    weight_col: str | None = None,
) -> SpecMenu:
    """LLM'siz minimal spec menüsü; ağırlıklandırma default nüfus-ağırlıklı."""
    control_sets: list[list[str]] = [[], controls] if controls else [[]]
    estimator_tuple = tuple(estimators) if estimators else ("OLS",)

    return SpecMenu(
        control_sets=control_sets,
        sample_filters=(None,),
        pre_period_windows=(None,),
        clustering_levels=(cluster_by,),
        never_treated_levels=(True,),
        estimators=estimator_tuple,
        weighting_levels=_resolve_weighting_levels(
            available_columns=available_columns,
            weight_col=weight_col,
        ),
    )


# -----------------------------
# LLM MENU GENERATION (JUDGE)
# -----------------------------


def generate_spec_menu(
    *,
    frozen: FrozenEstimand,
    available_columns: list[str],
) -> SpecMenuProposal:
    """JUDGE: frozen estimand + kolonlar → her eksende savunulabilir seviyeler + baseline + gerekçe."""
    if not available_columns:
        raise ValueError("Spec menu proposal needs at least one available column.")

    estimand = frozen.estimand
    prompt = (
        "Frozen estimand:\n"
        f"{prompt_json(estimand.model_dump())}\n\n"
        "Available columns:\n"
        f"{prompt_json(available_columns)}\n\n"
        "Supported estimators: OLS, TWFE.\n\n"
        "Create a SpecMenuProposal with exactly these 7 axes:\n"
        "control_set, sample, pre_period, clustering, never_treated, estimator, weighting.\n\n"
        "For each axis provide baseline_level, candidate_levels, and rationale.\n"
        "This is NOT a closed list — propose defensible levels grounded in the estimand and columns.\n"
        "Be conservative: never invent columns or unsupported estimators.\n\n"
        "Encoding rules:\n"
        "- control_set: 'none' for no controls, or 'col1+col2' for control sets\n"
        "- sample: 'none' for full sample, or a pandas query string\n"
        "- pre_period: 'none' or integer years (e.g. '3')\n"
        "- clustering: column name\n"
        "- never_treated: 'true' or 'false'\n"
        "- estimator: 'OLS' or 'TWFE'\n"
        f"- weighting: '{DEFAULT_WEIGHT_COL}' (population-weighted default) or 'none' for unweighted\n"
    )

    agent = build_agent(
        ModelRole.JUDGE,
        system_prompt=(
            "You design specification-curve robustness menus for causal inference. "
            "For each axis, recommend a defensible baseline and candidate levels with rationale — "
            "not a fixed closed list. Never invent columns or unsupported estimators. "
            "If the estimand is unclear for menu design, set needs_clarification=true."
        ),
        output_type=SpecMenuProposal,
    )
    return agent.run_sync(prompt).output


# -----------------------------
# PROPOSAL → MENU
# -----------------------------


def spec_menu_proposal_to_menu(
    proposal: SpecMenuProposal,
    *,
    available_columns: list[str],
) -> SpecMenu:
    cols = set(available_columns)

    def axis(name: str) -> list[str]:
        return _axis_levels(proposal, name)

    control_sets = [_parse_control_set(v) for v in axis("control_set")]
    sample_filters = tuple(None if v == "none" else v for v in axis("sample"))
    pre_period_windows = tuple(None if v == "none" else int(v) for v in axis("pre_period"))
    clustering_levels = tuple(axis("clustering"))
    never_treated_levels = tuple(v.lower() == "true" for v in axis("never_treated"))
    weighting_levels = tuple(None if v == "none" else v for v in axis("weighting"))
    estimator_levels = tuple(axis("estimator"))

    if estimator_levels[0] not in SUPPORTED_ESTIMATORS:
        raise ValueError(f"Unsupported estimator: {estimator_levels[0]}")
    if clustering_levels[0] not in cols:
        raise ValueError(f"Invalid clustering column: {clustering_levels[0]}")

    for w in weighting_levels:
        if w is not None and w not in cols:
            raise ValueError(f"Invalid weight column: {w}")

    return SpecMenu(
        control_sets=control_sets,
        sample_filters=sample_filters,
        pre_period_windows=pre_period_windows,
        clustering_levels=clustering_levels,
        never_treated_levels=never_treated_levels,
        estimators=estimator_levels,
        weighting_levels=weighting_levels,
        rationale=proposal.overall_rationale,
    )


# -----------------------------
# FREEZE FLOW
# -----------------------------


def freeze_spec_menu(
    proposal: SpecMenuProposal,
    *,
    available_columns: list[str],
    approved: bool,
) -> FrozenSpecMenu:
    if proposal.needs_clarification:
        raise ValueError(proposal.clarification_question or "Clarification required")
    if not approved:
        raise ValueError("User approval required to freeze spec menu")

    menu = spec_menu_proposal_to_menu(proposal, available_columns=available_columns)
    return FrozenSpecMenu(menu=menu, menu_hash=_menu_hash(menu))


# -----------------------------
# VALIDATION (SpecMenu → Specification, fail-loud)
# -----------------------------


def validate_spec_menu_to_specs(
    frozen_menu: FrozenSpecMenu,
    specs: list[Specification],
) -> None:
    menu = frozen_menu.menu
    errors: list[str] = []

    for spec in specs:
        if list(spec.controls) not in menu.control_sets:
            errors.append(f"{spec.spec_id}: invalid controls {list(spec.controls)!r}")
        if spec.sample_filter not in menu.sample_filters:
            errors.append(f"{spec.spec_id}: invalid sample_filter {spec.sample_filter!r}")
        if spec.pre_period_window not in menu.pre_period_windows:
            errors.append(f"{spec.spec_id}: invalid pre_period {spec.pre_period_window!r}")
        if spec.cluster_by not in menu.clustering_levels:
            errors.append(f"{spec.spec_id}: invalid cluster_by {spec.cluster_by!r}")
        if spec.include_never_treated not in menu.never_treated_levels:
            errors.append(f"{spec.spec_id}: invalid never_treated {spec.include_never_treated!r}")
        if spec.estimator not in menu.estimators:
            errors.append(f"{spec.spec_id}: invalid estimator {spec.estimator!r}")
        if spec.weight_col not in menu.weighting_levels:
            errors.append(f"{spec.spec_id}: invalid weight_col {spec.weight_col!r}")

    if errors:
        raise ValueError("Spec validation failed: " + "; ".join(errors))


# -----------------------------
# EXPANSION (CARTESIAN PRODUCT)
# -----------------------------


def expand_to_specs(
    frozen_menu: FrozenSpecMenu,
    *,
    outcome: str,
    treatment: str,
    unit_col: str,
    time_col: str,
) -> list[Specification]:
    """Aktif eksenlerin faktöriyel çarpımı. Sessiz eksenler baseline'a pinli. Sert tavan 24."""
    menu = frozen_menu.menu
    active = set(menu.active_axes) if menu.active_axes else _default_active(menu)

    def levels(axis: str, all_levels: list):  # noqa: ANN001, ANN202
        return all_levels if axis in active else all_levels[:1]

    combos = list(
        itertools.product(
            levels("control_set", menu.control_sets),
            levels("sample", list(menu.sample_filters)),
            levels("pre_period", list(menu.pre_period_windows)),
            levels("clustering", list(menu.clustering_levels)),
            levels("never_treated", list(menu.never_treated_levels)),
            levels("estimator", list(menu.estimators)),
            levels("weighting", list(menu.weighting_levels)),
        )
    )

    if len(combos) > HARD_CAP:
        raise ValueError(
            f"{len(combos)} spesifikasyon üretildi, sert tavan {HARD_CAP}. "
            "Aktif eksen sayısını azaltın — sessiz kırpma yapılmaz."
        )

    specs: list[Specification] = []
    for i, (controls, sample, pre, cluster, never, est, weight) in enumerate(combos):
        if est not in SUPPORTED_ESTIMATORS:
            raise ValueError(f"Desteklenmeyen estimator: {est}")
        specs.append(
            Specification(
                spec_id=f"spec_{i:04d}",
                outcome=outcome,
                treatment=treatment,
                controls=tuple(controls),
                unit_fe=unit_col if est == "TWFE" else None,
                time_fe=time_col if est == "TWFE" else None,
                pre_period_window=pre,
                cluster_by=cluster,
                estimator=est,
                sample_filter=sample,
                include_never_treated=never,
                weight_col=weight,
            )
        )

    logger.info("expand_to_specs: %d spec (aktif eksenler: %s)", len(specs), sorted(active))
    return specs
