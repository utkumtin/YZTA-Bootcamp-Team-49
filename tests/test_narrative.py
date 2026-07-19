"""Varyans anlatısı testleri: TestModel ile API yakmadan JUDGE dikişi doğrulanır."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from pareto.llm.narrative import VarianceNarrative, generate_narrative
from pareto.llm.router import use_test_model


def _summary() -> dict[str, object]:
    return {
        "n_total": 10,
        "n_ok": 10,
        "n_failed": 0,
        "sign_agreement": 0.8,
        "significance_rate": 0.7,
        "point_min": -0.021,
        "point_max": 0.034,
        "modal_sign": 1,
        "band": "fragile",
    }


def _diagnosis() -> dict[str, object]:
    return {
        "n_results": 10,
        "n_specs": 10,
        "n_used": 10,
        "n_excluded": {"failed": 0, "missing_coefficient": 0, "missing_spec": 0},
        "axes": ["estimator", "clustering"],
        "matched_pairs": {
            "estimator": {"n_pairs": 5, "sign_flip_rate": 0.4},
            "clustering": {"n_pairs": 5, "sign_flip_rate": 0.0},
        },
        "anova_partial_r2": {"estimator": 0.82, "clustering": 0.03},
        "warnings": [],
    }


def _narrative_args() -> dict[str, object]:
    return {
        "ozet": "10 spesifikasyonun 8'i pozitif; işaret dönüşlerini estimator ekseni sürüklüyor.",
        "eksen_yorumlari": [
            {
                "axis": "estimator",
                "yorum": "Eşleşmiş çiftlerde işaret dönüşü yalnız estimator değişiminde görülüyor.",
            }
        ],
    }


def test_testmodel_fake_diagnosis_returns_expected_narrative_skeleton() -> None:
    with use_test_model(TestModel(custom_output_args=_narrative_args())):
        narrative = generate_narrative(_summary(), _diagnosis())

    assert isinstance(narrative, VarianceNarrative)
    assert "estimator" in narrative.ozet
    assert [c.axis for c in narrative.eksen_yorumlari] == ["estimator"]
    assert narrative.eksen_yorumlari[0].yorum


def test_unknown_axis_in_narrative_fails_loud() -> None:
    args = _narrative_args()
    args["eksen_yorumlari"] = [{"axis": "weather", "yorum": "uydurma eksen"}]

    with use_test_model(TestModel(custom_output_args=args)):
        with pytest.raises(ValueError, match="olmayan eksen"):
            generate_narrative(_summary(), _diagnosis())


def test_no_successful_results_raises_without_calling_model() -> None:
    empty_summary = {**_summary(), "n_ok": 0}

    with pytest.raises(ValueError, match="başarılı sonuç yok"):
        generate_narrative(empty_summary, _diagnosis())
