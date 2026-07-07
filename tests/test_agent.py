from typing import get_args

import pandas as pd
import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from pareto.cleaning.agent import (
    CleaningProposal,
    TransformCall,
    TransformDecision,
    _build_judge_prompt,
    generate_ledger,
)
from pareto.cleaning.transforms import ALLOWED_TRANSFORMS, apply_transform
from pareto.llm.guardrails import spotlight
from pareto.llm.router import use_test_model


def _fake_profile() -> dict:
    return {
        "n_rows": 100,
        "n_cols": 3,
        "columns": {
            "county_fips": {
                "dtype": "object",
                "n_missing": 0,
                "pct_missing": 0.0,
                "n_unique": 90,
                "top_values": {"1001": 2, "1003": 2},
                "looks_like_date": False,
            },
            "deaths": {
                "dtype": "object",
                "n_missing": 60,
                "pct_missing": 0.6,
                "n_unique": 40,
                "top_values": {"Suppressed": 30, "12": 3},
                "looks_like_date": False,
            },
            "year": {
                "dtype": "int64",
                "n_missing": 0,
                "pct_missing": 0.0,
                "n_unique": 10,
                "stats": {"min": 2010.0, "max": 2019.0, "mean": 2014.5, "std": 2.9},
            },
        },
        "potential_join_keys": ["county_fips"],
        "duplicate_row_count": 0,
    }


def _decision(transform: dict, *, confidence: str = "high") -> dict:
    return {
        "bulgu": "profilde tespit edilen bulgu",
        "transform": transform,
        "gerekce": "vetted transform gerekçesi",
        "confidence": confidence,
    }


def test_testmodel_fake_profile_selects_vetted_transforms_and_flags():
    # NEDEN: JUDGE çıktısı kapalı sözlükten tipli karara inmeli; yüksek güven
    # otomatik yola (bayrak False), gerçek yargı gatekeeper'a (bayrak True) düşmeli.
    args = {
        "decisions": [
            _decision(
                {"transform_name": "preserve_leading_zeros", "col": "county_fips", "width": 5},
                confidence="high",
            ),
            _decision(
                {"transform_name": "rename_column", "old": "year", "new": "yil"},
                confidence="low",
            ),
        ]
    }

    with use_test_model(TestModel(custom_output_args=args)):
        entries = generate_ledger(_fake_profile())

    assert [e.transform_name for e in entries] == ["preserve_leading_zeros", "rename_column"]
    assert entries[0].params == {"col": "county_fips", "width": 5}
    assert entries[0].belirsizlik_bayragi is False
    assert entries[1].belirsizlik_bayragi is True
    assert all(e.timestamp for e in entries)  # denetim izi zaman damgalı


def test_high_missing_column_is_always_gatekept():
    # NEDEN: eşik üstü eksik oranlı kolonda karar, model "emin" olsa bile insana
    # sorulur; gatekeeper eşiği LLM güvenine devredilmez.
    args = {
        "decisions": [
            _decision(
                {"transform_name": "standardize_na", "col": "deaths", "markers": ["Suppressed"]},
                confidence="high",
            )
        ]
    }

    with use_test_model(TestModel(custom_output_args=args)):
        entries = generate_ledger(_fake_profile())

    assert entries[0].belirsizlik_bayragi is True


def test_schema_rejects_unvetted_transform():
    # NEDEN: L3 savunması; kapalı sözlük dışı ad (keyfi kod yolu) şemada ifade edilemez.
    with pytest.raises(ValidationError):
        CleaningProposal(
            decisions=[_decision({"transform_name": "exec_python", "code": "import os"})]
        )


def test_closed_schema_matches_vetted_registry_exactly():
    # NEDEN: şema ile vetted registry aynı kapalı taksonomiyi taşımalı; yeni bir
    # transform eklendiğinde ya da adı değiştiğinde sapma burada patlar.
    union = get_args(TransformCall)[0]
    schema_names = {
        get_args(member.model_fields["transform_name"].annotation)[0] for member in get_args(union)
    }
    assert schema_names == set(ALLOWED_TRANSFORMS)


def test_typed_params_bind_to_vetted_apply_signatures():
    # NEDEN: şemadaki tipli parametreler gerçek apply imzalarına birebir oturmalı;
    # şema-imza kayması üretimde patlamadan burada yakalanır.
    calls: dict[str, dict] = {
        "rename_column": {"old": "a", "new": "b"},
        "coerce_numeric": {"col": "a"},
        "preserve_leading_zeros": {"col": "a", "width": 5},
        "parse_date": {"col": "a", "fmt": "%Y"},
        "standardize_na": {"col": "a", "markers": ["-999"]},
        "drop_duplicates": {},
    }
    assert set(calls) == set(ALLOWED_TRANSFORMS)

    df = pd.DataFrame({"a": ["1", "-999"]})
    for name, params in calls.items():
        decision = TransformDecision(
            bulgu="b",
            transform={"transform_name": name, **params},
            gerekce="g",
            confidence="high",
        )
        apply_transform(df, decision.transform.transform_name, decision.transform.params())


def test_generate_ledger_fails_loud_on_invented_column():
    # NEDEN: JUDGE kolon uyduramaz; profilde olmayan kolon sessizce ledger'a giremez.
    args = {"decisions": [_decision({"transform_name": "coerce_numeric", "col": "ghost"})]}

    with (
        use_test_model(TestModel(custom_output_args=args)),
        pytest.raises(ValueError, match="ghost"),
    ):
        generate_ledger(_fake_profile())


def test_judge_prompt_spotlights_untrusted_content():
    # NEDEN: L2 katmanı; kullanıcı verisinden gelen kolon adları JUDGE istemine
    # işaretsiz (talimat gibi okunabilir) giremez.
    prompt = _build_judge_prompt(_fake_profile())

    assert spotlight("county_fips") in prompt
    assert "county_fips" not in prompt.replace(spotlight("county_fips"), "")


def test_clean_profile_yields_empty_ledger():
    # NEDEN: kanıt yoksa karar yok; JUDGE zorlama transform üretmez.
    with use_test_model(TestModel(custom_output_args={"decisions": []})):
        assert generate_ledger(_fake_profile()) == []


def test_generate_ledger_requires_columns():
    # NEDEN: boş profil sessizce boş ledger döndüremez; fail-loud.
    with pytest.raises(ValueError, match="kolon yok"):
        generate_ledger({"columns": {}})
