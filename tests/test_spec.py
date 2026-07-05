import pytest

from pareto.spec import Specification


def test_twfe_requires_fixed_effects():
    # NEDEN: TWFE = OLS + unit/time FE. FE'siz TWFE sessizce OLS'e düşerse estimator
    # ekseni anlamsızlaşır (sign-flip demosu çürür). Fail-loud olmalı.
    with pytest.raises(ValueError):
        Specification(spec_id="s", outcome="y", treatment="d", cluster_by="g", estimator="TWFE")


def test_ols_valid_without_fe():
    spec = Specification(spec_id="s", outcome="y", treatment="d", cluster_by="g", estimator="OLS")
    assert spec.estimator == "OLS"


def test_content_hash_ignores_spec_id_but_reflects_content():
    # NEDEN: dedup + reprodüksiyon spec_id'den değil, içerik hash'inden gelmeli.
    a = Specification(spec_id="a", outcome="y", treatment="d", cluster_by="g")
    b = Specification(spec_id="b", outcome="y", treatment="d", cluster_by="g")
    c = Specification(spec_id="a", outcome="y", treatment="d", cluster_by="g", controls=("x",))
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()
