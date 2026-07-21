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


def test_cluster_by_none_is_valid_and_distinct_from_clustered():
    # NEDEN: "kümeleme yok" savunulabilir bir clustering seviyesi; JUDGE bunu önerebilmeli.
    # Ayrı bir hash üretmeli, yoksa iki seviye multiverse'te aynı spec'e çöker.
    unclustered = Specification(spec_id="s", outcome="y", treatment="d", cluster_by=None)
    clustered = Specification(spec_id="s", outcome="y", treatment="d", cluster_by="g")
    assert unclustered.cluster_by is None
    assert unclustered.content_hash() != clustered.content_hash()


def test_content_hash_stable_after_optional_cluster_by():
    # NEDEN: cluster_by'ın opsiyonelleşmesi mevcut donmuş hash'leri KIRMAMALI.
    # Altın değer, cluster_by'ın `str` olduğu sürümde üretildi.
    spec = Specification(spec_id="a", outcome="y", treatment="d", cluster_by="g")
    assert spec.content_hash() == "86a56fa79042cf0b"
