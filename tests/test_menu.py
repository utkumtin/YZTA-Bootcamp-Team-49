import pytest

from pareto.analysis.menu import SpecMenu, expand_to_specs


def _frozen(**kw):
    base = {"control_sets": [["x1"], ["x1", "x2"]], "clustering_levels": ["g"]}
    base.update(kw)
    return SpecMenu(**base).freeze()


def test_freeze_is_deterministic_16char_hash():
    # NEDEN: reprodüksiyon garantisi dondurmadan gelir; aynı menü → aynı hash.
    assert _frozen().menu_hash == _frozen().menu_hash
    assert len(_frozen().menu_hash) == 16


def test_silent_axis_pinned_to_baseline():
    # NEDEN: aktif olmayan eksen baseline'a (ilk seviye) pinlenir → okunaklı + tekrarlanabilir.
    frozen = _frozen(estimators=["OLS"], active_axes=["control_set"])
    specs = expand_to_specs(frozen, outcome="y", treatment="d", unit_col="u", time_col="t")
    assert len(specs) == 2  # yalnız kontrol seti ekseni açık
    assert {s.estimator for s in specs} == {"OLS"}


def test_hard_cap_24_fails_loud():
    # NEDEN: sert tavan 24. Aşımda sessiz kırpma YOK — patlar.
    frozen = SpecMenu(
        control_sets=[[f"x{i}"] for i in range(30)],
        clustering_levels=["g"],
        active_axes=["control_set"],
    ).freeze()
    with pytest.raises(ValueError):
        expand_to_specs(frozen, outcome="y", treatment="d", unit_col="u", time_col="t")
