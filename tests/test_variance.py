from pareto.analysis.variance import Band, diagnose_axes, summarize
from pareto.contracts import EstimationResult
from pareto.spec import Specification


def _r(
    spec_id: str,
    coef: float | None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    *,
    status: str = "ok",
) -> EstimationResult:
    return EstimationResult(
        spec_id=spec_id,
        estimator="OLS",
        coefficient=coef,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=0.01,
        status=status,
    )


def test_robust_when_sign_and_significance_high():
    # NEDEN: işaret-uyumu ≥%95 VE anlamlılık ≥%70 → Robust.
    results = [_r(f"s{i}", 1.0, 0.5, 1.5) for i in range(10)]
    s = summarize(results)
    assert s["sign_agreement"] == 1.0
    assert s["band"] == Band.ROBUST.value


def test_fragile_when_sign_flips_across_specs():
    # NEDEN: işaret-uyumu <%90 → Kırılgan; ürünün "yayından önce bilmem lazım" mesajı.
    results = [_r(f"p{i}", 1.0, 0.5, 1.5) for i in range(6)]
    results += [_r(f"n{i}", -1.0, -1.5, -0.5) for i in range(4)]
    s = summarize(results)
    assert s["band"] == Band.FRAGILE.value


def test_failed_specs_excluded_from_summary():
    ok = [_r("a", 1.0, 0.5, 1.5)]
    failed = [EstimationResult(spec_id="b", estimator="OLS", status="failed", error="singular")]
    s = summarize(ok + failed)
    assert s["n_ok"] == 1 and s["n_failed"] == 1


def _spec(
    spec_id: str,
    *,
    estimator: str = "OLS",
    controls: tuple[str, ...] = (),
    sample_filter: str | None = None,
    cluster_by: str = "g",
    include_never_treated: bool = True,
    weight_col: str | None = None,
) -> Specification:
    return Specification(
        spec_id=spec_id,
        outcome="y",
        treatment="d",
        controls=controls,
        unit_fe="unit" if estimator == "TWFE" else None,
        time_fe="year" if estimator == "TWFE" else None,
        cluster_by=cluster_by,
        estimator=estimator,
        sample_filter=sample_filter,
        include_never_treated=include_never_treated,
        weight_col=weight_col,
    )


def test_diagnose_axes_attributes_known_sign_flip_to_estimator_axis():
    specs = [
        _spec("ols", estimator="OLS"),
        _spec("twfe", estimator="TWFE"),
    ]
    results = [_r("ols", 1.0, 0.5, 1.5), _r("twfe", -1.0, -1.5, -0.5)]

    out = diagnose_axes(results, specs)

    assert out["matched_pairs"]["estimator"]["n_pairs"] == 1
    assert out["matched_pairs"]["estimator"]["sign_flip_count"] == 1
    assert out["matched_pairs"]["estimator"]["sign_flip_rate"] == 1.0
    assert out["matched_pairs"]["control_set"]["n_pairs"] == 0


def test_diagnose_axes_treats_zero_coefficients_as_sign_neutral():
    specs = [
        _spec("zero_pos_a", estimator="OLS", controls=("zero_pos",)),
        _spec("zero_pos_b", estimator="TWFE", controls=("zero_pos",)),
        _spec("zero_neg_a", estimator="OLS", controls=("zero_neg",)),
        _spec("zero_neg_b", estimator="TWFE", controls=("zero_neg",)),
        _spec("pos_neg_a", estimator="OLS", controls=("pos_neg",)),
        _spec("pos_neg_b", estimator="TWFE", controls=("pos_neg",)),
    ]
    results = [
        _r("zero_pos_a", 0.0),
        _r("zero_pos_b", 1.0),
        _r("zero_neg_a", 0.0),
        _r("zero_neg_b", -1.0),
        _r("pos_neg_a", 1.0),
        _r("pos_neg_b", -1.0),
    ]

    out = diagnose_axes(results, specs)
    estimator_pairs = out["matched_pairs"]["estimator"]

    assert estimator_pairs["n_pairs"] == 3
    assert estimator_pairs["sign_comparable_pairs"] == 1
    assert estimator_pairs["sign_flip_count"] == 1
    assert estimator_pairs["sign_flip_rate"] == 1.0


def test_diagnose_axes_sign_flip_rate_none_when_no_sign_comparable_pairs():
    specs = [
        _spec("zero_pos_a", estimator="OLS", controls=("zero_pos",)),
        _spec("zero_pos_b", estimator="TWFE", controls=("zero_pos",)),
        _spec("zero_neg_a", estimator="OLS", controls=("zero_neg",)),
        _spec("zero_neg_b", estimator="TWFE", controls=("zero_neg",)),
    ]
    results = [
        _r("zero_pos_a", 0.0),
        _r("zero_pos_b", 1.0),
        _r("zero_neg_a", 0.0),
        _r("zero_neg_b", -1.0),
    ]

    out = diagnose_axes(results, specs)
    estimator_pairs = out["matched_pairs"]["estimator"]

    assert estimator_pairs["n_pairs"] == 2
    assert estimator_pairs["sign_comparable_pairs"] == 0
    assert estimator_pairs["sign_flip_count"] == 0
    assert estimator_pairs["sign_flip_rate"] is None


def test_diagnose_axes_counts_only_pairs_with_one_axis_different():
    specs = [
        _spec("base", estimator="OLS"),
        _spec("est", estimator="TWFE"),
        _spec("est_controls", estimator="TWFE", controls=("x1",)),
    ]
    results = [_r("base", 1.0), _r("est", 2.0), _r("est_controls", 3.0)]

    out = diagnose_axes(results, specs)

    assert out["matched_pairs"]["estimator"]["n_pairs"] == 1
    assert out["matched_pairs"]["control_set"]["n_pairs"] == 1


def test_diagnose_axes_excludes_failed_and_missing_coefficients():
    specs = [_spec("a", estimator="OLS"), _spec("b", estimator="TWFE"), _spec("c", controls=("x",))]
    results = [
        _r("a", 1.0),
        _r("b", 2.0, status="failed"),
        _r("c", None),
    ]

    out = diagnose_axes(results, specs)

    assert out["n_used"] == 1
    assert out["n_excluded"] == {"failed": 1, "missing_coefficient": 1, "missing_spec": 0}
    assert out["matched_pairs"]["estimator"]["n_pairs"] == 0
    assert any("başarısız" in warning for warning in out["warnings"])
    assert any("coefficient=None" in warning for warning in out["warnings"])


def test_diagnose_axes_returns_zero_pairs_when_no_matches_exist():
    specs = [
        _spec("a", estimator="OLS", controls=("x1",)),
        _spec("b", estimator="TWFE", controls=("x2",)),
    ]
    results = [_r("a", 1.0), _r("b", -1.0)]

    out = diagnose_axes(results, specs)

    assert out["matched_pairs"]["estimator"]["n_pairs"] == 0
    assert out["matched_pairs"]["control_set"]["n_pairs"] == 0


def test_diagnose_axes_anova_partial_r2_detects_dominant_estimator_axis():
    specs = []
    results = []
    i = 0
    for estimator in ("OLS", "TWFE"):
        for controls in ((), ("x1",)):
            for sample in (None, "region == 'A'"):
                for noise in (-0.05, 0.05):
                    spec_id = f"s{i}"
                    coef = (-5.0 if estimator == "OLS" else 5.0)
                    coef += 0.2 if controls else 0.0
                    coef += 0.1 if sample else 0.0
                    coef += noise
                    specs.append(
                        _spec(
                            spec_id,
                            estimator=estimator,
                            controls=controls,
                            sample_filter=sample,
                        )
                    )
                    results.append(_r(spec_id, coef))
                    i += 1

    out = diagnose_axes(results, specs)
    partial = out["anova_partial_r2"]

    assert partial["estimator"] is not None
    assert partial["control_set"] is not None
    assert partial["sample"] is not None
    assert partial["estimator"] > partial["control_set"]
    assert partial["estimator"] > partial["sample"]


def test_diagnose_axes_anova_partial_r2_is_none_for_single_level_axis():
    specs = [
        _spec("a", controls=()),
        _spec("b", controls=("x1",)),
        _spec("c", sample_filter="region == 'A'"),
        _spec("d", controls=("x1",), sample_filter="region == 'A'"),
    ]
    results = [_r("a", 1.0), _r("b", 1.2), _r("c", 1.1), _r("d", 1.3)]

    out = diagnose_axes(results, specs)

    assert out["anova_partial_r2"]["estimator"] is None


def test_diagnose_axes_excludes_unmatched_spec_id_with_warning():
    specs = [_spec("a"), _spec("b", estimator="TWFE")]
    results = [_r("a", 1.0), _r("missing", -1.0), _r("b", 2.0)]

    out = diagnose_axes(results, specs)

    assert out["n_used"] == 2
    assert out["n_excluded"]["missing_spec"] == 1
    assert any("eşleşen spec_id" in warning for warning in out["warnings"])
