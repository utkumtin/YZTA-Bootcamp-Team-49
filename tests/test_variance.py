from pareto.analysis.variance import Band, summarize
from pareto.contracts import EstimationResult


def _r(spec_id: str, coef: float, ci_low: float, ci_high: float) -> EstimationResult:
    return EstimationResult(
        spec_id=spec_id,
        estimator="OLS",
        coefficient=coef,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=0.01,
        status="ok",
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
