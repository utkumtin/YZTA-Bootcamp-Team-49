import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")  # tek-lib estimator; dep yoksa atla, CI'da koşar

from pareto.analysis.estimators import estimate_one  # noqa: E402
from pareto.spec import Specification  # noqa: E402


def _cross(effect: float = 0.8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    d = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    y = effect * d + 0.3 * x1 + rng.normal(0, 0.5, size=n)
    g = rng.integers(0, 15, n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "g": g})


def _panel(effect: float = 0.8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(40):
        treated = unit % 2
        u_fe = rng.normal()
        for year in range(6):
            post = 1 if year >= 3 else 0
            d = treated * post
            y = effect * d + u_fe + 0.1 * year + rng.normal(0, 0.3)
            rows.append({"y": y, "d": d, "unit": unit, "year": year})
    return pd.DataFrame(rows)


def test_ols_recovers_known_effect():
    # NEDEN: estimator kütüphaneden doğru besleniyor mu — bilinen 0.8 etkisini geri almalı.
    df = _cross(0.8)
    spec = Specification(
        spec_id="s", outcome="y", treatment="d", controls=("x1",), cluster_by="g", estimator="OLS"
    )
    res = estimate_one(spec, df)
    assert res.status == "ok"
    assert res.coefficient == pytest.approx(0.8, abs=0.15)


def test_twfe_recovers_known_effect_same_lib():
    # NEDEN: AYNI lib (pyfixest) TWFE'yi de koşar (tek-lib kararı); FE'ler etki+trendi soğurur.
    df = _panel(0.8)
    spec = Specification(
        spec_id="s",
        outcome="y",
        treatment="d",
        cluster_by="unit",
        unit_fe="unit",
        time_fe="year",
        estimator="TWFE",
    )
    res = estimate_one(spec, df)
    assert res.status == "ok"
    assert res.coefficient == pytest.approx(0.8, abs=0.2)


def test_bad_spec_fails_soft_not_crash():
    # NEDEN: per-spec izolasyon — hata koşuyu çökertmez, status='failed' döner.
    df = _cross()
    spec = Specification(
        spec_id="s", outcome="y", treatment="d", cluster_by="NOPE", estimator="OLS"
    )
    res = estimate_one(spec, df)
    assert res.status == "failed"
    assert res.error
