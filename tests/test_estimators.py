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


def test_none_clustering_runs_robust_se():
    # NEDEN: cluster_by=None sessizce bir kolona pinlenmemeli. Katsayı aynı kalır,
    # SE değişir — clustered ile robust yol gerçekten ayrışıyor mu, onu ölçer. Ayrıca
    # robust yolun spesifik olarak HC1 kullandığını pinler: "iid" (klasik OLS SE) ile
    # CRV1 arasındaki fark 1e-9 toleransının çok üzerinde olduğundan, vcov sessizce
    # "iid"ye regrese olsa bile eski assertion'lar yine geçerdi.
    df = _cross(0.8)
    shared = {"outcome": "y", "treatment": "d", "controls": ("x1",), "estimator": "OLS"}
    robust = estimate_one(Specification(spec_id="robust", cluster_by=None, **shared), df)
    clustered = estimate_one(Specification(spec_id="clustered", cluster_by="g", **shared), df)

    assert robust.status == "ok"
    assert robust.coefficient == pytest.approx(clustered.coefficient, abs=1e-9)
    assert robust.std_error != pytest.approx(clustered.std_error, abs=1e-9)

    import pyfixest as pf

    hc1_fit = pf.feols("y ~ d + x1", data=df, vcov="HC1")
    assert robust.std_error == pytest.approx(float(hc1_fit.se()["d"]), abs=1e-9)


def test_none_clustering_does_not_require_cluster_column():
    # NEDEN: kümeleme kolonu dropna listesine girerse, kümeleme YOKKEN bile eksik
    # değerler yüzünden satır düşerdi; efektif N sessizce sapar.
    df = _cross(0.8)
    df.loc[:49, "g"] = np.nan
    spec = Specification(
        spec_id="s", outcome="y", treatment="d", controls=("x1",), cluster_by=None, estimator="OLS"
    )
    res = estimate_one(spec, df)
    assert res.status == "ok"
    assert res.n_obs == len(df)


def test_twfe_none_clustering_overrides_fe_cluster_default():
    # NEDEN: _feols_kwargs, OLSEstimator ve TWFEEstimator arasında paylaşılır ama yalnız
    # OLS + cluster_by=None yolu test ediliyordu. FE'li modellerde vcov="hetero" açıkça
    # geçilmezse, _feols_kwargs unit_fe üzerinden CRV1 kümelemeye (R fixest'in klasik
    # davranışı) sessizce dönebilir ve "kümeleme yok" ekseni bir yalana dönüşür. Bu test
    # override'ın TWFE yolunda da gerçekten gerçekleştiğini kanıtlar: SE, explicit HC1
    # fit'iyle birebir eşleşmeli ve unit üzerinde açıkça kümelenmiş (CRV1) bir fit'ten
    # farklı olmalı.
    df = _panel(0.8)
    spec = Specification(
        spec_id="s",
        outcome="y",
        treatment="d",
        cluster_by=None,
        unit_fe="unit",
        time_fe="year",
        estimator="TWFE",
    )
    res = estimate_one(spec, df)
    assert res.status == "ok"
    assert res.coefficient == pytest.approx(0.8, abs=0.2)

    import pyfixest as pf

    hc1_fit = pf.feols("y ~ d | unit + year", data=df, vcov="HC1")
    assert res.std_error == pytest.approx(float(hc1_fit.se()["d"]), abs=1e-9)

    unit_clustered_fit = pf.feols("y ~ d | unit + year", data=df, vcov={"CRV1": "unit"})
    assert res.std_error != pytest.approx(float(unit_clustered_fit.se()["d"]), abs=1e-9)


def test_bad_spec_fails_soft_not_crash():
    # NEDEN: per-spec izolasyon — hata koşuyu çökertmez, status='failed' döner.
    df = _cross()
    spec = Specification(
        spec_id="s", outcome="y", treatment="d", cluster_by="NOPE", estimator="OLS"
    )
    res = estimate_one(spec, df)
    assert res.status == "failed"
    assert res.error
