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


# ---------------------------------------------------------------------------
# Örneklem eksenleri: pre_period ve never_treated
#
# NEDEN bu testler var: her iki alan da `Specification`'da tanımlıydı, menüde
# üretiliyordu ve varyans panelinde eksen olarak etiketleniyordu ama hiçbir
# estimator okumuyordu. Sonuç: aynı eğrinin katları birbirinin birebir kopyası
# oluyordu (ölçüldü: demo koşusunda 24 spec'in 12'si kopya). Bu testler eksenin
# veriyi gerçekten değiştirdiğini bağlar; sessiz no-op'a geri dönüş fark edilir.
# ---------------------------------------------------------------------------


def _twfe_spec(**overrides) -> Specification:
    base = {
        "spec_id": "s",
        "outcome": "y",
        "treatment": "d",
        "unit_fe": "unit",
        "time_fe": "year",
        "cluster_by": "unit",
        "estimator": "TWFE",
    }
    return Specification(**{**base, **overrides})


def test_pre_period_window_trims_sample_and_moves_estimate():
    """Dar pencere = tedaviye uzak yılları dışarıda bırakmak.

    Panelde tedavi year=3'te başlıyor; pencere 1 ise year>=2 kalmalı, yani 6
    dönemden 4'ü düşer. Katsayının kendisi de değişmeli — aksi halde eksen
    kullanıcıya dayanıklılık kontrolü gibi görünüp hiçbir şey sınamıyor olur.
    """
    df = _panel()

    full = estimate_one(_twfe_spec(spec_id="full"), df)
    trimmed = estimate_one(_twfe_spec(spec_id="win1", pre_period_window=1), df)

    assert full.status == "ok" and trimmed.status == "ok"
    # year >= 3 - 1 = 2 → 6 dönemden 4'ü kalır
    assert trimmed.n_obs == int((df["year"] >= 2).sum())
    assert trimmed.n_obs < full.n_obs
    assert trimmed.coefficient != full.coefficient


def test_pre_period_window_none_leaves_sample_untouched():
    """Varsayılan yol değişmedi: pencere yoksa tek satır bile düşmez."""
    df = _panel()
    result = estimate_one(_twfe_spec(pre_period_window=None), df)

    assert result.status == "ok"
    assert result.n_obs == len(df)


def test_pre_period_window_without_time_column_fails_loud():
    """Zaman kolonu yoksa sessizce no-op'a düşmek yasak.

    Sessiz dönüş tam olarak düzeltilen hatanın kendisiydi: eksen menüde
    görünürken veriye hiç dokunmuyordu.
    """
    df = _panel()
    spec = Specification(
        spec_id="ols_win",
        outcome="y",
        treatment="d",
        cluster_by="unit",
        estimator="OLS",
        pre_period_window=1,
    )
    result = estimate_one(spec, df)

    assert result.status == "failed"
    assert "pre_period_window" in (result.error or "")


def _staggered(seed: int = 0) -> pd.DataFrame:
    """Kademeli benimseme + hiç tedavi görmeyen birimler.

    never_treated ekseninin ucunda tanımlama hâlâ mümkün olmalı; eşzamanlı
    benimsemede never-treated'ı atmak tedaviyi yıl FE'siyle eşdoğrusal yapar.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(45):
        adopt = {0: None, 1: 2, 2: 4}[unit % 3]  # üçte biri hiç tedavi görmez
        u_fe = rng.normal()
        for year in range(6):
            d = 1 if adopt is not None and year >= adopt else 0
            rows.append(
                {
                    "y": 0.8 * d + u_fe + 0.1 * year + rng.normal(0, 0.3),
                    "d": d,
                    "unit": unit,
                    "year": year,
                }
            )
    return pd.DataFrame(rows)


def test_excluding_never_treated_drops_units_that_are_never_treated():
    """Karşılaştırma grubunu 'hiç tedavi görmeyen'den 'henüz görmeyen'e çevirir.

    Panelde birimlerin üçte biri hiç tedavi görmüyor; eksen kapatıldığında tam
    olarak onlar düşmeli ve tahmin yine de koşabilmeli.
    """
    df = _staggered()
    treated_units = set(df.loc[df["d"] == 1, "unit"].unique())

    kept = estimate_one(_twfe_spec(spec_id="with", include_never_treated=True), df)
    dropped = estimate_one(_twfe_spec(spec_id="without", include_never_treated=False), df)

    assert kept.status == "ok" and dropped.status == "ok"
    assert kept.n_obs == len(df)
    assert dropped.n_obs == int(df["unit"].isin(treated_units).sum())
    assert dropped.n_obs < kept.n_obs


def test_excluding_never_treated_fails_loud_when_it_removes_the_comparison_group():
    """Eşzamanlı benimsemede karşılaştırma grubu kalmaz.

    Doğru davranış sessizce tam örneklem sonucunu döndürmek değil, spec'i
    `status='failed'` ile düşürmek: kullanıcı eksenin bu ucunun tanımlanamaz
    olduğunu görmeli.
    """
    df = _panel()  # birimlerin yarısı tedavi, hepsi aynı yıl

    result = estimate_one(_twfe_spec(spec_id="no_control", include_never_treated=False), df)

    assert result.status == "failed"
