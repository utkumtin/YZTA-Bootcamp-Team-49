"""Estimator dikişi: tipli `Protocol` + TEK LİB (pyfixest) OLS/TWFE.

Estimator kütüphaneden gelir, **tek lib = pyfixest** (`feols`;
OLS ve TWFE aynı API). Prototipteki statsmodels+linearmodels ikilisi review'in "tek lib"
kararını ihlal ediyordu → pyfixest'e konsolide edildi. Runner/variance/panel
estimator-agnostik kalır; staggered (Callaway-Sant'Anna / SA / BJS) fast-follow'da yalnız
yeni bir Protocol implementasyonu ekler, çağıran katmanlar değişmez.

pyfixest lazy import edilir: dep yoksa `estimate_one` per-spec status='failed' döndürür
(koşu çökmez); testler `importorskip` ile atlar.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..contracts import EstimationResult
from ..spec import Specification


class Estimator(Protocol):
    """estimate(spec, panel) → EstimationResult. Staggered-ready sözleşme."""

    def estimate(self, spec: Specification, df: pd.DataFrame) -> EstimationResult: ...


def _apply_sample(df: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    """Örneklem ekseni: güvenli pandas query. Boş filtre → dokunmaz."""
    if not spec.sample_filter:
        return df
    try:
        return df.query(spec.sample_filter)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Geçersiz sample_filter '{spec.sample_filter}': {exc}") from exc


def _apply_pre_period(df: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    """Ön-dönem ekseni: ilk tedavi döneminden `pre_period_window` dönem öncesine kırpar.

    Eksenin amacı parallel-trends varsayımını pencere uzunluğuna karşı sınamak:
    dar bir pencere, tedaviye uzak yılların trendi sürüklemesini engeller.

    İlk tedavi dönemi veriden türetilir (`treatment == 1` olan en erken dönem),
    domain varsayımı ya da ek kolon gerektirmez.
    """
    if spec.pre_period_window is None:
        return df
    if spec.time_fe is None:
        raise ValueError(
            f"{spec.spec_id}: pre_period_window bir zaman kolonu gerektirir (time_fe yok)."
        )
    treated_times = df.loc[df[spec.treatment] == 1, spec.time_fe]
    if treated_times.empty:
        raise ValueError(
            f"{spec.spec_id}: pre_period_window uygulanamıyor — "
            f"'{spec.treatment}' hiçbir satırda 1 değil, ilk tedavi dönemi bulunamıyor."
        )
    return df[df[spec.time_fe] >= treated_times.min() - spec.pre_period_window]


def _apply_never_treated(df: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    """Never-treated ekseni: `include_never_treated=False` → hiç tedavi görmemiş birimleri düşürür.

    Karşılaştırma grubunu "hiç tedavi görmeyenler"den "henüz tedavi
    görmeyenler"e çevirir; hangi karşılaştırma grubunun seçildiği kimlik
    varsayımını değiştirdiği için bu bir dayanıklılık ekseni.
    """
    if spec.include_never_treated:
        return df
    if spec.unit_fe is None:
        raise ValueError(
            f"{spec.spec_id}: never_treated ekseni bir birim kolonu gerektirir (unit_fe yok)."
        )
    treated_units = df.loc[df[spec.treatment] == 1, spec.unit_fe].unique()
    return df[df[spec.unit_fe].isin(treated_units)]


def _prepare(df: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    """Tüm örneklem eksenlerini sırayla uygular.

    Her estimator bunu çağırmalı: bir eksen burada bağlanmazsa spec'ler
    birbirinin birebir kopyası olur ve dayanıklılık eğrisi sahte katlar
    gösterir (ölçüldü: pre_period bağlı değilken 24 spec'in 12'si kopyaydı).
    """
    return _apply_never_treated(_apply_pre_period(_apply_sample(df, spec), spec), spec)


def _fit_columns(spec: Specification) -> list[str]:
    cols = [spec.outcome, spec.treatment, *spec.controls]
    if spec.cluster_by is not None:
        cols.append(spec.cluster_by)
    if spec.weight_col:
        cols.append(spec.weight_col)
    return cols


def _rhs(spec: Specification) -> str:
    """Formül sağ tarafı: tedavi + kontroller (pyfixest/R sözdizimi)."""
    return " + ".join([spec.treatment, *spec.controls])


def _feols_kwargs(spec: Specification) -> dict:
    # Clustering ekseni: kolon verildiyse cluster-robust (CRV1), verilmediyse
    # heteroskedastisiteye dayanıklı SE (HC1). "Kümeleme yok" savunulabilir bir
    # seviyedir; sessizce bir kolona pinlenmez.
    vcov: dict | str = {"CRV1": spec.cluster_by} if spec.cluster_by is not None else "hetero"
    kwargs: dict = {"vcov": vcov}
    if spec.weight_col:
        kwargs["weights"] = spec.weight_col
    return kwargs


def _extract(fit, spec: Specification, n_obs: int) -> EstimationResult:  # noqa: ANN001
    ci = fit.confint().loc[spec.treatment]
    return EstimationResult(
        spec_id=spec.spec_id,
        estimator=spec.estimator,
        coefficient=float(fit.coef()[spec.treatment]),
        std_error=float(fit.se()[spec.treatment]),
        p_value=float(fit.pvalue()[spec.treatment]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        n_obs=n_obs,
    )


class OLSEstimator:
    """Kesitsel / havuzlanmış OLS, clustered ya da robust SE — pyfixest.feols (FE'siz formül)."""

    def estimate(self, spec: Specification, df: pd.DataFrame) -> EstimationResult:
        import pyfixest as pf

        sub = _prepare(df, spec).dropna(subset=_fit_columns(spec))
        fit = pf.feols(f"{spec.outcome} ~ {_rhs(spec)}", data=sub, **_feols_kwargs(spec))
        return _extract(fit, spec, int(len(sub)))


class TWFEEstimator:
    """İki-yönlü sabit etki DiD — pyfixest.feols (FE'ler '|' sonrası)."""

    def estimate(self, spec: Specification, df: pd.DataFrame) -> EstimationResult:
        import pyfixest as pf

        assert spec.unit_fe and spec.time_fe  # spec validator garanti eder
        sub = _prepare(df, spec).dropna(subset=_fit_columns(spec))
        fml = f"{spec.outcome} ~ {_rhs(spec)} | {spec.unit_fe} + {spec.time_fe}"
        fit = pf.feols(fml, data=sub, **_feols_kwargs(spec))
        return _extract(fit, spec, int(len(sub)))


_REGISTRY: dict[str, Estimator] = {"OLS": OLSEstimator(), "TWFE": TWFEEstimator()}


def get_estimator(name: str) -> Estimator:
    if name not in _REGISTRY:
        raise ValueError(
            f"Bilinmeyen estimator: {name}. Committed: {list(_REGISTRY)}; "
            "staggered {CS,SA,BJS} = fast-follow (default kapalı)."
        )
    return _REGISTRY[name]


def estimate_one(spec: Specification, df: pd.DataFrame) -> EstimationResult:
    """Tek spec'i koşar; hata yutulur → status='failed' (per-spec izolasyon)."""
    try:
        return get_estimator(spec.estimator).estimate(spec, df)
    except Exception as exc:  # noqa: BLE001
        return EstimationResult(
            spec_id=spec.spec_id, estimator=spec.estimator, status="failed", error=str(exc)
        )
