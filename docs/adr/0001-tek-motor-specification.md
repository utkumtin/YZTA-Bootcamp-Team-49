# 1. Tek motor: Specification atom birimi

- Durum: accepted
- Tarih: 2026-07-05

## Bağlam
OLS, TWFE-DiD ve staggered ayrı sistemler gibi görünür. İki ayrı motor yazmak 5 haftanın
en büyük katili ("yanlışlıkla iki ayrı sistem = fork").

## Karar
Çekirdek soyutlama `Specification` (`pareto/spec.py`): `{outcome, regressors, fixed_effects,
clustering, sample, estimator}`. Estimator yalnız bir eksen. Runner/variance/panel
estimator-agnostik.

## Sonuç
Tek mimari fork'u öldürür. Prototipteki dataclass Pydantic'e (frozen, hash'lenebilir)
taşındı; `sample_filter` ve `include_never_treated` eksenleri eklendi.
