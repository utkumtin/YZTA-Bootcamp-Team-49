# 5. Tek estimator kütüphanesi: pyfixest

- Durum: accepted
- Tarih: 2026-07-05
- İlgili: SCOPE §6 ("tek lib"), review.md sorun (statsmodels+linearmodels ikilisi)

## Bağlam
SCOPE §6 committed estimator için **tek lib** diyor ("young-lib riski yok"). Prototip
OLS'i `statsmodels`, TWFE'yi `linearmodels` ile koşuyordu — iki lib, iki API, iki çıkarım
yolu. Bu "tek lib" kararını ihlal ediyor ve dikişi çatallaştırıyor.

## Karar
Tek lib = **pyfixest** (`pf.feols`). OLS ve TWFE aynı API'de:
- OLS:  `feols("y ~ d + controls", data, vcov={"CRV1": cluster})`
- TWFE: `feols("y ~ d + controls | unit + time", data, vcov={"CRV1": cluster})`
- WLS:  `weights=...` · clustered SE built-in · `.coef()/.se()/.pvalue()/.confint()`.

`statsmodels` + `linearmodels` bağımlılıklardan çıkarıldı (pyproject + requirements).

## Gerekçe
- SCOPE §2 katman 5 zaten pyfixest'i runner'ın çağırdığı lib olarak adlandırıyor.
- Tek API → estimator `Protocol` dikişi temiz; OLS/TWFE aynı kod yolunu paylaşır.
- Staggered fast-follow (`diff-diff`) ayrı lib olsa da yalnız yeni Protocol impl'i;
  committed çekirdek tek-lib kalır.

## Sonuç
`estimators.py` pyfixest'e taşındı; lazy import (dep yoksa per-spec status='failed').
Testler `pytest.importorskip("pyfixest")` ile CI'da (uv sync sonrası) koşar. R (`did`/
`differences`) yalnız bağımsız doğrulama, ürün bağımlılığı değil.
