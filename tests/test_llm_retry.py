from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel

from pareto.llm import retry as retry_mod
from pareto.llm.retry import RetryingModel


class _FlakyModel(TestModel):
    """İlk `fail_times` çağrıda verilen hatayı atar, sonra normal yanıt döner."""

    def __init__(self, exc: Exception, fail_times: int) -> None:
        super().__init__()
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return await super().request(*args, **kwargs)


@pytest.fixture(autouse=True)
def _beklemeyi_kaldir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Testler gerçek backoff süresini beklemesin; ölçülen şey süre değil davranış."""
    monkeypatch.setattr(retry_mod, "_delay_for", lambda _attempt: 0.0)


def _calistir(model: RetryingModel) -> Any:
    """Senkron sürücü: projede async test eklentisi yok, `run_sync` kendi döngüsünü kurar."""
    from pydantic_ai import Agent

    return Agent(model).run_sync("selam")


def test_gecici_hata_ayni_modelde_yeniden_denenir():
    """429 tek başına akışı düşürmemeli; aynı pinli modelde tekrar denenmeli.

    Maddenin çıkış noktası buydu: "tek hatadan sonra direkt fallback'e geçilmeyecek,
    öncesinde pin'li modelde birkaç deneme yapılacak." Serbest katmanda 429 rutin.
    """
    flaky = _FlakyModel(ModelHTTPError(status_code=429, model_name="m"), fail_times=2)

    _calistir(RetryingModel(flaky, max_attempts=3))

    assert flaky.calls == 3, "iki geçici hatadan sonra üçüncü denemede başarılı olmalı"


def test_yeniden_denemeler_zinciri_degistirmez():
    """Retry FAILOVER DEĞİL: sarılan model baştan sona aynı kalmalı.

    ADR 0004'ün "judge pinli" kararı retry eklenirken sessizce delinmemeli;
    bu test sarmalayıcının ikinci bir uca kaymadığını bağlar.
    """
    flaky = _FlakyModel(ModelHTTPError(status_code=503, model_name="m"), fail_times=1)
    sarmalanan = RetryingModel(flaky, max_attempts=3)

    _calistir(sarmalanan)

    assert sarmalanan.wrapped is flaky


def test_kalici_hata_yeniden_denenmez():
    """401/400 kalıcıdır: yeniden denemek yalnız kullanıcıyı bekletir ve kotayı yakar.

    Kalıcı hatanın gecikmeli olarak aynı hataya dönmesi, sebebi gizlediği için
    geçici hatayı emmekten daha kötüdür.
    """
    kalici = _FlakyModel(ModelHTTPError(status_code=401, model_name="m"), fail_times=5)

    with pytest.raises(ModelHTTPError):
        _calistir(RetryingModel(kalici, max_attempts=3))

    assert kalici.calls == 1, "kalıcı hatada tek deneme yapılmalı"


def test_deneme_hakki_bitince_hata_yukselir():
    """Sonsuza kadar denenmez; hak bitince hata fail-loud yukarı çıkar."""
    hep_dusen = _FlakyModel(ModelHTTPError(status_code=500, model_name="m"), fail_times=99)

    with pytest.raises(ModelHTTPError):
        _calistir(RetryingModel(hep_dusen, max_attempts=3))

    assert hep_dusen.calls == 3


def test_retry_sarmalayicisi_cache_anahtarini_degistirmez():
    """En pahalı regresyon: cache anahtarı değişirse anahtarsız demo tamamen ölür.

    `CachedModel` anahtarı `model_name` üzerinden kuruyor ve `runs/llm_cache`'te
    commit'lenmiş golden-path dosyaları bu anahtara bağlı. Sarmalayıcı `model_name`'i
    değiştirseydi her istek cache-miss olur, canned modda `CannedModeCacheMissError`
    ile düşerdi. Bu test o bağı açıkça tutar.
    """
    from pathlib import Path

    from pareto.llm.cache import CachedModel

    base = TestModel()

    sarmasiz = CachedModel(base, Path("/tmp/pareto-test-cache"))
    sarmali = CachedModel(RetryingModel(base), Path("/tmp/pareto-test-cache"))

    assert sarmali.model_name == sarmasiz.model_name


def test_baglanti_hatasi_da_gecici_sayilir():
    """Timeout/bağlantı kopması durum kodu taşımaz ama tekrarlanabilir."""
    kopan = _FlakyModel(httpx.ConnectError("bağlantı yok"), fail_times=1)

    _calistir(RetryingModel(kopan, max_attempts=3))

    assert kopan.calls == 2
