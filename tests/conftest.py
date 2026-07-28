"""Test oturumu geneli izolasyon.

`test_privacy_routing` ve `test_router_smoke` import anında `.env`'i yüklüyor,
yani anahtarlar tüm oturuma yayılıyor. L7 tarayıcısı gerçek bir HTTP çağrısı
yaptığı için, `generate_ledger` çağıran her test farkında olmadan sağlayıcıya
istek atardı. `pyproject.toml`'daki `-m "not live"` varsayılanının amacı tam
olarak bunu engellemek; bu fixture aynı sözleşmeyi L7 için de zorlar.
"""

from __future__ import annotations

import pytest

from pareto.llm.guardrails import _PROMPT_GUARD_ENABLED_ENV


@pytest.fixture(autouse=True)
def _l7_agdan_izole(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """`live` işaretli olmayan testlerde L7'nin ağ çağrısını kapat.

    Kapatma anahtarı üretim kodunun kendi anahtarı; test için ayrı bir kaçış
    yolu açılmıyor. `live` testleri gerçek çağrıyı bilerek istediği için
    dokunulmadan geçer.
    """
    if "live" in request.keywords:
        return
    monkeypatch.setenv(_PROMPT_GUARD_ENABLED_ENV, "0")
