from __future__ import annotations

from typing import Any

import pytest
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

from pareto import streamlit_ui


class _FakeStatus:
    """`st.status` yerine geçen minimal sahte; yalnız durum geçişlerini kaydeder."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.updates: list[dict[str, Any]] = []

    def __enter__(self) -> _FakeStatus:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    @property
    def states(self) -> list[str]:
        return [u["state"] for u in self.updates if "state" in u]


@pytest.fixture
def fake_status(monkeypatch: pytest.MonkeyPatch) -> list[_FakeStatus]:
    created: list[_FakeStatus] = []

    def _status(label: str, **_kwargs: Any) -> _FakeStatus:
        status = _FakeStatus(label)
        created.append(status)
        return status

    monkeypatch.setattr(streamlit_ui.st, "status", _status)
    return created


def test_llm_cagrisi_bitince_gosterge_tamamlandi_durumuna_gecer(fake_status):
    """Gösterge iz bırakmadan kaybolmamalı, tamamlandığını söylemeli.

    Maddenin çıkış noktası serbest katman gecikmelerinde "uygulama dondu mu"
    belirsizliğiydi; yanıt geldiğinde bunu açıkça bildiren bir durum gerekiyor.
    """
    with streamlit_ui.llm_call_status("JUDGE çalışıyor…"):
        pass

    assert fake_status[0].states == ["complete"]


def test_llm_cagrisi_hata_verirse_gosterge_hata_durumuna_gecer_ve_istisna_yukselir(fake_status):
    """Hata yutulmamalı: sayfa kendi `except` dalıyla kullanıcıya mesaj veriyor.

    Gösterge hatayı işaretler ama istisnayı yukarı geçirir; aksi halde çağrı
    başarısızken akış sessizce devam eder.
    """
    with pytest.raises(ValueError):
        with streamlit_ui.llm_call_status("JUDGE çalışıyor…"):
            raise ValueError("sağlayıcı düştü")

    assert fake_status[0].states == ["error"]


def test_st_rerun_gosterge_tarafindan_hata_sayilmaz(fake_status):
    """`st.rerun()` akış kontrolüdür, başarısızlık değil.

    Analiz sayfası TAC önerisinden sonra blok içinde `st.rerun()` çağırıyor.
    `ScriptControlException` BaseException türevi olduğu için `except Exception`
    onu görmez; bu test o davranışı bağlar, yoksa başarılı bir çağrı kullanıcıya
    kırmızı hata olarak görünür.
    """
    with pytest.raises(RerunException):
        with streamlit_ui.llm_call_status("JUDGE çalışıyor…"):
            raise RerunException(None)  # type: ignore[arg-type]

    assert fake_status[0].states == []
