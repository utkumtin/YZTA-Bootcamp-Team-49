from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pareto.cleaning.agent import Resolution
from pareto.cleaning.ledger import LedgerEntry

PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "1_cleaning.py"
UPLOADER_KEY = "cleaning_file_uploader"


class _UploadedFile:
    """AppTest'in henüz doğrudan set edemediği UploadedFile için minimal sahte."""

    def __init__(self, file_id: str = "stable-upload-id") -> None:
        self.file_id = file_id
        self.name = "input.csv"
        # `size` tek kaynaktan türetiliyor: gerçek `UploadedFile`ta boyut ile
        # içerik tanım gereği tutarlıdır. Fixture'ın kendi içinde tutarsız
        # olması, boyuta bakan bir regresyonu gizlerdi.
        self._payload = b"value\n1\n2\n"
        self.size = len(self._payload)

    def getvalue(self) -> bytes:
        return self._payload


def _patch_uploader(monkeypatch: pytest.MonkeyPatch) -> None:
    """`st.file_uploader`'ı gerçek widget semantiğiyle taklit eder.

    NEDEN: `key` kwarg'ını yok sayan düz bir lambda ile widget hiç register
    olmuyor, dosya değeri `session_state[UPLOADER_KEY]`te yaşamıyor ve "Veriyi
    oturumdan sil" akışının o anahtarı pop etmesinin uploader'ı gerçekten
    temizleyip temizlemediği test edilemiyor — sıfırlama sözleşmesinin yarısı
    testin göremediği yerde kalıyor.

    Gerçek uploader'da widget değeri kendi `key`i altında session_state'te
    tutulur; anahtar silinince widget boşalır. Bu sahte tam olarak onu yapar:
    okuduğu tek kaynak `session_state[key]`.
    """

    def _uploader(*_args: Any, key: str | None = None, **_kwargs: Any) -> Any:
        if key is None:
            return None
        return st.session_state.get(key)

    monkeypatch.setattr("streamlit.file_uploader", _uploader)


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, entry: LedgerEntry) -> None:
    monkeypatch.setattr("pareto.profiling.load_raw_file", lambda _: pd.DataFrame({"value": [1, 2]}))
    monkeypatch.setattr("pareto.profiling.profile_dataframe", lambda _: {"columns": []})
    monkeypatch.setattr("pareto.cleaning.agent.generate_ledger", lambda _: [entry])


def _flagged_entry() -> LedgerEntry:
    return LedgerEntry(
        bulgu="İnceleme gerekli",
        transform_name="drop_duplicates",
        params={"subset": None},
        gerekce="Test kararı",
        belirsizlik_bayragi=True,
    )


def _run_judge_round(app: AppTest) -> str:
    """JUDGE turunu koşturur ve turun `run_id`'sini döndürür.

    `run_id` widget anahtarlarının parçası (`resolution_choice_{run_id}_{i}`),
    bu yüzden testler widget'a konumla değil bu kimlikle erişir.
    """
    next(button for button in app.button if "JUDGE" in button.label).click()
    app.run()
    return str(app.session_state["run_id"])


def test_same_upload_preserves_ledger_when_resolution_is_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2-13: widget rerun'ları aynı dosyanın karar defterini sıfırlamamalı."""
    _patch_uploader(monkeypatch)
    _patch_pipeline(monkeypatch, _flagged_entry())

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state[UPLOADER_KEY] = _UploadedFile()
    app.run()
    run_id = _run_judge_round(app)

    original_ledger = app.session_state["ledger"]

    # Widget'a konumla değil anahtarla eriş: konum bağımlı erişim, sayfaya
    # ikinci bir radio eklendiği gün sessizce yanlış widget'ı sürer.
    app.radio(key=f"resolution_choice_{run_id}_0").set_value(Resolution.REJECTED.value)
    app.run()
    next(button for button in app.button if "kararı kaydet" in button.label).click()
    app.run()

    assert app.session_state["ledger"] == original_ledger
    assert app.session_state["run_id"] == run_id
    assert app.session_state["resolutions"][0].resolution == Resolution.REJECTED


def test_new_upload_resets_ledger_and_resolution_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2-1/S2-13: farklı dosya eski karar turunu geçersiz kılmalı."""
    _patch_uploader(monkeypatch)
    _patch_pipeline(monkeypatch, _flagged_entry())

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state[UPLOADER_KEY] = _UploadedFile("first-upload")
    app.run()
    run_id = _run_judge_round(app)

    app.radio(key=f"resolution_choice_{run_id}_0").set_value(Resolution.REJECTED.value)
    app.run()
    next(button for button in app.button if "kararı kaydet" in button.label).click()
    app.run()

    assert app.session_state["resolutions"][0].resolution == Resolution.REJECTED

    app.session_state[UPLOADER_KEY] = _UploadedFile("second-upload")
    app.run()

    assert app.session_state["cleaning_uploaded_file_id"] == "id:second-upload"
    assert "ledger" not in app.session_state
    assert "resolutions" not in app.session_state
    assert "run_id" not in app.session_state


def test_failed_upload_shows_cached_error_without_reparsing_on_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hata yolunda banner sticky kalmalı ama load_raw_file her rerun'da tekrar
    çağrılmamalı; yalnızca açık 'Tekrar dene' ile yeniden denenmeli.

    Aksi halde bozuk bir dosya her widget etkileşiminde baştan parse edilir —
    büyük dosyada her tıklama tam parse maliyeti demektir.
    """
    call_count = {"n": 0}

    def _boom(_uploaded):
        call_count["n"] += 1
        raise ValueError("Dosya okunamadı")

    _patch_uploader(monkeypatch)
    monkeypatch.setattr("pareto.profiling.load_raw_file", _boom)

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state[UPLOADER_KEY] = _UploadedFile("failing-upload")
    app.run()
    assert call_count["n"] == 1
    assert any("Dosya okunamadı" in err.value for err in app.error)

    # Otomatik rerun (widget değeri aynı) — reparse OLMAMALI.
    app.run()
    assert call_count["n"] == 1
    assert any("Dosya okunamadı" in err.value for err in app.error)

    # Açık "Tekrar dene" — şimdi reparse OLMALI.
    next(button for button in app.button if "Tekrar dene" in button.label).click()
    app.run()
    assert call_count["n"] == 2


def test_page_renders_without_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dosya yokken sayfa ne çökmeli ne de oturuma veri yazmalı.

    Karar defteri bölümü de açılmamalı: gatekeeper `clean_df`e bağlı, yani
    veri olmadan JUDGE turu başlatılabiliyorsa sözleşme kırılmış demektir.
    """
    _patch_uploader(monkeypatch)
    _patch_pipeline(monkeypatch, _flagged_entry())

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.run()

    assert not app.exception
    assert "clean_df" not in app.session_state
    assert not any("JUDGE" in button.label for button in app.button)


def test_clearing_session_empties_uploader_so_file_is_not_reingested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Veriyi oturumdan sil", uploader widget'ını da boşaltmalı.

    NEDEN kritik: uploader'ın değeri kendi `key`i altında yaşıyor. O anahtar
    pop edilmezse bir sonraki rerun'da `uploaded` yine dolu gelir, dosya
    sessizce yeniden yüklenir ve "sil" fiilen hiçbir şey silmemiş olur —
    kullanıcı verisini sildiğini sanırken oturum eski veriyle devam eder.

    Dinamik widget anahtarları (`resolution_choice_*` vb.) burada bilerek
    iddia EDİLMİYOR: ölçüldüğünde Streamlit'in kendi widget-state toplayıcısı
    onları zaten düşürüyor, yani böyle bir assertion sayfadaki temizlik
    çağrısı kaldırılsa bile yeşil kalırdı (bkz. `_purge_dynamic_widget_keys`
    docstring'i). Düşemeyen assertion test değildir.
    """
    _patch_uploader(monkeypatch)
    _patch_pipeline(monkeypatch, _flagged_entry())

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state[UPLOADER_KEY] = _UploadedFile()
    app.run()
    # "Veriyi oturumdan sil" bloğu script'in en başında, uploader'dan ÖNCE
    # değerlendiriliyor; dosyanın yüklendiği koşuda buton henüz çizilmemiş olur.
    app.run()
    assert app.session_state["clean_df"] is not None

    next(button for button in app.button if "Veriyi oturumdan sil" in button.label).click()
    app.run()

    assert UPLOADER_KEY not in app.session_state
    assert "clean_df" not in app.session_state
    # Silme gerçekten kalıcı: bir sonraki rerun dosyayı geri getirmemeli.
    app.run()
    assert "clean_df" not in app.session_state
