from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pareto.cleaning.agent import Resolution
from pareto.cleaning.ledger import LedgerEntry

PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "1_cleaning.py"


class _UploadedFile:
    """AppTest'in henüz doğrudan set edemediği UploadedFile için minimal sahte."""

    def __init__(self, file_id: str = "stable-upload-id") -> None:
        self.file_id = file_id
        self.name = "input.csv"
        self.size = 14

    def getvalue(self) -> bytes:
        return b"value\n1\n2\n"


def test_same_upload_preserves_ledger_when_resolution_is_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2-13: widget rerun'ları aynı dosyanın karar defterini sıfırlamamalı."""
    uploaded = _UploadedFile()
    entry = LedgerEntry(
        bulgu="İnceleme gerekli",
        transform_name="drop_duplicates",
        params={"subset": None},
        gerekce="Test kararı",
        belirsizlik_bayragi=True,
    )

    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(
        "pareto.profiling.load_raw_file", lambda _: pd.DataFrame({"value": [1, 2]})
    )
    monkeypatch.setattr("pareto.profiling.profile_dataframe", lambda _: {"columns": []})
    monkeypatch.setattr("pareto.cleaning.agent.generate_ledger", lambda _: [entry])

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.run()
    next(button for button in app.button if "JUDGE" in button.label).click()
    app.run()

    original_ledger = app.session_state["ledger"]
    original_run_id = app.session_state["run_id"]

    app.radio[0].set_value(Resolution.REJECTED.value)
    app.run()
    next(button for button in app.button if "kararı kaydet" in button.label).click()
    app.run()

    assert app.session_state["ledger"] == original_ledger
    assert app.session_state["run_id"] == original_run_id
    assert app.session_state["resolutions"][0].resolution == Resolution.REJECTED


def test_new_upload_resets_ledger_and_resolution_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2-1/S2-13: farklı dosya eski karar turunu geçersiz kılmalı."""
    uploads = [_UploadedFile("first-upload")]
    entry = LedgerEntry(
        bulgu="İnceleme gerekli",
        transform_name="drop_duplicates",
        params={"subset": None},
        gerekce="Test kararı",
        belirsizlik_bayragi=True,
    )

    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploads[0])
    monkeypatch.setattr(
        "pareto.profiling.load_raw_file", lambda _: pd.DataFrame({"value": [1, 2]})
    )
    monkeypatch.setattr("pareto.profiling.profile_dataframe", lambda _: {"columns": []})
    monkeypatch.setattr("pareto.cleaning.agent.generate_ledger", lambda _: [entry])

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.run()
    next(button for button in app.button if "JUDGE" in button.label).click()
    app.run()
    app.radio[0].set_value(Resolution.REJECTED.value)
    app.run()
    next(button for button in app.button if "kararı kaydet" in button.label).click()
    app.run()

    assert app.session_state["resolutions"][0].resolution == Resolution.REJECTED

    uploads[0] = _UploadedFile("second-upload")
    app.run()

    assert app.session_state["cleaning_uploaded_file_id"] == "second-upload"
    assert "ledger" not in app.session_state
    assert "resolutions" not in app.session_state
    assert "run_id" not in app.session_state
