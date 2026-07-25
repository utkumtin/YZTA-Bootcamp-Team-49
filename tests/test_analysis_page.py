from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pareto.analysis.hypothesis import TACProposal, freeze_estimand

PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "2_analysis.py"


class _FakeProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class _FakeHandle:
    def __init__(self, *, returncode: int, stderr: str = "") -> None:
        self.process = _FakeProcess(returncode)
        self.run_dir = Path("runs") / "fake-run"
        self.results_path = self.run_dir / "results.json"
        self._stderr = stderr

    def read_progress(self) -> dict[str, int]:
        return {"done": 1, "total": 1}

    def is_done(self) -> bool:
        return True

    def read_stderr(self) -> str:
        return self._stderr


def _frozen_estimand():
    proposal = TACProposal(
        estimand_type="ATT",
        treatment="Medicaid expansion adoption",
        treatment_coding="expanded",
        outcome="uninsured_rate",
        outcome_unit="percentage points",
        population="US states",
        time_scope="2010-2020",
        expected_sign="negative",
        identification_assumption="parallel_trends",
        h0="ATT = 0",
        h1="ATT < 0",
        implied_result_translation="Lower uninsured rates in expansion states.",
        confirmation_question="Freeze?",
    )
    return freeze_estimand(proposal, approved=True)


def _analysis_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["a", "b", "c"],
            "year": [2010, 2010, 2010],
            "expanded": [0, 1, 0],
            "uninsured_rate": [10.0, 11.0, 12.0],
            "population": [100, 120, 130],
        }
    )


def _prepare_analysis_page(app: AppTest) -> None:
    app.session_state["clean_df"] = _analysis_df()
    app.session_state["frozen_estimand"] = _frozen_estimand()
    app.session_state["analysis_state"] = {
        "unit_col": "state",
        "time_col": "year",
        "cluster_by": "state",
        "controls": (),
    }


def test_analysis_page_shows_multiverse_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    app = AppTest.from_file(PAGE_PATH, default_timeout=20)
    _prepare_analysis_page(app)
    app.session_state["multiverse_handle"] = _FakeHandle(returncode=0)

    calls: list[tuple[str, str | None]] = []

    def _fake_page_link(page: str, *_, **kwargs) -> None:
        calls.append((page, kwargs.get("label")))

    # AppTest.from_file runs the page script without the full multipage registry.
    # Stubbing page_link keeps this test focused on progress/success flow.
    monkeypatch.setattr("streamlit.page_link", _fake_page_link)

    app.run()

    assert any("Multiverse tamamlandı." in item.value for item in app.success)
    assert any("Sonuçlar:" in item.value for item in app.caption)
    assert calls == [("pages/3_variance_panel.py", "Varyans panelini aç")]
    # NEDEN str(Path(...)): sabit "runs\\fake-run\\results.json" yalnızca
    # Windows'ta doğruydu; POSIX'te Path str() '/' ayracı üretir, bu yüzden
    # beklenen değeri platforma göre inşa ediyoruz.
    assert app.session_state["multiverse_results_path"] == str(
        Path("runs") / "fake-run" / "results.json"
    )
    assert not app.exception


def test_analysis_page_shows_multiverse_failure_path(monkeypatch: pytest.MonkeyPatch) -> None:
    app = AppTest.from_file(PAGE_PATH, default_timeout=20)
    _prepare_analysis_page(app)
    app.session_state["multiverse_handle"] = _FakeHandle(returncode=1, stderr="boom")

    calls: list[tuple[str, str | None]] = []

    def _fake_page_link(page: str, *_, **kwargs) -> None:
        calls.append((page, kwargs.get("label")))

    monkeypatch.setattr("streamlit.page_link", _fake_page_link)

    app.run()

    assert any("Multiverse başarısız oldu." in item.value for item in app.error)
    assert any("boom" in item.value for item in app.code)
    assert calls == []
    assert not app.exception
