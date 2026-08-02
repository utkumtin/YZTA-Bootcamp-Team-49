from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.models.test import TestModel

from pareto.config import ModelRole
from pareto.llm.router import build_agent, last_used_model, use_test_model
from pareto.memory.frozen_menu import build_frozen_menu_record
from pareto.repro.package import ReproInputs, _build_manifest


def _record(judge_model: dict[str, str] | None) -> dict[str, Any]:
    return build_frozen_menu_record(
        estimand_hash="abc123",
        menu_hash="def456",
        spec_count=8,
        run_id="run-1",
        estimand={"treatment": "t", "outcome": "y"},
        menu={"axes": []},
        judge_model=judge_model,
    )


def test_dondurma_kaydi_menuyu_ureten_modeli_tasir():
    """Denetlenebilirlik: "bu spec menüsünü hangi model üretti" cevaplanabilmeli.

    Model hem `.env`'den hem kullanıcı seçiminden gelebiliyor; kayıt olmadan bu
    soru hiçbir artefakttan cevaplanamıyor ve "savunulabilir sonuç" tezi
    reprodüksiyon tarafında eksik kalıyor.
    """
    record = _record({"provider": "groq", "model_id": "llama-3.3-70b-versatile"})

    assert record["judge_model"] == {"provider": "groq", "model_id": "llama-3.3-70b-versatile"}


def test_model_kimligi_yoksa_uydurulmaz():
    """Kimlik bilinmiyorsa None yazılır; uydurulmuş bir kimlik kimliksizlikten kötüdür."""
    assert _record(None)["judge_model"] is None


def test_repro_manifesti_modeli_dondurma_kaydindan_okur():
    """Manifest modeli canlı çözmemeli, dondurma kaydından almalı.

    Paket aylar sonra da üretilebiliyor; o an etkin olan model menüyü üreten model
    değildir. Bu test canlı çözüme kayışı yakalar.
    """

    inputs = ReproInputs(run_id="run-1", results_path=Path("yok/results.json"))

    manifest = _build_manifest(
        inputs,
        results=[],
        specs=[],
        frozen=_record({"provider": "google", "model_id": "gemini-3.6-flash"}),
        decisions=[],
        provenance={},
        contents=[],
    )

    assert manifest["judge_model"] == {"provider": "google", "model_id": "gemini-3.6-flash"}


def test_test_modelinde_provenance_gercek_model_iddia_etmez():
    """Test koşusunda üretilen artefakt gerçek bir model adı taşımamalı.

    `use_test_model` yolunda kayıt hiç güncellenmezse, aynı süreçte daha önce
    kurulmuş gerçek bir modelin kimliği test çıktısına yapışır ve provenance
    sessizce yalan söyler. Bu testin ölçtüğü şey tam olarak o sızıntı.
    """
    with use_test_model(TestModel()):
        build_agent(ModelRole.JUDGE, system_prompt="x")

    kayit = last_used_model(ModelRole.JUDGE)

    assert kayit is not None
    assert kayit["provider"] == "test"
    assert kayit["model_id"] == "TestModel"
