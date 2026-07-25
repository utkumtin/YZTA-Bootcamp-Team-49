"""`runs/latest` aynasının tek bir koşuyu taşıdığının testi.

Ayna reprodüksiyon paketinin varsayılan girdi yoludur ve paneldeki `run_id.txt`
okuyucusu koşunun kimliğini oradan çözer. Ayna iki koşuyu birden taşırsa paket
bir koşunun sonuçlarını başka bir koşunun donmuş menüsüyle çiftler — üstelik
hiçbir artefakt eksik olmadığı için paket "eksiksiz" görünür.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace

import pandas as pd
import pytest

from pareto.analysis import runner


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    # Testler repo'nun gerçek runs/ dizinini kirletmesin.
    patched = replace(runner.SETTINGS, runs_dir=str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "SETTINGS", patched)
    return tmp_path / "runs"


def _seed_run(runs_dir, run_id: str, *, with_results: bool) -> None:
    """Bir koşu dizinini `launch_multiverse`in bıraktığı hâle getirir."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
    (run_dir / "panel.pkl").write_bytes(pickle.dumps(pd.DataFrame({"y": [1.0], "d": [0]})))
    # Ayna dosyaları KOPYALAR, okumaz: içerik burada yalnız koşuları ayırt etmeye yarar.
    (run_dir / "specs.json").write_text(
        json.dumps([{"spec_id": f"{run_id}-s1"}], ensure_ascii=False), encoding="utf-8"
    )
    if with_results:
        # Sonuçlar koşunun EN SONUNDA yazılır; yeni başlayan bir koşuda yoktur.
        (run_dir / "results.json").write_text(
            json.dumps([{"spec_id": f"{run_id}-s1"}], ensure_ascii=False), encoding="utf-8"
        )


def test_mirror_drops_the_previous_runs_results(runs_dir):
    # NEDEN: `results.json` koşunun sonunda yazılır. Ayna yalnız kopyalasaydı B
    # başlarken aynada A'nın sonuçları kalır ve B'nin `run_id.txt`si ile eşleşirdi;
    # o pencerede kurulan paket A'nın sonuçlarını B'nin donmuş menüsüyle çiftler.
    _seed_run(runs_dir, "run-a", with_results=True)
    runner._mirror_latest_run(runs_dir / "run-a")

    latest = runs_dir / "latest"
    assert (latest / "results.json").exists()

    _seed_run(runs_dir, "run-b", with_results=False)
    runner._mirror_latest_run(runs_dir / "run-b")

    assert (latest / "run_id.txt").read_text(encoding="utf-8") == "run-b"
    assert not (latest / "results.json").exists(), "ayna hâlâ A'nın sonuçlarını taşıyor"


def test_mirror_reflects_the_run_it_was_given(runs_dir):
    # NEDEN: silme mantığı aynayı kırpmamalı — koşunun sahip olduğu her artefakt
    # aynada da bulunmalı, yoksa paket kendi kaynağını eksik sanar.
    _seed_run(runs_dir, "run-a", with_results=True)
    runner._mirror_latest_run(runs_dir / "run-a")

    latest = runs_dir / "latest"
    assert (latest / "run_id.txt").read_text(encoding="utf-8") == "run-a"
    for name in ("panel.pkl", "specs.json", "results.json"):
        assert (latest / name).read_bytes() == (runs_dir / "run-a" / name).read_bytes()
