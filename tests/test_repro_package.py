"""S3-04 reprodüksiyon paketi testleri.

Paketin sözü iki tanedir ve ikisi de burada sınanır: (1) denetim izinin hiçbir
parçası sessizce düşmez, eksik varsa manifest'e yazılır; (2) paketteki tek
komutluk script paketlenmiş sonuçları gerçekten yeniden üretir, uyuşmazlıkta
sıfırdan farklı çıkış koduyla biter.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pareto.repro import (
    ReproInputs,
    ReproPackageError,
    build_reproduction_package,
    missing_artifacts,
    render_methods_section,
)
from pareto.spec import Specification

pyfixest = pytest.importorskip("pyfixest")  # estimator dep yoksa atla, CI'da koşar

from pareto.analysis.runner import run_specs  # noqa: E402
from pareto.cleaning.codegen import render_audit_script  # noqa: E402
from pareto.cleaning.ledger import LedgerEntry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _panel(effect: float = 0.8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(40):
        treated = unit % 2
        u_fe = rng.normal()
        for year in range(6):
            post = 1 if year >= 3 else 0
            d = treated * post
            y = effect * d + u_fe + 0.1 * year + rng.normal(0, 0.3)
            rows.append({"y": y, "d": d, "unit": unit, "year": year})
    return pd.DataFrame(rows)


def _specs() -> list[Specification]:
    return [
        Specification(spec_id="s1", outcome="y", treatment="d", cluster_by="unit"),
        Specification(
            spec_id="s2", outcome="y", treatment="d", controls=("year",), cluster_by=None
        ),
    ]


def _make_run(tmp_path: Path, *, with_cleaning: bool = True) -> ReproInputs:
    """Gerçek bir koşunun diskteki artefakt düzenini kurar."""
    panel = _panel()
    specs = _specs()
    results = run_specs(panel, specs)

    run_dir = tmp_path / "runs" / "demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "panel.pkl").write_bytes(pickle.dumps(panel))
    (run_dir / "specs.json").write_text(
        json.dumps([s.model_dump() for s in specs], ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False), encoding="utf-8"
    )

    store_dir = tmp_path / "store" / "demo-run"
    store_dir.mkdir(parents=True)
    (store_dir / "frozen_menu.json").write_text(
        json.dumps(
            {
                "estimand_hash": "abc123deadbeef01",
                "menu_hash": "feed0123cafe4567",
                "spec_count": len(specs),
                "estimand": {
                    "estimand_type": "ATT",
                    "outcome": "y",
                    "outcome_unit": "puan",
                    "treatment": "d",
                    "treatment_coding": "d",
                    "population": "tedavi edilen birimler",
                    "time_scope": "1-6 dönem",
                    "expected_sign": "positive",
                    "identification_assumption": "parallel_trends",
                    "h0": "etki yok",
                    "h1": "etki pozitif",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ledger_path: Path | None = None
    script_path: Path | None = None
    raw_path: Path | None = None
    if with_cleaning:
        audit_dir = tmp_path / "audit_trail"
        audit_dir.mkdir(parents=True)
        entry = LedgerEntry(
            bulgu="Panelde tekrar eden satırlar var.",
            transform_name="drop_duplicates",
            params={"subset": None},
            gerekce="Tekrar eden satırlar tahmini yanlı hale getirir.",
            belirsizlik_bayragi=False,
            resolution="approved",
        )
        script_path = audit_dir / "clean_cleaning_steps.py"
        script_path.write_text(render_audit_script([entry]), encoding="utf-8")
        ledger_path = audit_dir / "clean_decision_ledger.jsonl"
        ledger_path.write_text(
            json.dumps(entry.model_dump(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw_path = audit_dir / "clean_repro" / "raw.pkl"
        raw_path.parent.mkdir(parents=True)
        # Ham veri = panel + tekrar eden satırlar; temizleme script'i tam olarak
        # paneli geri vermeli.
        raw = pd.concat([panel, panel.head(5)], ignore_index=True)
        raw_path.write_bytes(pickle.dumps(raw))

    return ReproInputs(
        run_id="demo-run",
        results_path=run_dir / "results.json",
        specs_path=run_dir / "specs.json",
        panel_path=run_dir / "panel.pkl",
        frozen_menu_path=store_dir / "frozen_menu.json",
        ledger_path=ledger_path,
        cleaning_script_path=script_path,
        raw_panel_path=raw_path,
        figures={"specification_curve.html": "<html><body>figure</body></html>"},
    )


def _names(payload: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return set(archive.namelist())


def _read(payload: bytes, name: str) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(name).decode("utf-8")


def test_package_carries_every_audit_trail_artifact(tmp_path):
    # NEDEN: ürün tezi "denetim izi = metot bölümü"; parçalardan biri pakete
    # girmezse paket tezi taşımıyor demektir.
    payload = build_reproduction_package(_make_run(tmp_path))
    assert _names(payload) >= {
        "README.md",
        "MANIFEST.json",
        "METHODS.md",
        "requirements.txt",
        "run_reproduction.py",
        "specs.json",
        "results.json",
        "data/panel.csv",
        "data/raw.csv",
        "cleaning/cleaning_steps.py",
        "cleaning/decision_ledger.jsonl",
        "figures/specification_curve.html",
        "figures/plotly.min.js",
    }

    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert manifest["missing"] == []
    assert manifest["estimand_hash"] == "abc123deadbeef01"
    assert manifest["menu_hash"] == "feed0123cafe4567"
    # Donmuş menünün her spec'i içerik hash'iyle kayıtlı → rapor edilen küme
    # sonradan kırpılamaz.
    assert set(manifest["spec_hashes"]) == {"s1", "s2"}
    assert manifest["determinism"]["seed"] == 20260704


def test_cleaning_script_is_copied_not_rerendered(tmp_path):
    # NEDEN: L4 kapısı diskteki script'i doğrular. Paket zamanında yeniden render
    # edilen script farklı bir artefakttır (zaman damgası değişir) ve doğrulanmamıştır.
    inputs = _make_run(tmp_path)
    payload = build_reproduction_package(inputs)
    assert inputs.cleaning_script_path is not None
    assert _read(payload, "cleaning/cleaning_steps.py") == inputs.cleaning_script_path.read_text(
        encoding="utf-8"
    )


def test_missing_artifacts_are_reported_not_hidden(tmp_path):
    # NEDEN: fail-loud. Karar defteri olmayan bir paket sessizce "eksiksiz" görünürse
    # kullanıcı olmayan bir denetim izine güvenir.
    inputs = _make_run(tmp_path, with_cleaning=False)
    assert set(missing_artifacts(inputs)) == {"decision_ledger", "cleaning_script", "raw_panel"}

    payload = build_reproduction_package(inputs)
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert set(manifest["missing"]) == {"decision_ledger", "cleaning_script", "raw_panel"}
    assert "karar defteri" in _read(payload, "README.md")
    assert "eksik" in _read(payload, "METHODS.md").lower()


def test_empty_spec_list_counts_as_missing(tmp_path):
    # NEDEN: boş bir specs.json pakete hiç girmez ve run script koşamaz. Yolun var
    # olması "eksik yok" demek için yetmez, yoksa paket doğrulanabilir görünür.
    inputs = _make_run(tmp_path)
    assert inputs.specs_path is not None
    inputs.specs_path.write_text("[]", encoding="utf-8")

    assert "specs" in missing_artifacts(inputs)
    payload = build_reproduction_package(inputs)
    assert "specs.json" not in _names(payload)
    assert "specs" in json.loads(_read(payload, "MANIFEST.json"))["missing"]


def test_column_dtypes_survive_the_csv_round_trip(tmp_path):
    # NEDEN: CSV tip taşımaz. FIPS gibi öndeki sıfırları korunan kolonlar okurken
    # sayıya dönerse temizleme karşılaştırması gerçek olmayan bir hatayla patlar.
    inputs = _make_run(tmp_path)
    assert inputs.panel_path is not None
    panel = pickle.loads(inputs.panel_path.read_bytes())
    panel["fips"] = ["05001"] * len(panel)
    inputs.panel_path.write_bytes(pickle.dumps(panel))

    payload = build_reproduction_package(inputs)
    declared = json.loads(_read(payload, "data/dtypes.json"))
    assert declared["panel"]["fips"] in {"object", "str"}

    restored = pd.read_csv(io.StringIO(_read(payload, "data/panel.csv")), dtype=declared["panel"])
    assert restored["fips"].iloc[0] == "05001"


def test_package_without_results_fails_loud(tmp_path):
    # NEDEN: sonuçsuz bir "reprodüksiyon paketi" yanlış güven üretir; sessiz boş
    # paket yerine patlamalı.
    with pytest.raises(ReproPackageError, match="Sonuç dosyası yok"):
        build_reproduction_package(
            ReproInputs(run_id="yok", results_path=tmp_path / "results.json")
        )


def test_package_bytes_are_deterministic(tmp_path):
    # NEDEN: reprodüksiyon paketinin kendisi de reprodüklenebilir olmalı; aynı
    # koşudan iki farklı zip, hash'lenerek atıf verilmesini imkansız kılar.
    inputs = _make_run(tmp_path)
    assert build_reproduction_package(inputs) == build_reproduction_package(inputs)


def test_methods_draft_renders_decisions_without_llm(tmp_path):
    # NEDEN: metot bölümü taslağı deterministik şablondan gelir; karar defterindeki
    # her satır taslakta görünür olmalı, yoksa "denetim izi = metot bölümü" boş bir iddia.
    payload = build_reproduction_package(_make_run(tmp_path))
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    methods = _read(payload, "METHODS.md")

    assert methods == render_methods_section(manifest)
    assert "drop_duplicates" in methods
    assert "Tekrar eden satırlar tahmini yanlı hale getirir." in methods
    assert "abc123deadbeef01" in methods
    assert "parallel_trends" in methods


def test_run_script_reproduces_packaged_results(tmp_path):
    # NEDEN: paketin tek gerçek sözü bu. Zip açılır, tek komut koşar ve paketlenmiş
    # katsayılar tolerans içinde geri gelir; gelmezse çıkış kodu sıfır olmaz.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    # `pareto` PyPI'da değil: sandbox'ta da repo kökü PYTHONPATH'e eklenir
    # (verify_reproduction ile aynı desen).
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [sys.executable, "run_reproduction.py"],
        cwd=extract_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TAMAM: temizleme adımı yeniden üretildi" in proc.stdout
    assert "Reprodüksiyon doğrulandı." in proc.stdout


def test_run_script_fails_when_packaged_results_are_tampered(tmp_path):
    # NEDEN: doğrulama gerçekten doğruluyor mu? Sonuç dosyası bozulduğunda script
    # sessizce "tamam" derse kapı yok demektir.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "tampered"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    results_file = extract_dir / "results.json"
    tampered = json.loads(results_file.read_text(encoding="utf-8"))
    tampered[0]["coefficient"] = float(tampered[0]["coefficient"]) + 1.0
    results_file.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [sys.executable, "run_reproduction.py"],
        cwd=extract_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "BAŞARISIZ" in proc.stdout
