from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pareto.analysis import runner


def _write_run(run_dir: Path, label: str, *, with_results: bool) -> None:
    run_dir.mkdir()
    (run_dir / "panel.pkl").write_bytes(f"panel-{label}".encode())
    (run_dir / "specs.json").write_text(f"specs-{label}", encoding="utf-8")
    (run_dir / "progress.json").write_text(f"progress-{label}", encoding="utf-8")
    if with_results:
        (run_dir / "results.json").write_text(f"results-{label}", encoding="utf-8")


def test_mirror_latest_run_replaces_the_entire_snapshot(tmp_path: Path, monkeypatch) -> None:
    settings = replace(runner.SETTINGS, runs_dir=str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "SETTINGS", settings)
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_run(run_a, "a", with_results=True)
    _write_run(run_b, "b", with_results=False)

    runner._mirror_latest_run(run_a)
    runner._mirror_latest_run(run_b)

    latest = tmp_path / "runs" / "latest"
    assert (latest / "panel.pkl").read_bytes() == b"panel-b"
    assert (latest / "specs.json").read_text(encoding="utf-8") == "specs-b"
    assert (latest / "progress.json").read_text(encoding="utf-8") == "progress-b"
    assert not (latest / "results.json").exists()
