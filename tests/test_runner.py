from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

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
    # Z8: kimse "latest/progress.json"ı okumuyordu (RunHandle yalnız kendi
    # run_dir'inin progress.json'ını okur), bu yüzden mirror'lanan dosya
    # listesinden çıkarıldı. Artık latest snapshot'ında bu dosya hiç yok.
    assert not (latest / "progress.json").exists()
    assert not (latest / "results.json").exists()


def test_mirror_latest_run_with_include_panel_false_skips_panel(
    tmp_path: Path, monkeypatch
) -> None:
    # Y7 artığı: `_run_job`'ın kullandığı `include_panel=False` yolu hiç test
    # edilmiyordu. `panel.pkl` diskte olsa bile mirror'a girmemeli.
    settings = replace(runner.SETTINGS, runs_dir=str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "SETTINGS", settings)
    run_dir = tmp_path / "run-a"
    _write_run(run_dir, "a", with_results=True)

    runner._mirror_latest_run(run_dir, include_panel=False)

    latest = tmp_path / "runs" / "latest"
    assert not (latest / "panel.pkl").exists()
    assert (latest / "specs.json").read_text(encoding="utf-8") == "specs-a"
    assert (latest / "results.json").read_text(encoding="utf-8") == "results-a"


def test_cleanup_panel_pickle_removes_single_file(tmp_path: Path) -> None:
    # Y7 artığı: yeniden adlandırılan (tekil) `_cleanup_panel_pickle`
    # hiç doğrudan test edilmiyordu.
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "panel.pkl").write_bytes(b"panel-a")

    runner._cleanup_panel_pickle(run_dir)

    assert not (run_dir / "panel.pkl").exists()


def test_cleanup_panel_pickle_is_noop_when_file_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    # unlink(missing_ok=True) sözleşmesi: dosya yoksa sessizce geçer.
    runner._cleanup_panel_pickle(run_dir)


def test_run_job_cleans_up_panel_pickle_even_when_run_specs_fails(
    tmp_path: Path, monkeypatch
) -> None:
    # Z4: eskiden temizlik yalnızca mutlu yolun sonundaydı; `run_specs`
    # patlarsa hem run_dir/panel.pkl hem de launch'ta kopyalanan
    # runs/latest/panel.pkl diskte kalıyordu. Artık `_run_job` try/finally
    # kullanıyor — bu test finally'nin gerçekten çalıştığını doğruluyor.
    settings = replace(runner.SETTINGS, runs_dir=str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "SETTINGS", settings)

    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "panel.pkl").write_bytes(b"panel-a")
    (run_dir / "specs.json").write_text("[]", encoding="utf-8")

    latest_dir = tmp_path / "runs" / "latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / "panel.pkl").write_bytes(b"panel-a")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("estimation patladı")

    monkeypatch.setattr(runner, "run_specs", _boom)

    with pytest.raises(RuntimeError, match="estimation patladı"):
        runner._run_job(run_dir)

    assert not (run_dir / "panel.pkl").exists()
    assert not (latest_dir / "panel.pkl").exists()
    # results.json hiç yazılmadı; mirror da hiç çağrılmadı (mutlu yol atlandı).
    assert not (run_dir / "results.json").exists()
