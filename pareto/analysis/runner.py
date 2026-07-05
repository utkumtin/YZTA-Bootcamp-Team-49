"""Multiverse Runner — ayrı subprocess, diske canlı-ilerleme.

Prototipteki `ProcessPoolExecutor` KALDIRILDI: Streamlit'te session_state child'a
geçmez → KeyError, ayrıca `asyncio.run()` Tornado loop'ta kırık (review sorun #2).
Yerine kilitlenen desen:
  - `run_specs()`  : sıralı, in-process, estimator-agnostik çekirdek (CLI + test).
  - `python -m pareto.analysis.runner --job ...` : standalone worker; sonuç+ilerleme
    diske yazar (Streamlit yalnız okur → session_state sorunu yok).
  - `launch_multiverse()` : Streamlit-facing; subprocess.Popen ile worker'ı başlatır,
    determinizm env pinleriyle (seed / PYTHONHASHSEED / OMP_NUM_THREADS).

Demo küçük → sıralı bile yeter. joblib-threading gerekirse subprocess *içinde*.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import SETTINGS
from ..contracts import EstimationResult
from ..spec import Specification
from .estimators import estimate_one

ProgressFn = Callable[[int, int, EstimationResult], None]


def run_specs(
    df: pd.DataFrame,
    specs: list[Specification],
    *,
    on_progress: ProgressFn | None = None,
) -> list[EstimationResult]:
    """Estimator-agnostik sıralı çekirdek. Her spec izole (fail → status='failed')."""
    results: list[EstimationResult] = []
    total = len(specs)
    for i, spec in enumerate(specs):
        res = estimate_one(spec, df)
        results.append(res)
        if on_progress is not None:
            on_progress(i + 1, total, res)
    return results


# --------------------------------------------------------------------------- #
# Streamlit-facing subprocess launcher + handle
# --------------------------------------------------------------------------- #
@dataclass
class RunHandle:
    """Subprocess koşusunun disk konumları — Streamlit bunları poll eder."""

    run_dir: Path
    process: subprocess.Popen

    @property
    def progress_path(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def results_path(self) -> Path:
        return self.run_dir / "results.json"

    def read_progress(self) -> dict:
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {"done": 0, "total": 0}

    def read_results(self) -> list[EstimationResult]:
        raw = json.loads(self.results_path.read_text(encoding="utf-8"))
        return [EstimationResult(**r) for r in raw]

    def is_done(self) -> bool:
        return self.process.poll() is not None


def launch_multiverse(df: pd.DataFrame, specs: list[Specification], run_id: str) -> RunHandle:
    """Worker'ı ayrı süreçte başlatır. Determinizm env pinlenir."""
    run_dir = Path(SETTINGS.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "panel.pkl").write_bytes(pickle.dumps(df))
    (run_dir / "specs.json").write_text(
        json.dumps([s.model_dump() for s in specs], ensure_ascii=False), encoding="utf-8"
    )

    env = {**os.environ, **SETTINGS.deterministic_env}
    proc = subprocess.Popen(  # noqa: S603  # sabit argüman listesi, shell yok; girdi kullanıcıdan gelmez
        [sys.executable, "-m", "pareto.analysis.runner", "--job", str(run_dir)],
        env=env,
    )
    return RunHandle(run_dir=run_dir, process=proc)


def _run_job(run_dir: Path) -> None:
    """Worker entrypoint: job'u okur, koşar, ilerleme + sonucu diske yazar."""
    df = pickle.loads(  # noqa: S301  # panel.pkl'i launch_multiverse yazar; aynı lokal güven sınırı
        (run_dir / "panel.pkl").read_bytes()
    )
    specs = [Specification(**s) for s in json.loads((run_dir / "specs.json").read_text())]
    progress_path = run_dir / "progress.json"
    results: list[EstimationResult] = []

    def _write_progress(done: int, total: int, _res: EstimationResult) -> None:
        progress_path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")

    results = run_specs(df, specs, on_progress=_write_progress)
    (run_dir / "results.json").write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pareto multiverse worker (subprocess)")
    parser.add_argument("--job", required=True, help="Run dizini (panel.pkl + specs.json içerir)")
    args = parser.parse_args()
    _run_job(Path(args.job))


if __name__ == "__main__":
    _cli()
