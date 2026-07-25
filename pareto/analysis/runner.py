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
from shutil import copy2, rmtree
from tempfile import mkdtemp

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

    def read_stderr(self) -> str:
        stderr_path = self.run_dir / "stderr.log"
        return stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""

    def is_done(self) -> bool:
        return self.process.poll() is not None


def _mirror_latest_run(run_dir: Path, *, include_panel: bool = True) -> None:
    """Publish a complete run snapshot without mixing files from separate runs.

    Bu fonksiyon atomik bir dizin takası (temp_dir -> latest) yapar.
    Bu sayede eski ve yeni koşu dosyaları asla birbirine karışmaz ve
    kopyalama sırası (örneğin kimlik damgasının sona bırakılması) önemsizleşir.

    NEDEN except-tipine göre dallanmıyoruz: os.replace() bir dizinin üzerine
    boş-olmayan bir hedef dizin varken yazamaz... (POSIX ENOTEMPTY detayı).
    Bunun yerine eskisini kenara alıp yenisini takas ediyor, sonra eskisini siliyoruz.
    """
    latest_dir = Path(SETTINGS.runs_dir) / "latest"
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(mkdtemp(prefix=".latest-", dir=latest_dir.parent))

    # Z8: "latest/progress.json"ı hiç kimse okumuyor; kopyalamaya dahil edilmedi.
    # Ancak run_id.txt okuyan katman (varyans paneli) için kritik, aksi halde
    # koşu "latest" sanılır ve run_id'ye bağlı artefaktlar bulunamaz.
    names = ["specs.json", "results.json", "run_id.txt"]
    if include_panel:
        names.insert(0, "panel.pkl")

    # Z6: copy2 sırasında bir hata olursa temp_dir sızıntısını önle.
    try:
        for name in names:
            source = run_dir / name
            if source.exists():
                copy2(source, temp_dir / name)

        if not latest_dir.exists():
            # İlk run: hedef yok, doğrudan atomik takas yeterli.
            os.replace(temp_dir, latest_dir)
            return
    except Exception:
        if temp_dir.exists():
            rmtree(temp_dir, ignore_errors=True)
        raise

    # latest_dir zaten var. Eskisini kenara taşı, yenisini yerine koy, eskisini sil.
    backup_dir = Path(mkdtemp(prefix=".latest-old-", dir=latest_dir.parent))
    backup_dir.rmdir()
    os.replace(latest_dir, backup_dir)
    try:
        os.replace(temp_dir, latest_dir)
    except OSError:
        # Takas başarısız oldu: latest_dir'i asla kayıp bırakma, eskisini geri koy.
        os.replace(backup_dir, latest_dir)
        raise
    else:
        rmtree(backup_dir, ignore_errors=True)
    finally:
        if temp_dir.exists():
            rmtree(temp_dir, ignore_errors=True)


def _cleanup_panel_pickle(run_dir: Path) -> None:
    """Drop the raw panel once the run directory no longer needs it."""
    (run_dir / "panel.pkl").unlink(missing_ok=True)


def launch_multiverse(df: pd.DataFrame, specs: list[Specification], run_id: str) -> RunHandle:
    """Worker'ı ayrı süreçte başlatır. Determinizm env pinlenir."""
    run_dir = Path(SETTINGS.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
    (run_dir / "panel.pkl").write_bytes(pickle.dumps(df))
    (run_dir / "specs.json").write_text(
        json.dumps([s.model_dump() for s in specs], ensure_ascii=False), encoding="utf-8"
    )
    _mirror_latest_run(run_dir)

    env = {**os.environ, **SETTINGS.deterministic_env}
    with (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_log:
        proc = subprocess.Popen(  # noqa: S603  # sabit argüman listesi, shell yok; girdi kullanıcıdan gelmez
            [sys.executable, "-m", "pareto.analysis.runner", "--job", str(run_dir)],
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
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

    # Z4: eskiden temizlik yalnızca mutlu yolun sonundaydı — run_specs (veya
    # results.json yazımı) patlarsa hem run_dir/panel.pkl hem de launch'ta
    # kopyalanan runs/latest/panel.pkl diskte kalıyordu. Artık finally'de,
    # başarı/başarısızlık fark etmeksizin, her iki konum da temizleniyor.
    try:
        results = run_specs(df, specs, on_progress=_write_progress)
        (run_dir / "results.json").write_text(
            json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _mirror_latest_run(run_dir, include_panel=False)
    finally:
        _cleanup_panel_pickle(run_dir)
        _cleanup_panel_pickle(Path(SETTINGS.runs_dir) / "latest")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pareto multiverse worker (subprocess)")
    parser.add_argument("--job", required=True, help="Run dizini (panel.pkl + specs.json içerir)")
    args = parser.parse_args()
    _run_job(Path(args.job))


if __name__ == "__main__":
    _cli()
