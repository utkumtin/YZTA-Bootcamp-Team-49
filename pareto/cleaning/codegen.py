"""Kod üretimi — deterministik RENDER (LLM YOK).

review'in kritik bulgusuna karşı yeniden yazıldı: prototipteki `code_executor.py`
LLM'e kod ürettirip `exec` ile çalıştırıyordu. Burada kod, karar defterindeki vetted
transform'ların SABİT şablonlarından RENDER edilir; uygulama da aynı vetted `apply`
fonksiyonlarıyla yapılır. Üretilen script = denetim izi = doğrudan metot bölümü.

Uygulama in-process (deterministik, saf fonksiyonlar, keyfi kod yok). L4 katmanı:
`verify_reproduction` diske yazılan script'i ayrı bir subprocess sandbox'ta
determinizm env pinleriyle koşar ve çıktıyı in-process sonuca tolerans eşitliğiyle
karşılaştırır. Uyuşmazlık sessizce geçilmez, `ReproductionError` ile patlar.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import SETTINGS
from .ledger import LedgerEntry
from .transforms import get_transform


def render_audit_script(entries: list[LedgerEntry]) -> str:
    """Onaylanmış transform'lardan yeniden-üretilebilir bir Python script'i render eder."""
    lines: list[str] = [
        f'"""Pareto denetim izi — otomatik render: {datetime.now(UTC).isoformat()}"""',
        "import pandas as pd",
        "",
        "def clean(df: pd.DataFrame) -> pd.DataFrame:",
    ]
    if not entries:
        lines.append("    return df")
        return "\n".join(lines) + "\n"

    for entry in entries:
        transform = get_transform(entry.transform_name)
        lines.append(f"    # {entry.bulgu} — {entry.gerekce}")
        rendered = transform.render(**entry.params)
        for rline in rendered.splitlines():
            lines.append(f"    {rline}")
    lines.append("    return df")
    return "\n".join(lines) + "\n"


def apply_ledger(
    df: pd.DataFrame, entries: list[LedgerEntry], run_id: str
) -> tuple[pd.DataFrame, Path]:
    """Vetted transform'ları sırayla uygular + audit script'i diske yazar.

    Döndürür: (temiz_df, audit_script_yolu).
    """
    current = df
    for entry in entries:
        transform = get_transform(entry.transform_name)
        current = transform.apply(current, **entry.params)

    out_dir = Path(SETTINGS.audit_trail_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"{run_id}_cleaning_steps.py"
    audit_path.write_text(render_audit_script(entries), encoding="utf-8")
    return current, audit_path


# --------------------------------------------------------------------------- #
# L4: subprocess sandbox reprodüksiyonu + tolerans eşitliği
# --------------------------------------------------------------------------- #
_REPRO_TIMEOUT_SECONDS = 120


class ReproductionError(RuntimeError):
    """L4 reprodüksiyon kapısı: sandbox çıktısı in-process sonucu doğrulayamadı."""


def verify_reproduction(
    raw_df: pd.DataFrame,
    audit_script: Path,
    expected_df: pd.DataFrame,
    run_id: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Path:
    """Diske yazılan audit script'ini sandbox'ta koşar, çıktıyı tolerans eşitliğiyle doğrular.

    Ham girdi, script kopyası ve sandbox çıktısı tek bir repro dizininde toplanır;
    bu dizin denetim izinin parçasıdır. Script koşamazsa ya da çıktı in-process
    sonuçla eşleşmezse sessizce geçilmez, `ReproductionError` fırlatılır.

    Döndürür: repro sandbox dizini.
    """
    job_dir = Path(SETTINGS.audit_trail_dir) / f"{run_id}_repro"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "raw.pkl").write_bytes(pickle.dumps(raw_df))
    shutil.copyfile(audit_script, job_dir / "cleaning_steps.py")

    env = {**os.environ, **SETTINGS.deterministic_env}
    try:
        proc = subprocess.run(  # noqa: S603  # sabit argüman listesi, shell yok; girdi kullanıcıdan gelmez
            [sys.executable, "-m", "pareto.cleaning.codegen", "--job", str(job_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=_REPRO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReproductionError(
            f"Audit script sandbox'ta {_REPRO_TIMEOUT_SECONDS}s içinde bitmedi "
            f"(sandbox: {job_dir})."
        ) from exc
    if proc.returncode != 0:
        raise ReproductionError(
            f"Audit script sandbox'ta koşamadı (exit={proc.returncode}, sandbox: {job_dir}).\n"
            f"stderr:\n{proc.stderr}"
        )

    reproduced = pickle.loads(  # noqa: S301  # reproduced.pkl'i worker yazar; aynı lokal güven sınırı
        (job_dir / "reproduced.pkl").read_bytes()
    )
    try:
        pd.testing.assert_frame_equal(
            reproduced, expected_df, check_exact=False, rtol=rtol, atol=atol
        )
    except AssertionError as exc:
        raise ReproductionError(
            f"L4 uyuşmazlığı: sandbox çıktısı in-process sonuçla tolerans içinde eşleşmiyor "
            f"(sandbox: {job_dir}).\n{exc}"
        ) from exc
    return job_dir


def _load_clean(script_path: Path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Audit script'ini modül olarak yükler ve `clean` fonksiyonunu döndürür.

    Script vetted şablonlardan render edilir (L3); burada yüklenmesi keyfi kod
    çalıştırmak değil, diske yazılan artefaktın kendisini doğrulamaktır (L4).
    """
    spec = importlib.util.spec_from_file_location("pareto_audit_script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Audit script modül olarak yüklenemedi: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean


def _run_repro_job(job_dir: Path) -> None:
    """Worker entrypoint: ham girdiyi okur, script'in `clean`'ini koşar, çıktıyı yazar."""
    raw = pickle.loads(  # noqa: S301  # raw.pkl'i verify_reproduction yazar; aynı lokal güven sınırı
        (job_dir / "raw.pkl").read_bytes()
    )
    clean = _load_clean(job_dir / "cleaning_steps.py")
    (job_dir / "reproduced.pkl").write_bytes(pickle.dumps(clean(raw)))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pareto codegen repro worker (subprocess)")
    parser.add_argument(
        "--job", required=True, help="Repro dizini (raw.pkl + cleaning_steps.py içerir)"
    )
    args = parser.parse_args()
    _run_repro_job(Path(args.job))


if __name__ == "__main__":
    _cli()
