"""Harness raporlarının ortak zemini: provenance damgası ve rapor yazımı.

Rapor üreten her harness aynı üç şeyi yapıyor: koşuyu üreten commit'i damgala,
platformu adlandır, raporu uzantıya göre Markdown ya da JSON yaz. Bu üçü kopya
kaldıkça biri düzelip diğerinin geride kalma riski taşıyordu. Markdown gövdesi
harness'a özgü kalır, yalnız ortak olan kısım buraya taşındı.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def jsonable(value: Any) -> Any:
    """Rapora giren değerleri `json.dumps`'ın kaldırabileceği tiplere indirger."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def source_commit(repo_root: Path = REPO_ROOT) -> str:
    """Raporu üreten commit; git yoksa `unknown` döner, rapor yine üretilir."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607  # provenance; PATH'teki git yeter
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def platform_name() -> str:
    system = platform.system()
    return "macOS" if system == "Darwin" else system or "unknown"


def write_report(
    report: dict[str, Any],
    out_path: Path,
    *,
    render: Callable[[dict[str, Any]], str],
) -> None:
    """Uzantı `.json` ise JSON, değilse `render`'ın ürettiği Markdown yazar.

    Renderer parametre olarak gelir: iki harness'ın Markdown gövdesi farklı,
    yazma davranışı (dizin açma, uzantıya göre biçim seçme) aynı.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        out_path.write_text(render(report), encoding="utf-8")
