"""Kod Üretici ve Çalıştırıcı (Execution).

Onaylanmış kararlara göre veriyi temizleyen Python/Pandas kodunu üretir,
bunu bir .py dosyası olarak (denetim izi) diske kaydeder, ardından bu kodu
İZOLE bir scope'ta çalıştırır. `exec` kullanımı kasıtlıdır; ama global scope'a
ASLA sızmaz -- her çalıştırma kendi temiz sözlüğünde olur ve sadece
`pandas` + `df` değişkeni expose edilir (rastgele import / dosya erişimi yok).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import SETTINGS
from ..llm_client import call_llm
from .decision_ledger import LedgerEntry
from .gatekeeper import Resolution, ResolvedDecision

_CODEGEN_SYSTEM_PROMPT = """\
Sen bir pandas kod üreticisisin. Sana bir veri temizleme kararı verilecek.
Görevin, bu kararı uygulayan SAF bir Python fonksiyonu üretmek.

Kesin kurallar:
- Fonksiyon imzası TAM OLARAK şu olmalı: def clean_step(df: pd.DataFrame) -> pd.DataFrame:
- Sadece pandas (pd) ve numpy (np) kullanabilirsin, başka import YAPMA.
- Dosya sistemine erişme, network çağrısı yapma, print dışında yan etki üretme.
- Fonksiyonun sonunda temizlenmiş df'i return et.
- SADECE kod döndür, açıklama veya markdown fence ekleme.
"""


def _generate_cleaning_code(decision: ResolvedDecision) -> str:
    instruction = decision.entry.onerilen_duzeltme
    if decision.resolution == Resolution.MODIFIED and decision.user_instruction:
        instruction = decision.user_instruction

    user_prompt = (
        f"Bulgu: {decision.entry.bulgu}\n"
        f"Uygulanacak düzeltme: {instruction}\n"
        f"Gerekçe: {decision.entry.gerekce}\n\n"
        "Bu düzeltmeyi uygulayan clean_step fonksiyonunu üret."
    )
    code = call_llm(_CODEGEN_SYSTEM_PROMPT, user_prompt)
    return code.strip().strip("`").removeprefix("python").strip()


def _safe_exec_clean_step(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Üretilen kodu izole bir namespace'te çalıştırır."""
    import numpy as np  # sadece bu scope'a expose edilecek

    local_ns: dict = {}
    global_ns = {"pd": pd, "np": np, "__builtins__": __builtins__}
    exec(code, global_ns, local_ns)  # noqa: S102 -- kasıtlı, izole scope

    clean_step = local_ns.get("clean_step")
    if clean_step is None:
        raise RuntimeError("Üretilen kodda `clean_step` fonksiyonu bulunamadı.")
    return clean_step(df)


def apply_decisions(
    df: pd.DataFrame,
    resolved_decisions: list[ResolvedDecision],
    run_id: str,
) -> tuple[pd.DataFrame, Path]:
    """Onaylanmış/değiştirilmiş her kararı sırayla uygular, reddedilenleri atlar.

    Tüm üretilen kod parçalarını tek bir denetim-izi .py dosyasında birleştirip
    diske yazar ve o dosyanın yolunu döndürür.
    """
    out_dir = Path(SETTINGS.audit_trail_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"{run_id}_cleaning_steps.py"

    generated_blocks: list[str] = [
        f'"""Pareto denetim izi - otomatik üretildi: {datetime.now(timezone.utc).isoformat()}"""',
        "import pandas as pd",
        "import numpy as np",
        "",
    ]

    current_df = df
    for i, decision in enumerate(resolved_decisions):
        if decision.resolution == Resolution.REJECTED:
            generated_blocks.append(f"# [ATLANDI] {decision.entry.bulgu}\n")
            continue

        code = _generate_cleaning_code(decision)
        renamed_code = code.replace("clean_step", f"clean_step_{i}")
        generated_blocks.append(f"# Karar: {decision.entry.bulgu}\n{renamed_code}\n")
        current_df = _safe_exec_clean_step(code, current_df)

    audit_path.write_text("\n".join(generated_blocks), encoding="utf-8")
    return current_df, audit_path
