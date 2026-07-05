"""Kod üretimi — deterministik RENDER (LLM YOK).

review'in kritik bulgusuna karşı yeniden yazıldı: prototipteki `code_executor.py`
LLM'e kod ürettirip `exec` ile çalıştırıyordu. Burada kod, karar defterindeki vetted
transform'ların SABİT şablonlarından RENDER edilir; uygulama da aynı vetted `apply`
fonksiyonlarıyla yapılır. Üretilen script = denetim izi = doğrudan metot bölümü.

Not: uygulama şu an in-process (deterministik, saf fonksiyonlar, keyfi kod yok).
Üretilen script'in subprocess sandbox'ta koşulup CleanPanel'e tolerans-eşitliği
assert'i (L4) reprodüksiyon paketinin parçası — Sprint-2/3 (bkz docs/scrum).
"""

from __future__ import annotations

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
