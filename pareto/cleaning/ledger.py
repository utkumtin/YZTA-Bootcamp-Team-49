"""Karar Defteri (Decision Ledger) — kara-kutu yasağının somutlaştığı yer.

Prototipteki `p1_cleaning/decision_ledger.py` şeması migre edildi ve GÜVENLİ hale
getirildi: her karar artık serbest-metin talimat değil, kapalı transform sözlüğünden
(`transforms.ALLOWED_TRANSFORMS`) bir `transform_name` + tipli `params` taşır. Böylece
ledger doğrudan `codegen.py`'nin deterministik render'ına beslenir — LLM keyfi kod
üretemez (L3). Tüm kayıtlar diske yazılır (audit trail = metot bölümü).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from ..config import SETTINGS
from .transforms import ALLOWED_TRANSFORMS


class LedgerEntry(BaseModel):
    """Bir temizleme kararı: bulgu → seçilmiş (vetted) transform + gerekçe."""

    bulgu: str
    transform_name: str
    params: dict[str, Any]
    gerekce: str
    belirsizlik_bayragi: bool
    timestamp: str = ""
    # İnsanın gatekeeper'daki kararı: "approved" | "modified" | "rejected" | None
    # (None = henüz çözülmemiş / audit dışı ham JUDGE önerisi). persist_ledger
    # bu alan sayesinde insan kararını da denetim izine yazabilir.
    resolution: str | None = None

    @field_validator("transform_name")
    @classmethod
    def _must_be_vetted(cls, v: str) -> str:
        if v not in ALLOWED_TRANSFORMS:
            raise ValueError(
                f"Transform {v!r} kapalı taksonomide değil: {ALLOWED_TRANSFORMS}. "
                "Ledger yalnız vetted transform kabul eder (L3)."
            )
        return v

    def stamped(self) -> LedgerEntry:
        return self.model_copy(update={"timestamp": datetime.now(UTC).isoformat()})


def persist_ledger(entries: list[LedgerEntry], run_id: str) -> Path:
    """Karar defterini denetim izi olarak diske yazar (JSON Lines)."""
    out_dir = Path(SETTINGS.audit_trail_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_decision_ledger.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
    return out_path
