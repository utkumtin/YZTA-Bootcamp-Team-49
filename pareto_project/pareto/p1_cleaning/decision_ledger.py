"""Karar Defteri (Decision Ledger) Agent'ı.

Profilleme çıktısını alır, LLM'e gönderir ve her bulgu için standart bir
karar objesi üretir:

    {"bulgu": str, "önerilen_duzeltme": str, "gerekce": str, "belirsizlik_bayragi": bool}

Bu kayıtların TAMAMI diske yazılır (audit trail) -- kara kutu yasağının
somutlaştığı yer burasıdır.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import SETTINGS
from ..llm_client import call_llm_json

_SYSTEM_PROMPT = """\
Sen kıdemli bir veri temizleme ekonometristisin. Sana bir dataframe'in
istatistiksel profili verilecek. Görevin, bu profildeki veri kalitesi
sorunlarını tespit edip her biri için bir KARAR ÖNERİSİ üretmek.

Kurallar:
- Sadece profildeki somut kanıtlara dayan, veriyi hayal etme.
- Yüksek güvenle otomatik düzeltilebilecek şeyler (ör. "id" kolonunda %0 eksik,
  açıkça sayısal bir kolonun string olarak okunması) için belirsizlik_bayragi=false ver.
- Gerçek yargı gerektiren durumlar (ör. tarih formatı belirsiz, aykırı değerin
  hata mı yoksa gerçek mi olduğu belli değil, %30+ eksik veri olan bir kolonun
  ne yapılacağı) için belirsizlik_bayragi=true ver.
- Her bulgu kısa ve spesifik olsun, kolon adlarını kullan.

Çıktı formatı: bir JSON listesi, her eleman şu şemada:
{"bulgu": "...", "onerilen_duzeltme": "...", "gerekce": "...", "belirsizlik_bayragi": true/false}
"""


@dataclass
class LedgerEntry:
    bulgu: str
    onerilen_duzeltme: str
    gerekce: str
    belirsizlik_bayragi: bool
    timestamp: str


def generate_decisions(profile: dict[str, Any]) -> list[LedgerEntry]:
    """Profili LLM'e gönderir ve LedgerEntry listesi döndürür."""
    user_prompt = (
        "Aşağıdaki veri profiline bakarak karar listesini üret:\n\n"
        f"{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    raw_decisions = call_llm_json(_SYSTEM_PROMPT, user_prompt)

    if not isinstance(raw_decisions, list):
        raise RuntimeError("Karar Defteri LLM'i beklenmedik bir format döndürdü (liste değil).")

    now = datetime.now(timezone.utc).isoformat()
    entries = [
        LedgerEntry(
            bulgu=item["bulgu"],
            onerilen_duzeltme=item["onerilen_duzeltme"],
            gerekce=item["gerekce"],
            belirsizlik_bayragi=bool(item["belirsizlik_bayragi"]),
            timestamp=now,
        )
        for item in raw_decisions
    ]
    return entries


def persist_ledger(entries: list[LedgerEntry], run_id: str) -> Path:
    """Karar defterini denetim izi olarak diske yazar (JSON Lines)."""
    out_dir = Path(SETTINGS.audit_trail_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_decision_ledger.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return out_path
