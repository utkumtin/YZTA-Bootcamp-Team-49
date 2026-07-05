"""Temizleme Agent'ı (human-in-the-loop).

LLM YARGISI + deterministik uygulama. Agent profili okur, kapalı transform
sözlüğünden (`transforms.ALLOWED_TRANSFORMS`) tipli kararlar (transform_name + params)
seçer, yüksek-güvenli olanları otomatik uygular, yalnız GERÇEK belirsizlikleri
kullanıcıya sorar (gatekeeper), her kararı ledger'a loglar.

Onay etkileşim ŞEMASI (onayla/değiştir/reddet) prototipten migre edildi ve burada
gerçek bir Sprint-1 dikişi olarak duruyor. LLM karar-ÜRETİMİ (profile → vetted ledger)
Sprint-2: router JUDGE rolü, PydanticAI ile tipli çıktı (kapalı sözlük şeması).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .ledger import LedgerEntry


class Resolution(StrEnum):
    APPROVED = "approved"  # önerilen transform'u aynen uygula
    MODIFIED = "modified"  # kullanıcı params'ı değiştirdi
    REJECTED = "rejected"  # bu düzeltmeyi uygulama


class ResolvedDecision(BaseModel):
    entry: LedgerEntry
    resolution: Resolution
    modified_params: dict | None = None


def resolve(entry: LedgerEntry, *, auto_approve: bool = True) -> ResolvedDecision:
    """Belirsizlik bayrağı yoksa otomatik onay; varsa UI kapısına bırakılır (Sprint-2)."""
    if entry.belirsizlik_bayragi and not auto_approve:
        raise NotImplementedError("Belirsiz kararlar için gatekeeper UI Sprint-2 (bkz docs/scrum).")
    return ResolvedDecision(entry=entry, resolution=Resolution.APPROVED)


def generate_ledger(profile: dict) -> list[LedgerEntry]:
    """LLM: profil → vetted transform kararları listesi.

    SPRINT-2: JUDGE modeli kapalı sözlükten transform_name + tipli params seçer
    (keyfi kod DEĞİL). Yüksek güven → belirsizlik_bayragi=false; gerçek yargı → true.
    """
    raise NotImplementedError("LLM karar üretimi Sprint-2 kapsamında (bkz docs/scrum).")
