"""Gatekeeper (Onay Kapısı).

belirsizlik_bayragi=true olan her karar için sistemi durdurur ve kullanıcıdan
onay/yönlendirme ister. false olanlar otomatik geçer.

İki arayüz destekler:
  - CLI (varsayılan, script/otomasyon dostu)
  - Streamlit (dashboard içine gömülebilir; `resolve_via_streamlit`)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .decision_ledger import LedgerEntry


class Resolution(str, Enum):
    APPROVED = "approved"          # önerilen düzeltmeyi aynen uygula
    MODIFIED = "modified"          # kullanıcı alternatif bir talimat verdi
    REJECTED = "rejected"          # bu düzeltmeyi uygulama


@dataclass
class ResolvedDecision:
    entry: LedgerEntry
    resolution: Resolution
    user_instruction: str | None = None  # MODIFIED durumunda kullanıcının kendi talimatı


def auto_pass(entry: LedgerEntry) -> ResolvedDecision:
    """Belirsizlik bayrağı olmayan kararlar için otomatik onay."""
    return ResolvedDecision(entry=entry, resolution=Resolution.APPROVED)


def resolve_via_cli(entry: LedgerEntry) -> ResolvedDecision:
    """Terminal üzerinden kullanıcıya belirsiz bir kararı sorar."""
    print("\n--- BELİRSİZLİK TESPİT EDİLDİ ---")
    print(f"Bulgu     : {entry.bulgu}")
    print(f"Öneri     : {entry.onerilen_duzeltme}")
    print(f"Gerekçe   : {entry.gerekce}")
    choice = input("[o]nayla / [d]eğiştir / [r]eddet? > ").strip().lower()

    if choice == "o":
        return ResolvedDecision(entry=entry, resolution=Resolution.APPROVED)
    if choice == "d":
        instruction = input("Alternatif talimatını yaz: ").strip()
        return ResolvedDecision(entry=entry, resolution=Resolution.MODIFIED, user_instruction=instruction)
    return ResolvedDecision(entry=entry, resolution=Resolution.REJECTED)


def resolve_via_streamlit(entry: LedgerEntry, st_module) -> ResolvedDecision | None:
    """Streamlit içinde kullanılacak eşdeğer; `st_module` = `import streamlit as st`.

    None dönerse kullanıcı henüz karar vermemiş demektir (UI tekrar render edilmeli).
    """
    st_module.warning(f"**Belirsizlik**: {entry.bulgu}")
    st_module.caption(f"Öneri: {entry.onerilen_duzeltme}  \nGerekçe: {entry.gerekce}")
    col1, col2, col3 = st_module.columns(3)
    if col1.button("Onayla", key=f"approve_{entry.bulgu}"):
        return ResolvedDecision(entry=entry, resolution=Resolution.APPROVED)
    if col2.button("Reddet", key=f"reject_{entry.bulgu}"):
        return ResolvedDecision(entry=entry, resolution=Resolution.REJECTED)
    instruction = col3.text_input("Alternatif talimat", key=f"mod_{entry.bulgu}")
    if instruction and col3.button("Gönder", key=f"submit_{entry.bulgu}"):
        return ResolvedDecision(entry=entry, resolution=Resolution.MODIFIED, user_instruction=instruction)
    return None


def run_gate(entries: list[LedgerEntry], *, interactive: bool = True) -> list[ResolvedDecision]:
    """Tüm karar listesini gatekeeper'dan geçirir (CLI modu)."""
    resolved: list[ResolvedDecision] = []
    for entry in entries:
        if entry.belirsizlik_bayragi and interactive:
            resolved.append(resolve_via_cli(entry))
        else:
            resolved.append(auto_pass(entry))
    return resolved
