"""Bir multiverse koşusunun dondurulmuş estimand/menu provenance kaydı.

Kaydı analiz sayfası yazar, varyans paneli okur. İki taraf sözlüğün şeklini
kendi başına kurduğu sürece anahtar adı ya da alanlardan biri tek tarafta
değiştiğinde okuma sessizce "kayıt bulunamadı"ya düşüyor ve testler yeşil
kalıyordu. Şekil burada tek yerde tanımlı; yazan, okuyan ve testler aynı
fonksiyonlardan geçer.
"""

from __future__ import annotations

from typing import Any

from .store import ProjectStore

FROZEN_MENU_KEY = "frozen_menu"


def build_frozen_menu_record(
    *,
    estimand_hash: str,
    menu_hash: str,
    spec_count: int,
    run_id: str | None,
    estimand: dict[str, Any],
    menu: dict[str, Any],
) -> dict[str, Any]:
    """Diske yazılacak provenance kaydını üretir."""
    return {
        "estimand_hash": estimand_hash,
        "menu_hash": menu_hash,
        "spec_count": spec_count,
        "run_id": run_id,
        "estimand": estimand,
        "menu": menu,
    }


def save_frozen_menu_record(run_id: str, record: dict[str, Any]) -> None:
    """Kaydı koşunun kendi kapsamına yazar."""
    ProjectStore(scope_id=run_id).save(FROZEN_MENU_KEY, record)


def load_frozen_menu_record(run_id: str | None) -> dict[str, Any] | None:
    """Koşunun provenance kaydını okur; yoksa None.

    `ProjectStore.load` var olmayan anahtarda None döner (exception atmaz), bu
    yüzden geniş bir try/except'e gerek yok — yalnız beklenmedik şekle (dict
    olmayan içerik) karşı savunuluyoruz.
    """
    if not run_id:
        return None
    record = ProjectStore(scope_id=str(run_id)).load(FROZEN_MENU_KEY)
    return record if isinstance(record, dict) else None
