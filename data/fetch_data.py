"""Veri-erişim yardımcısı.

Beklenen ham dosyaların varlığını doğrular ve eksikse nasıl edinileceğini söyler.
DÜRÜSTLÜK: CDC WONDER interaktif bir sorgu aracıdır — programatik indirme YOK; manuel
export gerekir (bkz SOURCES.md). KFF küçük tablo repo'da gelir. Census/ACS Sprint-2.

Kullanım:  python data/fetch_data.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPECTED = {
    "medicaid/raw/cdc_wonder_mortality_1999_2020.tsv": (
        "CDC WONDER Multiple Cause of Death (ilçe × yıl, 1999-2020) — MANUEL export. "
        "http://wonder.cdc.gov/mcd.html · export'u bu yola koy (SOURCES.md)."
    ),
    "medicaid/raw/kff_expansion_dates.csv": (
        "KFF eyalet genişleme tarihleri — repo ile gelir (küçük, redistribute)."
    ),
}


def check() -> int:
    missing = 0
    for rel, how in EXPECTED.items():
        path = HERE / rel
        status = "OK " if path.exists() else "EKSİK"
        if not path.exists():
            missing += 1
        print(f"[{status}] {rel}")
        if not path.exists():
            print(f"        → {how}")
    if missing:
        print(f"\n{missing} dosya eksik. Yukarıdaki yönergelerle edinin.")
    else:
        print("\nTüm beklenen ham dosyalar mevcut.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto veri-erişim yardımcısı")
    parser.add_argument("--check", action="store_true", help="Beklenen ham dosyaları doğrula")
    args = parser.parse_args()
    if args.check or True:  # şimdilik tek eylem: check
        raise SystemExit(check())


if __name__ == "__main__":
    main()
