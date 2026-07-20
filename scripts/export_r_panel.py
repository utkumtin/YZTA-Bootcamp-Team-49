"""S2-14 — Medicaid panelini R referans koşusu için CSV'ye aktarır.

Committed baseline (data/medicaid/config.yaml): 2014 genişleme kohortu vs
never-treated. `treated_post` göstergesi burada türetilir çünkü merge katmanı
bilinçli olarak türetmez (estimand aşamasının işi); R referansı ile pyfixest
aynı göstergeyi görmeli.

Çıktı: runs/r_fixture/medicaid_panel.csv (gitignored — deterministik, yeniden üretilebilir).
Devamı: scripts/r_reference_medicaid.R bu CSV'yi okuyup fixture üretir.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pareto.cleaning.merge import build_panel  # noqa: E402

OUT = REPO_ROOT / "runs" / "r_fixture" / "medicaid_panel.csv"

EXPORT_COLS = [
    "county_fips",
    "state_fips",
    "year",
    "treated_post",
    "crude_rate",
    "pct_uninsured",
    "median_hh_income",
    "poverty_rate",
    "unemployment_rate",
    "population",
]


def main() -> None:
    panel = build_panel(REPO_ROOT / "data" / "medicaid")
    df = panel.df
    in_2014_cohort = df["treatment_cohort"].eq(2014).fillna(False)
    sample = df[in_2014_cohort | df["never_treated"]].copy()
    sample["treated_post"] = (
        sample["treatment_cohort"].eq(2014).fillna(False) & (sample["year"] >= 2014)
    ).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample[EXPORT_COLS].to_csv(OUT, index=False)
    print(f"yazıldı: {OUT}")
    print(f"satır: {len(sample)} · treated_post=1: {int(sample['treated_post'].sum())}")


if __name__ == "__main__":
    main()
