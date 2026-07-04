"""Pareto CLI girişi.

Kullanım:
    python main.py --input veri.csv --outcome y --treatment d \
        --unit firm_id --time year --research-context "H0: ... H1: ..."

Akış:
    1) P1 Temizleme Motoru  -> temiz dataframe + denetim izleri
    2) P2 Spec Generator     -> LLM'den serbestlik dereceleri menüsü
    3) P3 Multiverse + Runner -> N spesifikasyon paralel koşulur, sonuç JSON'a yazılır
    4) Kullanıcı sonrasında `streamlit run pareto/p4_dashboard/app.py -- --results ...` çalıştırır.
"""

from __future__ import annotations

import argparse
import uuid

from pareto.p1_cleaning.pipeline import run_cleaning_pipeline
from pareto.p2_p3_analysis.multiverse import build_multiverse
from pareto.p2_p3_analysis.runner import persist_results, run_multiverse
from pareto.p2_p3_analysis.spec_generator import generate_spec_menu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto: uçtan uca temizleme + çokluevren analizi")
    parser.add_argument("--input", required=True, help="Ham .csv veya .dta dosyası")
    parser.add_argument("--outcome", required=True, help="Sonuç değişkeni kolonu")
    parser.add_argument("--treatment", required=True, help="Tedavi/etki değişkeni kolonu")
    parser.add_argument("--unit", required=True, help="Panel birim kolonu (ör. firma id)")
    parser.add_argument("--time", required=True, help="Panel zaman kolonu (ör. yıl)")
    parser.add_argument(
        "--research-context",
        default="Tedavi değişkeninin sonuç üzerindeki nedensel etkisini tahmin ediyoruz.",
        help="H0/H1 ve araştırma bağlamının kısa açıklaması",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Gatekeeper'ı otomatik-onayla modunda çalıştır (belirsizlikleri de geçir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = uuid.uuid4().hex[:10]

    print("== P1: Temizleme Motoru ==")
    clean_df, ledger_path, audit_path = run_cleaning_pipeline(
        args.input, interactive=not args.non_interactive
    )
    print(f"Karar defteri : {ledger_path}")
    print(f"Denetim izi   : {audit_path}")
    print(f"Temiz veri    : {len(clean_df)} satır, {clean_df.shape[1]} kolon")

    print("\n== P2: Spec Generator ==")
    baseline_spec = {
        "outcome": args.outcome,
        "treatment": args.treatment,
        "unit": args.unit,
        "time": args.time,
    }
    spec_menu = generate_spec_menu(
        research_context=args.research_context,
        baseline_spec=baseline_spec,
        available_columns=list(clean_df.columns),
    )
    print(f"Kontrol seti sayısı : {len(spec_menu.control_sets)}")
    print(f"Pre-period pencereleri : {spec_menu.pre_period_windows}")
    print(f"Clustering seviyeleri  : {spec_menu.clustering_levels}")
    print(f"Gerekçe: {spec_menu.gerekce}")

    print("\n== P3: Multiverse + Runner ==")
    specs = build_multiverse(
        outcome=args.outcome,
        treatment=args.treatment,
        unit_col=args.unit,
        time_col=args.time,
        spec_menu=spec_menu,
    )
    print(f"{len(specs)} spesifikasyon üretildi. Çalıştırılıyor...")

    results = run_multiverse(clean_df, specs)
    results_path = persist_results(results, run_id)
    print(f"Sonuçlar yazıldı: {results_path}")

    print("\nDashboard için:")
    print(f"  streamlit run pareto/p4_dashboard/app.py -- --results {results_path}")


if __name__ == "__main__":
    main()
