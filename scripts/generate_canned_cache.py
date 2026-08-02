"""Golden-path canned-mode cache'ini gerçek Sonnet 5 (high effort) çağrılarıyla doldurur.

`app/home.py`'deki "Demo moduna gir" butonu, committed `app/demo.py`
sabitleriyle tam bu script'in izlediği yolu (temizleme → hipotez → menü →
multiverse → varyans anlatısı) yürütür. Buradaki her JUDGE çağrısı gerçek bir
`claude -p --effort high` alt süreci çalıştırır (bkz. pareto/llm/demo_sonnet_5.py)
ve `runs/llm_cache/`e (SETTINGS.llm_cache_dir, committed) yazar — deploy edilen
ortamda `DEMO_SONNET_5_SESSION` hiç tanımlı olmadığı için aynı istekler oradan
yalnız bu üretilen cache'i okuyarak karşılanır.

Kullanım:
    DEMO_SONNET_5_SESSION=1 PARETO_JUDGE_PROVIDER=demo_sonnet_5 \\
        .venv/bin/python scripts/generate_canned_cache.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DEMO_SONNET_5_SESSION", "1")
os.environ.setdefault("PARETO_JUDGE_PROVIDER", "demo_sonnet_5")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.demo import (  # noqa: E402
    DEMO_ANALYSIS_STATE,
    DEMO_DATASET_PATH,
    DEMO_DECLARATION,
    DEMO_RESEARCH_STORY,
)
from pareto.analysis.hypothesis import draft_tac_proposal, freeze_estimand  # noqa: E402
from pareto.analysis.menu import (  # noqa: E402
    ALL_AXES,
    evaluate_menu_defensibility,
    expand_to_specs,
    freeze_spec_menu,
    generate_spec_menu,
    validate_spec_menu_to_specs,
)
from pareto.analysis.runner import run_specs  # noqa: E402
from pareto.analysis.variance import diagnose_axes, summarize  # noqa: E402
from pareto.cleaning.agent import (  # noqa: E402
    Resolution,
    entries_to_apply,
    generate_ledger,
    resolve,
)
from pareto.cleaning.codegen import apply_ledger, verify_reproduction  # noqa: E402
from pareto.llm.narrative import generate_narrative  # noqa: E402
from pareto.profiling import load_raw_file, profile_dataframe  # noqa: E402

RUN_ID = "generate-canned-cache"


def main() -> int:
    print(f"[1/6] Veri yükleniyor: {DEMO_DATASET_PATH}")
    raw_df = load_raw_file(DEMO_DATASET_PATH)
    raw_profile = profile_dataframe(raw_df)
    print(f"      {raw_df.shape[0]} satır x {raw_df.shape[1]} kolon")

    print("[2/6] JUDGE: temizlik kararları (cleaning) — claude -p --effort high…")
    entries = generate_ledger(raw_profile)
    print(f"      {len(entries)} karar üretildi")
    resolutions = {}
    for i, entry in enumerate(entries):
        if entry.belirsizlik_bayragi:
            resolutions[i] = resolve(entry, auto_approve=False, resolution=Resolution.APPROVED)
        else:
            resolutions[i] = resolve(entry, auto_approve=True)
    to_apply = entries_to_apply(entries, resolutions)
    cleaned_df, audit_path = apply_ledger(raw_df, to_apply, RUN_ID)
    verify_reproduction(raw_df, audit_path, cleaned_df, RUN_ID)
    columns = [str(c) for c in cleaned_df.columns]
    print(f"      temizlendi: {cleaned_df.shape[0]} satır x {cleaned_df.shape[1]} kolon")

    print("[3/6] JUDGE: TAC proposal (hipotez/estimand) — claude -p --effort high…")
    proposal = draft_tac_proposal(
        research_story=DEMO_RESEARCH_STORY,
        available_columns=columns,
        declaration=DEMO_DECLARATION,
    )
    if proposal.needs_clarification:
        print(f"      DURDU: needs_clarification=True — {proposal.clarification_question}")
        return 1
    print(f"      treatment={proposal.treatment} outcome={proposal.outcome}")
    frozen_estimand = freeze_estimand(proposal, approved=True)

    print("[4/6] JUDGE: spec menu — claude -p --effort high…")
    menu_proposal = generate_spec_menu(frozen=frozen_estimand, available_columns=columns)
    if menu_proposal.needs_clarification:
        print(f"      DURDU: needs_clarification=True — {menu_proposal.clarification_question}")
        return 1
    ok, reasons, spec_count = evaluate_menu_defensibility(
        menu_proposal,
        available_columns=columns,
        outcome=frozen_estimand.estimand.outcome,
        treatment=frozen_estimand.estimand.treatment,
        unit_col=DEMO_ANALYSIS_STATE["unit_col"],
        time_col=DEMO_ANALYSIS_STATE["time_col"],
    )
    print(f"      savunulabilirlik: {ok} · spec_count={spec_count} · reasons={reasons}")
    if not ok:
        return 1
    frozen_menu = freeze_spec_menu(
        menu_proposal,
        available_columns=columns,
        approved=True,
        active_axes=tuple(ALL_AXES),
    ).menu.freeze()

    print("[5/6] Multiverse (deterministik OLS/TWFE, LLM yok)…")
    specs = expand_to_specs(
        frozen_menu,
        outcome=frozen_estimand.estimand.outcome,
        treatment=frozen_estimand.estimand.treatment,
        unit_col=DEMO_ANALYSIS_STATE["unit_col"],
        time_col=DEMO_ANALYSIS_STATE["time_col"],
        available_columns=columns,
    )
    validate_spec_menu_to_specs(frozen_menu, specs)
    print(f"      {len(specs)} spesifikasyon")
    results = run_specs(cleaned_df, specs)
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"      {n_ok}/{len(results)} başarılı")

    print("[6/6] JUDGE: varyans anlatısı (opsiyonel) — claude -p --effort high…")
    summary = dict(summarize(results))
    diagnosis = diagnose_axes(results, specs)
    if not summary.get("n_ok"):
        print("      atlandı: başarılı sonuç yok.")
        for r in results:
            if r.status != "ok":
                print(f"        {r.spec_id}: {r.error}")
        return 0
    try:
        narrative = generate_narrative(summary, diagnosis)
        print(f"      ozet: {narrative.ozet[:120]}")
    except Exception as exc:  # narrative opsiyonel — cache.py/3_variance_panel.py ile tutarlı
        print(f"      atlandı (opsiyonel, hata): {exc}")

    print("\nTamamlandı. runs/llm_cache/ içindeki yeni dosyaları commit'lemeyi unutmayın.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
