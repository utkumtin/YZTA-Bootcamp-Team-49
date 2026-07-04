"""P1 Temizleme Motorunu uçtan uca bağlayan orkestratör.

ingestion -> decision_ledger -> gatekeeper -> code_executor
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from . import code_executor, decision_ledger, gatekeeper, ingestion


def run_cleaning_pipeline(
    input_path: str | Path,
    *,
    interactive: bool = True,
) -> tuple[pd.DataFrame, Path, Path]:
    """Ham dosyadan temiz dataframe'e giden tüm P1 akışını çalıştırır.

    Döndürür: (temiz_df, karar_defteri_yolu, denetim_izi_kod_yolu)
    """
    run_id = uuid.uuid4().hex[:10]

    raw_df = ingestion.load_raw_file(input_path)
    profile = ingestion.profile_dataframe(raw_df)

    entries = decision_ledger.generate_decisions(profile)
    ledger_path = decision_ledger.persist_ledger(entries, run_id)

    resolved = gatekeeper.run_gate(entries, interactive=interactive)

    clean_df, audit_path = code_executor.apply_decisions(raw_df, resolved, run_id)
    return clean_df, ledger_path, audit_path
