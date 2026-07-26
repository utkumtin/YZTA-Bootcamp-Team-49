"""Commit'li türev extract'lerin kaynaklarından yeniden üretilebilirliği.

Bir türev dosya repoda dururken, onu üreten fonksiyon commit'li kaynaktan farklı bir
çıktı verirse provenance iddiası sessizce yalanlanır: dosyayı yeniden üreten biri
analizin kullandığından başka sayılar elde eder. Üretici `dest.exists()` gördüğünde
atladığı için bu kayma kendiliğinden fark edilmez, yani kontrol testte durmak zorunda.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_KRUEGER_RAW = REPO_ROOT / "data" / "card_krueger" / "raw"


def _fetch_data_module():
    """`data/fetch_data.py` paket içinde değil, dosya yolundan yüklenir."""
    spec = importlib.util.spec_from_file_location("fetch_data", REPO_ROOT / "data/fetch_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_long_extract_reproduces_from_its_committed_source(tmp_path: Path) -> None:
    """Türev geniş kaynaktan byte-byte üretilemiyorsa kaynak gösterimi yanlıştır."""
    wide = CARD_KRUEGER_RAW / "card_krueger.csv"
    committed = CARD_KRUEGER_RAW / "card_krueger_long.csv"
    if not wide.exists() or not committed.exists():
        pytest.skip("Card-Krueger extract'leri lokalde yok")

    # Üretici çıktı yolunu `data/` köküne göre raporlar, o yüzden tmp dizini oraya açılır.
    scratch = CARD_KRUEGER_RAW / f"_{tmp_path.name}_long.csv"
    try:
        _fetch_data_module()._card_krueger_long(wide, scratch)
        assert scratch.read_bytes() == committed.read_bytes(), (
            "Commit'li uzun panel, geniş kaynaktan yeniden üretilenle aynı değil: "
            "türev kaynağından kopmuş."
        )
    finally:
        scratch.unlink(missing_ok=True)


def test_the_long_extract_keeps_every_store_in_both_waves() -> None:
    """Satır düşerse örneklem daralması temizleme aşamasının kararı olmaktan çıkar."""
    import pandas as pd

    committed = CARD_KRUEGER_RAW / "card_krueger_long.csv"
    if not committed.exists():
        pytest.skip("Card-Krueger uzun extract'i lokalde yok")

    df = pd.read_csv(committed, dtype=str)

    assert set(df["wave"]) == {"1", "2"}
    assert df["store_id"].nunique() == 410  # codebook: 410 mağaza
    assert len(df) == 820
    assert not df.duplicated(["store_id", "wave"]).any()
