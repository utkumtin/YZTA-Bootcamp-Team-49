"""Codegen L4 testleri: sandbox reprodüksiyonu + tolerans eşitliği (fail-loud)."""

import subprocess
from dataclasses import replace

import pandas as pd
import pytest

from pareto.cleaning import codegen
from pareto.cleaning.codegen import ReproductionError, apply_ledger, verify_reproduction
from pareto.cleaning.ledger import LedgerEntry


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    # Testler repo'nun gerçek runs/ dizinini kirletmesin; audit izleri tmp'e gitsin.
    patched = replace(codegen.SETTINGS, audit_trail_dir=str(tmp_path / "audit_trail"))
    monkeypatch.setattr(codegen, "SETTINGS", patched)
    return tmp_path / "audit_trail"


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame({"x": ["1,234.5", "5,678.25", "9.0"], "fips": ["1001", "2013", "40143"]})


def _entries() -> list[LedgerEntry]:
    return [
        LedgerEntry(
            bulgu="x kolonunda binlik ayraç var",
            transform_name="coerce_numeric",
            params={"col": "x"},
            gerekce="Sayısal analiz için numeric tip gerekli",
            belirsizlik_bayragi=False,
        ),
        LedgerEntry(
            bulgu="fips kodlarında öndeki sıfırlar düşmüş",
            transform_name="preserve_leading_zeros",
            params={"col": "fips", "width": 5},
            gerekce="FIPS eşleşmesi 5 haneli string ister",
            belirsizlik_bayragi=False,
        ),
    ]


def test_verify_reproduction_passes_when_script_reproduces_output(audit_dir):
    # NEDEN: L4 kapısının pozitif yolu. Diske yazılan audit script tek başına
    # (subprocess sandbox'ta) koşulduğunda in-process sonucu üretebilmeli;
    # üretemiyorsa denetim izi metot bölümü olarak güvenilmezdir.
    raw = _raw_df()
    cleaned, audit_path = apply_ledger(raw, _entries(), "run_ok")
    job_dir = verify_reproduction(raw, audit_path, cleaned, "run_ok")
    assert (job_dir / "reproduced.pkl").exists()


def test_verify_reproduction_fails_loud_on_injected_deviation(audit_dir):
    # NEDEN: kartın kabul kriteri. Script'e kasıtlı sapma enjekte edilir;
    # sandbox çıktısı in-process sonuçtan ayrışır ve kapı sessizce geçmek
    # yerine ReproductionError ile patlamalıdır.
    raw = _raw_df()
    cleaned, audit_path = apply_ledger(raw, _entries(), "run_dev")
    script = audit_path.read_text(encoding="utf-8")
    tampered = script.replace("    return df", "    df['x'] = df['x'] + 1\n    return df")
    assert tampered != script
    audit_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ReproductionError, match="tolerans"):
        verify_reproduction(raw, audit_path, cleaned, "run_dev")


def test_verify_reproduction_fails_loud_when_script_crashes(audit_dir, tmp_path):
    # NEDEN: fail-loud yalnız uyuşmazlıkta değil; script hiç koşamıyorsa da
    # kapı hatayı yutmadan, stderr'i mesaja taşıyarak patlamalıdır.
    raw = _raw_df()
    broken = tmp_path / "broken_steps.py"
    broken.write_text("def clean(df):\n    raise RuntimeError('kasitli cokme')\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match="kasitli cokme"):
        verify_reproduction(raw, broken, raw, "run_crash")


def test_verify_reproduction_fails_loud_when_worker_writes_no_output(audit_dir, monkeypatch):
    # NEDEN: exit=0 dönen ama reproduced.pkl yazmayan bir worker, kapının
    # ReproductionError sözleşmesi dışında ham FileNotFoundError sızdırmamalı;
    # UI yalnız ReproductionError yakaladığı için bu da tipli hatayla patlamalıdır.
    raw = _raw_df()
    cleaned, audit_path = apply_ledger(raw, _entries(), "run_noout")
    monkeypatch.setattr(
        codegen.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(ReproductionError, match="reproduced.pkl"):
        verify_reproduction(raw, audit_path, cleaned, "run_noout")


def test_verify_reproduction_tolerates_tiny_float_noise(audit_dir):
    # NEDEN: L4 bit eşitliği değil tolerans eşitliğidir; platform/BLAS kaynaklı
    # son-basamak float gürültüsü reprodüksiyonu geçersiz kılmamalıdır.
    raw = _raw_df()
    cleaned, audit_path = apply_ledger(raw, _entries(), "run_tol")
    noisy = cleaned.copy()
    noisy["x"] = noisy["x"] * (1 + 1e-9)
    verify_reproduction(raw, audit_path, noisy, "run_tol")
