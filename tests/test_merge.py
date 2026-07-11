"""merge_to_panel testleri — küçük fixture ile merge şekli + FIPS leading-zero + NA kuralı."""

from pathlib import Path

import pandas as pd
import pytest

from pareto.cleaning.merge import (
    build_panel,
    load_dataset_config,
    load_sources,
    merge_to_panel,
)
from pareto.contracts import Tier

# --------------------------------------------------------------------------- fixtures


def _cdc_df() -> pd.DataFrame:
    # 2 ilçe (AL=01, CO=08) × 2 yıl; her şey str (ham okuma davranışı).
    return pd.DataFrame(
        {
            "County Code": ["01001", "01001", "08001", "08001"],
            "Year Code": ["2013", "2014", "2013", "2014"],
            "Deaths": ["412", "430", "300", "Suppressed"],
            "Population": ["54135", "54571", "40000", "41000"],
            "Crude Rate": ["761.1", "788.0", "750.0", "Unreliable"],
        }
    )


def _kff_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "08"],
            "status": ["Not Adopted", "Adopted"],
            "implementation_date": [None, "2014-01-01"],
        }
    )


def _mini_config() -> dict:
    """Medicaid config.yaml yapısının küçültülmüş aynası (CDC + KFF)."""
    return {
        "sources": {
            "cdc": {
                "file": "raw/cdc.tsv",
                "read": {"na_markers": ["Suppressed", "Unreliable"]},
                "columns": {
                    "county_fips": "County Code",
                    "year": "Year Code",
                    "deaths": "Deaths",
                    "population": "Population",
                    "crude_rate": "Crude Rate",
                },
            },
            "kff": {
                "file": "raw/kff.csv",
                "columns": {
                    "state_fips": "state_fips",
                    "status": "status",
                    "implementation_date": "implementation_date",
                },
            },
        },
        "merge": {
            "derive": {
                "state_fips": {"from": "county_fips", "take_first": 2},
                "county_fips": {"concat": ["state_fips", "county_fips_3"]},
            },
            "on": "state_fips",
            "how": "left",
        },
        "panel": {
            "unit": "county_fips",
            "time": "year",
            "outcome": ["crude_rate"],
            "covariates": [],
            "weight": "population",
        },
        "treatment": {
            "cohort_from": "implementation_date",
            "never_treated_when": {"status": "Not Adopted"},
        },
    }


def _merge(cdc: pd.DataFrame | None = None, kff: pd.DataFrame | None = None):
    sources = {
        "cdc": cdc if cdc is not None else _cdc_df(),
        "kff": kff if kff is not None else _kff_df(),
    }
    return merge_to_panel(sources, _mini_config())


# --------------------------------------------------------------------------- merge şekli


def test_left_merge_preserves_spine_shape_and_broadcasts_state_rows():
    # NEDEN: panel = spine'ın (ilçe-yıl) satır kümesi; eyalet-seviye KFF satır
    # ÇOĞALTMADAN (m:1) her ilçe-yıla yayılmalı. Fan-out olursa DiD örneklemi şişer.
    panel = _merge()
    assert len(panel.df) == 4
    al = panel.df[panel.df["county_fips"] == "01001"]
    assert (al["status"] == "Not Adopted").all()


def test_fips_leading_zero_preserved_and_state_derived_from_first_two():
    # NEDEN: FIPS int'e düşerse "01001"→1001 olur ve ilk-2-hane eyalet eşlemesi
    # sessizce yanlış eyalete bağlar. Leading-zero bu dikişin bel kemiği.
    panel = _merge()
    assert set(panel.df["county_fips"]) == {"01001", "08001"}
    assert set(panel.df["state_fips"]) == {"01", "08"}


def test_suppressed_markers_become_na_and_outcome_is_numeric():
    # NEDEN: DUA-uyumu — Suppressed/Unreliable hücreler NA'dır, yeniden kurulmaz;
    # outcome sayısal dtype'a inmeli ki estimator doğrudan tüketebilsin.
    panel = _merge()
    row = panel.df[(panel.df["county_fips"] == "08001") & (panel.df["year"] == 2014)]
    assert row["crude_rate"].isna().all()
    assert row["deaths"].isna().all()
    assert pd.api.types.is_float_dtype(panel.df["crude_rate"])
    assert pd.api.types.is_numeric_dtype(panel.df["population"])


def test_rows_without_panel_keys_are_dropped():
    # NEDEN: LAUS-tarzı dosya sonu boş satırlar (unit/time NA) panele sızmamalı.
    cdc = pd.concat([_cdc_df(), pd.DataFrame({"County Code": [None], "Year Code": [None]})])
    panel = _merge(cdc=cdc)
    assert len(panel.df) == 4


def test_county_level_source_joins_on_two_part_fips_times_year():
    # NEDEN: SAHIE/SAIPE FIPS'i iki parça (statefips+countyfips) gelir; concat
    # türetmesi + unit×time join bu merge'ün çekirdek numarası — değer doğru
    # ilçe-yıla oturmalı, yanlış hizalanma sessiz veri bozar.
    config = _mini_config()
    config["sources"]["sahie"] = {
        "file": "raw/sahie.csv",
        "columns": {
            "state_fips": "statefips",
            "county_fips_3": "countyfips",
            "year": "year",
            "pct_uninsured": "pctui",
        },
    }
    config["panel"]["outcome"] = ["crude_rate", "pct_uninsured"]
    sahie = pd.DataFrame(
        {
            "statefips": ["01", "08"],
            "countyfips": ["001", "001"],
            "year": ["2014", "2014"],
            "pctui": ["18.0", "11.5"],
        }
    )
    panel = merge_to_panel({"cdc": _cdc_df(), "kff": _kff_df(), "sahie": sahie}, config)
    df = panel.df.set_index(["county_fips", "year"])
    assert df.loc[("01001", 2014), "pct_uninsured"] == 18.0
    assert df.loc[("08001", 2014), "pct_uninsured"] == 11.5
    assert pd.isna(df.loc[("01001", 2013), "pct_uninsured"])  # left-merge: eşleşmeyen yıl NA


# --------------------------------------------------------------------------- tedavi türetmesi


def test_treatment_cohort_and_never_treated_derived_to_tier1():
    # NEDEN: config'in treatment bloğu deterministik kohort/never-treated üretir;
    # bunlar manifest'e yazılmazsa panel Tier2'ye düşer ve DiD zinciri kopar.
    panel = _merge()
    df = panel.df.set_index(["county_fips", "year"])
    assert df.loc[("08001", 2014), "treatment_cohort"] == 2014
    assert bool(df.loc[("01001", 2014), "never_treated"]) is True
    assert pd.isna(df.loc[("01001", 2014), "treatment_cohort"])  # adopte etmeyen: kohort NA
    assert panel.manifest.treatment_cohort_col == "treatment_cohort"
    assert panel.manifest.never_treated_col == "never_treated"
    assert panel.validate_contract() is Tier.TIER1_PANEL_DID


# --------------------------------------------------------------------------- fail-loud


def test_missing_declared_source_column_fails_loud():
    # NEDEN: sessiz şema kayması (kaynak dosya formatı değişti) anında patlamalı,
    # eksik kolonla yarım panel üretilmemeli.
    with pytest.raises(ValueError, match="cdc"):
        _merge(cdc=_cdc_df().drop(columns=["Deaths"]))


def test_duplicate_spine_rows_fail_loud():
    # NEDEN: spine unit×time başına tekil değilse her merge katı sessizce çoğalır.
    with pytest.raises(ValueError, match="tekil"):
        _merge(cdc=pd.concat([_cdc_df(), _cdc_df().iloc[[0]]]))


def test_non_unique_right_side_fails_loud_as_m1_violation():
    # NEDEN: KFF'te bir eyaletin iki satırı olursa left-merge satırları fan-out
    # eder; m:1 ihlali gürültüyle durmalı.
    with pytest.raises(ValueError, match="m:1"):
        _merge(kff=pd.concat([_kff_df(), _kff_df().iloc[[0]]]))


# --------------------------------------------------------------------------- I/O katmanı


def test_load_sources_truncates_at_footer_marker(tmp_path: Path):
    # NEDEN: CDC WONDER export'unun "---" sonrası metadata footer'ı veri değildir;
    # sızarsa FIPS kolonuna metin karışır ve merge patlar.
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "cdc.tsv").write_text(
        '"County Code"\t"Year Code"\tDeaths\n'
        '"01001"\t"2013"\t412\n'
        '"01001"\t"2014"\t430\n'
        '"---"\n'
        '"Dataset: Multiple Cause of Death"\n',
        encoding="utf-8",
    )
    config = {
        "sources": {
            "cdc": {
                "file": "raw/cdc.tsv",
                "format": "tsv",
                "read": {"sep": "\t", "skipfooter_marker": "---"},
                "columns": {"county_fips": "County Code", "year": "Year Code"},
            }
        },
        "panel": {"unit": "county_fips", "time": "year", "outcome": []},
    }
    sources = load_sources(config, tmp_path)
    assert len(sources["cdc"]) == 2
    assert sources["cdc"]["County Code"].tolist() == ["01001", "01001"]


def test_load_sources_glob_concats_per_year_files(tmp_path: Path):
    # NEDEN: LAUS yıl-başına ayrı xlsx gelir; glob sıralı okunup tek frame olmalı.
    (tmp_path / "raw").mkdir()
    for year in ("2013", "2014"):
        pd.DataFrame({"fips": ["01001"], "Year": [year]}).to_excel(
            tmp_path / "raw" / f"laus{year}.xlsx", index=False
        )
    config = {
        "sources": {
            "laus": {
                "file": "raw/laus*.xlsx",
                "format": "xlsx",
                "columns": {"county_fips": "fips", "year": "Year"},
            }
        },
        "panel": {"unit": "county_fips", "time": "year", "outcome": []},
    }
    sources = load_sources(config, tmp_path)
    assert len(sources["laus"]) == 2
    assert sorted(sources["laus"]["Year"]) == ["2013", "2014"]


def test_load_dataset_config_requires_sources_and_panel(tmp_path: Path):
    (tmp_path / "config.yaml").write_text("dataset: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zorunlu"):
        load_dataset_config(tmp_path)


# --------------------------------------------------------------------------- gerçek veri (lokal)

_MEDICAID_DIR = Path(__file__).resolve().parents[1] / "data" / "medicaid"
_CDC_TSV = _MEDICAID_DIR / "raw" / "cdc_wonder_mortality_2009_2019.tsv"


@pytest.mark.skipif(not _CDC_TSV.exists(), reason="Medicaid ham verisi lokalde yok (CI'da atlanır)")
def test_medicaid_end_to_end_panel_is_tier1():
    # NEDEN: hero config gerçek dosyalarla uçtan uca Tier1 panel üretmeli —
    # config.yaml ile kodun sözleşmesi ancak gerçek veride kanıtlanır.
    panel = build_panel(_MEDICAID_DIR)
    assert panel.validate_contract() is Tier.TIER1_PANEL_DID
    df = panel.df
    assert len(df) > 30_000
    assert df["year"].between(2009, 2019).all()
    assert "01001" in set(df["county_fips"])  # leading-zero hayatta
    assert (df["treatment_cohort"] == 2014).any()  # 2014 genişleme kohortu
    assert df["never_treated"].any()
    assert df["crude_rate"].isna().any()  # Unreliable → NA (DUA)
    assert df["pct_uninsured"].notna().mean() > 0.9
    assert df["unemployment_rate"].notna().mean() > 0.9
