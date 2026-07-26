"""S3-08 smoke matrisinin sözleşmeleri.

Matrisin yeşil raporu ancak şunlar doğruysa bir şey kanıtlar: JUDGE stub'ları
şemayla uyumlu, profiller dataset descriptor'larıyla senkron, tedavi göstergesi
gerçekten doğru satırları işaretliyor ve eksik veri/kırık adım sessizce geçmiyor.

Zincirin kendisi burada koşmaz (gerçek veri + subprocess ister); onun kanıtı
CI'da koşan harness'ın kendisidir. Burada korunan şey harness'ın iskeletidir.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from pydantic import BaseModel

import scripts.run_smoke_matrix as smoke
from pareto.analysis.hypothesis import TACProposal
from pareto.analysis.menu import SpecMenuProposal
from pareto.cleaning.agent import CleaningProposal
from pareto.cleaning.merge import build_panel
from pareto.config import SETTINGS
from pareto.contracts import Tier
from pareto.profiling import profile_dataframe
from scripts.run_smoke_matrix import (
    PROFILES,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    TREATMENT_FROM_COHORT,
    TREATMENT_FROM_PANEL,
    DatasetProfile,
    DatasetRun,
    main,
    missing_source_files,
    resolve_treatment,
    run_dataset,
    run_smoke_matrix,
    selected_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(profile: DatasetProfile) -> dict[str, Any]:
    path = REPO_ROOT / "data" / profile.key / "config.yaml"
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _declared_columns(config: dict[str, Any]) -> set[str]:
    return {role for source in config["sources"].values() for role in (source.get("columns") or {})}


def _cohort_panel() -> pd.DataFrame:
    """İki kohortlu birim + bir never-treated birim, 2013-2015."""
    units = (("a", 2014, False), ("b", 2015, False), ("c", None, True))
    return pd.DataFrame(
        [
            {"unit": unit, "year": year, "treatment_cohort": cohort, "never_treated": never}
            for unit, cohort, never in units
            for year in (2013, 2014, 2015)
        ]
    ).astype({"treatment_cohort": "Int64"})


def _profile(**overrides: Any) -> DatasetProfile:
    """Test için türetilmiş profil; alan eklendiğinde elle güncelleme gerektirmesin."""
    return replace(smoke.CARD_KRUEGER, **overrides)


def _dataset_dir_without_raw(tmp_path: Path, key: str) -> Path:
    """Config'i olan ama ham dosyası olmayan bir dataset dizini (CI checkout'u taklidi)."""
    dataset_dir = tmp_path / "data" / key
    dataset_dir.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "data" / key / "config.yaml", dataset_dir / "config.yaml")
    return dataset_dir


# --------------------------------------------------------------------------- #
# JUDGE stub'ları ve profil ↔ descriptor senkronu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", PROFILES, ids=[p.key for p in PROFILES])
@pytest.mark.parametrize(
    ("attribute", "model"),
    [
        ("judge_ledger", CleaningProposal),
        ("judge_estimand", TACProposal),
        ("judge_menu", SpecMenuProposal),
    ],
    ids=["ledger", "estimand", "menu"],
)
def test_judge_stubs_match_committed_output_schemas(
    profile: DatasetProfile, attribute: str, model: type[BaseModel]
) -> None:
    """Stub bir JUDGE şemasından koparsa matris çekirdeği değil kendini doğrular."""
    model.model_validate(getattr(profile, attribute))


@pytest.mark.parametrize("profile", PROFILES, ids=[p.key for p in PROFILES])
def test_profile_outcome_matches_the_dataset_descriptor(profile: DatasetProfile) -> None:
    """Descriptor'daki sonuç kolonu değişirse stub eski kolonu ölçmeye devam eder."""
    outcomes = _config(profile)["panel"]["outcome"]

    assert profile.judge_estimand["outcome"] in outcomes


@pytest.mark.parametrize("profile", PROFILES, ids=[p.key for p in PROFILES])
def test_profile_treatment_column_is_declared_or_derivable(profile: DatasetProfile) -> None:
    """Tedavi göstergesi ne descriptor'da ne kohortta varsa zincir menü adımında kopar."""
    config = _config(profile)
    treatment = config.get("treatment") or {}
    declared = profile.treatment_col in _declared_columns(config)
    derivable = bool(treatment.get("cohort_from"))

    assert declared or derivable
    # Descriptor'da hazır gelen gösterge `indicator_from` ile bildirilmezse merge onu
    # manifest'e taşımaz ve panel yalnız bu yüzden bir tier aşağı raporlar.
    if declared and not derivable:
        assert treatment.get("indicator_from") == profile.treatment_col


@pytest.mark.parametrize("profile", PROFILES, ids=[p.key for p in PROFILES])
def test_menu_stub_only_names_columns_the_dataset_declares(profile: DatasetProfile) -> None:
    """Menü dondurma uydurma kolonu reddeder; stub kolonu descriptor'dan gelmeli."""
    config = _config(profile)
    known = _declared_columns(config) | {"treatment_cohort", "never_treated", profile.treatment_col}

    named: set[str] = set()
    for axis in profile.judge_menu["axes"]:
        if axis["axis_name"] not in {"control_set", "clustering", "weighting"}:
            continue
        for level in [axis["baseline_level"], *axis["candidate_levels"]]:
            named.update(part for part in str(level).split("+") if part.lower() != "none")

    assert named <= known, f"stub bilinmeyen kolon anıyor: {sorted(named - known)}"


@pytest.mark.parametrize("profile", PROFILES, ids=[p.key for p in PROFILES])
def test_menu_stub_opens_more_than_one_axis(profile: DatasetProfile) -> None:
    """Tek eksen oynarsa faktöriyel açılım ve eksen atfı test edilmemiş kalır."""
    playing = [axis["axis_name"] for axis in profile.judge_menu["axes"] if axis["candidate_levels"]]

    assert len(playing) >= 2


def test_matrix_covers_structurally_different_datasets() -> None:
    """Matris aynı şekilli setlerden oluşursa genelleme garantisi diye bir şey kalmaz."""
    sources = {profile.key: len(_config(profile)["sources"]) for profile in PROFILES}

    assert len(PROFILES) >= 3
    assert min(sources.values()) == 1 and max(sources.values()) > 1


# --------------------------------------------------------------------------- #
# Tedavi göstergesi
# --------------------------------------------------------------------------- #
def test_treatment_is_derived_only_for_cohort_units_after_their_cohort_year() -> None:
    """Kontrol satırı tedavi işaretlenirse tüm matris yanlış bir şeyi doğrular."""
    df = _cohort_panel()

    column, source = resolve_treatment(df, _profile(), "year")

    assert (column, source) == ("treated_post", TREATMENT_FROM_COHORT)
    treated = {(row.unit, row.year) for row in df.itertuples() if row.treated_post == 1}
    assert treated == {("a", 2014), ("a", 2015), ("b", 2015)}


def test_existing_treatment_column_is_used_as_is() -> None:
    """Panelde hazır dummy varken kohorttan yeniden türetmek göstergeyi sessizce değiştirirdi."""
    df = _cohort_panel()
    # Kohorttan türetilse (a,2014), (a,2015), (b,2015) işaretlenirdi; buradaki desen
    # bilerek başka, "olduğu gibi kullanıldı" iddiası ancak böyle ayırt edilebilir.
    df["treated_post"] = [1, 0] * (len(df) // 2) + [0] * (len(df) % 2)

    column, source = resolve_treatment(df, _profile(), "year")

    assert (column, source) == ("treated_post", TREATMENT_FROM_PANEL)
    assert df["treated_post"].tolist() == [1, 0] * (len(df) // 2) + [0] * (len(df) % 2)


@pytest.mark.parametrize("level", [0, 1], ids=["hepsi-kontrol", "hepsi-tedavi"])
def test_panel_supplied_indicator_must_carry_both_arms(level: int) -> None:
    """Tek kollu gösterge zinciri geçer ama hiçbir karşılaştırma ölçmez; sessiz yeşil olur."""
    df = _cohort_panel()
    df["treated_post"] = level

    with pytest.raises(ValueError, match="0/1 ikilisini taşımıyor"):
        resolve_treatment(df, _profile(), "year")


def test_derived_indicator_must_carry_both_arms() -> None:
    """Kohort yolu da aynı korumayı ister: her birim kohortluysa kontrol grubu kalmaz."""
    df = _cohort_panel()
    df["treatment_cohort"] = 2013  # her birim ilk yıldan itibaren tedavili
    df["never_treated"] = False

    with pytest.raises(ValueError, match="0/1 ikilisini taşımıyor"):
        resolve_treatment(df, _profile(), "year")


def test_treatment_derivation_fails_loud_on_units_that_are_neither_cohorted_nor_control() -> None:
    """Belirsiz satır sessizce kontrol sayılırsa tedavi etkisi aşağı doğru sapar."""
    df = _cohort_panel()
    df.loc[df["unit"].eq("c"), "never_treated"] = False

    with pytest.raises(ValueError, match="ne kohort taşıyor ne never-treated"):
        resolve_treatment(df, _profile(), "year")


def test_treatment_derivation_fails_loud_without_cohort_columns() -> None:
    """Kohort kolonları yoksa gösterge uydurulamaz; sessiz sıfır kolonu üretilmemeli."""
    df = _cohort_panel().drop(columns=["treatment_cohort"])

    with pytest.raises(ValueError, match="eksik kolon"):
        resolve_treatment(df, _profile(), "year")


# --------------------------------------------------------------------------- #
# Eksik ham veri ve kırık adım
# --------------------------------------------------------------------------- #
def test_missing_source_files_reports_both_plain_and_glob_patterns(tmp_path: Path) -> None:
    """Glob'lu kaynak (yıl başına dosya) atlanırsa eksik veri var yok sanılır."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/present.csv").write_text("a\n", encoding="utf-8")
    config = {
        "sources": {
            "one": {"file": "raw/present.csv"},
            "two": {"file": "raw/absent.csv"},
            "many": {"file": "raw/series/part*.xlsx"},
        }
    }

    assert missing_source_files(config, tmp_path) == ["raw/absent.csv", "raw/series/part*.xlsx"]


def test_local_only_dataset_is_skipped_with_its_reason(tmp_path: Path) -> None:
    """Ham verisi repoda olmayan set CI'ı kırmamalı ama neden atlandığı raporda durmalı."""
    _dataset_dir_without_raw(tmp_path, "medicaid")

    run = run_dataset(smoke.MEDICAID, repo_root=tmp_path)

    assert run.status == STATUS_SKIPPED
    assert "ham veri eksik" in run.reason and smoke.MEDICAID.missing_data_note in run.reason
    assert run.steps == []


def test_missing_data_fails_loud_for_a_dataset_that_should_ship_with_the_repo(
    tmp_path: Path,
) -> None:
    """İşaretsiz bir setin verisi kaybolduğunda matris sessizce küçülmemeli."""
    _dataset_dir_without_raw(tmp_path, "card_krueger")

    run = run_dataset(_profile(local_only=False), repo_root=tmp_path)

    assert run.status == STATUS_FAILED
    assert "ham veri eksik" in run.reason


def test_an_unreadable_descriptor_breaks_only_its_own_dataset(tmp_path: Path) -> None:
    """Descriptor okunamayınca matris traceback ile ölerse kalan setler raporsuz kalır."""
    (tmp_path / "data" / "card_krueger").mkdir(parents=True)  # config.yaml bilerek yok
    _dataset_dir_without_raw(tmp_path, "divorce")

    report = run_smoke_matrix((_profile(local_only=False), smoke.DIVORCE), repo_root=tmp_path)

    assert [run["status"] for run in report["datasets"]] == [STATUS_FAILED, STATUS_FAILED]
    assert "descriptor okunamadı" in report["datasets"][0]["reason"]
    # divorce descriptor'ı okundu ve kendi eksik verisiyle kırıldı: matris gerçekten devam etti.
    assert "ham veri eksik" in report["datasets"][1]["reason"]


def test_a_broken_dataset_does_not_stop_the_rest_of_the_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """İlk kırılmada durulursa kalan setlerin durumu bilinmez ve rapor eksik kanıt olur."""
    calls: list[str] = []

    def fake_run(profile: DatasetProfile, **_kwargs: Any) -> DatasetRun:
        calls.append(profile.key)
        status = STATUS_FAILED if profile.key == PROFILES[0].key else STATUS_OK
        return DatasetRun(key=profile.key, status=status, reason="test", seconds=1.0)

    monkeypatch.setattr(smoke, "run_dataset", fake_run)

    report = run_smoke_matrix()

    assert calls == [profile.key for profile in PROFILES]
    assert report["overall"]["status"] == STATUS_FAILED
    assert PROFILES[0].key in report["overall"]["reason"]


def test_exit_code_reports_a_broken_dataset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sıfır çıkış kodu CI'yı yeşile boyar; kırık zincir buradan sızmamalı."""

    def fake_run(profile: DatasetProfile, **_kwargs: Any) -> DatasetRun:
        return DatasetRun(key=profile.key, status=STATUS_FAILED, reason="menu adımı kırıldı")

    monkeypatch.setattr(smoke, "run_dataset", fake_run)

    exit_code = main(["--dataset", "divorce"])

    assert exit_code == 1
    assert "KIRIK" in capsys.readouterr().out


def test_exit_code_is_zero_when_only_local_only_datasets_are_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CI'da medicaid'in atlanması kırık değildir; ama raporda görünmek zorundadır."""

    def fake_run(profile: DatasetProfile, **_kwargs: Any) -> DatasetRun:
        if profile.local_only:
            return DatasetRun(key=profile.key, status=STATUS_SKIPPED, reason="ham veri eksik")
        return DatasetRun(key=profile.key, status=STATUS_OK, reason="koştu", seconds=2.0)

    monkeypatch.setattr(smoke, "run_dataset", fake_run)

    exit_code = main([])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ATLANDI" in output and "medicaid" in output


def test_time_threshold_is_reported_as_a_warning_and_not_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eşik raporlanmazsa performans regresyonu görünmez; zorlanırsa yeşil zincir kırılır."""

    def fake_run(profile: DatasetProfile, **_kwargs: Any) -> DatasetRun:
        return DatasetRun(key=profile.key, status=STATUS_OK, reason="koştu", seconds=1000.0)

    monkeypatch.setattr(smoke, "run_dataset", fake_run)

    report = run_smoke_matrix()

    assert report["budget"]["total_seconds"] == 1000.0 * len(PROFILES)
    assert report["budget"]["within_threshold"] is False
    assert "EŞİK AŞILDI" in smoke.render_markdown(report)
    # Aşım yalnız uyarıdır: zincirin kendisi koştuğu için koşu geçerli sayılır.
    assert report["overall"]["status"] == STATUS_OK


def test_a_matrix_where_nothing_ran_is_reported_as_broken(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Boşalmış matris yeşil dönerse iddia doğrulanmamışken doğrulanmış görünür."""

    def fake_run(profile: DatasetProfile, **_kwargs: Any) -> DatasetRun:
        return DatasetRun(key=profile.key, status=STATUS_SKIPPED, reason="ham veri eksik")

    monkeypatch.setattr(smoke, "run_dataset", fake_run)

    exit_code = main([])

    assert exit_code == 1
    assert "hiçbir dataset koşmadı" in capsys.readouterr().out


def test_an_empty_dataset_selection_is_reported_as_broken() -> None:
    """Sıfır profille koşan bir matris de hiçbir şey kanıtlamaz; yeşil dönmemeli."""
    report = run_smoke_matrix(())

    assert report["overall"]["status"] == STATUS_FAILED


def test_per_dataset_timeout_stays_under_the_ci_job_limit() -> None:
    """Harness GitHub'dan önce konuşmalı: job kesilirse ne rapor ne de neden kalır."""
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    job_limit_seconds = int(workflow["jobs"]["smoke-matrix"]["timeout-minutes"]) * 60

    worst_case = smoke.MULTIVERSE_TIMEOUT_SECONDS * len(PROFILES)
    assert worst_case < job_limit_seconds


def test_report_states_its_side_effects() -> None:
    """runs/latest'in ezildiği ve raporun log olduğu söylenmezse okuyucu yanılır."""

    report = run_smoke_matrix(())

    rendered = smoke.render_markdown(report)
    assert report["notes"]
    assert "runs/latest" in rendered and "log" in rendered


def test_repeated_dataset_selection_runs_the_chain_once() -> None:
    """Aynı set iki kez koşarsa süre bütçesi ve rapor iki kat şişer."""
    assert selected_profiles(["divorce", "divorce"]) == (smoke.DIVORCE,)
    assert selected_profiles(None) == PROFILES


# --------------------------------------------------------------------------- #
# Gerçek zincir (repoda verisi olan en küçük set)
# --------------------------------------------------------------------------- #
def test_the_real_chain_runs_end_to_end_for_the_smallest_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zincir yalnız CI'da koşarsa pytest yeşilken kırık olabilir; yerel döngü de görmeli.

    Artefaktlar geliştiricinin `runs/` dizinini kirletmesin diye çalışma dizini
    değiştirilir: `SETTINGS.runs_dir` göreli olduğu için hem bu süreç hem de
    multiverse worker'ı tmp altına yazar.
    """
    monkeypatch.chdir(tmp_path)

    run = run_dataset(smoke.CARD_KRUEGER)

    assert run.status == STATUS_OK, run.reason
    assert [step["status"] for step in run.steps] == [STATUS_OK] * len(smoke.STEPS)
    menu_step = next(step for step in run.steps if step["key"] == "menu")
    assert menu_step["metrics"]["n_specs"] > 1


def test_load_step_fails_loud_when_the_panel_tier_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier sessizce değişirse aynı rapor başka bir analiz sınıfını doğrulamış olur."""
    monkeypatch.chdir(tmp_path)

    run = run_dataset(replace(smoke.CARD_KRUEGER, expected_tier=Tier.TIER2_CROSS_OLS))

    assert run.status == STATUS_FAILED
    assert "tier'ı beklenenden farklı" in run.reason


@pytest.mark.parametrize(
    "profile",
    [p for p in PROFILES if not p.local_only],
    ids=[p.key for p in PROFILES if not p.local_only],
)
def test_stub_decisions_stay_below_the_gatekeeper_threshold(profile: DatasetProfile) -> None:
    """Eşiği aşan kolona atıf yapan stub, koşuyu veriyi anlatmayan bir onay hatasıyla kırar."""
    panel = build_panel(REPO_ROOT / "data" / profile.key)
    columns = profile_dataframe(panel.df).get("columns", {})

    referenced = [
        str(value)
        for decision in profile.judge_ledger["decisions"]
        for key, value in decision["transform"].items()
        if key != "transform_name" and isinstance(value, str) and value in columns
    ]
    assert referenced, "Stub hiçbir gerçek kolona atıf yapmıyor; kontrol boşa koşuyor."
    for column in referenced:
        pct_missing = float(columns[column]["pct_missing"])
        assert pct_missing < SETTINGS.missing_value_hard_threshold, (
            f"{profile.key}/{column}: %{pct_missing * 100:.0f} eksik, gatekeeper bayrağı kalkar"
        )
