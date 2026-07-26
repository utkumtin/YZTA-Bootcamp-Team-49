"""S2-13 uçtan uca doğrulama harness'ının sözleşmeleri.

Harness'ın yeşil raporunun anlamını taşıyan şeyleri korur: JUDGE stub'ları
şemayla uyumlu kalmalı, stub'lar doğrulanmak istenen yolları gerçekten
uyarmalı, dikişler çekirdeğe doğru bağlanmalı ve kırık bir dikiş sessizce
geçmemeli.

Tam akış burada koşmaz: ham CDC dosyası repoda tutulmuyor, o yüzden `load`
dikişi ve gerçek multiverse koşusu dışarıda kalır. Kalan dikişler sentetik
panelle gerçekten çalıştırılır; kapsanmayan kısmın kanıtı, harness'ın yerelde
üretip commit'lediği doğrulama raporudur.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import BaseModel

import scripts.run_e2e_check as e2e
from pareto.analysis.hypothesis import TACProposal
from pareto.analysis.menu import SpecMenuProposal
from pareto.cleaning import codegen, ledger
from pareto.cleaning.agent import CleaningProposal, Resolution, ResolvedDecision
from pareto.contracts import EstimationResult
from pareto.llm.narrative import VarianceNarrative
from pareto.profiling import profile_dataframe
from scripts.run_e2e_check import (
    STATUS_BLOCKED,
    STATUS_EXTERNAL,
    STATUS_FAILED,
    STATUS_OK,
    build_checklist,
    committed_baseline_sample,
    main,
    render_markdown,
    run_seams,
)


def _seam_report(key: object, status: str, detail: str = "") -> dict[str, object]:
    return {
        "key": key,
        "title": f"{key} dikişi",
        "status": status,
        "detail": detail,
        "metrics": {},
    }


def _all_seams_passing() -> list[dict[str, object]]:
    return [_seam_report(key, STATUS_OK) for key, _title, _fn in e2e.SEAMS]


def _committed_config(cohorts: list[int] | None = None) -> dict[str, object]:
    return {
        "treatment": {"committed_baseline": {"cohorts": cohorts or [2014]}},
        "panel": {"unit": "county_fips", "time": "year", "weight": "population"},
    }


def _committed_panel() -> pd.DataFrame:
    """İki genişleme ilçesi (2014 kohortu) + bir hiç genişlemeyen ilçe, 2013-2015.

    Kolon kümesi JUDGE stub'larının adlandırdığı her kolonu taşır: menü dondurma
    adımı mevcut olmayan bir kolona atıf yapan seviyeyi reddettiği için, eksik
    kolon dikiş testini gerçek hata yerine kurulum hatasıyla düşürürdü.
    """
    units = (("01001", 2014, False), ("01003", 2014, False), ("02001", None, True))
    rows = [
        {
            "county_fips": unit,
            "year": year,
            "treatment_cohort": cohort,
            "never_treated": never,
            "pct_uninsured": 20.0 - (year - 2013) - (2.0 if cohort and year >= cohort else 0.0),
            "median_hh_income": 45000.0 + 1000.0 * (year - 2013),
            "poverty_rate": 15.0 - 0.5 * (year - 2013),
            "unemployment_rate": 6.0,
            "population": 100000.0,
        }
        for unit, cohort, never in units
        for year in (2013, 2014, 2015)
    ]
    return pd.DataFrame(rows).astype({"treatment_cohort": "Int64"})


# --------------------------------------------------------------------------- #
# JUDGE stub'ları: şema uyumu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "model"),
    [
        (e2e.JUDGE_LEDGER_OUTPUT, CleaningProposal),
        (e2e.JUDGE_ESTIMAND_OUTPUT, TACProposal),
        (e2e.JUDGE_MENU_OUTPUT, SpecMenuProposal),
        (e2e.JUDGE_NARRATIVE_OUTPUT, VarianceNarrative),
    ],
    ids=["ledger", "estimand", "menu", "narrative"],
)
def test_judge_stubs_match_committed_output_schemas(payload: dict, model: type[BaseModel]) -> None:
    """Stub bir JUDGE şemasından koparsa harness gerçek dikişi değil kendini doğrular."""
    model.model_validate(payload)


def test_ledger_stub_exercises_both_confidence_paths() -> None:
    """Tek güven seviyesi kalırsa gatekeeper yolu hiç uyarılmaz, yeşil rapor eksik kanıt olur."""
    confidences = {decision["confidence"] for decision in e2e.JUDGE_LEDGER_OUTPUT["decisions"]}
    assert confidences == {"high", "low"}


def test_menu_stub_opens_more_than_one_axis() -> None:
    """Tek eksen oynarsa matched-pair atfı boş kalır ve panel dikişi anlamsız yeşile döner."""
    multi_level = [
        axis["axis_name"] for axis in e2e.JUDGE_MENU_OUTPUT["axes"] if axis["candidate_levels"]
    ]
    assert len(multi_level) >= 2


def test_narrative_stub_only_names_axes_the_core_can_report() -> None:
    """Anlatı, deterministik teşhiste olmayan bir eksen anarsa çekirdek fail-loud eder."""
    from pareto.analysis.variance import AXES

    named = {comment["axis"] for comment in e2e.JUDGE_NARRATIVE_OUTPUT["eksen_yorumlari"]}
    assert named <= set(AXES)


# --------------------------------------------------------------------------- #
# Committed baseline türetmesi
# --------------------------------------------------------------------------- #
def test_committed_baseline_marks_only_treated_units_after_the_cohort_year() -> None:
    """Kontrol satırı tedavi işaretlenirse tüm uçtan uca yeşil yanlış bir şeyi doğrular."""
    sample, cohort = committed_baseline_sample(_committed_panel(), _committed_config())

    assert cohort == 2014
    treated = sample[sample["treated_post"] == 1]
    assert set(treated["county_fips"]) == {"01001", "01003"}
    assert set(treated["year"]) == {2014, 2015}
    assert not sample.loc[sample["never_treated"], "treated_post"].any()


def test_committed_baseline_keeps_never_treated_control_group() -> None:
    """Kontrol grubu düşerse DiD karşılaştırması kalmaz; örneklem sessizce daralmamalı."""
    sample, _cohort = committed_baseline_sample(_committed_panel(), _committed_config())

    assert set(sample["county_fips"]) == {"01001", "01003", "02001"}


def test_committed_baseline_rejects_multi_cohort_config() -> None:
    """Committed baseline tek kohortludur; çok kohortlu config sessizce ilkine düşmemeli."""
    with pytest.raises(ValueError, match="tek kohort"):
        committed_baseline_sample(_committed_panel(), _committed_config([2014, 2015]))


def test_committed_baseline_requires_treatment_columns() -> None:
    """Merge tedavi kolonlarını türetmediyse eksik kolon fail-loud raporlanmalı."""
    panel = _committed_panel().drop(columns=["never_treated"])

    with pytest.raises(ValueError, match="never_treated"):
        committed_baseline_sample(panel, _committed_config())


# --------------------------------------------------------------------------- #
# Kırık dikişin raporlanması
# --------------------------------------------------------------------------- #
def test_broken_seam_blocks_the_rest_and_is_never_reported_as_pass(monkeypatch) -> None:
    """Kırık dikişten sonrası koşulmaz; koşulmayan adım "geçti" sayılırsa rapor yalan söyler."""

    def _ok(_state: dict) -> tuple[str, dict]:
        return "tamam", {}

    def _boom(_state: dict) -> tuple[str, dict]:
        raise ValueError("dikiş koptu")

    def _should_not_run(_state: dict) -> tuple[str, dict]:
        raise AssertionError("kırık dikişten sonraki adım çalıştırılmamalıydı")

    monkeypatch.setattr(
        e2e,
        "SEAMS",
        (("load", "Yükle", _ok), ("clean", "Temizle", _boom), ("panel", "Panel", _should_not_run)),
    )

    reports = run_seams({})

    assert [report["status"] for report in reports] == [STATUS_OK, STATUS_FAILED, STATUS_BLOCKED]
    assert "dikiş koptu" in reports[1]["detail"]


def test_checklist_covers_every_seam_plus_llm_and_quality_gate() -> None:
    """Kontrol listesi dikişlerin bir alt kümesine daralırsa "DoS yeşil" iddiası eksik kalır."""
    items = build_checklist(_all_seams_passing())

    assert len(items) == len(e2e.SEAMS) + 2
    assert items[-2]["status"] == STATUS_OK
    assert items[-1]["status"] == STATUS_EXTERNAL


def test_quality_gate_item_is_never_claimed_as_verified_here() -> None:
    """Harness lint/tip/test koşmaz; bunları geçmiş göstermek uydurma yeşil olur."""
    items = build_checklist(_all_seams_passing())
    quality_gate = items[-1]

    assert quality_gate["status"] != STATUS_OK
    for command in e2e.QUALITY_GATE_COMMANDS:
        assert command in quality_gate["note"]


def test_llm_checklist_item_fails_when_a_judge_seam_breaks() -> None:
    """JUDGE dikişi kırıkken "her LLM adımı koşulabilir" maddesi yeşil kalmamalı."""
    seams = _all_seams_passing()
    for report in seams:
        if report["key"] == "menu":
            report["status"] = STATUS_FAILED

    items = build_checklist(seams)

    assert items[-2]["status"] == STATUS_FAILED


def test_markdown_shows_the_broken_seam_reason() -> None:
    """Kırık dikişin nedeni raporda görünmezse hata sessizce kaybolur."""
    seams = _all_seams_passing()
    seams[2] = _seam_report(seams[2]["key"], STATUS_FAILED, "ValueError: dikiş koptu")
    report = {
        "check": "S2-13",
        "dataset": "medicaid",
        "run_id": "s2-13-e2e",
        "provenance": {
            "generation_command": "cmd",
            "source_commit": "abc1234",
            "python": "3.11.7",
            "platform": "Linux",
            "judge_mode": "test modeli",
        },
        "overall": {"status": STATUS_FAILED, "reason": "en az bir dikiş kırıldı"},
        "seams": seams,
        "checklist": build_checklist(seams),
        "limitations": ["kapsam notu"],
    }

    markdown = render_markdown(report)

    assert "ValueError: dikiş koptu" in markdown
    assert "KIRIK" in markdown


def test_process_exits_non_zero_when_a_seam_breaks(monkeypatch, capsys) -> None:
    """CI kırık akışı yakalayabilsin diye kırık rapor sıfır olmayan çıkış kodu vermeli."""
    seams = _all_seams_passing()
    seams[0] = _seam_report(seams[0]["key"], STATUS_FAILED, "ValueError: dikiş koptu")
    failing = {
        "check": "S2-13",
        "dataset": "medicaid",
        "run_id": "s2-13-e2e",
        "provenance": {
            "generation_command": "cmd",
            "source_commit": "abc1234",
            "python": "3.11.7",
            "platform": "Linux",
            "judge_mode": "test modeli",
        },
        "overall": {"status": STATUS_FAILED, "reason": "en az bir dikiş kırıldı"},
        "seams": seams,
        "checklist": build_checklist(seams),
        "limitations": [],
    }
    monkeypatch.setattr(e2e, "run_e2e_check", lambda **_kwargs: failing)
    monkeypatch.setattr("sys.argv", ["run_e2e_check.py"])

    exit_code = main()

    assert exit_code != 0
    assert "KIRIK" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Dikişlerin kendisi
#
# Buradaki testler ham CDC dosyası gerektiren `load` dikişini atlar; kalan
# dikişleri sentetik panelle gerçekten koşar. Amaç pareto çekirdeğini yeniden
# test etmek değil, harness'ın çekirdeğe doğru bağlandığını ve kendi fail-loud
# kapılarının çalıştığını göstermek: aksi halde yeşil rapor yalnız stub'ların
# birbiriyle tutarlı olduğunu kanıtlar.
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Dikiş testleri repodaki `runs/` dizinine yazmasın, oradan da okumasın."""
    for module in (codegen, ledger):
        patched = replace(module.SETTINGS, audit_trail_dir=str(tmp_path / "audit_trail"))
        monkeypatch.setattr(module, "SETTINGS", patched)
    monkeypatch.setattr(e2e, "SETTINGS", replace(e2e.SETTINGS, runs_dir=str(tmp_path / "runs")))
    return tmp_path


def _profiled_state() -> dict[str, object]:
    panel = _committed_panel()
    return {"panel": SimpleNamespace(df=panel), "profile": profile_dataframe(panel)}


def _analysis_state(isolated: object) -> dict[str, object]:
    """Estimand ve menü dikişlerini gerçekten koşturup panel dikişinin girdisini üretir."""
    state: dict[str, object] = {"clean_df": _committed_panel(), "config": _committed_config()}
    e2e._seam_estimand(state)
    e2e._seam_menu(state)
    return state


def test_profile_seam_threads_the_profile_into_state_for_cleaning() -> None:
    """Temizleme dikişi profili state'ten okur; burada kopan zincir sonraki dikişte görünmez."""
    state = _profiled_state()
    del state["profile"]

    _detail, metrics = e2e._seam_profile(state)

    assert state["profile"]["columns"], "profil state'e yazılmadı, temizleme beslenemez"
    assert metrics["n_rows"] == len(_committed_panel())


def test_profile_seam_fails_loud_when_nothing_was_profiled() -> None:
    """Kolonsuz profil temizlemeyi boş besler; dikiş durmazsa yeşil rapor anlamsızlaşır."""
    state = {"panel": SimpleNamespace(df=pd.DataFrame())}

    with pytest.raises(ValueError, match="Profil kolon üretmedi"):
        e2e._seam_profile(state)


def test_clean_seam_gates_the_flagged_decision_and_applies_the_rest(isolated_artifacts) -> None:
    """Temizleme dikişi JUDGE defterini gerçekten uygulayıp L4 reprodüksiyonu doğrulamalı."""
    state = _profiled_state()

    _detail, metrics = e2e._seam_clean(state)

    assert metrics["n_decisions"] == 2
    assert metrics["n_gated"] == 1, "bayraklı karar gatekeeper'a düşmedi"
    assert metrics["n_applied"] == 2
    assert isinstance(state["clean_df"], pd.DataFrame)


def test_clean_seam_fails_when_the_gatekeeper_stops_gating(isolated_artifacts, monkeypatch) -> None:
    """Bayraklı karar otomatik onaylanabiliyorsa gatekeeper kırıktır, dikiş yeşil olamaz."""
    monkeypatch.setattr(
        e2e,
        "resolve",
        lambda entry, **_kwargs: ResolvedDecision(entry=entry, resolution=Resolution.APPROVED),
    )
    state = _profiled_state()

    with pytest.raises(ValueError, match="Gatekeeper kırık"):
        e2e._seam_clean(state)


def test_clean_seam_fails_when_no_decision_is_flagged(isolated_artifacts, monkeypatch) -> None:
    """Bayraksız defter gatekeeper yolunu hiç uyarmaz; kanıtsız yeşil rapor olur."""
    only_confident = {
        "decisions": [dict(e2e.JUDGE_LEDGER_OUTPUT["decisions"][0], confidence="high")]
    }
    monkeypatch.setattr(e2e, "JUDGE_LEDGER_OUTPUT", only_confident)
    state = _profiled_state()

    with pytest.raises(ValueError, match="Belirsizlik bayrağı hiç kalkmadı"):
        e2e._seam_clean(state)


def test_estimand_seam_freezes_from_the_committed_baseline_sample(isolated_artifacts) -> None:
    """Dondurma temizlenmiş panelden türetilmezse sonraki her adım başka bir örneklemi ölçer."""
    state: dict[str, object] = {"clean_df": _committed_panel(), "config": _committed_config()}

    _detail, metrics = e2e._seam_estimand(state)

    assert metrics["committed_cohort"] == 2014
    # İki genişleme ilçesi x 2014-2015 = 4 tedavi-sonrası satır; kontrol grubu işaretlenmez.
    assert metrics["n_treated_post_rows"] == 4
    assert state["frozen_estimand"].freeze_hash


def test_menu_seam_expands_the_frozen_menu_and_runs_both_validators(isolated_artifacts) -> None:
    """Menü stub'u iki eksende oynar; faktöriyel açılım sert tavanın altında kalmalı."""
    state = _analysis_state(isolated_artifacts)

    specs = state["specs"]
    assert len(specs) == 8, "iki kontrol seti x iki kestirici x iki ağırlık bekleniyor"
    assert len(specs) <= e2e.SETTINGS.max_specifications
    assert state["frozen_menu"].menu_hash


def test_panel_seam_fails_when_no_specification_succeeded(isolated_artifacts) -> None:
    """Tek bir sonuç bile yokken panel gösterecek şey yoktur; boş panel yeşil sayılamaz."""
    state = _analysis_state(isolated_artifacts)
    state["results"] = [
        EstimationResult(spec_id=spec.spec_id, estimator=spec.estimator, status="failed")
        for spec in state["specs"]
    ]

    with pytest.raises(ValueError, match="Hiçbir spesifikasyon başarılı olmadı"):
        e2e._seam_panel(state)


def test_panel_seam_fails_when_a_successful_spec_hides_its_effective_n(
    isolated_artifacts,
) -> None:
    """Efektif N yoksa örneklem etkileşimi şeffaf değil; kart bunu açıkça istiyor."""
    state = _analysis_state(isolated_artifacts)
    state["results"] = [
        EstimationResult(
            spec_id=spec.spec_id,
            estimator=spec.estimator,
            coefficient=-1.0,
            std_error=0.2,
            p_value=0.01,
            n_obs=None,
        )
        for spec in state["specs"]
    ]

    with pytest.raises(ValueError, match="efektif N taşımıyor"):
        e2e._seam_panel(state)


def test_multiverse_seam_clears_a_stale_run_dir_before_launching(
    isolated_artifacts, monkeypatch
) -> None:
    """Sabit run id ile önceki koşunun sonucu kalırsa dikiş başka bir koşuyu doğrular."""
    stale = Path(e2e.SETTINGS.runs_dir) / e2e.RUN_ID / "results.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("[]", encoding="utf-8")

    seen: dict[str, bool] = {}

    def _fake_launch(_df: object, _specs: object, _run_id: str) -> None:
        seen["stale_survived"] = stale.exists()
        raise RuntimeError("launch burada durur; ilgilendiğimiz şey öncesi")

    monkeypatch.setattr(e2e, "launch_multiverse", _fake_launch)

    with pytest.raises(RuntimeError):
        e2e._seam_multiverse({"specs": [], "sample": _committed_panel(), "timeout": 1.0})

    assert seen["stale_survived"] is False
