"""S2-13 uçtan uca entegrasyon doğrulaması (Medicaid hero verisi).

Committed çekirdeğin yedi dikişini sırayla ve gerçek veriyle koşar: yükle,
profille, temizle, estimand, menü, multiverse, varyans paneli. Yeni analiz
mantığı yazmaz; yalnız hazır fonksiyonları birbirine bağlayıp her dikişin
gerçekten çalıştığını doğrular.

JUDGE adımları PydanticAI test modeliyle koşar: sahte ama şemaya uygun çıktı
verilir, böylece doğrulama deterministik kalır ve API anahtarı gerektirmez.
Doğrulanan şey model kalitesi değil, dikişlerin bağlantısıdır; canlı sağlayıcı
çağrıları ayrı bir kartın konusudur.

Kırık dikiş sessizce geçilmez: hata yakalanır, raporda hangi dikişin nasıl
kırıldığı yazılır, sonraki dikişler "bloke" işaretlenir ve süreç sıfırdan
farklı bir çıkış koduyla biter.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Doğrudan çalıştırma her çalışma dizininden kararlı kalsın: bu script paket
# giriş noktası olarak kurulmuyor.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Doğrudan çalıştırma, Pareto importlarından önce repo kökünü sys.path'e ister.
from pydantic_ai.models.test import TestModel  # noqa: E402

from pareto.analysis.event_study import estimate_pretrend_event_study  # noqa: E402
from pareto.analysis.hypothesis import (  # noqa: E402
    SocraticDeclaration,
    draft_tac_proposal,
    freeze_estimand,
    validate_estimand_spec_mapping,
)
from pareto.analysis.menu import (  # noqa: E402
    expand_to_specs,
    freeze_spec_menu,
    generate_spec_menu,
    validate_spec_menu_to_specs,
)
from pareto.analysis.runner import launch_multiverse  # noqa: E402
from pareto.analysis.variance import diagnose_axes, summarize  # noqa: E402
from pareto.cleaning.agent import (  # noqa: E402
    Resolution,
    audit_entries,
    entries_to_apply,
    generate_ledger,
    resolve,
)
from pareto.cleaning.codegen import apply_ledger, verify_reproduction  # noqa: E402
from pareto.cleaning.ledger import persist_ledger  # noqa: E402
from pareto.cleaning.merge import build_panel, load_dataset_config  # noqa: E402
from pareto.config import SETTINGS  # noqa: E402
from pareto.llm.narrative import generate_narrative  # noqa: E402
from pareto.llm.router import use_test_model  # noqa: E402
from pareto.profiling import profile_dataframe  # noqa: E402

DATASET = "medicaid"
RUN_ID = "s2-13-e2e"  # sabit: her koşu aynı çalışma dizinini tazeler, artık birikmez
MULTIVERSE_TIMEOUT_SECONDS = 900
MULTIVERSE_POLL_SECONDS = 0.5
GENERATION_COMMAND = ".venv/bin/python scripts/run_e2e_check.py"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_EXTERNAL = "external"

# CI'nin koştuğu kalite kapısı; bu harness bunları çalıştırmaz, yalnız listeler.
QUALITY_GATE_COMMANDS = (
    "ruff format --check .",
    "ruff check .",
    "mypy pareto",
    "pytest -q",
)


# --------------------------------------------------------------------------- #
# JUDGE stub'ları: şemaya uygun sahte çıktılar (deterministik, API'siz)
# --------------------------------------------------------------------------- #
JUDGE_LEDGER_OUTPUT: dict[str, Any] = {
    "decisions": [
        {
            "bulgu": "İlçe FIPS kodu metin taşınıyor, baştaki sıfırlar kaybolabilir.",
            "transform": {
                "transform_name": "preserve_leading_zeros",
                "col": "county_fips",
                "width": 5,
            },
            "gerekce": "Beş haneli FIPS panelin birim anahtarıdır; zero-pad korunmalıdır.",
            "confidence": "high",
        },
        {
            "bulgu": "Birim-yıl kırılımında yinelenen satır olup olmadığı profilden okunamıyor.",
            "transform": {
                "transform_name": "drop_duplicates",
                "subset": ["county_fips", "year"],
            },
            "gerekce": "Yinelenen birim-yıl satırı panel dengesini bozar; insana sorulmalıdır.",
            "confidence": "low",
        },
    ]
}

JUDGE_ESTIMAND_OUTPUT: dict[str, Any] = {
    "estimand_type": "ATT",
    "treatment": "treated_post",
    "treatment_coding": "treated_post",
    "outcome": "pct_uninsured",
    "outcome_unit": "yüzde puan",
    "population": "2014 genişleme kohortundaki ilçeler ile hiç genişlemeyen ilçeler",
    "time_scope": "2009-2019",
    "expected_sign": "negative",
    "identification_assumption": "parallel_trends",
    "h0": "Medicaid genişlemesinin sigortasızlık oranına etkisi yoktur.",
    "h1": "Medicaid genişlemesi sigortasızlık oranını düşürür.",
    "implied_result_translation": (
        "Genişleme yapan ilçelerde sigortasızlık oranı, hiç genişlemeyen ilçelere kıyasla "
        "genişleme sonrasında daha çok düşmüştür."
    ),
    "confirmation_question": "Tedaviyi 2014 sonrası genişleme göstergesi olarak okumak doğru mu?",
    "needs_clarification": False,
    "clarification_question": None,
}

JUDGE_MENU_OUTPUT: dict[str, Any] = {
    "axes": [
        {
            "axis_name": "control_set",
            "baseline_level": "none",
            "candidate_levels": ["median_hh_income+poverty_rate+unemployment_rate"],
            "rationale": "Gelir, yoksulluk ve işsizlik kontrolleri savunulabilir ikinci seviyedir.",
        },
        {
            "axis_name": "sample",
            "baseline_level": "none",
            "candidate_levels": [],
            "rationale": "Committed baseline zaten kohort ve never-treated ile sınırlıdır.",
        },
        {
            "axis_name": "pre_period",
            "baseline_level": "none",
            "candidate_levels": [],
            "rationale": "Tam pencere korunur; pre-trend kanıtı ayrı bir teşhistir.",
        },
        {
            "axis_name": "clustering",
            "baseline_level": "county_fips",
            "candidate_levels": [],
            "rationale": "Tedavi ilçe düzeyinde tanımlıdır, kümeleme aynı düzeyde kalır.",
        },
        {
            "axis_name": "never_treated",
            "baseline_level": "true",
            "candidate_levels": [],
            "rationale": "Hiç genişlemeyen ilçeler kontrol grubunu oluşturur.",
        },
        {
            "axis_name": "estimator",
            "baseline_level": "OLS",
            "candidate_levels": ["TWFE"],
            "rationale": "Havuzlanmış OLS ile iki yönlü sabit etki karşılaştırması savunulabilir.",
        },
        {
            "axis_name": "weighting",
            "baseline_level": "population",
            "candidate_levels": ["none"],
            "rationale": "Nüfus ağırlığı varsayılandır, ağırlıksız kestirim ikinci seviyedir.",
        },
    ],
    "overall_rationale": (
        "Kontrol seti, kestirici ve ağırlıklandırma eksenleri açık; kalan eksenler "
        "committed baseline gereği baseline seviyesine pinlidir."
    ),
    "needs_clarification": False,
    "clarification_question": None,
}

JUDGE_NARRATIVE_OUTPUT: dict[str, Any] = {
    "ozet": (
        "Spesifikasyonların çoğu aynı işareti paylaşıyor; dönüşler kestirici ekseninde toplanıyor."
    ),
    "eksen_yorumlari": [
        {
            "axis": "estimator",
            "yorum": "Eşleşmiş çiftlerde işaret ve anlamlılık değişimi kestirici ekseninde.",
        }
    ],
}

SOCRATIC_DECLARATION = SocraticDeclaration(
    conceptual_treatment="Eyaletin ACA kapsamında Medicaid'i genişletmesi",
    conceptual_outcome="Çalışma çağındaki nüfusta sigortasızlık oranı",
    expected_sign="negative",
)

RESEARCH_STORY = (
    "ACA kapsamında Medicaid'i 2014'te genişleten eyaletlerdeki ilçelerde, hiç genişletmeyen "
    "eyaletlerdeki ilçelere kıyasla sigortasızlık oranının nasıl değiştiğini ölçmek istiyorum."
)


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _repo_path(path: Path) -> str:
    """Artefakt yolunu raporda repo köküne göre yazar.

    Ayar dizinleri göreli tanımlıdır; mutlak yol geldiğinde repo kökü kırpılır,
    kök dışında kalan bir yol olduğu gibi yazılır.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def committed_baseline_sample(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Committed baseline örneklemi + `treated_post` göstergesi.

    Merge katmanı bu göstergeyi bilinçli olarak türetmez; kohort ile hiç
    genişlemeyen kontrol grubunu seçmek ve tedavi göstergesini kurmak estimand
    aşamasının işidir. Aynı türetme R referans akışında da yapılır; ortak bir
    yardımcıya çıkarmak bu kartın kapsamı dışında bırakıldı.
    """
    treatment_cfg = config.get("treatment") or {}
    baseline_cfg = treatment_cfg.get("committed_baseline") or {}
    cohorts = baseline_cfg.get("cohorts") or []
    if len(cohorts) != 1:
        raise ValueError(f"Committed baseline tek kohort bekler, gelen: {cohorts!r}")
    cohort = int(cohorts[0])

    missing = [col for col in ("treatment_cohort", "never_treated") if col not in df.columns]
    if missing:
        raise ValueError(f"Committed baseline için gerekli kolon(lar) panelde yok: {missing}")

    in_cohort = df["treatment_cohort"].eq(cohort).fillna(False)
    never_treated = df["never_treated"].fillna(False).astype(bool)
    sample = df[in_cohort | never_treated].copy()
    if sample.empty:
        raise ValueError(f"Committed baseline örneklemi boş: {cohort} kohortu ve kontrol yok.")

    sample["treated_post"] = (in_cohort[sample.index] & (sample["year"] >= cohort)).astype(int)
    if int(sample["treated_post"].sum()) == 0:
        raise ValueError(f"Committed baseline örneklemi tedavi edilmiş satır taşımıyor ({cohort}).")
    return sample, cohort


# --------------------------------------------------------------------------- #
# Dikişler: her biri hata fırlatarak kırılır, dönüş değeri rapora girer
# --------------------------------------------------------------------------- #
def _seam_load(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dataset_dir = state["dataset_dir"]
    config = load_dataset_config(dataset_dir)
    panel = build_panel(dataset_dir)
    tier = panel.validate_contract()

    state["config"] = config
    state["panel"] = panel
    return (
        f"{len(panel.df)} satır, {panel.df.shape[1]} kolon, tier {tier.value}",
        {
            "n_rows": int(len(panel.df)),
            "n_cols": int(panel.df.shape[1]),
            "tier": tier.value,
            "n_sources": len(config["sources"]),
            "unit_col": config["panel"]["unit"],
            "time_col": config["panel"]["time"],
        },
    )


def _seam_profile(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = profile_dataframe(state["panel"].df)
    if not profile.get("columns"):
        raise ValueError("Profil kolon üretmedi; temizleme adımı beslenemez.")

    state["profile"] = profile
    return (
        f"{profile['n_cols']} kolon profillendi",
        {
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "duplicate_row_count": profile["duplicate_row_count"],
            "n_potential_join_keys": len(profile["potential_join_keys"]),
        },
    )


def _seam_clean(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    with use_test_model(TestModel(custom_output_args=JUDGE_LEDGER_OUTPUT)):
        entries = generate_ledger(state["profile"])
    if not entries:
        raise ValueError("JUDGE hiç karar üretmedi; temizleme dikişi doğrulanamaz.")

    flagged = [entry for entry in entries if entry.belirsizlik_bayragi]
    if not flagged:
        raise ValueError("Belirsizlik bayrağı hiç kalkmadı; gatekeeper yolu doğrulanamıyor.")

    # Gatekeeper sözleşmesi: bayraklı karar, insan çözümü olmadan otomatik onaylanamaz.
    for entry in flagged:
        try:
            resolve(entry, auto_approve=True)
        except ValueError:
            continue
        raise ValueError("Gatekeeper kırık: bayraklı karar resolution olmadan otomatik onaylandı.")

    resolutions = {
        i: resolve(entry, auto_approve=False, resolution=Resolution.APPROVED)
        if entry.belirsizlik_bayragi
        else resolve(entry, auto_approve=True)
        for i, entry in enumerate(entries)
    }

    raw_df = state["panel"].df
    to_apply = entries_to_apply(entries, resolutions)
    cleaned_df, audit_path = apply_ledger(raw_df, to_apply, RUN_ID)
    repro_dir = verify_reproduction(raw_df, audit_path, cleaned_df, RUN_ID)
    ledger_path = persist_ledger(audit_entries(entries, resolutions), RUN_ID)

    state["clean_df"] = cleaned_df
    return (
        f"{len(entries)} karar ({len(flagged)} tanesi gatekeeper'dan geçti), "
        "L4 reprodüksiyon doğrulandı",
        {
            "n_decisions": len(entries),
            "n_gated": len(flagged),
            "n_applied": len(to_apply),
            "audit_script": _repo_path(audit_path),
            "repro_dir": _repo_path(repro_dir),
            "ledger": _repo_path(ledger_path),
            "n_rows_after_clean": int(len(cleaned_df)),
        },
    )


def _seam_estimand(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sample, cohort = committed_baseline_sample(state["clean_df"], state["config"])
    columns = [str(col) for col in sample.columns]

    with use_test_model(TestModel(custom_output_args=JUDGE_ESTIMAND_OUTPUT)):
        proposal = draft_tac_proposal(
            research_story=RESEARCH_STORY,
            available_columns=columns,
            declaration=SOCRATIC_DECLARATION,
        )
    frozen_estimand = freeze_estimand(proposal, approved=True)

    state["sample"] = sample
    state["columns"] = columns
    state["cohort"] = cohort
    state["frozen_estimand"] = frozen_estimand
    return (
        f"estimand donduruldu (hash {frozen_estimand.freeze_hash})",
        {
            "committed_cohort": cohort,
            "n_rows": int(len(sample)),
            "n_treated_post_rows": int(sample["treated_post"].sum()),
            "freeze_hash": frozen_estimand.freeze_hash,
            "outcome": frozen_estimand.estimand.outcome,
            "treatment_coding": frozen_estimand.estimand.treatment_coding,
        },
    )


def _seam_menu(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    frozen_estimand = state["frozen_estimand"]
    columns = state["columns"]
    panel_cfg = state["config"]["panel"]

    with use_test_model(TestModel(custom_output_args=JUDGE_MENU_OUTPUT)):
        proposal = generate_spec_menu(frozen=frozen_estimand, available_columns=columns)

    frozen_menu = freeze_spec_menu(proposal, available_columns=columns, approved=True)
    specs = expand_to_specs(
        frozen_menu,
        outcome=frozen_estimand.estimand.outcome,
        treatment=frozen_estimand.estimand.treatment_coding,
        unit_col=str(panel_cfg["unit"]),
        time_col=str(panel_cfg["time"]),
    )
    validate_spec_menu_to_specs(frozen_menu, specs)

    warnings: list[str] = []
    for spec in specs:
        warnings.extend(
            validate_estimand_spec_mapping(frozen_estimand, spec, available_columns=columns)
        )

    state["frozen_menu"] = frozen_menu
    state["specs"] = specs
    return (
        f"menü donduruldu (hash {frozen_menu.menu_hash}), {len(specs)} spesifikasyon açıldı",
        {
            "menu_hash": frozen_menu.menu_hash,
            "n_specs": len(specs),
            "hard_cap": SETTINGS.max_specifications,
            # Menüde birden çok seviyesi olan eksenler faktöriyel açılıma girer;
            # hangi eksenin gerçekten oynadığı panel adımındaki partial-R² tablosunda görünür.
            "n_control_sets": len(frozen_menu.menu.control_sets),
            "n_estimator_levels": len(frozen_menu.menu.estimators),
            "n_weighting_levels": len(frozen_menu.menu.weighting_levels),
            "estimators": sorted({spec.estimator for spec in specs}),
            "mapping_warnings": sorted(set(warnings)),
        },
    )


def _seam_multiverse(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    specs = state["specs"]
    timeout = float(state["timeout"])

    handle = launch_multiverse(state["sample"], specs, RUN_ID)
    deadline = time.monotonic() + timeout
    while not handle.is_done():
        if time.monotonic() > deadline:
            handle.process.kill()
            raise TimeoutError(f"Multiverse worker {timeout:.0f}s içinde bitmedi.")
        time.sleep(MULTIVERSE_POLL_SECONDS)

    if handle.process.returncode != 0:
        stderr = handle.read_stderr().strip()
        raise RuntimeError(f"Multiverse worker exit={handle.process.returncode}.\n{stderr}")

    progress = handle.read_progress()
    if int(progress.get("done", 0)) != len(specs) or int(progress.get("total", 0)) != len(specs):
        raise ValueError(f"Diske yazılan ilerleme spec sayısıyla uyuşmuyor: {progress}")

    results = handle.read_results()
    if len(results) != len(specs):
        raise ValueError(f"{len(specs)} spec koşuldu ama {len(results)} sonuç yazıldı.")

    # Panel varsayılan olarak aynanın sonucunu okur; ayna kurulmazsa dikiş kopar.
    mirror = Path(SETTINGS.runs_dir) / "latest" / "results.json"
    if not mirror.exists():
        raise FileNotFoundError(f"Runner aynası yazılmadı: {mirror}")

    failed = [result for result in results if result.status != "ok"]
    state["results"] = results
    return (
        f"{len(results)} sonuç yazıldı ({len(failed)} başarısız spec)",
        {
            "run_dir": _repo_path(handle.run_dir),
            "n_results": len(results),
            "n_failed_specs": len(failed),
            "failed_specs": [
                {"spec_id": result.spec_id, "error": result.error} for result in failed
            ],
            "progress": progress,
            "mirror": _repo_path(mirror),
        },
    )


def _seam_panel(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = state["results"]
    specs = state["specs"]
    sample = state["sample"]
    panel_cfg = state["config"]["panel"]

    summary = dict(summarize(results))
    if not summary["n_ok"]:
        raise ValueError("Hiçbir spesifikasyon başarılı olmadı; panel gösterecek sonuç yok.")

    diagnosis = diagnose_axes(results, specs)
    with use_test_model(TestModel(custom_output_args=JUDGE_NARRATIVE_OUTPUT)):
        narrative = generate_narrative(summary, diagnosis)

    effective_n = {result.spec_id: result.n_obs for result in results if result.status == "ok"}
    if any(value is None for value in effective_n.values()):
        raise ValueError("Başarılı bir spesifikasyon efektif N taşımıyor; örneklem şeffaf değil.")

    event_study = estimate_pretrend_event_study(
        sample,
        outcome_col=state["frozen_estimand"].estimand.outcome,
        unit_col=str(panel_cfg["unit"]),
        time_col=str(panel_cfg["time"]),
        cohort_col="treatment_cohort",
        never_treated_col="never_treated",
        weight_col=str(panel_cfg["weight"]),
        treated_cohorts=(state["cohort"],),
    )
    if event_study["status"] != "ok":
        raise ValueError(f"Pre-trend event-study koşamadı: {event_study['error']}")

    return (
        f"bant {summary['band']}, {len(narrative.eksen_yorumlari)} eksen yorumu, "
        f"{len(event_study['series'])} pre-trend katsayısı",
        {
            "summary": summary,
            "n_used_in_diagnosis": diagnosis["n_used"],
            "anova_partial_r2": diagnosis["anova_partial_r2"],
            "diagnosis_warnings": diagnosis["warnings"],
            "narrative_axes": [comment.axis for comment in narrative.eksen_yorumlari],
            "effective_n": effective_n,
            "event_study_n_obs": event_study["n_obs"],
            "event_study_points": len(event_study["series"]),
        },
    )


SeamFn = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]

SEAMS: tuple[tuple[str, str, SeamFn], ...] = (
    ("load", "Yükle: config güdümlü panel merge", _seam_load),
    ("profile", "Profille: deterministik kolon profili", _seam_profile),
    ("clean", "Temizle: JUDGE karar defteri, gatekeeper, codegen", _seam_clean),
    ("estimand", "Estimand: TAC önerisi ve dondurma", _seam_estimand),
    ("menu", "Menü: JUDGE spec menüsü, dondurma, faktöriyel açılım", _seam_menu),
    ("multiverse", "Multiverse: subprocess runner ve disk çıktıları", _seam_multiverse),
    ("panel", "Varyans paneli: özet, eksen atfı, narrative, efektif N, pre-trend", _seam_panel),
)


# --------------------------------------------------------------------------- #
# Koşu ve rapor
# --------------------------------------------------------------------------- #
def run_seams(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Dikişleri sırayla koşar. İlk kırılmadan sonrakiler bloke işaretlenir."""
    reports: list[dict[str, Any]] = []
    broken = False
    for key, title, seam in SEAMS:
        if broken:
            reports.append(
                {
                    "key": key,
                    "title": title,
                    "status": STATUS_BLOCKED,
                    "detail": "önceki dikiş kırıldığı için çalıştırılmadı",
                    "metrics": {},
                }
            )
            continue
        try:
            detail, metrics = seam(state)
        except Exception as exc:
            broken = True
            reports.append(
                {
                    "key": key,
                    "title": title,
                    "status": STATUS_FAILED,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                }
            )
            continue
        reports.append(
            {
                "key": key,
                "title": title,
                "status": STATUS_OK,
                "detail": detail,
                "metrics": _jsonable(metrics),
            }
        )
    return reports


def build_checklist(seam_reports: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Definition of Success kontrol listesi: dikişler + LLM testlenebilirliği + CI kapısı."""
    by_key = {report["key"]: report for report in seam_reports}
    items = [
        {"item": report["title"], "status": report["status"], "note": report["detail"]}
        for report in seam_reports
    ]

    llm_keys = ("clean", "estimand", "menu", "panel")
    llm_statuses = [by_key[key]["status"] for key in llm_keys]
    if all(status == STATUS_OK for status in llm_statuses):
        llm_status, llm_note = STATUS_OK, "dört JUDGE adımı da test modeliyle koştu"
    elif STATUS_FAILED in llm_statuses:
        llm_status, llm_note = STATUS_FAILED, "en az bir JUDGE adımı kırıldı"
    else:
        llm_status, llm_note = STATUS_BLOCKED, "JUDGE adımları önceki kırılma yüzünden koşmadı"
    items.append(
        {
            "item": "Her LLM adımı test modeliyle koşulabilir",
            "status": llm_status,
            "note": llm_note,
        }
    )

    items.append(
        {
            "item": "CI kalite kapısı",
            "status": STATUS_EXTERNAL,
            "note": "bu harness çalıştırmaz, CI doğrular: " + ", ".join(QUALITY_GATE_COMMANDS),
        }
    )
    return items


def _source_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607  # provenance; PATH'teki git yeter
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _platform_name() -> str:
    system = platform.system()
    return "macOS" if system == "Darwin" else system or "unknown"


def _report_provenance(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    # Zaman damgası yok: kod, veri ve commit değişmediğinde rapor tekrar üretilebilir kalsın.
    return {
        "generation_command": GENERATION_COMMAND,
        "source_commit": _source_commit(repo_root=repo_root),
        "python": platform.python_version(),
        "platform": _platform_name(),
        "judge_mode": "PydanticAI test modeli (sahte tipli çıktı, canlı sağlayıcı çağrısı yok)",
    }


def run_e2e_check(
    *,
    repo_root: Path = REPO_ROOT,
    timeout: float = MULTIVERSE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "dataset_dir": repo_root / "data" / DATASET,
        "timeout": timeout,
    }
    seam_reports = run_seams(state)
    checklist = build_checklist(seam_reports)
    passed = all(report["status"] == STATUS_OK for report in seam_reports)

    return {
        "check": "S2-13 uçtan uca entegrasyon doğrulaması",
        "dataset": DATASET,
        "run_id": RUN_ID,
        "provenance": _report_provenance(repo_root=repo_root),
        "overall": {
            "status": STATUS_OK if passed else STATUS_FAILED,
            "reason": (
                "yedi dikiş de koştu ve beklenen çıktıyı üretti"
                if passed
                else "en az bir dikiş kırıldı; ayrıntı dikiş tablosunda"
            ),
        },
        "seams": seam_reports,
        "checklist": checklist,
        "limitations": [
            "JUDGE adımları test modeliyle koşar, canlı sağlayıcı davranışı ölçülmez",
            "gatekeeper kararları programatik onaylanır, arayüz etkileşimi kapsam dışı",
            "Streamlit sayfaları render edilmez, doğrulanan şey sayfaların çağırdığı zincir",
            "ham CDC dosyası repoda tutulmadığından bu koşu yerel veri gerektirir",
        ],
    }


_STATUS_LABELS = {
    STATUS_OK: "GEÇTİ",
    STATUS_FAILED: "KIRIK",
    STATUS_BLOCKED: "BLOKE",
    STATUS_EXTERNAL: "DIŞ KONTROL",
}


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    return [f"  - {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metrics.items()]


def render_markdown(report: dict[str, Any]) -> str:
    provenance = report["provenance"]
    overall = report["overall"]
    lines = [
        "# S2-13 Uçtan Uca Entegrasyon Doğrulaması",
        "",
        "## Provenance",
        "",
        "Şu komutla üretildi:",
        "",
        "```bash",
        provenance["generation_command"],
        "```",
        "",
        f"- Veri seti: {report['dataset']}",
        f"- Run: {report['run_id']}",
        f"- Kaynak commit: {provenance['source_commit']}",
        f"- Python: {provenance['python']}",
        f"- Platform: {provenance['platform']}",
        f"- JUDGE modu: {provenance['judge_mode']}",
        "",
        "## Genel Sonuç",
        "",
        f"**{_status_label(overall['status'])}**: {overall['reason']}",
        "",
        "## Definition of Success Kontrol Listesi",
        "",
        "| Madde | Durum | Not |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {item['item']} | {_status_label(item['status'])} | {item['note']} |"
        for item in report["checklist"]
    )

    lines.extend(["", "## Dikişler", ""])
    for seam in report["seams"]:
        lines.append(f"### {seam['title']}")
        lines.append("")
        lines.append(f"- Durum: **{_status_label(seam['status'])}**")
        lines.append(f"- Sonuç: {seam['detail']}")
        if seam["metrics"]:
            lines.append("- Ölçümler:")
            lines.extend(_format_metrics(seam["metrics"]))
        lines.append("")

    lines.extend(["## Kapsam Sınırları", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        out_path.write_text(render_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto S2-13 uçtan uca entegrasyon doğrulaması")
    parser.add_argument(
        "--timeout",
        type=float,
        default=MULTIVERSE_TIMEOUT_SECONDS,
        help="Multiverse worker için saniye cinsinden üst sınır",
    )
    parser.add_argument("--out", help="Varsayılan Markdown; uzantı .json ise JSON yazar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_e2e_check(timeout=args.timeout)
    if args.out:
        write_report(report, Path(args.out))
    else:
        print(render_markdown(report), end="")
    return 0 if report["overall"]["status"] == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())
