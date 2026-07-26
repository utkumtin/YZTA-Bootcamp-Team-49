"""S3-08 CI smoke matrisi — çekirdeğin veri-agnostik kaldığının uçtan uca kanıtı.

SCOPE §11 "genelleme smoke": jüri kendi verisini yükleyebilir, bu yüzden çekirdek tek
bir dataset'e göre şekillenmemeli. Bu harness aynı zinciri (yükle → temizle → estimand →
menü → multiverse → özet) yapısal olarak farklı üç veri setinde koşar:

- card_krueger: iki dönemli 2×2, ağırlıksız, hazır tedavi göstergesi
- divorce: kademeli yasalaşan eyalet-yıl paneli, ağırlıklı, hazır tedavi göstergesi
- medicaid: çok kaynaklı ilçe-yıl merge'i, tedavi göstergesi kohort kolonlarından türetilir

JUDGE adımları PydanticAI test modeliyle koşar: şemaya uygun sahte çıktı verilir, böylece
matris deterministik kalır ve API anahtarı gerektirmez. Ölçülen şey model kalitesi değil,
çekirdeğin farklı veri şekillerini taşıyabilmesidir.

Ham verisi repoda olmayan dataset (medicaid'in CDC export'u gitignore'lu) `local_only`
işaretlidir: CI'da nedeni raporlanarak atlanır, lokalde koşar. Sessiz atlama yoktur:
işaretsiz bir dataset'in verisi eksikse koşu kırık sayılır, ve hiçbir dataset koşmadıysa
matris atlananların hepsi işaretli olsa bile kırık sayılır. Her iki durumda da çıkış kodu
sıfırdan farklı olur, çünkü boş bir matris genelleme hakkında hiçbir şey kanıtlamaz.

İki yan etki bilinçli: (1) her multiverse koşusu `runs/latest` aynasını günceller, yani
matrisin ardından varyans paneli son dataset'in smoke sonucunu gösterir; (2) rapor duvar
saati süreleri taşır, bu yüzden bir LOG'dur, uçtan uca doğrulama raporunun aksine
byte-tekrarlanabilir bir artefakt değildir ve commit edilmesi beklenmez.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
from pareto.cleaning.agent import entries_to_apply, generate_ledger, resolve  # noqa: E402
from pareto.cleaning.codegen import apply_ledger  # noqa: E402
from pareto.cleaning.ledger import LedgerEntry  # noqa: E402
from pareto.cleaning.merge import (  # noqa: E402
    build_panel,
    load_dataset_config,
    resolve_source_paths,
)
from pareto.config import SETTINGS  # noqa: E402
from pareto.contracts import Tier  # noqa: E402
from pareto.llm.router import use_test_model  # noqa: E402
from pareto.profiling import profile_dataframe  # noqa: E402
from scripts._report import jsonable, platform_name, source_commit, write_report  # noqa: E402

GENERATION_COMMAND = "uv run python scripts/run_smoke_matrix.py"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"

TREATMENT_FROM_PANEL = "panelde hazır"
TREATMENT_FROM_COHORT = "kohort kolonlarından türetildi"

# Uyarı eşiği, zorlanan bir sınır DEĞİL: aşım koşuyu kırmaz, yalnız raporda görünür.
# Amacı bir performans regresyonunu görünür kılmak ve ağır setleri nightly'ye ayırma
# kararını beslemek; bir dataset yavaşladı diye yeşil bir zinciri kırmak istemiyoruz.
TIME_BUDGET_WARN_SECONDS = 300.0

# Dataset başına multiverse üst sınırı. CI job'ının timeout'undan (15 dk) küçük kalmalı:
# üç dataset × 240 sn = 720 sn, yani takılan bir koşuda harness kendi TimeoutError'ını
# raporlamaya yetişir. Aksi halde job'ı GitHub nedensiz keser ve rapor hiç basılmaz.
MULTIVERSE_TIMEOUT_SECONDS = 240
MULTIVERSE_POLL_SECONDS = 0.5

# Koşunun okuyucuya söylemesi gereken yan etkiler; raporun her formatında görünür.
RUN_NOTES: tuple[str, ...] = (
    "Her multiverse koşusu runs/latest aynasını günceller: matristen sonra varyans "
    "paneli son dataset'in smoke sonucunu gösterir.",
    "Bu rapor süre alanları taşır, yani bir log'dur; uçtan uca doğrulama raporunun "
    "aksine tekrar-üretilebilir artefakt değildir ve commit edilmesi beklenmez.",
)


# --------------------------------------------------------------------------- #
# Dataset profilleri: JUDGE stub'ları + tedavi göstergesi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetProfile:
    """Bir dataset'i matriste koşturmak için gereken her şey.

    `treatment_col` panelde varsa doğrudan kullanılır (hazır tedavi dummy'si olan
    setler); yoksa merge'ün türettiği kohort kolonlarından üretilir. `expected_tier`
    descriptor'ın tasarım kararını koşulabilir sözleşmeye çevirir: tier sessizce
    değişirse yükleme adımı kırılır. `local_only`, ham verisi repoda olmayan setleri
    işaretler.
    """

    key: str
    research_story: str
    declaration: SocraticDeclaration
    treatment_col: str
    expected_tier: Tier
    judge_ledger: dict[str, Any]
    judge_estimand: dict[str, Any]
    judge_menu: dict[str, Any]
    local_only: bool = False
    missing_data_note: str = ""


def _menu_axes(
    *,
    control_candidates: list[str],
    clustering: str,
    weighting: str,
    weighting_candidates: list[str],
    rationale: str,
) -> dict[str, Any]:
    """Yedi eksenli JUDGE menü çıktısı; oynayan eksenler kontrol seti + kestirici.

    Matrisin işi menü kalitesini ölçmek değil, faktöriyel açılımın her veri şeklinde
    aynı biçimde çalıştığını görmek: bu yüzden eksen deseni datasetler arasında sabit,
    yalnız kolon adları değişir.
    """
    return {
        "axes": [
            {
                "axis_name": "control_set",
                "baseline_level": "none",
                "candidate_levels": control_candidates,
                "rationale": "Kontrolsüz baseline ile deklare edilen kontrol seti karşılaştırılır.",
            },
            {
                "axis_name": "sample",
                "baseline_level": "none",
                "candidate_levels": [],
                "rationale": "Smoke koşusu tam örneklemde kalır.",
            },
            {
                "axis_name": "pre_period",
                "baseline_level": "none",
                "candidate_levels": [],
                "rationale": "Pencere kısaltması bu koşunun konusu değil.",
            },
            {
                "axis_name": "clustering",
                "baseline_level": clustering,
                "candidate_levels": [],
                "rationale": "Kümeleme tedavinin atandığı düzeyde sabitlenir.",
            },
            {
                "axis_name": "never_treated",
                "baseline_level": "true",
                "candidate_levels": [],
                "rationale": "Kontrol grubu örneklemde kalır.",
            },
            {
                "axis_name": "estimator",
                "baseline_level": "OLS",
                "candidate_levels": ["TWFE"],
                "rationale": "Havuzlanmış OLS ile iki yönlü sabit etki karşılaştırılır.",
            },
            {
                "axis_name": "weighting",
                "baseline_level": weighting,
                "candidate_levels": weighting_candidates,
                "rationale": "Ağırlıklandırma seçimi veri setinin taşıdığı ağırlık kolonuna bağlı.",
            },
        ],
        "overall_rationale": rationale,
        "needs_clarification": False,
        "clarification_question": None,
    }


CARD_KRUEGER = DatasetProfile(
    key="card_krueger",
    research_story=(
        "New Jersey'nin Nisan 1992'de asgari ücreti artırmasının fast-food restoranlarındaki "
        "tam zamanlı eşdeğer istihdamı, Pennsylvania restoranlarına kıyasla nasıl "
        "değiştirdiğini ölçmek istiyorum."
    ),
    declaration=SocraticDeclaration(
        conceptual_treatment="New Jersey'de asgari ücretin artırılması",
        conceptual_outcome="Restoran başına tam zamanlı eşdeğer istihdam",
        expected_sign="negative",
    ),
    treatment_col="treated_post",
    # Kohort yılı yok ama gösterge descriptor'da bildirildiği için panel Tier1 raporlar.
    expected_tier=Tier.TIER1_PANEL_DID,
    judge_ledger={
        "decisions": [
            {
                "bulgu": "Tedavi göstergesi metin olarak taşınıyor.",
                "transform": {"transform_name": "coerce_numeric", "col": "treated_post"},
                "gerekce": "Gösterge 0/1 kodlu; sayısala çevrilmeden regresyona giremez.",
                "confidence": "high",
            }
        ]
    },
    judge_estimand={
        "estimand_type": "ATT",
        "treatment": "treated_post",
        "treatment_coding": "treated_post",
        "outcome": "fte_employment",
        "outcome_unit": "tam zamanlı eşdeğer çalışan",
        "population": "New Jersey ve Pennsylvania'daki fast-food restoranları",
        "time_scope": "1992 iki görüşme dalgası",
        "expected_sign": "negative",
        "identification_assumption": "parallel_trends",
        "h0": "Asgari ücret artışının istihdama etkisi yoktur.",
        "h1": "Asgari ücret artışı istihdamı düşürür.",
        "implied_result_translation": (
            "New Jersey restoranlarında istihdam, Pennsylvania restoranlarına kıyasla "
            "artış sonrasında farklı bir seyir izlemiştir."
        ),
        "confirmation_question": (
            "Tedaviyi ikinci dalgadaki New Jersey mağazaları olarak okumak doğru mu?"
        ),
        "needs_clarification": False,
        "clarification_question": None,
    },
    judge_menu=_menu_axes(
        control_candidates=["chain+co_owned"],
        clustering="store_id",
        weighting="none",
        weighting_candidates=[],
        rationale="Zincir ve mülkiyet kontrolleri açık; ağırlık kolonu olmadığı için tek seviye.",
    ),
)

DIVORCE = DatasetProfile(
    key="divorce",
    research_story=(
        "Tek taraflı boşanma yasalarını yürürlüğe koyan eyaletlerde kadın intihar oranının, "
        "yasayı henüz koymamış eyaletlere kıyasla nasıl değiştiğini ölçmek istiyorum."
    ),
    declaration=SocraticDeclaration(
        conceptual_treatment="Eyalette tek taraflı boşanma yasasının yürürlüğe girmesi",
        conceptual_outcome="Kadınlarda yaş düzeltilmiş intihar oranı",
        expected_sign="negative",
    ),
    treatment_col="post",
    expected_tier=Tier.TIER1_PANEL_DID,
    judge_ledger={
        "decisions": [
            {
                "bulgu": "Tedavi göstergesi metin olarak taşınıyor.",
                "transform": {"transform_name": "coerce_numeric", "col": "post"},
                "gerekce": "Gösterge 0/1 kodlu; sayısala çevrilmeden regresyona giremez.",
                "confidence": "high",
            }
        ]
    },
    judge_estimand={
        "estimand_type": "ATT",
        "treatment": "post",
        "treatment_coding": "post",
        "outcome": "suicide_rate_f",
        "outcome_unit": "milyon kadın başına intihar",
        "population": "ABD eyaletleri, 1964-1996",
        "time_scope": "1964-1996",
        "expected_sign": "negative",
        "identification_assumption": "parallel_trends",
        "h0": "Tek taraflı boşanma yasasının kadın intihar oranına etkisi yoktur.",
        "h1": "Tek taraflı boşanma yasası kadın intihar oranını düşürür.",
        "implied_result_translation": (
            "Yasayı yürürlüğe koyan eyaletlerde kadın intihar oranı, yasayı henüz koymamış "
            "eyaletlere kıyasla farklı bir seyir izlemiştir."
        ),
        "confirmation_question": (
            "Tedaviyi yasanın yürürlüğe girdiği yıl ve sonrası olarak okumak doğru mu?"
        ),
        "needs_clarification": False,
        "clarification_question": None,
    },
    judge_menu=_menu_axes(
        control_candidates=["per_capita_income+homicide_rate"],
        clustering="state_fips",
        weighting="weight",
        weighting_candidates=["none"],
        rationale="Gelir ve cinayet oranı kontrolleri açık; ağırlıklı baseline, ağırlıksız aday.",
    ),
)

MEDICAID = DatasetProfile(
    key="medicaid",
    research_story=(
        "ACA kapsamında Medicaid'i genişleten eyaletlerdeki ilçelerde, hiç genişletmeyen "
        "eyaletlerdeki ilçelere kıyasla sigortasızlık oranının nasıl değiştiğini ölçmek istiyorum."
    ),
    declaration=SocraticDeclaration(
        conceptual_treatment="Eyaletin ACA kapsamında Medicaid'i genişletmesi",
        conceptual_outcome="Çalışma çağındaki nüfusta sigortasızlık oranı",
        expected_sign="negative",
    ),
    treatment_col="treated_post",
    expected_tier=Tier.TIER1_PANEL_DID,
    judge_ledger={
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
            }
        ]
    },
    judge_estimand={
        "estimand_type": "ATT",
        "treatment": "treated_post",
        "treatment_coding": "treated_post",
        "outcome": "pct_uninsured",
        "outcome_unit": "yüzde puan",
        "population": "Genişleme yapan ve hiç genişletmeyen eyaletlerdeki ilçeler",
        "time_scope": "2009-2019",
        "expected_sign": "negative",
        "identification_assumption": "parallel_trends",
        "h0": "Medicaid genişlemesinin sigortasızlık oranına etkisi yoktur.",
        "h1": "Medicaid genişlemesi sigortasızlık oranını düşürür.",
        "implied_result_translation": (
            "Genişleme yapan ilçelerde sigortasızlık oranı, hiç genişletmeyen ilçelere kıyasla "
            "genişleme sonrasında daha çok düşmüştür."
        ),
        "confirmation_question": "Tedaviyi genişleme yılı ve sonrası olarak okumak doğru mu?",
        "needs_clarification": False,
        "clarification_question": None,
    },
    judge_menu=_menu_axes(
        control_candidates=["median_hh_income+poverty_rate"],
        clustering="county_fips",
        weighting="population",
        weighting_candidates=["none"],
        rationale="Gelir ve yoksulluk kontrolleri açık; nüfus ağırlığı baseline, ağırlıksız aday.",
    ),
    local_only=True,
    missing_data_note=(
        "CDC WONDER export'u gitignore'lu (data/**/raw/*.tsv), temiz bir CI checkout'unda "
        "bulunmaz; bu set lokalde koşulur"
    ),
)

PROFILES: tuple[DatasetProfile, ...] = (CARD_KRUEGER, DIVORCE, MEDICAID)
PROFILE_BY_KEY = {profile.key: profile for profile in PROFILES}


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def missing_source_files(config: dict[str, Any], dataset_dir: Path) -> list[str]:
    """Config'in deklare ettiği ama diskte olmayan kaynak dosyalar.

    Yol çözümü okuma yolunun kullandığı fonksiyonun aynısıdır: iki kural ayrışırsa
    "veri var" denip okuma aşamasında kırılan ya da tersi bir sessiz sapma doğardı.
    Kontrol koşudan önce yapılır, böylece "veri yok" durumu akışın ortasında çıkan
    bir merge hatasından ayrılabilir.
    """
    missing: list[str] = []
    for src in config["sources"].values():
        pattern = str(src["file"])
        paths = resolve_source_paths(pattern, dataset_dir)
        if not paths or any(not path.exists() for path in paths):
            missing.append(pattern)
    return missing


def _require_both_arms(values: pd.Series, col: str, origin: str) -> None:
    """Gösterge hem tedavi hem kontrol satırı taşımalı; tek kollu bir gösterge sessizdir.

    Tek seviyeye çökmüş bir dummy ile zincirin altı adımı da geçer: menü açılır,
    spesifikasyonlar koşar, özet yazılır. Ama karşılaştırılacak iki grup olmadığı için
    o yeşil rapor hiçbir şey ölçmemiş olur. Kontrol iki yolda da aynıdır: göstergenin
    panelden gelmesi doğru olduğu anlamına gelmez.
    """
    levels = set(pd.to_numeric(values, errors="raise").dropna().unique())
    if levels != {0, 1}:
        raise ValueError(
            f"'{col}' göstergesi ({origin}) 0/1 ikilisini taşımıyor, bulunan: {sorted(levels)}. "
            "Tedavi ile kontrol ayrışmadan multiverse hiçbir karşılaştırma ölçmez."
        )


def resolve_treatment(df: pd.DataFrame, profile: DatasetProfile, time_col: str) -> tuple[str, str]:
    """Tedavi göstergesini panelde bulur ya da kohort kolonlarından türetir.

    Hazır dummy taşıyan setlerde (2×2 fixture, kademeli panel) kolon olduğu gibi
    kullanılır. Taşımayanlarda gösterge merge'ün ürettiği `treatment_cohort` +
    `never_treated` kolonlarından üretilir: birim kohortluysa ve zaman kohorta
    ulaştıysa tedavili. Kohortsuz ama never-treated de olmayan satır belirsizdir,
    sessizce kontrol sayılmaz. Hangi yoldan gelirse gelsin gösterge iki kollu olmalıdır.
    """
    col = profile.treatment_col
    if col in df.columns:
        _require_both_arms(df[col], col, TREATMENT_FROM_PANEL)
        return col, TREATMENT_FROM_PANEL

    required = ["treatment_cohort", "never_treated"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            f"'{col}' panelde yok ve kohorttan türetilemiyor; eksik kolon(lar): {missing}"
        )

    cohort = pd.to_numeric(df["treatment_cohort"], errors="raise")
    never_treated = df["never_treated"].astype(bool)
    ambiguous = int((cohort.isna() & ~never_treated).sum())
    if ambiguous:
        raise ValueError(
            f"{ambiguous} satır ne kohort taşıyor ne never-treated; tedavi göstergesi türetilemez."
        )

    time = pd.to_numeric(df[time_col], errors="raise")
    df[col] = (cohort.notna() & (time >= cohort)).astype(int)
    _require_both_arms(df[col], col, TREATMENT_FROM_COHORT)
    return col, TREATMENT_FROM_COHORT


def _flagged_detail(entry: LedgerEntry, column_profile: dict[str, Any]) -> str:
    """Gatekeeper'a düşen kararı, tetikleyen kolonun eksik oranıyla birlikte anlatır."""
    columns = column_profile.get("columns", {})
    referenced: list[str] = []
    for value in entry.params.values():
        candidates = value if isinstance(value, list) else [value]
        referenced.extend(str(c) for c in candidates if isinstance(c, str) and c in columns)

    rates = ", ".join(
        f"{col} %{float(columns[col].get('pct_missing', 0.0)) * 100:.0f} eksik"
        for col in referenced
    )
    return f"{entry.transform_name} ({rates or 'kolon profilde bulunamadı'})"


def _repo_path(path: Path) -> str:
    """Artefakt yolunu raporda repo köküne göre yazar."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Adımlar: her biri hata fırlatarak kırılır, dönüş değeri rapora girer
# --------------------------------------------------------------------------- #
def _step_load(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile: DatasetProfile = state["profile"]
    dataset_dir = state["dataset_dir"]
    config = load_dataset_config(dataset_dir)
    panel = build_panel(dataset_dir)
    tier = panel.validate_contract()
    if tier is not profile.expected_tier:
        # Tier tasarım kararıdır: sessiz bir değişim, aynı raporun altında başka bir
        # analiz sınıfını doğrulamak demek olur.
        raise ValueError(
            f"Panel tier'ı beklenenden farklı: {tier.value} "
            f"(beklenen {profile.expected_tier.value})"
        )

    state["config"] = config
    state["df"] = panel.df
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


def _step_clean(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile: DatasetProfile = state["profile"]
    raw_df = state["df"]

    column_profile = profile_dataframe(raw_df)
    with use_test_model(TestModel(custom_output_args=profile.judge_ledger)):
        entries = generate_ledger(column_profile)
    if not entries:
        raise ValueError("JUDGE hiç karar üretmedi; temizleme adımı doğrulanamaz.")

    # Stub kararları bilinçli olarak bayraksız seçildi; matris insan onayı bekleyemez.
    # Bayrak kalkmışsa sebep verinin kendisidir (kolonun eksik oranı gatekeeper eşiğini
    # aşmıştır) — gatekeeper'ın genel "otomatik onaylanamaz" mesajı bunu göstermez.
    flagged = [entry for entry in entries if entry.belirsizlik_bayragi]
    if flagged:
        detail = "; ".join(_flagged_detail(entry, column_profile) for entry in flagged)
        raise ValueError(f"Stub kararı gatekeeper'a düştü: {detail}")

    resolutions = {i: resolve(entry, auto_approve=True) for i, entry in enumerate(entries)}
    to_apply = entries_to_apply(entries, resolutions)
    cleaned_df, audit_path = apply_ledger(raw_df, to_apply, state["run_id"])

    treatment_col, treatment_source = resolve_treatment(
        cleaned_df, profile, str(state["config"]["panel"]["time"])
    )

    state["df"] = cleaned_df
    state["treatment_col"] = treatment_col
    return (
        f"{len(entries)} karar uygulandı, tedavi göstergesi {treatment_source}",
        {
            "n_decisions": len(entries),
            "n_applied": len(to_apply),
            "n_rows_after_clean": int(len(cleaned_df)),
            "audit_script": _repo_path(audit_path),
            "treatment_col": treatment_col,
            "treatment_source": treatment_source,
            "n_treated_rows": int(pd.to_numeric(cleaned_df[treatment_col], errors="raise").sum()),
        },
    )


def _step_estimand(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile: DatasetProfile = state["profile"]
    columns = [str(col) for col in state["df"].columns]

    with use_test_model(TestModel(custom_output_args=profile.judge_estimand)):
        proposal = draft_tac_proposal(
            research_story=profile.research_story,
            available_columns=columns,
            declaration=profile.declaration,
        )
    frozen_estimand = freeze_estimand(proposal, approved=True)

    state["columns"] = columns
    state["frozen_estimand"] = frozen_estimand
    return (
        f"estimand donduruldu (hash {frozen_estimand.freeze_hash})",
        {
            "freeze_hash": frozen_estimand.freeze_hash,
            "outcome": frozen_estimand.estimand.outcome,
            "treatment_coding": frozen_estimand.estimand.treatment_coding,
        },
    )


def _step_menu(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile: DatasetProfile = state["profile"]
    frozen_estimand = state["frozen_estimand"]
    columns = state["columns"]
    panel_cfg = state["config"]["panel"]

    with use_test_model(TestModel(custom_output_args=profile.judge_menu)):
        proposal = generate_spec_menu(frozen=frozen_estimand, available_columns=columns)

    frozen_menu = freeze_spec_menu(proposal, available_columns=columns, approved=True)
    specs = expand_to_specs(
        frozen_menu,
        outcome=frozen_estimand.estimand.outcome,
        treatment=state["treatment_col"],
        unit_col=str(panel_cfg["unit"]),
        time_col=str(panel_cfg["time"]),
    )
    validate_spec_menu_to_specs(frozen_menu, specs)

    warnings: list[str] = []
    for spec in specs:
        warnings.extend(
            validate_estimand_spec_mapping(frozen_estimand, spec, available_columns=columns)
        )

    state["specs"] = specs
    return (
        f"menü donduruldu (hash {frozen_menu.menu_hash}), {len(specs)} spesifikasyon açıldı",
        {
            "menu_hash": frozen_menu.menu_hash,
            "n_specs": len(specs),
            "hard_cap": SETTINGS.max_specifications,
            "estimators": sorted({spec.estimator for spec in specs}),
            "mapping_warnings": sorted(set(warnings)),
        },
    )


def _step_multiverse(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    specs = state["specs"]
    run_id = state["run_id"]
    timeout = float(state["timeout"])

    # Runner çalışma dizinini exist_ok ile açar; sabit run id'de önceki koşunun
    # sonucu yerinde kalır ve yanlışlıkla okunabilir.
    shutil.rmtree(Path(SETTINGS.runs_dir) / run_id, ignore_errors=True)

    handle = launch_multiverse(state["df"], specs, run_id)
    deadline = time.monotonic() + timeout
    while not handle.is_done():
        if time.monotonic() > deadline:
            handle.process.kill()
            handle.process.wait()  # kill sinyali yeter değil; reap edilmezse zombie kalır
            raise TimeoutError(f"Multiverse worker {timeout:.0f}s içinde bitmedi.")
        time.sleep(MULTIVERSE_POLL_SECONDS)

    if handle.process.returncode != 0:
        stderr = handle.read_stderr().strip()
        raise RuntimeError(f"Multiverse worker exit={handle.process.returncode}.\n{stderr}")

    results = handle.read_results()
    if len(results) != len(specs):
        raise ValueError(f"{len(specs)} spec koşuldu ama {len(results)} sonuç yazıldı.")

    failed = [result for result in results if result.status != "ok"]
    if failed:
        detail = "; ".join(f"{result.spec_id}: {result.error}" for result in failed[:3])
        raise ValueError(f"{len(failed)} spesifikasyon koşamadı — {detail}")

    state["results"] = results
    return (
        f"{len(results)} sonuç yazıldı",
        {
            "run_dir": _repo_path(handle.run_dir),
            "n_results": len(results),
            "progress": handle.read_progress(),
        },
    )


def _step_summarize(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = state["results"]
    summary = dict(summarize(results))
    if not summary["n_ok"]:
        raise ValueError("Hiçbir spesifikasyon başarılı olmadı; özetlenecek sonuç yok.")

    diagnosis = diagnose_axes(results, state["specs"])
    effective_n = [result.n_obs for result in results if result.status == "ok"]
    if any(value is None for value in effective_n):
        raise ValueError("Başarılı bir spesifikasyon efektif N taşımıyor; örneklem şeffaf değil.")

    return (
        f"bant {summary['band']}, {summary['n_ok']} sonuç özetlendi",
        {
            "summary": summary,
            "n_used_in_diagnosis": diagnosis["n_used"],
            "anova_partial_r2": diagnosis["anova_partial_r2"],
            "diagnosis_warnings": diagnosis["warnings"],
            "min_effective_n": min(effective_n),
        },
    )


StepFn = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]

STEPS: tuple[tuple[str, str, StepFn], ...] = (
    ("load", "Yükle", _step_load),
    ("clean", "Temizle", _step_clean),
    ("estimand", "Estimand", _step_estimand),
    ("menu", "Menü", _step_menu),
    ("multiverse", "Multiverse", _step_multiverse),
    ("summarize", "Özet", _step_summarize),
)


# --------------------------------------------------------------------------- #
# Koşu
# --------------------------------------------------------------------------- #
@dataclass
class DatasetRun:
    """Tek dataset'in matristeki sonucu."""

    key: str
    status: str
    reason: str
    seconds: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)


def run_dataset(
    profile: DatasetProfile,
    *,
    repo_root: Path = REPO_ROOT,
    timeout: float = MULTIVERSE_TIMEOUT_SECONDS,
) -> DatasetRun:
    """Tek dataset'i zincir boyunca koşar; ilk kırılmadan sonrakiler bloke işaretlenir."""
    dataset_dir = repo_root / "data" / profile.key
    started = time.monotonic()

    # Pre-flight de adımlar gibi korunur: descriptor'ı okunamayan bir dataset yalnız
    # kendi satırını kırmalı, matrisi traceback ile düşürüp diğerlerini raporsuz
    # bırakmamalı.
    try:
        config = load_dataset_config(dataset_dir)
        missing = missing_source_files(config, dataset_dir)
    except Exception as exc:
        return DatasetRun(
            key=profile.key,
            status=STATUS_FAILED,
            reason=f"descriptor okunamadı — {type(exc).__name__}: {exc}",
            seconds=time.monotonic() - started,
        )

    if missing:
        reason = f"ham veri eksik: {', '.join(missing)}"
        if profile.local_only:
            return DatasetRun(
                key=profile.key,
                status=STATUS_SKIPPED,
                reason=f"{reason} ({profile.missing_data_note})",
                seconds=time.monotonic() - started,
            )
        return DatasetRun(
            key=profile.key,
            status=STATUS_FAILED,
            reason=reason,
            seconds=time.monotonic() - started,
        )

    state: dict[str, Any] = {
        "profile": profile,
        "dataset_dir": dataset_dir,
        "run_id": f"s3-08-smoke-{profile.key}",
        "timeout": timeout,
    }

    steps: list[dict[str, Any]] = []
    broken = ""
    for key, title, step in STEPS:
        if broken:
            steps.append(
                {
                    "key": key,
                    "title": title,
                    "status": STATUS_BLOCKED,
                    "detail": f"'{broken}' kırıldığı için çalıştırılmadı",
                    "metrics": {},
                }
            )
            continue
        try:
            detail, metrics = step(state)
        except Exception as exc:
            broken = key
            steps.append(
                {
                    "key": key,
                    "title": title,
                    "status": STATUS_FAILED,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                }
            )
            continue
        steps.append(
            {
                "key": key,
                "title": title,
                "status": STATUS_OK,
                "detail": detail,
                "metrics": jsonable(metrics),
            }
        )

    seconds = time.monotonic() - started
    if broken:
        detail = next(s["detail"] for s in steps if s["key"] == broken)
        return DatasetRun(
            key=profile.key,
            status=STATUS_FAILED,
            reason=f"{broken} adımı kırıldı — {detail}",
            seconds=seconds,
            steps=steps,
        )
    return DatasetRun(
        key=profile.key,
        status=STATUS_OK,
        reason="zincirin altı adımı da koştu",
        seconds=seconds,
        steps=steps,
    )


def _report_provenance(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    return {
        "generation_command": GENERATION_COMMAND,
        "source_commit": source_commit(repo_root=repo_root),
        "python": platform.python_version(),
        "platform": platform_name(),
        "judge_mode": "PydanticAI test modeli (sahte tipli çıktı, canlı sağlayıcı çağrısı yok)",
    }


def run_smoke_matrix(
    profiles: tuple[DatasetProfile, ...] = PROFILES,
    *,
    repo_root: Path = REPO_ROOT,
    timeout: float = MULTIVERSE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Matrisi koşar. Bir dataset kırılsa da kalanlar koşar; rapor hepsini taşır."""
    runs = [run_dataset(profile, repo_root=repo_root, timeout=timeout) for profile in profiles]
    failed = [run for run in runs if run.status == STATUS_FAILED]
    skipped = [run for run in runs if run.status == STATUS_SKIPPED]
    completed = [run for run in runs if run.status == STATUS_OK]
    total_seconds = sum(run.seconds for run in runs)

    if failed:
        overall_reason = f"{len(failed)} dataset kırıldı: {', '.join(r.key for r in failed)}"
    elif not completed:
        # Boşalmış bir matris yeşil dönerse en tehlikeli hâline gelir: kimse log'a bakmaz
        # ve genelleme iddiası doğrulanmamışken doğrulanmış görünür.
        atlanan = ", ".join(r.key for r in skipped) or "yok"
        overall_reason = (
            "hiçbir dataset koşmadı, matris bu hâliyle çekirdek hakkında hiçbir şey "
            f"kanıtlamıyor (atlanan: {atlanan})"
        )
    elif skipped:
        overall_reason = (
            f"{len(completed)} dataset zinciri yürüttü; "
            f"atlanan: {', '.join(r.key for r in skipped)}"
        )
    else:
        overall_reason = f"koşan {len(runs)} dataset zinciri baştan sona yürüttü"

    return {
        "check": "S3-08 CI smoke matrisi",
        "provenance": _report_provenance(repo_root=repo_root),
        "overall": {
            "status": STATUS_OK if completed and not failed else STATUS_FAILED,
            "reason": overall_reason,
        },
        "budget": {
            "total_seconds": round(total_seconds, 1),
            "warn_threshold_seconds": TIME_BUDGET_WARN_SECONDS,
            "within_threshold": total_seconds <= TIME_BUDGET_WARN_SECONDS,
        },
        "notes": list(RUN_NOTES),
        "datasets": [
            {
                "key": run.key,
                "status": run.status,
                "reason": run.reason,
                "seconds": round(run.seconds, 1),
                "steps": run.steps,
            }
            for run in runs
        ],
    }


# --------------------------------------------------------------------------- #
# Rapor
# --------------------------------------------------------------------------- #
_STATUS_LABELS = {
    STATUS_OK: "GEÇTİ",
    STATUS_FAILED: "KIRIK",
    STATUS_BLOCKED: "BLOKE",
    STATUS_SKIPPED: "ATLANDI",
}


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def render_markdown(report: dict[str, Any]) -> str:
    provenance = report["provenance"]
    overall = report["overall"]
    budget = report["budget"]
    lines = [
        "# S3-08 CI Smoke Matrisi",
        "",
        "## Provenance",
        "",
        "Şu komutla üretildi:",
        "",
        "```bash",
        provenance["generation_command"],
        "```",
        "",
        f"- Kaynak commit: {provenance['source_commit']}",
        f"- Python: {provenance['python']}",
        f"- Platform: {provenance['platform']}",
        f"- JUDGE modu: {provenance['judge_mode']}",
        "",
        "## Genel Sonuç",
        "",
        f"**{_status_label(overall['status'])}**: {overall['reason']}",
        "",
        "## Matris",
        "",
        "| Dataset | Durum | Süre (sn) | Not |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {run['key']} | {_status_label(run['status'])} | {run['seconds']} | {run['reason']} |"
        for run in report["datasets"]
    )

    budget_label = "eşik altında" if budget["within_threshold"] else "UYARI: EŞİK AŞILDI"
    lines.extend(
        [
            "",
            "## Süre Eşiği",
            "",
            f"- Toplam: {budget['total_seconds']} sn "
            f"(uyarı eşiği {budget['warn_threshold_seconds']} sn, {budget_label})",
            "- Eşik aşımı koşuyu kırmaz; ağır setleri nightly'ye ayırma kararı içindir.",
            "",
            "## Koşu Notları",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## Dataset Detayları", ""])
    for run in report["datasets"]:
        lines.append(f"### {run['key']}")
        lines.append("")
        lines.append(f"- Durum: **{_status_label(run['status'])}** ({run['reason']})")
        for step in run["steps"]:
            lines.append(
                f"- {step['title']}: **{_status_label(step['status'])}** — {step['detail']}"
            )
            lines.extend(
                f"  - {key}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in step["metrics"].items()
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto S3-08 CI smoke matrisi")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(PROFILE_BY_KEY),
        help="Yalnız bu dataset'i koş (tekrarlanabilir); verilmezse matrisin tamamı",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=MULTIVERSE_TIMEOUT_SECONDS,
        help="Multiverse worker için dataset başına saniye cinsinden üst sınır",
    )
    parser.add_argument(
        "--out",
        help="Varsayılan Markdown; uzantı .json ise JSON yazar. Çıktı bir log'dur, "
        "süre taşıdığı için tekrar-üretilebilir artefakt değildir",
    )
    return parser.parse_args(argv)


def selected_profiles(keys: list[str] | None) -> tuple[DatasetProfile, ...]:
    """Seçilen dataset'ler; tekrar eden anahtar zinciri iki kez koşturmaz."""
    if not keys:
        return PROFILES
    return tuple(PROFILE_BY_KEY[key] for key in dict.fromkeys(keys))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = selected_profiles(args.dataset)
    report = run_smoke_matrix(selected, timeout=args.timeout)
    # `--out` verilse de rapor stdout'ta kalır: dosyaya yazdırıp log'u boşaltmak,
    # kırık bir koşuda nedeni okunabilir tek yerden kaldırırdı.
    if args.out:
        write_report(report, Path(args.out), render=render_markdown)
    print(render_markdown(report), end="")
    return 0 if report["overall"]["status"] == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())
