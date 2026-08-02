"""JUDGE benchmark koşucusunun testleri — API yakmadan.

İddia "puanlayıcı çalıştı" değil, "puanlayıcı ŞU ihlali yakalar". Her test
bilerek bozuk bir model çıktısı kurar ve puanlayıcının onu işaretlediğini
doğrular; puanlama mantığı gevşerse test düşer.

Neden bu kadar önemli: bozuk bir altın set ya da gevşemiş bir puanlayıcı,
benchmark'ı sessizce yanlış ölçtürür ve `providers.py`'ye yanlış model listesi
girer. Bu dosya o sessiz hatayı gürültülü hale getirir.

Canlı sağlayıcı çağrısı yok: uçtan uca testler PydanticAI `TestModel` ile koşar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

import scripts.run_model_benchmark as bench
from pareto.analysis.hypothesis import TACProposal
from pareto.analysis.menu import ALL_AXES, SpecMenuAxis, SpecMenuProposal
from pareto.cleaning.ledger import LedgerEntry
from pareto.cleaning.merge import build_panel, load_dataset_config, resolve_source_paths
from pareto.cleaning.transforms import REGISTRY
from pareto.llm.narrative import AxisComment, VarianceNarrative
from pareto.llm.router import use_test_model
from pareto.profiling import profile_dataframe
from scripts.run_model_benchmark import (
    GOLD_DIR,
    ORDER_CHECK_VARIANT,
    REPO_ROOT,
    Meter,
    QuotaExhausted,
    Throttle,
    _run_cleaning,
    _run_estimand,
    _run_narrative,
    answer_fingerprint,
    build_report,
    by_priority,
    classify_error,
    fabricated_numbers,
    load_gold,
    load_models,
    looks_turkish,
    pinned_model,
    plan_lines,
    preflight_one,
    quota_cap,
    quota_pool,
    read_done,
    read_many,
    read_rows,
    require_cache_disabled,
    result_id,
    run_matrix,
    schema_verdict,
    score_cleaning,
    score_estimand,
    score_narrative,
    score_spec_menu,
)


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _entry(transform_name: str, params: dict, *, flagged: bool = False) -> LedgerEntry:
    return LedgerEntry(
        bulgu="test bulgusu",
        transform_name=transform_name,
        params=params,
        gerekce="test gerekçesi",
        belirsizlik_bayragi=flagged,
    ).stamped()


def _case(task: str, case_id: str) -> dict:
    for case in load_gold(task):
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"{task} altın setinde {case_id} yok")


def _missing_source(dataset_dir: str) -> str | None:
    """Config'te tanımlı ama diskte olmayan ilk kaynağın adını döndürür, yoksa None.

    `load_sources` ile aynı ön-kontrolü yapar (bkz. `resolve_source_paths`), ama
    hata fırlatmak yerine karar çağırana bırakılır.
    """
    root = REPO_ROOT / dataset_dir
    for name, src in load_dataset_config(root)["sources"].items():
        paths = resolve_source_paths(str(src["file"]), root)
        if not paths or any(not p.exists() for p in paths):
            return name
    return None


def _panel_root(dataset_dir: str) -> Path:
    """Ham verisi olan veri seti kökünü döndürür; eksikse testi atlar.

    Medicaid'in CDC WONDER extract'i DUA kısıtı yüzünden repo'ya konmuyor
    (.gitignore: `data/**/raw/*.tsv`). Panel gerçek dosyalardan kurulduğu için
    bu testler ham veri olmadan koşamaz; CI'da beklenen sonuç hata değil atlamadır.
    """
    missing = _missing_source(dataset_dir)
    if missing is not None:
        pytest.skip(f"{dataset_dir}: '{missing}' ham kaynağı lokalde yok (CI'da atlanır)")
    return REPO_ROOT / dataset_dir


# --------------------------------------------------------------------------- #
# Temizleme puanlayıcısı
# --------------------------------------------------------------------------- #
def test_cleaning_scorer_catches_forbidden_transform() -> None:
    """county_fips'i sayıya çevirmek baştaki sıfırı siler ve join anahtarını öldürür.

    Bu, şema açısından tamamen geçerli bir karardır — hiçbir üretim doğrulayıcısı
    yakalamaz. Yakalayan tek şey altın setteki `forbidden` etiketidir; puanlayıcı
    bunu kaçırırsa veri bozan model 'temiz' görünür.
    """
    case = _case("cleaning", "medicaid")
    scores = score_cleaning([_entry("coerce_numeric", {"col": "county_fips"})], case)

    assert scores["forbidden_hits"] == ["coerce_numeric:county_fips"]


def test_cleaning_scorer_counts_missed_must_fix() -> None:
    """Hiçbir şey önermeyen model 'hata yapmamış' sayılmamalı: recall 0."""
    case = _case("cleaning", "medicaid")
    scores = score_cleaning([], case)

    assert scores["must_fix_recall"] == 0.0
    assert "coerce_numeric:deaths" in scores["must_fix_missed"]
    assert "parse_date:implementation_date" in scores["must_fix_missed"]


def test_cleaning_scorer_full_recall_and_clean_precision() -> None:
    case = _case("cleaning", "medicaid")
    scores = score_cleaning(
        [
            _entry("coerce_numeric", {"col": "deaths"}),
            _entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=True),
        ],
        case,
    )

    assert scores["must_fix_recall"] == 1.0
    assert scores["forbidden_hits"] == []
    assert scores["precision"] == 1.0
    assert scores["extraneous"] == []


def test_cleaning_scorer_flags_overconfidence_on_structural_missing() -> None:
    """implementation_date'in %31 eksiği kusur değil, never-treated sinyali.

    Eksiklik oranı `missing_value_hard_threshold`ın (0.5) ALTINDA olduğu için
    deterministik gatekeeper devreye girmez — kararın insana gidip gitmeyeceği
    yalnız modelin güvenine kalır. Bayraksız gelen karar gatekeeper'ı atlar.
    """
    case = _case("cleaning", "medicaid")
    confident = score_cleaning(
        [_entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=False)], case
    )
    humble = score_cleaning(
        [_entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=True)], case
    )

    assert confident["overconfident_on_structural_missing"] == ["parse_date:implementation_date"]
    assert humble["overconfident_on_structural_missing"] == []


def test_cleaning_scorer_counts_forbidden_against_precision() -> None:
    """Yasak öneri hem ağır ihlal hem isabetsizliktir; precision'ı 1.0 bırakamaz.

    Önceki sürümde forbidden çiftleri `extraneous`tan dışlanıyordu, dolayısıyla
    yalnız veri bozan bir transform öneren model precision=1.0 alıyordu —
    tabloya bakan kişi modeli isabetli sanırdı.
    """
    case = _case("cleaning", "medicaid")
    scores = score_cleaning([_entry("coerce_numeric", {"col": "county_fips"})], case)

    assert scores["forbidden_hits"] == ["coerce_numeric:county_fips"]
    assert scores["precision"] == 0.0


def test_cleaning_scorer_counts_column_less_transform_as_a_proposal() -> None:
    """drop_duplicates(subset=None) kolon anmaz ama yine de bir öneridir.

    Kolon başına döngü onu tamamen görünmez bırakıyordu: model istediği kadar
    kolonsuz transform önerip precision'ını hiç düşürmeden gürültü üretebilirdi.
    """
    case = _case("cleaning", "medicaid")
    scores = score_cleaning(
        [
            _entry("coerce_numeric", {"col": "deaths"}),
            _entry("drop_duplicates", {"subset": None}),
        ],
        case,
    )

    assert scores["precision"] == 0.5
    assert "drop_duplicates:" in scores["extraneous"]


def test_cleaning_scorer_penalises_extraneous_proposals() -> None:
    """Altın sette geçmeyen öneri precision'ı düşürmeli: gürültü bedava olmamalı."""
    case = _case("cleaning", "castle")
    scores = score_cleaning(
        [
            _entry("coerce_numeric", {"col": "effective_year"}),
            _entry("coerce_numeric", {"col": "poverty"}),
        ],
        case,
    )

    assert scores["extraneous"] == ["coerce_numeric:poverty"]
    assert scores["precision"] == 0.5


def test_cleaning_scorer_reports_full_proposed_set() -> None:
    """`proposed`, cevap parmak izinin (answer_fingerprint) girdisi.

    Tekrarlar-arası tutarlılık ölçümü tam bu alanı karşılaştırır — eksik ya da
    yanlış sıralı gelirse iki özdeş öneri seti 'farklı cevap' sayılır.
    """
    case = _case("cleaning", "medicaid")
    scores = score_cleaning(
        [_entry("coerce_numeric", {"col": "deaths"}), _entry("drop_duplicates", {"subset": None})],
        case,
    )

    assert scores["proposed"] == ["coerce_numeric:deaths", "drop_duplicates:"]


def test_cleaning_scorer_flags_unnecessary_hesitation_on_known_correct_decision() -> None:
    """Bilinen-doğru (must_fix) bir kararı yine de insana sormak gereksiz çekingenliktir.

    `structurally_missing`'in TAMAMLAYICISI yanlış çapa olurdu: çoğu kolon ne
    must_fix'te ne structurally_missing'te, oraya flag koymak meşru bir
    belirsizlik olabilir. Yalnız gold'un zaten 'belirsizliksiz' dediği bir çifti
    (must_fix ya da acceptable) flag'lemek ölçülür — README:166-172'deki
    bad_control/config çelişkisiyle aynı hata sınıfını (yanlış çapa, doğru
    davranışı cezalandırma) abstention yönünde tekrarlamamak için.
    """
    case = _case("cleaning", "medicaid")
    hesitant = score_cleaning([_entry("coerce_numeric", {"col": "deaths"}, flagged=True)], case)
    confident = score_cleaning([_entry("coerce_numeric", {"col": "deaths"}, flagged=False)], case)

    assert hesitant["unnecessary_flags"] == ["coerce_numeric:deaths"]
    assert confident["unnecessary_flags"] == []


def test_cleaning_scorer_does_not_call_structural_hesitation_unnecessary() -> None:
    """implementation_date hem must_fix HEM structurally_missing.

    Flag'lemek gatekeeper'ın istediği TAM davranış (bkz.
    test_cleaning_scorer_flags_overconfidence_on_structural_missing — 'humble'
    varyantı burada `overconfident_on_structural_missing == []` alıyor). İki sayaç
    aynı eylemi biri ödüllendirip diğeri cezalandıramaz; `unnecessary_flags`
    structural kolonları hariç tutmalı.
    """
    case = _case("cleaning", "medicaid")
    scores = score_cleaning(
        [_entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=True)], case
    )

    assert scores["overconfident_on_structural_missing"] == []
    assert scores["unnecessary_flags"] == []


# --------------------------------------------------------------------------- #
# Estimand puanlayıcısı
# --------------------------------------------------------------------------- #
def _tac(**overrides) -> TACProposal:
    base = dict(
        # `treatment` kolon adı, `treatment_coding` kodlamanın serbest metin
        # tarifi. Bu fixture eskiden ikisini ters tutuyordu ve puanlayıcı da
        # kolonu treatment_coding'den okuduğu için tutarlı görünüyordu.
        treatment="treated_post",
        treatment_coding="1 = New Jersey mağazası x dalga 2, 0 = diğer",
        outcome="fte_employment",
        outcome_unit="çalışan",
        population="NJ/PA restoranları",
        time_scope="1992",
        expected_sign="negative",
        h0="Etki yoktur.",
        h1="İstihdamı düşürür.",
        implied_result_translation="...",
        confirmation_question="...",
    )
    base.update(overrides)
    return TACProposal(**base)


def test_estimand_scorer_catches_flipped_expected_sign() -> None:
    """Model literatürü 'düzeltip' işareti çevirirse yakalanmalı.

    Estimand'ın işi çıpalamayı öldürmek: kullanıcının teorik beyanı sonuç
    görülmeden dondurulur. Card-Krueger vakası bilerek seçildi — ünlü bulgu
    kullanıcının beyanının tersi, yani modelin 'düzeltme' dürtüsü en yüksek.
    """
    case = _case("estimand", "card_krueger")
    columns = ["treated_post", "fte_employment"]

    honest = score_estimand(_tac(), case, available_columns=columns)
    flipped = score_estimand(_tac(expected_sign="positive"), case, available_columns=columns)

    assert honest["sign_preserved"] is True
    assert flipped["sign_preserved"] is False


def test_estimand_scorer_catches_hallucinated_column() -> None:
    case = _case("estimand", "card_krueger")
    scores = score_estimand(
        _tac(outcome="istihdam_orani"), case, available_columns=["treated_post", "fte_employment"]
    )

    assert scores["hallucinated_columns"] == ["istihdam_orani"]
    assert scores["outcome_ok"] is False


def test_estimand_scorer_requires_clarification_on_adversarial_case() -> None:
    """Panelde olmayan bir sonuç istendiğinde doğru cevap kolon uydurmak değil, sormak."""
    case = _case("estimand", "medicaid_olmayan_kolon")
    columns = ["crude_rate", "pct_uninsured", "treatment_cohort"]

    guessed = score_estimand(
        _tac(outcome="crude_rate", expected_sign="positive"), case, available_columns=columns
    )
    asked = score_estimand(
        _tac(
            outcome="crude_rate",
            expected_sign="positive",
            needs_clarification=True,
            clarification_question="Hastane kapasitesi hangi kolonda?",
        ),
        case,
        available_columns=columns,
    )

    assert guessed["clarification_correct"] is False
    assert asked["clarification_correct"] is True


def test_estimand_scorer_does_not_reward_clarification_on_clean_case() -> None:
    """Her vakada 'bilmiyorum' demek kaçamaktır; temiz vakada da hata sayılmalı."""
    case = _case("estimand", "card_krueger")
    scores = score_estimand(
        _tac(needs_clarification=True, clarification_question="?"),
        case,
        available_columns=["treated_post", "fte_employment"],
    )

    assert scores["clarification_correct"] is False


def test_estimand_scorer_echoes_gold_expectation_for_confusion_matrix() -> None:
    """Rapor confusion matrix'i (TP/FP/FN/TN) bu alandan okur, gold'u ikinci kez yüklemez.

    Diğer puanlayıcılar zaten aynı deseni kullanıyor (narrative'de
    expected_driving_axes) — saf 'çıktı + altın kayıt -> metrik sözlüğü'
    sözleşmesini bozmuyor.
    """
    adversarial = score_estimand(
        _tac(outcome="crude_rate", expected_sign="positive"),
        _case("estimand", "medicaid_olmayan_kolon"),
        available_columns=["crude_rate"],
    )
    clean = score_estimand(
        _tac(),
        _case("estimand", "card_krueger"),
        available_columns=["treated_post", "fte_employment"],
    )

    assert adversarial["expect_needs_clarification"] is True
    assert clean["expect_needs_clarification"] is False


# --------------------------------------------------------------------------- #
# Spec menüsü puanlayıcısı
# --------------------------------------------------------------------------- #
def _menu(**levels) -> SpecMenuProposal:
    defaults = {
        "control_set": "none",
        "sample": "none",
        "pre_period": "none",
        "clustering": "state",
        "never_treated": "true",
        "estimator": "TWFE",
        "weighting": "none",
    }
    defaults.update(levels)
    return SpecMenuProposal(
        axes=[
            SpecMenuAxis(axis_name=name, baseline_level=defaults[name], rationale="test")
            for name in ALL_AXES
        ],
        overall_rationale="test",
    )


def _menu_kwargs(case_id: str) -> dict:
    """Koşucunun `dataset_inputs`'ı ile AYNI kaynaklardan beslenir.

    panel_cfg elle yazılmaz: config.yaml'dan okunur, yoksa test üretimde
    kullanılmayan bir unit/time çiftiyle geçer ve yanlış güven verirdi
    (castle'ın unit'i 'state' değil 'state_id').
    """
    case = _case("spec_menu", case_id)
    root = _panel_root(case["dataset_dir"])
    return {
        "case": case,
        "available_columns": list(build_panel(root).df.columns),
        "panel_cfg": dict(load_dataset_config(root)["panel"]),
    }


def test_spec_menu_scorer_catches_unclustered_panel() -> None:
    """clustering='none' eyalet-yıl panelinde savunulamaz (Bertrand ve ark. 2004).

    Deterministik doğrulayıcı bunu YAKALAMAZ — 'none' geçerli bir seviye. Kırılgan
    olan şey standart hataların güvenilirliği, ve onu yalnız altın etiket biliyor.
    """
    kwargs = _menu_kwargs("castle")
    scores = score_spec_menu(_menu(clustering="none"), **kwargs)

    assert scores["indefensible_levels"] == ["clustering=none"]
    assert scores["baseline_violations"] == ["clustering=none"]


def test_spec_menu_scorer_catches_bad_control() -> None:
    """pct_uninsured genişlemenin birincil sonucu; mortalite regresyonunda kontrol edilemez.

    Kontrol olarak koymak, tam da ölçülmek istenen mekanizmayı (kapsam artışı →
    mortalite) kapatır. Deterministik doğrulayıcı bunu görmez: `pct_uninsured`
    panelde var olan, tamamen geçerli bir kolon adı.
    """
    kwargs = _menu_kwargs("medicaid")
    proposal = _menu(clustering="state_fips", weighting="population")
    for axis in proposal.axes:
        if axis.axis_name == "control_set":
            axis.baseline_level = "poverty_rate+pct_uninsured"
    scores = score_spec_menu(proposal, **kwargs)

    assert scores["bad_control_hits"] == ["pct_uninsured (poverty_rate+pct_uninsured)"]


def test_spec_menu_scorer_catches_indefensible_candidate_not_only_baseline() -> None:
    """İhlal yalnız baseline'da değil, aday seviyelerde de aranmalı.

    Çokluevren adayların ÇARPIMINI koşar; savunulamaz bir aday seviye baseline
    savunulabilir olsa bile sonuç dağılımına girer.
    """
    kwargs = _menu_kwargs("castle")
    proposal = _menu()
    for axis in proposal.axes:
        if axis.axis_name == "clustering":
            axis.candidate_levels = ["state", "none"]
    scores = score_spec_menu(proposal, **kwargs)

    assert scores["indefensible_levels"] == ["clustering=none"]
    assert scores["baseline_violations"] == []


def test_spec_menu_scorer_reports_missing_axis() -> None:
    kwargs = _menu_kwargs("castle")
    proposal = _menu()
    proposal.axes = [a for a in proposal.axes if a.axis_name != "weighting"]
    scores = score_spec_menu(proposal, **kwargs)

    assert scores["missing_axes"] == ["weighting"]
    assert scores["deterministic_gate_passed"] is False


def test_spec_menu_scorer_accepts_defensible_menu() -> None:
    kwargs = _menu_kwargs("castle")
    scores = score_spec_menu(_menu(), **kwargs)

    assert scores["indefensible_levels"] == []
    assert scores["bad_control_hits"] == []
    assert scores["baseline_violations"] == []
    assert scores["deterministic_gate_passed"] is True


def test_spec_menu_scorer_reports_baseline_by_axis() -> None:
    """Cevap parmak izinin (answer_fingerprint) ve order-check karşılaştırmasının girdisi."""
    kwargs = _menu_kwargs("castle")
    scores = score_spec_menu(_menu(clustering="state"), **kwargs)

    assert scores["baseline_by_axis"]["clustering"] == "state"
    assert set(scores["baseline_by_axis"]) == set(ALL_AXES)


# --------------------------------------------------------------------------- #
# Anlatı puanlayıcısı
# --------------------------------------------------------------------------- #
def test_narrative_scorer_catches_hallucinated_axis() -> None:
    """Teşhiste olmayan eksene yorum yazmak halüsinasyondur."""
    case = _case("narrative", "estimator_driven")
    narrative = VarianceNarrative(
        ozet="12 koşunun 9'u pozitif; dönüşleri estimator sürüklüyor.",
        eksen_yorumlari=[AxisComment(axis="sample", yorum="Örneklem dönüşü sürüklüyor.")],
    )
    scores = score_narrative(narrative, case)

    assert scores["forbidden_axes_mentioned"] == ["sample"]


def test_narrative_scorer_catches_fabricated_number() -> None:
    """Sistem promptu sayı üretmeyi yasaklıyor ama üretimde bunu zorlayan doğrulayıcı YOK.

    Girdide 0.41 diye bir şey yok (point_max 0.041); model virgülü kaydırırsa
    panelde yanlış bir büyüklük görünür ve hiçbir üretim kontrolü uyarmaz.
    """
    case = _case("narrative", "estimator_driven")
    narrative = VarianceNarrative(
        ozet="Katsayı en fazla 0.41 seviyesine çıkıyor; dönüşleri estimator sürüklüyor.",
        eksen_yorumlari=[],
    )
    scores = score_narrative(narrative, case)

    assert "0.41" in scores["fabricated_numbers"]


def test_narrative_scorer_allows_derived_counts() -> None:
    """'12 koşunun 9'u' meşru bir türetme (12 x 0.75); uydurma sayılmamalı."""
    case = _case("narrative", "estimator_driven")
    narrative = VarianceNarrative(
        ozet="12 koşunun 9'u pozitif, 3'ü ters; dönüşleri estimator sürüklüyor.",
        eksen_yorumlari=[AxisComment(axis="estimator", yorum="Eşleşmiş çiftlerde dönüş burada.")],
    )
    scores = score_narrative(narrative, case)

    assert scores["fabricated_numbers"] == []
    assert scores["driver_named_in_ozet"] is True
    assert scores["forbidden_axes_mentioned"] == []


def test_narrative_scorer_catches_invented_signal_when_robust() -> None:
    """Sinyal yokken eksen yorumu yazmak: prompt 'skip axes with no signal' diyor."""
    case = _case("narrative", "no_signal_robust")
    narrative = VarianceNarrative(
        ozet="8 koşunun tamamı pozitif ve anlamlı.",
        eksen_yorumlari=[AxisComment(axis="estimator", yorum="Estimator dönüşü sürüklüyor.")],
    )
    scores = score_narrative(narrative, case)

    assert scores["commented_without_signal"] is True


def test_narrative_scorer_catches_english_output() -> None:
    """Prompt 'Write ozet and yorum in Turkish' diyor; İngilizce çıktı panelde kırık görünür."""
    case = _case("narrative", "estimator_driven")
    english = VarianceNarrative(
        ozet="9 of 12 runs are positive; the estimator axis drives the flips.",
        eksen_yorumlari=[],
    )
    turkish = VarianceNarrative(
        ozet="12 koşunun 9'u pozitif; dönüşleri estimator ekseni sürüklüyor.",
        eksen_yorumlari=[],
    )

    assert score_narrative(english, case)["is_turkish"] is False
    assert score_narrative(turkish, case)["is_turkish"] is True


def test_fabricated_numbers_matches_percent_form_of_rates() -> None:
    """0.75 oranı metinde '%75' diye yazılabilir; bu uydurma değildir."""
    assert fabricated_numbers("sign agreement %75", {75.0}) == []
    assert fabricated_numbers("sign agreement %76", {75.0}) == ["76"]


def test_looks_turkish_needs_more_than_one_common_word() -> None:
    """Tek bir 've' Türkçe kanıtı değil; sezgisel eşik iki sinyal."""
    assert looks_turkish("alpha and beta ve gamma") is False
    assert looks_turkish("koşu ve işaret") is True


# --------------------------------------------------------------------------- #
# Throttle / resume
# --------------------------------------------------------------------------- #
def test_throttle_raises_when_pool_cap_reached() -> None:
    clock = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    for _ in range(3):
        clock.acquire("havuz", rpm=None, cap=3)

    with pytest.raises(QuotaExhausted):
        clock.acquire("havuz", rpm=None, cap=3)


def test_throttle_shares_cap_across_models_in_one_pool() -> None:
    """OpenRouter'ın 50/gün kotası tüm :free modellerde ORTAK.

    Model başına saymak kotayı model sayısı kadar büyük gösterir ve koşu
    sağlayıcıdan 429 yer. Havuz kavramının varlık sebebi bu.
    """
    clock = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    clock.acquire("openrouter:free", rpm=None, cap=2)  # ling
    clock.acquire("openrouter:free", rpm=None, cap=2)  # north-mini

    with pytest.raises(QuotaExhausted):
        clock.acquire("openrouter:free", rpm=None, cap=2)


def test_throttle_sleeps_to_respect_rpm() -> None:
    slept: list[float] = []
    # now() sırası: (1) acquire#1 damgası, (2) acquire#2 bekleme hesabı, (3) acquire#2 damgası
    ticks = iter([0.0, 1.0, 1.0])
    clock = Throttle(now=lambda: next(ticks), sleep=slept.append)

    clock.acquire("havuz", rpm=30, cap=None)  # ilk çağrı beklemez
    clock.acquire("havuz", rpm=30, cap=None)  # 30 RPM -> 2 sn aralık, 1 sn geçmiş

    assert slept == [pytest.approx(1.0)]


def test_throttle_reports_rpm_wait_before_sleeping() -> None:
    """RPM beklemesi terminale basılmalı — sessiz kalırsa çok günlük bir koşuda

    (ör. gemini-3.6-flash rpm=5 -> istekler arası 12sn) kullanıcı script'in
    donduğunu sanır. `on_wait` `sleep`'ten ÖNCE çağrılmalı: kullanıcı ne kadar
    bekleyeceğini beklemeden önce görmeli.
    """
    waits: list[tuple[str, float]] = []
    order: list[str] = []
    ticks = iter([0.0, 1.0, 1.0])

    def _on_wait(pool: str, seconds: float) -> None:
        waits.append((pool, seconds))
        order.append("wait")

    def _sleep(_seconds: float) -> None:
        order.append("sleep")

    clock = Throttle(now=lambda: next(ticks), sleep=_sleep, on_wait=_on_wait)

    clock.acquire("havuz", rpm=30, cap=None)
    clock.acquire("havuz", rpm=30, cap=None)

    assert waits == [("havuz", pytest.approx(1.0))]
    assert order == ["wait", "sleep"]


def test_throttle_does_not_report_wait_on_pools_first_call() -> None:
    """İlk çağrıda önceki damga yok, dolayısıyla beklenecek bir şey de yok."""
    waits: list[tuple[str, float]] = []
    clock = Throttle(
        now=lambda: 0.0,
        sleep=lambda _s: None,
        on_wait=lambda pool, seconds: waits.append((pool, seconds)),
    )

    clock.acquire("havuz", rpm=30, cap=None)

    assert waits == []


def test_throttle_seed_counts_resumed_calls_against_cap() -> None:
    """`seed` ham bir sayaçtır: kendisine verilen her çağrı tavandan düşer.

    Hangi eski çağrının seed edileceği burada DEĞİL `seed_from_prior()`'da kararlaşır
    (günlük havuzda yalnız bugünkü, ömür-boyu kredi havuzunda hepsi) — bkz.
    test_seed_counts_prior_days_only_for_lifetime_credit_pools. İki sorumluluk
    ayrıldı: bu test sayacın sızdırmadığını, o test gün ayrımını doğruluyor.
    """
    clock = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    clock.seed("havuz", 5)

    with pytest.raises(QuotaExhausted):
        clock.acquire("havuz", rpm=None, cap=5)


def _report_row(
    model: str, task: str, case_id: str, repeat: int, *, error: str | None = None
) -> dict[str, Any]:
    return {
        "result_id": f"{model}|{task}|{case_id}|{repeat}",
        "model": model,
        "provider": "google",
        "model_id": model.split("/")[-1],
        "served_by": model.split("/")[-1],
        "task": task,
        "case_id": case_id,
        "error": error,
        "scores": None if error else {},
        "latency_s": 1.0,
        "requests": 1,
    }


def test_read_many_does_not_merge_the_same_cell_across_two_runs(tmp_path) -> None:
    """İki koşunun aynı hücresi TEK satıra inmemeli.

    `result_id` koşu tarihini içermiyor (`model|task|case_id|repeat`), yani
    `latest_per_cell` iki koşunun aynı hücresini dedupe eder ve 3-tekrarlı eski
    koşu ile 2-tekrarlı yeni koşu sessizce birleşir. Sonuç EKSİK değil YANLIŞ
    olurdu: n küçülür, oranlar kayar, kimse fark etmez.
    """
    eski, yeni = tmp_path / "eski.jsonl", tmp_path / "yeni.jsonl"
    row = _report_row("google/gemini-3.6-flash", "narrative", "estimator_driven", 0)
    eski.write_text(json.dumps({**row, "error": "sema_tutmadi"}) + "\n", encoding="utf-8")
    yeni.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = bench.latest_per_cell(read_many([eski, yeni]))

    assert len(rows) == 2, "iki koşunun aynı hücresi birleştirilmiş"
    assert {r["source"] for r in rows} == {str(eski), str(yeni)}


def test_report_names_its_source_runs_when_it_merges_several(tmp_path) -> None:
    """Birleşik rapor hangi koşudan kaç satır aldığını YAZMALI.

    Yazmazsa farklı tekrar sayısı/prompt sürümüyle üretilmiş satırlar tek bir
    ölçüm gibi okunur — teslimde savunulamaz bir tablo.
    """
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_report_row("google/x", "narrative", "c1", 0)) + "\n", encoding="utf-8")
    b.write_text(json.dumps(_report_row("google/y", "narrative", "c1", 0)) + "\n", encoding="utf-8")

    report = build_report(read_many([a, b]), ("narrative",))

    assert "## Kaynak koşular" in report
    assert str(a) in report and str(b) in report


def test_report_breaks_down_by_task_only_for_a_small_candidate_set() -> None:
    """Görev kırılımı teslim setinde VAR, tam keşif matrisinde YOK.

    Havuzlu tablo bir eleme aracı ve öyle kalıyor; 17 modelde görev kırılımı
    gürültü. Ama teslimde soru "hangi model hangi dikişte iyi" — cleaning'de
    üstün olup narrative'de düşen bir modeli havuzlu oran gizler.
    """
    tasks = ("cleaning", "narrative")
    few = [
        _report_row("google/a", "cleaning", "medicaid", 0),
        _report_row("google/a", "narrative", "c1", 0, error="sema_tutmadi"),
        _report_row("google/b", "cleaning", "medicaid", 0),
    ]
    assert "## Görev kırılımı" in build_report(few, tasks)
    assert "| `google/a` | 1/1 | 0/1 |" in build_report(few, tasks)

    many = [_report_row(f"google/m{i}", "cleaning", "medicaid", 0) for i in range(8)]
    assert "## Görev kırılımı" not in build_report(many, tasks)


def test_throttle_token_limit_binds_when_it_is_tighter_than_rpm() -> None:
    """Bağlayıcı kısıt istek değil TOKEN olabilir; throttle büyük olanı uygulamalı.

    Canlı kanıt (2026-07-30): gemma-4-31b-it'in rpd'si 14.400 ve rpm'i 30, ama
    tpm'i 16.000. rpm'e bakan throttle 2sn aralıkla gider; ~7K'lık bir cleaning
    çağrısında bu dakikada ~30 çağrı = 210K token demek ve koşu 7. çağrıda 429
    yedi. Doğru aralık 60*7000/16000 = 26,25sn.
    """
    assert Throttle.interval(rpm=30, tpm=16_000, est_tokens=7_000) == pytest.approx(26.25)
    # TPM gevşekse rpm bağlar (gemini-3.6-flash: 250K tpm, 5 rpm -> 12sn)
    assert Throttle.interval(rpm=5, tpm=250_000, est_tokens=7_000) == pytest.approx(12.0)
    # tpm bildirilmemişse davranış eskisi gibi: yalnız rpm
    assert Throttle.interval(rpm=30, tpm=None, est_tokens=7_000) == pytest.approx(2.0)


def test_throttle_waits_the_token_interval_not_the_request_interval() -> None:
    """`acquire` gerçekten TPM aralığı kadar bekler (yalnız hesaplamakla kalmaz)."""
    waited: list[float] = []
    now = {"t": 0.0}
    clock = Throttle(
        now=lambda: now["t"], sleep=lambda s: waited.append(s), on_wait=lambda _p, _s: None
    )
    clock.acquire("gemma", rpm=30, cap=None, tpm=16_000, est_tokens=7_000)
    now["t"] = 2.0  # rpm aralığı (2sn) doldu, tpm aralığı (26,25sn) dolmadı
    clock.acquire("gemma", rpm=30, cap=None, tpm=16_000, est_tokens=7_000)

    assert waited == [pytest.approx(24.25)]


def test_throttle_estimate_prefers_the_larger_of_table_and_observation() -> None:
    """Tahmin tablosu ESKİ bir koşudan; gerçek maliyet büyürse throttle onu almalı.

    Tablo (TASK_TOKEN_ESTIMATE) prompt değişmeden önce ölçüldü. Prompt büyüyünce
    tabloya sadık kalan bir throttle gerçekte harcanandan az bekler ve 429 yer.
    """
    clock = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    assert clock.estimate("cleaning") == bench.TASK_TOKEN_ESTIMATE["cleaning"]

    clock.observe("cleaning", bench.TASK_TOKEN_ESTIMATE["cleaning"] + 5_000)
    assert clock.estimate("cleaning") == bench.TASK_TOKEN_ESTIMATE["cleaning"] + 5_000

    clock.observe("cleaning", 10)  # küçük gözlem tahmini DÜŞÜRMEZ
    assert clock.estimate("cleaning") == bench.TASK_TOKEN_ESTIMATE["cleaning"] + 5_000


def test_read_done_skips_completed_results(tmp_path) -> None:
    """Yarım kalan koşu tekrarlanmamalı — aynı çağrı iki kez kotadan yemesin."""
    path = tmp_path / "results.jsonl"
    model = {"provider": "groq", "id": "openai/gpt-oss-120b"}
    rid = result_id(model, "narrative", "estimator_driven", 0)
    path.write_text(json.dumps({"result_id": rid}) + "\n", encoding="utf-8")

    assert rid in read_done(path)
    assert result_id(model, "narrative", "estimator_driven", 1) not in read_done(path)


def test_read_done_tolerates_truncated_last_line(tmp_path) -> None:
    """Koşu kill edilirse son satır yarım kalabilir; bu tüm resume'u çöpe atmamalı."""
    path = tmp_path / "results.jsonl"
    path.write_text('{"result_id": "a|narrative|x|0"}\n{"result_id": "b|nar', encoding="utf-8")

    assert read_done(path) == {"a|narrative|x|0": 1}


def test_read_rows_returns_empty_for_missing_file(tmp_path) -> None:
    """İlk çağrıdan önce kota biterse dosya hiç oluşmaz; rapor adımı çökmemeli."""
    assert read_rows(tmp_path / "yok.jsonl") == []


# --------------------------------------------------------------------------- #
# Hata sınıflandırma (preflight'ın eleme gerekçeleri)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (OSError("OPENAI_API_KEY tanımlı değil. Şunlardan biriyle ayarlayın:"), "anahtar_yok"),
        (RuntimeError("Error code: 429 - rate limit exceeded"), "kota"),
        (RuntimeError("Error code: 404 - model not found"), "model_yok"),
        (RuntimeError("Error code: 401 - Unauthorized"), "yetki"),
        (ValueError("JUDGE profilde olmayan kolon andı: ['xyz']"), "dogrulayici_reddetti"),
    ],
)
def test_classify_error_labels_preflight_outcomes(exc: BaseException, expected: str) -> None:
    """Elenen model 'hata' diye tek torbaya atılmamalı: gerekçe eylem belirler.

    'anahtar_yok' ops işidir, 'model_yok' katalog yanlış demektir,
    'sema_tutmadi' modelin kendi kusurudur. Rapor bunları ayırmazsa
    aday listesi yanlış nedenle daraltılır.
    """
    assert classify_error(exc) == expected


def test_classify_error_prefers_type_over_message_substring() -> None:
    """pydantic-ai'nin şema retry mesajı 'not found' içerebilir.

    Metne önce bakılırsa modelin kendi şema kusuru 'model_yok' diye raporlanır ve
    model listeden yanlış nedenle çıkarılır — üstelik gerçekten var olan bir uç
    'katalogda yok' diye işaretlenir.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    exc = UnexpectedModelBehavior("Exceeded maximum retries: tool 'final_result' not found")

    assert classify_error(exc) == "sema_tutmadi"


def test_schema_verdict_is_unknown_when_call_never_reached_the_model() -> None:
    """Anahtar yoksa şema hakkında hiçbir şey öğrenmedik; True demek yalan olur."""
    assert schema_verdict(None) is True
    assert schema_verdict("sema_tutmadi") is False
    assert schema_verdict("anahtar_yok") is None
    assert schema_verdict("model_yok") is None
    assert schema_verdict("dogrulayici_reddetti") is True


# --------------------------------------------------------------------------- #
# Uçtan uca: üretim fonksiyonu -> puanlayıcı bağlantısı
# --------------------------------------------------------------------------- #
def test_run_narrative_wires_production_function_to_scorer() -> None:
    case = load_gold("narrative")[0]
    output = {
        "ozet": "12 koşunun 9'u pozitif; dönüşleri estimator ekseni sürüklüyor.",
        "eksen_yorumlari": [{"axis": "estimator", "yorum": "Dönüş burada yoğunlaşıyor."}],
    }

    with use_test_model(TestModel(custom_output_args=output)):
        scores = _run_narrative(case)

    assert scores["driver_named_in_ozet"] is True
    assert scores["fabricated_numbers"] == []


def test_run_cleaning_surfaces_forbidden_transform_end_to_end() -> None:
    """generate_ledger -> LedgerEntry -> referenced_columns -> altın etiket zinciri."""
    case = _case("cleaning", "card_krueger")
    output = {
        "decisions": [
            {
                "bulgu": "store_id sayısal görünüyor",
                "transform": {"transform_name": "coerce_numeric", "col": "store_id"},
                "gerekce": "sayıya çevrilsin",
                "confidence": "high",
            }
        ]
    }

    with use_test_model(TestModel(custom_output_args=output)):
        scores = _run_cleaning(case)

    assert scores["forbidden_hits"] == ["coerce_numeric:store_id"]


def test_run_estimand_uses_real_panel_columns() -> None:
    """available_columns panelden geliyor; altın set kolon listesini tekrarlamıyor."""
    case = _case("estimand", "divorce")
    output = {
        "treatment": "post",
        "treatment_coding": "1 = tek-taraflı boşanma yasası o eyalet-yılda yürürlükte, 0 = değil",
        "outcome": "suicide_rate_f",
        "outcome_unit": "1M kadın başına",
        "population": "ABD eyaletleri",
        "time_scope": "1964-1996",
        "expected_sign": "negative",
        "h0": "Etki yoktur.",
        "h1": "Düşürür.",
        "implied_result_translation": "...",
        "confirmation_question": "...",
    }

    with use_test_model(TestModel(custom_output_args=output)):
        scores = _run_estimand(case)

    assert scores["treatment_ok"] is True
    assert scores["outcome_ok"] is True
    assert scores["hallucinated_columns"] == []
    assert scores["sign_preserved"] is True


# --------------------------------------------------------------------------- #
# Altın set bütünlüğü — bozuk altın set sessizce yanlış ölçer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", load_gold("cleaning"), ids=lambda c: c["case_id"])
def test_cleaning_gold_columns_and_transforms_are_real(case: dict) -> None:
    """Altın setteki her kolon gerçekten profilde, her transform REGISTRY'de olmalı."""
    profile = profile_dataframe(build_panel(_panel_root(case["dataset_dir"])).df)
    columns = set(profile["columns"])

    for bucket in ("must_fix", "forbidden", "acceptable"):
        for record in case.get(bucket, []):
            assert record["transform"] in REGISTRY, f"{bucket}: bilinmeyen transform {record}"
            assert record["column"] in columns, f"{bucket}: panelde olmayan kolon {record}"
    for column in case.get("structurally_missing", []):
        assert column in columns, f"structurally_missing: panelde yok {column}"


def test_cleaning_gold_must_fix_columns_really_need_fixing() -> None:
    """MUST_FIX iddiası veriyle uyuşmalı: sayıya çevrilecek kolon gerçekten metin olmalı.

    Bu test altın seti VERİYE bağlar. Panel pipeline'ı ileride dtype'ı düzeltirse
    (ör. merge katmanı coerce etmeye başlarsa) burası düşer ve altın set
    güncellenir — yoksa benchmark var olmayan bir kusuru aramaya devam ederdi.
    """
    checked = 0
    for case in load_gold("cleaning"):
        if _missing_source(case["dataset_dir"]) is not None:
            continue  # ham verisi repo'da olmayan set (bkz. _panel_root)
        checked += 1
        profile = profile_dataframe(build_panel(REPO_ROOT / case["dataset_dir"]).df)
        for record in case["must_fix"]:
            if record["transform"] != "coerce_numeric":
                continue
            dtype = profile["columns"][record["column"]]["dtype"]
            assert dtype in ("object", "str"), (
                f"{case['case_id']}.{record['column']}: dtype={dtype} — "
                "artık metin değil, coerce_numeric MUST_FIX olmaktan çıkmış"
            )

    # Tek vaka bile koşmadıysa test sessizce yeşil kalırdı; atlama olduğunu söyle.
    if checked == 0:
        pytest.skip("hiçbir temizleme vakasının ham verisi lokalde yok")


@pytest.mark.parametrize("case", load_gold("estimand"), ids=lambda c: c["case_id"])
def test_estimand_gold_accepted_columns_exist(case: dict) -> None:
    columns = set(build_panel(_panel_root(case["dataset_dir"])).df.columns)
    for name in [*case["accept_treatment"], *case["accept_outcome"]]:
        assert name in columns, f"{case['case_id']}: panelde olmayan kolon kabul ediliyor: {name}"
    if case["adversarial"]:
        assert not case["accept_treatment"] and not case["accept_outcome"]


@pytest.mark.parametrize("case", load_gold("spec_menu"), ids=lambda c: c["case_id"])
def test_spec_menu_gold_bad_controls_do_not_contradict_config(case: dict) -> None:
    """Altın set, veri setinin kendi config.yaml'ıyla çelişemez.

    İlk taslakta castle'da `prisoner`/`police`, divorce'ta `homicide_rate`/
    `afdc_cases` bad control diye etiketlenmişti — oysa ikisi de o veri setlerinin
    `panel.covariates` listesinde ve kaynak makaleler bunları kontrol olarak
    kullanıyor. Böyle bir etiket, doğru davranan modeli cezalandırır ve
    benchmark'ı sessizce ters çevirir.
    """
    config = load_dataset_config(REPO_ROOT / case["dataset_dir"])
    covariates = set(config["panel"]["covariates"] or [])
    labelled = set(case.get("bad_control_columns", {}))

    assert not (labelled & covariates), (
        f"{case['case_id']}: config.yaml covariate olarak listeliyor ama altın set "
        f"bad control diyor: {sorted(labelled & covariates)}"
    )


@pytest.mark.parametrize("case", load_gold("spec_menu"), ids=lambda c: c["case_id"])
def test_spec_menu_gold_labels_reference_real_axes_and_columns(case: dict) -> None:
    columns = set(build_panel(_panel_root(case["dataset_dir"])).df.columns)

    for axis in [*case.get("baseline_must_be_in", {}), *case.get("indefensible_levels", {})]:
        assert axis in ALL_AXES, f"{case['case_id']}: bilinmeyen eksen {axis}"
    for column in case.get("bad_control_columns", {}):
        assert column in columns, f"{case['case_id']}: panelde olmayan bad control {column}"
    for column in case["baseline_must_be_in"].get("clustering", []):
        assert column in columns, f"{case['case_id']}: panelde olmayan cluster kolonu {column}"
    # Kolon adını taşıyan alan `treatment`. `treatment_coding` kodlamanın serbest
    # metin tarifi ve KOLON OLMAMALI: altın kayıt onu kolon adıyla doldurursa
    # spec_menu promptu modele üretimin üretemeyeceği bir sözleşme gösterir.
    assert case["estimand"]["treatment"] in columns
    assert case["estimand"]["treatment_coding"] not in columns
    assert case["estimand"]["outcome"] in columns


@pytest.mark.parametrize("case", load_gold("narrative"), ids=lambda c: c["case_id"])
def test_narrative_gold_axes_are_consistent(case: dict) -> None:
    """Beklenen eksen teşhiste OLMALI, yasak eksen teşhiste OLMAMALI."""
    diagnosed = set(case["diagnosis"]["axes"])

    for axis in case["expected_driving_axes"]:
        assert axis in diagnosed, f"{case['case_id']}: beklenen eksen teşhiste yok: {axis}"
    for axis in case["forbidden_axes"]:
        assert axis in ALL_AXES, f"{case['case_id']}: {axis} gerçek bir eksen adı değil"
        assert axis not in diagnosed, f"{case['case_id']}: yasak eksen teşhiste var: {axis}"


def test_narrative_gold_summary_shape_matches_production() -> None:
    """Dondurulmuş fixture, summarize()'ın gerçek anahtarlarını taşımalı.

    Fixture üretimden kayarsa benchmark modele üretimde hiç görmeyeceği bir
    payload gösterir ve ölçüm anlamını yitirir.
    """
    required = {"n_total", "n_ok", "n_failed", "sign_agreement", "significance_rate", "band"}
    for case in load_gold("narrative"):
        assert required <= set(case["summary"]), case["case_id"]
        assert {"axes", "matched_pairs", "anova_partial_r2"} <= set(case["diagnosis"])


# --------------------------------------------------------------------------- #
# Aday matrisi bütünlüğü
# --------------------------------------------------------------------------- #
def test_models_json_providers_are_registered_public_judge_slots() -> None:
    """Matristeki her sağlayıcı gerçekten kayıtlı bir PUBLIC judge slotu olmalı.

    Aksi halde hata koşunun ortasında, kota harcandıktan sonra çıkardı.
    """
    from pareto.config import PrivacyMode
    from pareto.llm.providers import judge_slots_for

    registered = set(judge_slots_for(PrivacyMode.PUBLIC))
    for model in load_models():
        assert model["provider"] in registered, model["id"]


def test_models_json_has_control_group_per_pinned_default() -> None:
    """Kontrol grubu olmadan skorlar okunamaz: neye göre daha iyi?"""
    controls = {m["id"] for m in load_models() if m.get("control")}

    assert "gemini-3.6-flash" in controls  # JUDGE_SLOT defaultu


def test_models_json_shared_pools_are_declared() -> None:
    """Aynı hesap kotasını paylaşan uçlar tek havuzda toplanmalı."""
    models = load_models()
    google = {quota_pool(m) for m in models if m["provider"] == "google"}

    assert len(google) == 4, "Google limitleri model başına, havuz paylaşılmamalı"


def test_models_json_shared_pool_members_declare_the_same_rpm() -> None:
    """Throttle RPM aralığını HAVUZ başına tutar (bkz. Throttle.acquire), model başına değil.

    Aynı havuzdaki iki uç farklı rpm bildirirse bekleme süresi, o an hangi
    modelin çağrıldığına göre rastgele geniş/dar hesaplanır — pool-level throttle
    yalnız havuzdaki TÜM üyeler aynı rpm'i paylaşırsa doğru çalışır. Matriste bugün
    çok-üyeli paylaşımlı havuz yok (OpenRouter çıkarıldı), ama bu test paylaşımlı
    bir havuz geri eklenirse (farklı rpm'li bir üyeyle) sessizce bozulmaya karşı
    kilitli kalsın diye duruyor.
    """
    by_pool: dict[str, set[int | None]] = {}
    for model in load_models():
        by_pool.setdefault(quota_pool(model), set()).add(model.get("rpm"))

    for pool, rpms in by_pool.items():
        assert len(rpms) == 1, f"{pool}: havuz üyeleri farklı rpm bildiriyor: {rpms}"


def test_models_json_every_model_declares_a_quota() -> None:
    """Kotasız model throttle edilemez; sessizce 429 yemektense burada patlasın."""
    for model in load_models():
        assert quota_cap(model) is not None, model["id"]
        assert model.get("rpm"), model["id"]


def test_gold_files_cover_every_task() -> None:
    for task in ("cleaning", "estimand", "spec_menu", "narrative"):
        cases = load_gold(task)
        assert cases, task
        assert len({c["case_id"] for c in cases}) == len(cases), f"{task}: yinelenen case_id"
        assert (GOLD_DIR / f"{task}.json").exists()


def test_meter_reports_retries_as_requests_minus_one() -> None:
    """İlk denemede şemayı tutturan model 1 istek yapar; retry sayısı 0 olmalı."""
    assert Meter(requests=1).retries == 0
    assert Meter(requests=3).retries == 2
    assert Meter(requests=0).retries == 0


# --------------------------------------------------------------------------- #
# Rapor
# --------------------------------------------------------------------------- #
def _row(model: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "task": "cleaning",
        "case_id": "medicaid",
        "error": None,
        "latency_s": 1.0,
        "retries": 0,
        "scores": {},
    }
    row.update(overrides)
    return row


def test_report_surfaces_severe_violations_not_just_rates() -> None:
    """Ağır ihlal başarı oranının içinde kaybolmamalı.

    Uydurulmuş bir kolon adı ya da değiştirilmiş bir `expected_sign`, şema uyumu
    %100 olsa bile modeli JUDGE slotundan diskalifiye eder. Rapor bunu ayrı
    listede göstermezse yanlış model seçilir.
    """
    rows = [
        _row("groq/x", scores={"forbidden_hits": ["coerce_numeric:county_fips"]}),
        _row("groq/x", task="estimand", scores={"sign_preserved": False}),
        _row("google/y", scores={"must_fix_recall": 1.0}),
    ]

    report = build_report(rows, ("cleaning", "estimand"))

    assert "coerce_numeric:county_fips" in report
    assert "expected_sign DEĞİŞTİRİLDİ" in report
    assert "`groq/x`" in report and "`google/y`" in report


def test_report_lists_failing_models_instead_of_dropping_them() -> None:
    """Hiç cevap veremeyen model tablodan düşmemeli: sessiz atlama = yanlış okuma."""
    rows = [
        _row("nvidia/z", error="model_yok", scores=None),
        _row("nvidia/z", error="model_yok", scores=None),
    ]

    report = build_report(rows, ("cleaning",))

    assert "`nvidia/z`" in report
    assert "model_yok" in report
    assert "| 2 | 0/2 |" in report


def test_report_says_none_when_nothing_severe() -> None:
    report = build_report([_row("google/y")], ("cleaning",))

    assert "- yok" in report


def test_report_median_latency_ignores_instant_failures() -> None:
    """Hemen dönen bir 404, yavaş ama çalışan modeli hızlı göstermemeli.

    Gecikme yalnız başarılı çağrılardan hesaplanır; aksi halde en çok hata veren
    model tabloda en hızlı görünür.
    """
    rows = [
        _row("groq/x", latency_s=4.0),
        _row("groq/x", latency_s=6.0),
        _row("groq/x", latency_s=0.01, error="model_yok", scores=None),
    ]

    report = build_report(rows, ("cleaning",))

    assert "| 5.00 |" in report


# --------------------------------------------------------------------------- #
# Cevap parmak izi + tekrar-tutarlılığı
# --------------------------------------------------------------------------- #
def test_answer_fingerprint_distinguishes_different_cleaning_proposals() -> None:
    case = _case("cleaning", "medicaid")
    same_a = score_cleaning([_entry("coerce_numeric", {"col": "deaths"})], case)
    same_b = score_cleaning([_entry("coerce_numeric", {"col": "deaths"})], case)
    different = score_cleaning([_entry("coerce_numeric", {"col": "county_fips"})], case)

    assert answer_fingerprint("cleaning", same_a) == answer_fingerprint("cleaning", same_b)
    assert answer_fingerprint("cleaning", same_a) != answer_fingerprint("cleaning", different)


def test_answer_fingerprint_estimand_tracks_treatment_and_outcome() -> None:
    case = _case("estimand", "card_krueger")
    columns = ["treated_post", "fte_employment"]
    a = score_estimand(_tac(), case, available_columns=columns)
    b = score_estimand(_tac(outcome="istihdam_orani"), case, available_columns=columns)

    assert answer_fingerprint("estimand", a) != answer_fingerprint("estimand", b)


def test_answer_fingerprint_spec_menu_tracks_baseline_choice() -> None:
    kwargs = _menu_kwargs("castle")
    a = score_spec_menu(_menu(clustering="state"), **kwargs)
    b = score_spec_menu(_menu(clustering="none"), **kwargs)

    assert answer_fingerprint("spec_menu", a) != answer_fingerprint("spec_menu", b)


def test_percentile_uses_nearest_rank_not_interpolation() -> None:
    """Nearest-rank yalnız GÖRÜLMÜŞ bir değer döndürür.

    `statistics.quantiles(..., n=100)` model başına ~onlarca çağrılık ölçekte
    enterpolasyonla var olmayan bir hassasiyet uydurur; bu benchmark'ın kendisi
    uydurma sayıyı ağır ihlal sayıyor (bkz. `fabricated_numbers`) — p95'in
    kendisi uydurulamaz.
    """
    assert bench._percentile([], 95) is None
    assert bench._percentile([3.0], 95) == 3.0
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


def test_answer_consistency_requires_at_least_two_successful_repeats() -> None:
    """Tek tekrarlı grup '%100 tutarlı' okunmamalı.

    README'nin kota takvimi gemini-3.6-flash'ı günde 20 istekle en dar havuz
    ilan ediyor — kota bitince tam da bu model tek-tekrarlı gruplarla kalır. Grup
    büyüklüğü sayılmadan ortalanırsa en çok kotaya çarpan model en tutarlı görünür.
    """
    rows = [_row("google/x", task="narrative", case_id="c1", scores={"axes_commented": ["sample"]})]

    rate, n = bench._answer_consistency(rows)

    assert (rate, n) == (None, 0)


def test_answer_consistency_flags_disagreement_across_repeats() -> None:
    rows = [
        _row("google/x", task="narrative", case_id="c1", scores={"axes_commented": ["sample"]}),
        _row("google/x", task="narrative", case_id="c1", scores={"axes_commented": ["sample"]}),
        _row("google/x", task="narrative", case_id="c2", scores={"axes_commented": ["sample"]}),
        _row("google/x", task="narrative", case_id="c2", scores={"axes_commented": ["estimator"]}),
    ]

    rate, n = bench._answer_consistency(rows)

    assert n == 2  # iki sayılabilir grup: c1, c2
    assert rate == 0.5  # yalnız c1 tekrarları aynı cevaba varıyor


def test_answer_consistency_excludes_errored_and_order_variant_rows() -> None:
    """429/model_yok bir 'tutarsızlık' değil, YOKLUKTUR; order-check ayrı bir soru sorar."""
    rows = [
        _row("google/x", task="narrative", case_id="c1", scores={"axes_commented": ["sample"]}),
        _row("google/x", task="narrative", case_id="c1", error="kota", scores=None),
        _row(
            "google/x",
            task="narrative",
            case_id="c1",
            scores={"axes_commented": ["farkli"]},
            order_variant="reversed",
        ),
    ]

    rate, n = bench._answer_consistency(rows)

    assert (rate, n) == (None, 0)  # hatasız/non-variant tek tekrar kaldı


# --------------------------------------------------------------------------- #
# Abstention sayaçları
# --------------------------------------------------------------------------- #
def test_cleaning_abstention_counts_separate_missed_gate_from_over_caution() -> None:
    case = _case("cleaning", "medicaid")
    overconfident = score_cleaning(
        [_entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=False)], case
    )
    hesitant = score_cleaning([_entry("coerce_numeric", {"col": "deaths"}, flagged=True)], case)
    rows = [
        _row("google/x", task="cleaning", scores=overconfident),
        _row("google/x", task="cleaning", scores=hesitant),
    ]

    missed_gate, over_caution = bench._cleaning_abstention_counts(rows)

    assert missed_gate == 1
    assert over_caution == 1


def test_estimand_confusion_counts_all_four_outcomes() -> None:
    clean_case = _case("estimand", "card_krueger")
    adversarial_case = _case("estimand", "medicaid_olmayan_kolon")
    clean_columns = ["treated_post", "fte_employment"]
    adv_columns = ["crude_rate"]

    tn = score_estimand(_tac(), clean_case, available_columns=clean_columns)
    fp = score_estimand(
        _tac(needs_clarification=True, clarification_question="?"),
        clean_case,
        available_columns=clean_columns,
    )
    fn = score_estimand(
        _tac(outcome="crude_rate", expected_sign="positive"),
        adversarial_case,
        available_columns=adv_columns,
    )
    tp = score_estimand(
        _tac(
            outcome="crude_rate",
            expected_sign="positive",
            needs_clarification=True,
            clarification_question="?",
        ),
        adversarial_case,
        available_columns=adv_columns,
    )
    rows = [
        _row("google/x", task="estimand", scores=tn),
        _row("google/x", task="estimand", scores=fp),
        _row("google/x", task="estimand", scores=fn),
        _row("google/x", task="estimand", scores=tp),
    ]

    counts = bench._estimand_confusion(rows)

    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "n": 4}


def test_estimand_confusion_excludes_order_check_variants() -> None:
    """--order-check'in 4. örneği farklı bir deneysel koşuldan gelir.

    Dahil edilirse rapordaki 'tekrar=3'te 9/3 örnek' notu yanlış olur — n görünen
    tekrar sayısıyla uyuşmalı.
    """
    case = _case("estimand", "card_krueger")
    tn = score_estimand(_tac(), case, available_columns=["treated_post", "fte_employment"])
    rows = [
        _row("google/x", task="estimand", scores=tn),
        _row("google/x", task="estimand", scores=tn, order_variant="reversed"),
    ]

    counts = bench._estimand_confusion(rows)

    assert counts["n"] == 1


def test_report_includes_token_p95_and_consistency_columns() -> None:
    rows = [
        _row("groq/x", latency_s=4.0, output_tokens=100),
        _row("groq/x", latency_s=6.0, output_tokens=200),
    ]

    report = build_report(rows, ("cleaning",))

    assert "p95 gecikme (s)" in report
    assert "Çıktı token (medyan)" in report
    assert "Cevap tutarlılığı" in report
    assert "maliyet hesaplanmadı" in report


def test_report_includes_abstention_counters_section() -> None:
    case = _case("cleaning", "medicaid")
    overconfident = score_cleaning(
        [_entry("parse_date", {"col": "implementation_date", "fmt": None}, flagged=False)], case
    )
    report = build_report([_row("google/x", task="cleaning", scores=overconfident)], ("cleaning",))

    assert "Abstention (gatekeeper) sayaçları" in report
    assert "| `google/x` | 1 | 0 | 0/0/0/0 (n=0) |" in report


# --------------------------------------------------------------------------- #
# Sıra duyarlılığı (--order-check)
# --------------------------------------------------------------------------- #
def test_report_omits_order_check_section_when_not_run() -> None:
    report = build_report([_row("google/x")], ("cleaning",))

    assert "Sıra duyarlılığı" not in report


def test_report_flags_unstable_answer_across_order_check_variant() -> None:
    """Kolon sırası ters çevrilince farklı outcome seçmek sıra duyarlılığıdır."""
    baseline = _row(
        "google/x",
        task="estimand",
        case_id="card_krueger",
        scores={
            "proposed_treatment": "treated_post",
            "proposed_outcome": "fte_employment",
            "needs_clarification": False,
        },
    )
    reversed_variant = _row(
        "google/x",
        task="estimand",
        case_id="card_krueger",
        order_variant="reversed",
        scores={
            "proposed_treatment": "treated_post",
            "proposed_outcome": "istihdam_orani",
            "needs_clarification": False,
        },
    )

    report = build_report([baseline, reversed_variant], ("estimand",))

    assert "Sıra duyarlılığı" in report
    assert "HAYIR" in report


def test_report_marks_stable_answer_across_order_check_variant() -> None:
    baseline = _row(
        "google/x",
        task="estimand",
        case_id="card_krueger",
        scores={
            "proposed_treatment": "treated_post",
            "proposed_outcome": "fte_employment",
            "needs_clarification": False,
        },
    )
    reversed_variant = _row(
        "google/x",
        task="estimand",
        case_id="card_krueger",
        order_variant="reversed",
        scores=dict(baseline["scores"]),
    )

    report = build_report([baseline, reversed_variant], ("estimand",))

    assert "| `google/x` | estimand/card_krueger | evet |" in report


def test_plan_lines_accounts_for_order_check_extra_calls() -> None:
    """--order-check tavan hesabına girmezse koşu ortasında beklenmedik 429 yer."""
    models = [{"id": "m", "provider": "google", "rpm": 5, "rpd": 20}]
    n_estimand = len(load_gold("estimand"))
    n_narrative = len(load_gold("narrative"))

    without = plan_lines(models, ("estimand", "narrative"), repeats=3, order_check=False)
    with_check = plan_lines(models, ("estimand", "narrative"), repeats=3, order_check=True)

    assert f"toplam: {(n_estimand + n_narrative) * 3}" in without[1]
    # narrative order-check'e dahil değil (yalnız estimand+spec_menu, bkz.
    # ORDER_CHECK_RUNNERS); yalnız estimand vaka sayısı kadar ekstra çağrı beklenir.
    assert f"+{n_estimand}/model" in with_check[0]
    assert f"toplam: {(n_estimand + n_narrative) * 3 + n_estimand}" in with_check[1]


def test_run_matrix_order_check_adds_a_reversed_column_variant(tmp_path) -> None:
    """--order-check estimand vakalarına, kolon sırası ters olan 1 ekstra çağrı ekler.

    Skorlayıcı kümeye bakıyor (score_estimand: `set(available_columns)`), sıraya
    değil — bu test doğru cevabın sıradan etkilenmediğini ve ekstra satırın
    doğru etiketlerle (result_id, order_variant) emitted olduğunu doğrular.
    """
    case = _case("estimand", "divorce")
    output = {
        "treatment": "post",
        "treatment_coding": "1 = tek-taraflı boşanma yasası o eyalet-yılda yürürlükte, 0 = değil",
        "outcome": "suicide_rate_f",
        "outcome_unit": "1M kadın başına",
        "population": "ABD eyaletleri",
        "time_scope": "1964-1996",
        "expected_sign": "negative",
        "h0": "Etki yoktur.",
        "h1": "Düşürür.",
        "implied_result_translation": "...",
        "confirmation_question": "...",
    }
    model = {"id": "gemma-4-31b-it", "provider": "google", "rpm": None, "rpd": None}

    with use_test_model(TestModel(custom_output_args=output)):
        rows = run_matrix(
            [model],
            ("estimand",),
            repeats=1,
            out_dir=tmp_path,
            throttle=Throttle(),
            order_check=True,
        )

    variant_rows = [r for r in rows if r.get("order_variant")]
    assert len(variant_rows) == len(load_gold("estimand"))
    divorce_variant = next(r for r in variant_rows if r["case_id"] == case["case_id"])
    assert divorce_variant["result_id"].endswith(f"|{ORDER_CHECK_VARIANT}")
    assert divorce_variant["scores"]["treatment_ok"] is True


# --------------------------------------------------------------------------- #
# Kuyruk sırası
# --------------------------------------------------------------------------- #
def test_by_priority_runs_high_before_low() -> None:
    """Paylaşımlı havuzda sıra sonucu belirler.

    NVIDIA'nın 1.000 kredisi biterse kuyruğun sonundaki modeller hiç koşmaz;
    kredinin asıl adaylara gitmesi gerekir. models.json `priority` alanını
    "koşu kuyruğu sırası" diye belgeliyor — bu test o sözü tutturur.
    """
    models = [
        {"id": "dusuk", "priority": "low"},
        {"id": "orta", "priority": "normal"},
        {"id": "yuksek", "priority": "high"},
        {"id": "belirtilmemis"},
    ]

    assert [m["id"] for m in by_priority(models)] == [
        "yuksek",
        "orta",
        "belirtilmemis",  # varsayılan normal
        "dusuk",
    ]


def test_by_priority_keeps_file_order_within_a_tier() -> None:
    """Eşit öncelikte dosya sırası korunmalı: matris tekrar koşulduğunda aynı sıra."""
    models = [{"id": "a", "priority": "high"}, {"id": "b", "priority": "high"}]

    assert [m["id"] for m in by_priority(models)] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Preflight ve model sabitleme
# --------------------------------------------------------------------------- #
_GEMINI = {"provider": "google", "id": "gemma-4-31b-it"}


def test_pinned_model_sets_and_restores_env() -> None:
    """Sızıntı olursa bir sonraki model yanlış ID ile ölçülür ve rapor sessizce yanlış olur."""
    before = dict(os.environ)

    with pinned_model(_GEMINI):
        assert os.environ["PARETO_JUDGE_PROVIDER"] == "google"
        assert os.environ["GEMINI_JUDGE_MODEL"] == "gemma-4-31b-it"

    assert os.environ.get("PARETO_JUDGE_PROVIDER") == before.get("PARETO_JUDGE_PROVIDER")
    assert os.environ.get("GEMINI_JUDGE_MODEL") == before.get("GEMINI_JUDGE_MODEL")


def test_pinned_model_rejects_unregistered_provider() -> None:
    """models.json'a kayıtsız bir sağlayıcı yazılırsa koşu ortasında değil, hemen patlasın."""
    with pytest.raises(SystemExit):
        with pinned_model({"provider": "cerebras", "id": "gpt-oss-120b"}):
            pass


def test_preflight_reports_ok_for_responsive_endpoint() -> None:
    output = {"ozet": "12 koşunun 9'u pozitif.", "eksen_yorumlari": []}

    with use_test_model(TestModel(custom_output_args=output)):
        row = preflight_one(_GEMINI)

    assert row["status"] == "ok"
    assert row["model"] == "google/gemma-4-31b-it"


def test_preflight_eliminates_with_a_reason_not_just_a_flag(monkeypatch) -> None:
    """Elenen model 'hata' diye tek torbaya atılırsa aday listesi yanlış nedenle daralır.

    'model_yok' katalog yanlış demektir (listeden çıkar), 'anahtar_yok' ops işidir
    (model masum). İkisi aynı satıra yazılırsa okuyan yanlış kararı verir.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Error code: 404 - model 'x' does not exist")

    monkeypatch.setattr(bench, "generate_narrative", _boom)
    row = preflight_one(_GEMINI)

    assert row["status"] == "elendi"
    assert row["detail"].startswith("model_yok:")


def test_require_cache_disabled_refuses_to_run_with_cache_on(monkeypatch) -> None:
    """Cache açıkken ölçüm anlamsız; sessizce yanlış sayı üretmektense durmalı."""
    monkeypatch.delenv("PARETO_LLM_CACHE", raising=False)
    with pytest.raises(SystemExit):
        require_cache_disabled()

    monkeypatch.setenv("PARETO_LLM_CACHE", "0")
    require_cache_disabled()  # patlamamalı


# --------------------------------------------------------------------------- #
# Kota günü, kota uzantısı (fallback_id) ve devre kesici
# --------------------------------------------------------------------------- #
def _flash_pair() -> list[dict[str, Any]]:
    """Sevk edilen kontrol grubu çifti: 3.6-flash + kota uzantısı 3.5-flash."""
    return by_priority(
        [m for m in load_models() if m["id"] in {"gemini-3.6-flash", "gemini-3.5-flash"}]
    )


def _run_days(
    monkeypatch,
    out_dir: Path,
    days: list[str],
    *,
    models: list[dict[str, Any]] | None = None,
    dead_ids: frozenset[str] = frozenset(),
    repeats: int = 3,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Aynı `--out` ile verilen takvim günlerinde run_matrix koşar (ağ yok).

    `_call_and_score` taklit edilir: ölçülen şey puanlayıcı değil, KOTA/RESUME
    kararları — gerçek çağrı yapmak bu kararları gizlerdi.

    `repeats` bilerek `bench.DEFAULT_REPEATS` DEĞİL: buradaki testlerin konusu gün
    geçişi ve kota uzantısı, yani matrisin bir günlük kotayı AŞMASI gerekiyor.
    Sevk varsayılanı (2) 16 vakayla 32 çağrı eder ve flash çiftinin birleşik
    40/gün kotasına tek günde sığar — o zaman gün geçişi hiç tetiklenmez ve bu
    testler sessizce hiçbir şey ölçmez hale gelir.
    """
    models = models if models is not None else _flash_pair()
    clock = {"day": days[0]}
    monkeypatch.setattr(bench, "local_date", lambda: clock["day"])

    def fake_call(model, task, case, *, rid, runner, extra, served=None):
        served = served or model
        failed = model["id"] in dead_ids
        return {
            "result_id": rid,
            "model": bench.model_key(model),
            "provider": model["provider"],
            "model_id": model["id"],
            "served_by": served["id"],
            "date": bench.local_date(),
            "task": task,
            "case_id": case["case_id"],
            "schema_ok": not failed,
            "validator_passed": not failed,
            "error": "timeout" if failed else None,
            "scores": None if failed else {},
            "latency_s": 0.1,
            "requests": 1,
            **extra,
        }

    monkeypatch.setattr(bench, "_call_and_score", fake_call)
    fresh: list[int] = []
    for day in days:
        clock["day"] = day
        throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None, on_wait=lambda _p, _s: None)
        fresh.append(len(run_matrix(models, bench.TASKS, repeats, out_dir, throttle)))
    return fresh, read_rows(out_dir / "results.jsonl")


def test_daily_quota_resumes_the_next_day_instead_of_stalling(monkeypatch, tmp_path) -> None:
    """Günlük kota ertesi gün YENİLENİR; koşu kaldığı yerden devam etmeli.

    Bu davranış kırıldığında koşu ilk günden sonra sessizce 0 çağrı yapıyor ve
    `report.md` yarım bir matrisi final okuma gibi gösteriyordu — kota sayacı
    dünkü çağrıları bugünün tavanından düştüğü için (bkz. seed_from_prior).
    Testin asıl iddiası: Gün 1'de yeni çağrı sayısı SIFIR OLMAMALI.
    """
    fresh, rows = _run_days(monkeypatch, tmp_path, ["2026-07-30", "2026-07-31"])

    assert fresh[1] > 0, "ertesi gün hiç çağrı yapılmadı: günlük kota yenilenmemiş sayılıyor"
    expected = sum(len(load_gold(t)) for t in bench.TASKS) * 3
    assert len(rows) == expected
    assert len({r["result_id"] for r in rows}) == expected, "aynı çağrı iki kez kaydedilmiş"


def test_same_day_rerun_does_not_spend_the_quota_twice(monkeypatch, tmp_path) -> None:
    """Aynı gün ikinci koşu yeni çağrı YAPMAMALI.

    Çökme sonrası yeniden başlatma sık: bugünün çağrıları bugünün kotasından
    gitti, tekrar sayılmazsa sağlayıcıdan gerçek 429 gelir ve o hatalar
    results.jsonl'e model kusuru gibi yazılır.
    """
    fresh, _rows = _run_days(monkeypatch, tmp_path, ["2026-07-30", "2026-07-30"])

    assert fresh[0] > 0
    assert fresh[1] == 0


def test_case_repeats_never_split_across_two_endpoints(monkeypatch, tmp_path) -> None:
    """Bir vakanın TÜM tekrarları aynı uçtan gitmeli.

    `repeats` en iç döngü olduğu için kota sınırı vaka ortasına düşebilir. Düşerse
    aynı vakanın tekrarları iki farklı modele dağılır ve `answer_consistency`
    model-İÇİ tutarlılık değil iki-model-uyuşması ölçmeye başlar — metrik sessizce
    başka bir şeyi ölçer.
    """
    _fresh, rows = _run_days(monkeypatch, tmp_path, ["2026-07-30", "2026-07-31"])

    endpoints: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row["model"], row["task"], row["case_id"])
        endpoints.setdefault(key, set()).add(row["served_by"])
    split = {k: v for k, v in endpoints.items() if len(v) > 1}

    assert not split, f"vaka ortasında uç değişmiş: {split}"


def test_quota_extension_rows_keep_the_primary_identity(monkeypatch, tmp_path) -> None:
    """Kota uzantısından gelen satırlar BİRİNCİL modelin başlığı altında toplanır.

    Amaç kotayı genişletmek, iki model ölçmek değil (models.json: fallback_id).
    Kimlik uzantıya kayarsa kontrol grubunun n'i ikiye bölünür ve rapor iki yarım
    tablo gösterir. `served_by` ise kaybolmamalı: rapor hangi satırın nereden
    geldiğini söyleyebilmek zorunda (bkz. _served_by_lines).
    """
    _fresh, rows = _run_days(monkeypatch, tmp_path, ["2026-07-30", "2026-07-31"])

    assert {r["model"] for r in rows} == {"google/gemini-3.6-flash"}
    assert {r["served_by"] for r in rows} == {"gemini-3.6-flash", "gemini-3.5-flash"}
    assert "gemini-3.5-flash" in "\n".join(bench._served_by_lines(rows))


def test_fallback_only_model_is_never_its_own_subject(monkeypatch, tmp_path) -> None:
    """`fallback_only` model kendi başına ÖLÇÜLMEZ.

    Ölçülürse kendi 48 çağrısını ister; kazanılan kota kadar iş eklendiği için
    koşu süresi hiç kısalmaz — kota uzantısının tüm amacı boşa gider.
    """
    _fresh, rows = _run_days(monkeypatch, tmp_path, ["2026-07-30", "2026-07-31"])

    assert not [r for r in rows if r["model"] == "google/gemini-3.5-flash"]


def test_schema_retries_are_charged_to_the_quota_pool(monkeypatch, tmp_path) -> None:
    """Şema retry'ı sağlayıcıya AYRI bir istek gider; tavandan düşmezse kota aşılır.

    `acquire` hücre başına bir kez çağrılıyor, ama pydantic-ai'nin retry'ı modele
    yeni bir istek yolluyor ve sağlayıcının sayacından düşüyor. Fark işlenmezse
    koşucu 8 hakkı kaldığını sanırken sağlayıcı 429 döndürür ve o hatalar
    results.jsonl'e MODEL KUSURU gibi yazılır. Ölçülen sevk matrisinde çarpan
    bugün 1.00, ama bütçe şansa değil doğruya dayanmalı.
    """
    model = {"id": "iki-istekli", "provider": "google", "rpm": None, "rpd": 8, "priority": "high"}

    def fake_call(m, task, case, *, rid, runner, extra, served=None):
        return {
            "result_id": rid,
            "model": bench.model_key(m),
            "provider": m["provider"],
            "model_id": m["id"],
            "served_by": (served or m)["id"],
            "date": bench.local_date(),
            "task": task,
            "case_id": case["case_id"],
            "error": None,
            "scores": {},
            "latency_s": 0.1,
            "requests": 2,  # 1 asıl + 1 şema retry
            **extra,
        }

    monkeypatch.setattr(bench, "_call_and_score", fake_call)
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None, on_wait=lambda _p, _s: None)
    rows = run_matrix([model], ("narrative",), 2, tmp_path, throttle)

    # 4 vaka x 2 tekrar = 8 hücre isterdi; her hücre 2 istek yaktığı için tavan
    # (rpd 8) yarısında dolar.
    assert len(rows) == 4
    assert throttle.used(bench.quota_pool(model)) == 8


def test_ship_matrix_keeps_the_quota_extension_of_a_shipped_model() -> None:
    """`--ship` uzantıyı da almalı, yoksa kota tavanı yarı görünür.

    `gemini-3.5-flash` `fallback_only`: kendi satırlarını üretmez ama tavanı
    birincilin havuzuna eklenir (bkz. plan_lines). Listeden düşerse `--dry-run`
    gemini havuzunu 40 değil 20 sanar ve teslim matrisi sığmıyor gibi görünür.
    """
    shipped = bench.ship_matrix(load_models())
    ids = {m["id"] for m in shipped}

    assert {"gemini-3.6-flash", "gemma-4-31b-it"} <= ids
    assert "gemini-3.5-flash" in ids, "kota uzantısı düşmüş"
    assert all(m.get("ship") or m.get("fallback_only") for m in shipped)


def test_ship_matrix_calls_fit_one_day_of_the_narrowest_pool() -> None:
    """Teslim matrisi TEK GÜNDE bitmeli: sevk edilen ayarlarla en dar havuz 1 gün.

    Bu testin varlık sebebi bir kaza: tekrar sayısı ya da vaka sayısı büyürse
    (16x3=48 > 40) koşu sessizce ikinci güne taşar ve "tek çalıştırmada sonuç"
    vaadi ölür. Sayı değil KOŞUL bağlanıyor.
    """
    shipped = bench.ship_matrix(load_models())
    per_model = sum(len(load_gold(t)) for t in bench.TASKS) * bench.DEFAULT_REPEATS

    for model in shipped:
        if model.get("fallback_only"):
            continue  # kendi çağrısı yok; tavanı aşağıda birincile eklenir
        cap = quota_cap(model)
        extender = next(
            (m for m in shipped if m["id"] == str(model.get("fallback_id") or "")), None
        )
        extra_cap = quota_cap(extender) if extender is not None else None
        if cap is not None and extra_cap is not None:
            cap += extra_cap
        assert cap is None or per_model <= cap, (
            f"{model['id']}: {per_model} çağrı {cap} tavanına sığmıyor — "
            "koşu ikinci güne taşar ve 'tek çalıştırmada sonuç' vaadi ölür"
        )


def test_plan_lines_minutes_reflect_the_token_limit() -> None:
    """Dakika sütunu TPM'i görmeli; görmezse --dry-run throttle'ı doğrulayamaz.

    Gün sütunu gemma'da her hâlükârda 1 çıkar (rpd 14.400), yani TPM throttle'ının
    olup olmadığını ayırt etmez. Süre tahmini token aralığından gelmezse
    `--dry-run` "her şey yolunda" der ve koşu 429'a çarpar.
    """
    gemma = [m for m in load_models() if m["id"] == "gemma-4-31b-it"]
    lines = plan_lines(gemma, ("cleaning",), 2)
    pool_row = next(line for line in lines if line.startswith("google/gemma-4-31b-it"))
    minutes = int(pool_row.split()[-1])

    calls = len(load_gold("cleaning")) * 2
    by_tpm = calls * Throttle.interval(rpm=30, tpm=16_000, est_tokens=7_000) / 60.0
    by_rpm = calls * (60.0 / 30) / 60.0

    assert minutes == round(by_tpm), "dakika sütunu token aralığından gelmiyor"
    assert round(by_rpm) == 0, "rpm'e göre hesaplansaydı 0 dk çıkardı — test ayırt etmiyor"


def _lifetime_credit_trio(dead_id: str) -> list[dict[str, Any]]:
    """Ömür-boyu (yenilenmeyen) kredi paylaşan, biri ölü 3 sentetik uç.

    Gerçek bir sağlayıcıya bağlı değil — yalnız `budget`/`pool` alanlarının
    şekli önemli (bkz. eski `nvidia:hesap` havuzu, artık matriste yok).
    """
    return [
        {
            "id": "lifetime-a",
            "provider": "test",
            "pool": "lifetime:test",
            "rpm": 40,
            "budget": 1000,
        },
        {"id": dead_id, "provider": "test", "pool": "lifetime:test", "rpm": 40, "budget": 1000},
        {
            "id": "lifetime-c",
            "provider": "test",
            "pool": "lifetime:test",
            "rpm": 40,
            "budget": 1000,
        },
    ]


def test_circuit_breaker_skips_the_rest_of_a_dead_model(monkeypatch, tmp_path) -> None:
    """Ölü uç her görevde en fazla `CIRCUIT_BREAK_ERRORS` çağrı harcar; sağlamlar tam koşar.

    Ömür-boyu (yenilenmeyen) kredi paylaşan bir havuzda ölü bir uç devre kesici
    olmadan tüm koşu boyunca kredi ve saat yakar. Eleme görev bazlı olduğu için
    tavan 2 değil 2×görev sayısı — karşılığında modelin hangi görevde boğulduğunu
    öğreniyoruz.
    """
    dead = "lifetime-b"
    trio = _lifetime_credit_trio(dead)
    _fresh, rows = _run_days(
        monkeypatch, tmp_path, ["2026-07-30"], models=trio, dead_ids=frozenset({dead})
    )

    per_model: dict[str, int] = {}
    for row in rows:
        per_model[row["model_id"]] = per_model.get(row["model_id"], 0) + 1
    full = sum(len(load_gold(t)) for t in bench.TASKS) * 3

    assert per_model[dead] == bench.CIRCUIT_BREAK_ERRORS * len(bench.TASKS)
    assert [r for r in rows if r.get("circuit_broken")], "eleme rapora iz bırakmamış"
    assert dead in "\n".join(bench._circuit_break_lines(rows))
    for model in trio:
        if model["id"] != dead:
            assert per_model[model["id"]] == full, "sağlam model de kesilmiş"


def test_seed_counts_prior_days_only_for_lifetime_credit_pools() -> None:
    """Dünkü ömür-boyu kredi geri gelmez, dünkü Gemini isteği bugünün 20'sinden düşmez.

    `quota_cap()` ikisini de tek sayıya indiriyor; ayrım yapılmazsa günlük havuzlar
    ömür-boyu tavan gibi davranır ve çok günlü koşu ilk günden sonra durur.
    """
    daily = {"id": "d", "provider": "google", "rpm": 5, "rpd": 20}
    lifetime = {"id": "l", "provider": "test", "pool": "lifetime:test", "rpm": 40, "budget": 1000}
    prior = [
        {"model_id": "d", "served_by": "d", "date": "2026-07-29"},
        {"model_id": "d", "served_by": "d", "date": "2026-07-30"},
        {"model_id": "l", "served_by": "l", "date": "2026-07-29"},
    ]
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None)

    bench.seed_from_prior(throttle, prior, {"d": daily, "l": lifetime}, "2026-07-30")

    assert throttle.used(quota_pool(daily)) == 1
    assert throttle.used(quota_pool(lifetime)) == 1


def test_case_server_leaves_a_half_done_case_to_its_original_endpoint() -> None:
    """Yarım kalmış vaka fallback'e KAYDIRILMAZ; ertesi güne bırakılır.

    Kaydırılırsa vaka saflığı günler arasında kırılır (bkz.
    test_case_repeats_never_split_across_two_endpoints) — aynı vakanın dünkü
    tekrarları bir uçtan, bugünkü tekrarları başka uçtan gelir.
    """
    primary = {"id": "p", "provider": "google", "rpm": None, "rpd": 20, "fallback_id": "f"}
    extender = {"id": "f", "provider": "google", "rpm": None, "rpd": 20, "fallback_only": True}
    lookup = {"p": primary, "f": extender}
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    throttle.seed(quota_pool(primary), 19)  # birincilde 1 hak kaldı, vaka 3 istiyor

    picked = bench.case_server(throttle, primary, needed=3, pinned_id=None, lookup=lookup)
    pinned = bench.case_server(throttle, primary, needed=3, pinned_id="p", lookup=lookup)

    assert picked is not None and picked["id"] == "f"
    assert pinned is None


def test_case_server_reserves_a_retry_margin_before_filling_a_pool() -> None:
    """Vaka SIFIR payla onaylanmaz: retry `acquire`'dan sonra gelir, tavanı taşırır.

    Ölçülen kaza (simülasyon, 2026-07-31): gemini-3.6-flash tavan 20, vaka başına
    2 çağrı -> 10. vaka tam `remaining == 2` ile onaylanıyordu. O vakanın ilk
    hücresinde tek bir şema retry'ı sayacı 20'ye çıkarıyor ve ikinci hücrenin
    `acquire`'ı QuotaExhausted atıyordu. Pay, vakayı bir erken devrettirir.
    """
    primary = {"id": "p", "provider": "google", "rpm": None, "rpd": 20, "fallback_id": "f"}
    extender = {"id": "f", "provider": "google", "rpm": None, "rpd": 20, "fallback_only": True}
    lookup = {"p": primary, "f": extender}
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    # Birincide tam `needed` kadar hak var ama pay yok: paysız seçim burayı
    # onaylardı ve tek retry koşuyu düşürürdü.
    throttle.seed(quota_pool(primary), 20 - 2)

    picked = bench.case_server(throttle, primary, needed=2, pinned_id=None, lookup=lookup)

    assert picked is not None and picked["id"] == "f", (
        "birincide pay kalmadı; vaka uzantıya devredilmeliydi"
    )


def test_case_server_spends_the_last_calls_when_no_endpoint_has_margin() -> None:
    """Pay ZORUNLU değil tercihli: hiçbir uçta pay yoksa havuzun artığı kullanılır.

    Payı zorunlu kılmak dar havuzlarda her koşuda `RETRY_MARGIN` kadar hakkı
    kalıcı olarak çöpe atardı (gemini'de gün başına 3 çağrı). Vaka ortasında
    dolma riskini `run_matrix`'teki QuotaExhausted yakalaması karşılıyor —
    bkz. test_mid_case_quota_exhaustion_does_not_kill_the_whole_run.
    """
    solo = {"id": "p", "provider": "google", "rpm": None, "rpd": 20}
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    throttle.seed(quota_pool(solo), 20 - 2)  # tam 2 hak: pay yok, uzantı da yok

    picked = bench.case_server(throttle, solo, needed=2, pinned_id=None, lookup={"p": solo})

    assert picked is not None and picked["id"] == "p"
    assert bench.case_server(throttle, solo, needed=3, pinned_id=None, lookup={"p": solo}) is None


def test_mid_case_quota_exhaustion_does_not_kill_the_whole_run(monkeypatch, tmp_path) -> None:
    """Vaka ortasında dolan kota koşuyu ÖLDÜRMEZ; sıradaki model koşmaya devam eder.

    Ölçülen kaza (simülasyon, 2026-07-31): teslim koşusunda gemini'nin 19.
    hücresindeki tek retry yakalanmamış QuotaExhausted üretiyor, `main()` bunu
    yakalamadığı için gemma ve inkling HİÇ koşmadan traceback'e düşülüyordu —
    yani tek bir retry, üç modelin tamamının ölçümünü siliyordu. Doğru davranış
    sağlayıcının 429'uyla aynı: o modeli durdur, diğerleri sürsün.
    """
    # Tavan 2 = tam bir vaka. İlk hücrenin retry'ı sayacı 2'ye çıkarır, İKİNCİ
    # hücrenin `acquire`'ı vaka ortasında QuotaExhausted atar — `case_server`'ın
    # None döndüğü zarif yol DEĞİL, tam olarak yakalanması gereken yol.
    dar = {"id": "dar", "provider": "google", "rpm": None, "rpd": 2, "priority": "high"}
    sonraki = {"id": "sonraki", "provider": "groq", "rpm": None, "rpd": 100, "priority": "normal"}

    def fake_call(m, task, case, *, rid, runner, extra, served=None):
        return {
            "result_id": rid,
            "model": bench.model_key(m),
            "provider": m["provider"],
            "model_id": m["id"],
            "served_by": (served or m)["id"],
            "date": bench.local_date(),
            "task": task,
            "case_id": case["case_id"],
            "error": None,
            "scores": {},
            "latency_s": 0.1,
            "requests": 2 if m["id"] == "dar" else 1,
            **extra,
        }

    monkeypatch.setattr(bench, "_call_and_score", fake_call)
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None, on_wait=lambda _p, _s: None)

    rows = run_matrix([dar, sonraki], ("narrative",), 2, tmp_path, throttle)

    assert len([r for r in rows if r["model_id"] == "dar"]) == 1, (
        "dar model kota dolmadan önceki tek satırını yazmalıydı"
    )
    assert len([r for r in rows if r["model_id"] == "sonraki"]) == 8, (  # 4 vaka x 2 tekrar
        "dar modelin kotası vaka ortasında dolunca koşu ölmüş — sıradaki model hiç koşmadı"
    )


# --------------------------------------------------------------------------- #
# Hata politikası: neyi ölçüm sayıyoruz, neyi elemeye sayıyoruz
# --------------------------------------------------------------------------- #
def test_fast_provider_errors_do_not_eliminate_the_model() -> None:
    """429/503 sağlayıcı kaynaklı; model masum ve maliyeti ~0,5sn.

    Canlı kanıt (2026-07-30): gemma-4-31b-it 3 başarılı çağrıdan sonra 429 yedi,
    dakikalar sonra aynı uç HTTP 200 döndü. Bunları elemeye saymak sağlam bir adayı
    ops sebebiyle listeden düşürür — classify_error'ın 'model masum' ayrımının
    devre kesiciye de uygulanması gerekiyor.
    """
    for kind in ("kota", "sunucu", "yetki", "anahtar_yok", "model_yok"):
        assert bench.counts_toward_elimination(kind) is False, kind


def test_schema_failures_do_not_eliminate_the_model() -> None:
    """Şema/doğrulayıcı hataları benchmark'ın ÖLÇTÜĞÜ şey.

    Eleyerek susturursak benchmark kendi sorusunu ('bu model şemayı tutturuyor mu')
    cevaplayamaz: 2 ardışık şema hatası tam olarak kaydedilmesi gereken sinyal.
    """
    for kind in ("sema_tutmadi", "dogrulayici_reddetti"):
        assert bench.counts_toward_elimination(kind) is False, kind
    assert bench.counts_toward_elimination("hata:TimeoutError") is True


def test_provider_errors_are_retried_but_measurements_are_not() -> None:
    """Geçici 429/503 matris hücresini kalıcı olarak zehirlememeli.

    Hücre 'yapıldı' sayılırsa o model o vakayı BİR DAHA hiç koşmaz ve rapor eksik
    n'i sessizce final okuma gibi gösterir. Modelin kendi kusurları ise tekrar
    DENENMEMELİ — ikinci çekilişte geçmesi ölçümü yumuşatır.
    """
    retryable: tuple[str | None, ...] = ("kota", "sunucu", "yetki", "anahtar_yok", "model_yok")
    measured: tuple[str | None, ...] = (
        None,
        "sema_tutmadi",
        "dogrulayici_reddetti",
        "hata:TimeoutError",
    )
    for kind in retryable:
        assert bench.is_retryable_error(kind) is True, kind
    for kind in measured:
        assert bench.is_retryable_error(kind) is False, kind


def test_slow_failures_count_even_when_the_class_is_infra() -> None:
    """504 @ 181sn 'altyapı' etiketli ama en pahalı sınıf: 48 çağrı × 180sn ≈ 2,4 saat.

    Devre kesicinin ölçütü hata etiketi değil MALİYET olmak zorunda; yoksa yavaş
    başarısızlık veren bir uç etiketinin arkasına saklanıp saatleri yakar.
    """
    assert bench.elimination_signal("sunucu", 181.0) is True
    assert bench.elimination_signal("sunucu", 0.5) is False


def test_slow_schema_failures_never_count_no_matter_how_long_they_take() -> None:
    """Yavaş bir ŞEMA hatası elemeye sayılmaz — yavaşlığın sebebi ölçülen kusurun kendisi.

    Canlı kanıt (2026-07-31): gpt-oss-120b `cleaning`'de 9/9 `sema_tutmadi` verdi,
    bazıları 72-90sn sürdü çünkü pydantic-ai şemayı yeniden denedi
    (`requests=2..4`). Bunları "pahalı hata" sayan ilk sürüm, "şema hataları
    ölçümdür" kuralını arka kapıdan iptal edip görevi haksız yere kesti.
    """
    for latency in (0.5, 72.5, 90.0, 600.0):
        assert bench.elimination_signal("sema_tutmadi", latency) is False, latency
        assert bench.elimination_signal("dogrulayici_reddetti", latency) is False, latency


def test_transport_errors_count_without_a_latency_gate() -> None:
    """`hata:*` sınıfında gecikme kapısı YOK: hızlı başarısızlık da kredi yakar.

    `nvidia:hesap` ömür-boyu 1.000 kredi. Hızlı hata veren bir uçta 2 çağrıda
    durmak ile 48 çağrıda durmak 46 kredi fark eder — maliyet yalnız süre değil.
    """
    assert bench.elimination_signal("hata:TimeoutError", 0.1) is True
    assert bench.elimination_signal("hata:ConnectError", 900.0) is True


def test_streak_resets_when_the_endpoint_produced_a_model_response() -> None:
    """Sıfırlama ölçütü: uç GERÇEKTEN bir cevap üretti mi.

    Başarı ve ölçüm hatası ikisi de üretti → sayaç sıfırlanır. Sıfırlamayı atlamak
    "ardışık" kelimesini yalan yapıyordu: araya 3 ölçüm hatası girmiş iki yavaş
    hata "2 ardışık" sayılıp gpt-oss-120b'nin cleaning görevini kesti (2026-07-31).
    Hızlı altyapı hatası ne sayar ne sıfırlar — iki timeout arasındaki 429 seriyi
    bozmamalı.
    """
    assert bench.resets_streak(None) is True
    assert bench.resets_streak("sema_tutmadi") is True
    assert bench.resets_streak("dogrulayici_reddetti") is True
    assert bench.resets_streak("kota") is False
    assert bench.resets_streak("hata:TimeoutError") is False


def test_interleaved_measurement_errors_prevent_a_false_elimination(monkeypatch, tmp_path) -> None:
    """gpt-oss-120b senaryosunun uçtan uca hali: kesilmemeli.

    Sıra: yavaş şema hatası (72sn) → 3 ölçüm hatası → yavaş şema hatası (90sn).
    Eski mantık bunu "2 ardışık pahalı hata" sayıp görevi kesiyordu; hiçbiri
    altyapı hatası olmadığı için hiçbiri sayılmamalı ve görev tam koşmalı.
    """
    model = {"id": "openai/gpt-oss-120b", "provider": "groq", "rpm": None, "rpd": None}
    # (hata sınıfı, gecikme): iki yavaş SAYAN hatanın arasına ölçüm hatası giriyor.
    # Doğru davranış: ölçüm hatası sayacı sıfırlar, ikinci 504 "2. ardışık" olmaz.
    script = [
        ("sunucu", 181.0),  # sayar   -> streak 1
        ("sema_tutmadi", 72.5),  # sıfırlar -> streak 0 (uç cevap üretti)
        ("sema_tutmadi", 90.0),  # sıfırlar -> streak 0
        ("sunucu", 181.0),  # sayar   -> streak 1, KESMEMELİ
    ]
    outcomes = iter(script + [(None, 30.0)] * 50)

    def fake_call(m, task, case, *, rid, runner, extra, served=None):
        kind, latency = next(outcomes)
        return {
            "result_id": rid,
            "model": bench.model_key(m),
            "provider": m["provider"],
            "model_id": m["id"],
            "served_by": (served or m)["id"],
            "date": bench.local_date(),
            "task": task,
            "case_id": case["case_id"],
            "schema_ok": kind is None,
            "validator_passed": kind is None,
            "error": kind,
            "scores": {} if kind is None else None,
            "latency_s": latency,
            "requests": 2,
            **extra,
        }

    monkeypatch.setattr(bench, "_call_and_score", fake_call)
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None, on_wait=lambda _p, _s: None)
    rows = run_matrix([model], ("cleaning",), 3, tmp_path, throttle)

    assert not [r for r in rows if r.get("circuit_broken")], "görev haksız yere kesilmiş"
    # `sunucu` satırları retry edilebilir olduğu için toplam satır sayısı 12'yi aşabilir;
    # asıl iddia görevin KESİLMEMESİ, yani 4 vakanın hepsine dokunulması.
    assert {r["case_id"] for r in rows} == {c["case_id"] for c in load_gold("cleaning")}


def test_circuit_breaker_is_scoped_to_one_task_not_the_whole_model(monkeypatch, tmp_path) -> None:
    """Bir görevde boğulan model diğer görevlerde ölçülmeye devam etmeli.

    gemma-4-31b-it `cleaning/medicaid`'i 3/3 geçip `cleaning/card_krueger`'da 504
    aldı. Modeli tümden elemek onun estimand/spec_menu/narrative performansını da
    silerdi — oysa 'şu görevde boğuluyor' rapora girmesi gereken bulgu.
    """
    model = {"id": "gemma-4-31b-it", "provider": "google", "rpm": None, "rpd": None}
    slow_task = "cleaning"

    def fake_call(m, task, case, *, rid, runner, extra, served=None):
        failed = task == slow_task
        return {
            "result_id": rid,
            "model": bench.model_key(m),
            "provider": m["provider"],
            "model_id": m["id"],
            "served_by": (served or m)["id"],
            "date": bench.local_date(),
            "task": task,
            "case_id": case["case_id"],
            "schema_ok": not failed,
            "validator_passed": not failed,
            "error": "sunucu" if failed else None,
            "scores": None if failed else {},
            "latency_s": 181.0 if failed else 1.0,
            "requests": 1,
            **extra,
        }

    monkeypatch.setattr(bench, "_call_and_score", fake_call)
    throttle = Throttle(now=lambda: 0.0, sleep=lambda _s: None, on_wait=lambda _p, _s: None)
    rows = run_matrix([model], bench.TASKS, 1, tmp_path, throttle)

    per_task: dict[str, int] = {}
    for row in rows:
        per_task[row["task"]] = per_task.get(row["task"], 0) + 1

    assert per_task[slow_task] == bench.CIRCUIT_BREAK_ERRORS
    for task in bench.TASKS:
        if task != slow_task:
            assert per_task[task] == len(load_gold(task)), f"{task} de kesilmiş"


def test_report_counts_a_retried_cell_once(tmp_path) -> None:
    """Tekrar denenen hücrenin iki satırı `n`'i şişirmemeli.

    429 yemiş bir çağrı + sonraki koşudaki başarılı çağrı aynı `result_id`'yi
    taşır. İkisini de saymak başarı oranını olduğundan düşük gösterir: sağlayıcı
    kaynaklı bir ret, modelin başarısızlığı gibi okunur.
    """
    rid = "google/x|cleaning|medicaid|0"
    base: dict[str, Any] = {
        "model": "google/x",
        "task": "cleaning",
        "case_id": "medicaid",
        "result_id": rid,
    }
    rows: list[dict[str, Any]] = [
        {**base, "error": "kota", "schema_ok": None, "validator_passed": False, "scores": None},
        {**base, "error": None, "schema_ok": True, "validator_passed": True, "scores": {}},
    ]

    assert len(bench.latest_per_cell(rows)) == 1
    assert bench.latest_per_cell(rows)[0]["error"] is None
    assert "| 1 | 1/1 |" in build_report(rows, ("cleaning",))


def test_unparseable_output_is_a_schema_failure_whatever_the_transport() -> None:
    """Ayrıştırılamayan structured output ŞEMA kusurudur, taşıyıcısı ne olursa olsun.

    Groq bunu HTTP 400 ile döndürüyor (canlı: gpt-oss-20b, 2026-07-31), pydantic-ai
    retry tükenmesiyle. Ayırmazsak `hata:*` sınıfına düşüyor ve iki yanlış sonuç
    doğuruyor: `schema_ok=True` (başarısızlık BAŞARI sayılır, birincil metrik bozulur)
    ve elemeye koşulsuz sayılma (şema hatası transport gibi davranır).
    """

    class ModelHTTPError(RuntimeError):
        pass

    exc = ModelHTTPError(
        "status_code: 400, model_name: openai/gpt-oss-20b, body: {'error': {'message': "
        '"Parsing failed. The model generated output that could not be parsed. Please '
        "adjust your prompt. See 'failed_generation' for more details.\", "
        "'type': 'invalid_request_error'}}"
    )
    kind = classify_error(exc)

    assert kind == "sema_tutmadi"
    assert schema_verdict(kind) is False, "ayrıştırılamayan çıktı 'şema geçti' sayılamaz"
    assert bench.elimination_signal(kind, 53.3) is False, "şema kusuru elemeye sayılmamalı"


def test_real_infra_signals_still_win_over_parse_text() -> None:
    """Gövdesinde 'parsing failed' geçen bir 429 yine `kota` sayılmalı.

    Altyapı sinyali daha bağlayıcı bilgi: 429 gelmişse modelin çıktısı hiç
    değerlendirilmedi. Yeni parse kontrolü bu sırayı bozmamalı.
    """
    exc = RuntimeError("status_code: 429, body: {'message': 'quota exceeded, parsing failed'}")

    assert classify_error(exc) == "kota"


# --------------------------------------------------------------------------- #
# Koşu ortasında sınıflandırma düzeltildiğinde satırların onarımı
# --------------------------------------------------------------------------- #
def test_repair_rederives_class_and_drops_a_now_invalid_break_flag(tmp_path) -> None:
    """Yanlış sınıfla yazılmış satır onarılır; geçersiz kalan eleme bayrağı kalkar.

    Koşu sırasında `classify_error` düzeltilirse çalışan süreç eski sürümü yüklü
    tutar, yani kalan satırlar eski sınıfla yazılır. `error_detail` ham mesajı
    sakladığı için sınıf sonradan doğru türetilebilir — ama `circuit_broken`
    bayrağı da düşmek zorunda: yeni sınıf elemeye saymıyorsa bayrak var olmayan
    bir elemeyi raporlar.
    """
    from scripts.repair_error_classes import main as repair_main

    path = tmp_path / "results.jsonl"
    row = {
        "result_id": "groq/x|spec_menu|castle|0",
        "model": "groq/x",
        "model_id": "x",
        "task": "spec_menu",
        "case_id": "castle",
        "repeat": 0,
        "error": "hata:ModelHTTPError",
        "error_detail": "status_code: 400, body: {'code': 'output_parse_failed'}",
        "schema_ok": True,
        "latency_s": 61.6,
        "circuit_broken": True,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    repair_main([str(path), "--apply"])
    fixed = json.loads(path.read_text(encoding="utf-8").strip())

    assert fixed["error"] == "sema_tutmadi"
    assert fixed["schema_ok"] is False
    assert "circuit_broken" not in fixed
    assert fixed["reclassified"] is True


def test_repair_keeps_a_still_valid_break_flag_and_is_idempotent(tmp_path) -> None:
    """Yeni sınıf hâlâ elemeye sayıyorsa bayrak KORUNUR; ikinci koşu hiçbir şeyi değiştirmez.

    İlk yazdığım sürüm bayrağı koşulsuz siliyordu — geçerli bir elemeyi de silerdi.
    İdempotentlik ayrıca şart: onarım kazayla iki kez koşulabilir.
    """
    from scripts.repair_error_classes import main as repair_main

    path = tmp_path / "results.jsonl"
    row = {
        "result_id": "nvidia/y|cleaning|castle|0",
        "model": "nvidia/y",
        "model_id": "y",
        "task": "cleaning",
        "case_id": "castle",
        "repeat": 0,
        "error": "hata:ModelHTTPError",
        # 504 → `sunucu`, 181sn ile hâlâ pahalı: eleme geçerli kalır.
        "error_detail": "status_code: 504, body: 'Deadline expired'",
        "schema_ok": True,
        "latency_s": 181.0,
        "circuit_broken": True,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    repair_main([str(path), "--apply"])
    first = path.read_text(encoding="utf-8")
    fixed = json.loads(first.strip())

    assert fixed["error"] == "sunucu"
    assert fixed["circuit_broken"] is True, "geçerli eleme bayrağı silinmiş"

    repair_main([str(path), "--apply"])
    assert path.read_text(encoding="utf-8") == first, "onarım idempotent değil"


def test_capacity_refusal_is_its_own_class_permanent_and_not_a_schema_pass() -> None:
    """413 "Request too large" KALICI bir kapasite reddi: 429/503'ten farklı davranmalı.

    Canlı örnek (2026-07-31): llama-3.1-8b-instant cleaning payload'unu kabul
    etmiyor. Payload küçülmediği sürece her denemede aynı sonuç geliyor, yani:
    - tekrar DENENMEZ (`kota`/`sunucu` gibi geçici değil; her koşuda 2 çağrı yakardı)
    - koşulsuz elemeye SAYAR (gecikme kapısına takılsa 12 çağrının hepsi aynı 413'ü yer)
    - `schema_ok=None` — model isteği hiç görmedi, şema hakkında kanıt yok
    """
    exc = RuntimeError(
        "status_code: 413, body: {'error': {'message': 'Request too large for model "
        "`llama-3.1-8b-instant`, please reduce'}}"
    )
    kind = classify_error(exc)

    assert kind == "kapasite"
    assert schema_verdict(kind) is None
    assert bench.is_retryable_error(kind) is False, "kalıcı hata her koşuda tekrar denenmemeli"
    assert bench.elimination_signal(kind, 5.5) is True, "hızlı da olsa koşulsuz saymalı"
    assert bench.resets_streak(kind) is False


def test_schema_verdict_defaults_to_no_evidence_for_unreached_models() -> None:
    """Modele ULAŞILAMAMIŞ çağrılar "şema geçti" sayılamaz — varsayılan None olmalı.

    Eski sürüm kara liste kullanıyordu (bilinmeyen her sınıf True), bu yüzden 413 ve
    `hata:ConnectError` birincil metriği sessizce şişiriyordu. Beyaz liste, ileride
    eklenecek sınıflarda da güvenli tarafta kalır.

    `dogrulayici_reddetti` istisnası bilinçli: orada şema TUTTU, reddedilen içerik.
    """
    assert schema_verdict(None) is True
    assert schema_verdict("sema_tutmadi") is False
    assert schema_verdict("dogrulayici_reddetti") is True
    for kind in ("kapasite", "kota", "sunucu", "yetki", "hata:ConnectError", "gelecek_sinif"):
        assert schema_verdict(kind) is None, kind
