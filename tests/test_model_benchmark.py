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
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

import scripts.run_model_benchmark as bench
from pareto.analysis.hypothesis import TACProposal
from pareto.analysis.menu import ALL_AXES, SpecMenuAxis, SpecMenuProposal
from pareto.cleaning.ledger import LedgerEntry
from pareto.cleaning.merge import build_panel, load_dataset_config
from pareto.cleaning.transforms import REGISTRY
from pareto.llm.narrative import AxisComment, VarianceNarrative
from pareto.llm.router import use_test_model
from pareto.profiling import profile_dataframe
from scripts.run_model_benchmark import (
    GOLD_DIR,
    REPO_ROOT,
    Meter,
    QuotaExhausted,
    Throttle,
    _run_cleaning,
    _run_estimand,
    _run_narrative,
    build_report,
    by_priority,
    classify_error,
    fabricated_numbers,
    load_gold,
    load_models,
    looks_turkish,
    pinned_model,
    preflight_one,
    quota_cap,
    quota_pool,
    read_done,
    read_rows,
    require_cache_disabled,
    result_id,
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


# --------------------------------------------------------------------------- #
# Estimand puanlayıcısı
# --------------------------------------------------------------------------- #
def _tac(**overrides) -> TACProposal:
    base = dict(
        treatment="asgari ücret artışı",
        treatment_coding="treated_post",
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
    root = REPO_ROOT / case["dataset_dir"]
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


def test_throttle_seed_counts_resumed_calls_against_cap() -> None:
    """Resume kotayı sıfırlamamalı: dünkü çağrılar da aynı günün kotasından gitti."""
    clock = Throttle(now=lambda: 0.0, sleep=lambda _s: None)
    clock.seed("havuz", 5)

    with pytest.raises(QuotaExhausted):
        clock.acquire("havuz", rpm=None, cap=5)


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
        (OSError("NVIDIA_API_KEY tanımlı değil. Şunlardan biriyle ayarlayın:"), "anahtar_yok"),
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
        "treatment": "tek-taraflı boşanma",
        "treatment_coding": "post",
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
    profile = profile_dataframe(build_panel(REPO_ROOT / case["dataset_dir"]).df)
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
    for case in load_gold("cleaning"):
        profile = profile_dataframe(build_panel(REPO_ROOT / case["dataset_dir"]).df)
        for record in case["must_fix"]:
            if record["transform"] != "coerce_numeric":
                continue
            dtype = profile["columns"][record["column"]]["dtype"]
            assert dtype in ("object", "str"), (
                f"{case['case_id']}.{record['column']}: dtype={dtype} — "
                "artık metin değil, coerce_numeric MUST_FIX olmaktan çıkmış"
            )


@pytest.mark.parametrize("case", load_gold("estimand"), ids=lambda c: c["case_id"])
def test_estimand_gold_accepted_columns_exist(case: dict) -> None:
    columns = set(build_panel(REPO_ROOT / case["dataset_dir"]).df.columns)
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
    columns = set(build_panel(REPO_ROOT / case["dataset_dir"]).df.columns)

    for axis in [*case.get("baseline_must_be_in", {}), *case.get("indefensible_levels", {})]:
        assert axis in ALL_AXES, f"{case['case_id']}: bilinmeyen eksen {axis}"
    for column in case.get("bad_control_columns", {}):
        assert column in columns, f"{case['case_id']}: panelde olmayan bad control {column}"
    for column in case["baseline_must_be_in"].get("clustering", []):
        assert column in columns, f"{case['case_id']}: panelde olmayan cluster kolonu {column}"
    assert case["estimand"]["treatment_coding"] in columns
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

    assert "gemini-3.5-flash" in controls  # JUDGE_SLOT defaultu
    assert "llama-3.3-70b-versatile" in controls  # JUDGE_GROQ_SLOT defaultu


def test_models_json_shared_pools_are_declared() -> None:
    """Aynı hesap kotasını paylaşan uçlar tek havuzda toplanmalı."""
    models = load_models()
    nvidia = {quota_pool(m) for m in models if m["provider"] == "nvidia"}
    openrouter = {quota_pool(m) for m in models if m["provider"] == "openrouter"}
    google = {quota_pool(m) for m in models if m["provider"] == "google"}

    assert len(nvidia) == 1, "NVIDIA kredileri hesap seviyesinde ortak"
    assert len(openrouter) == 1, "OpenRouter :free kotası ortak havuz"
    assert len(google) == 5, "Google limitleri model başına, havuz paylaşılmamalı"


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


def test_shipped_matrix_puts_strongest_nvidia_candidates_first() -> None:
    """NVIDIA havuzu tek ve tükenebilir; kredinin nereye gittiği önem taşır."""
    nvidia = [m for m in by_priority(load_models()) if m["provider"] == "nvidia"]
    first_ids = {m["id"] for m in nvidia[:4]}

    assert "stepfun-ai/step-3.7-flash" not in first_ids
    assert first_ids & {"deepseek-ai/deepseek-v4-pro", "thinkingmachines/inkling"}


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
