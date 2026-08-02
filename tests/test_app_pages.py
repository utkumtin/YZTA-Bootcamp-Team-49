from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pareto.analysis.hypothesis import TACProposal, freeze_estimand
from pareto.analysis.variance import SIG_ROBUST, SIGN_FRAGILE, SIGN_ROBUST
from pareto.config import SETTINGS as _BASE_SETTINGS
from pareto.contracts import EstimationResult
from pareto.memory import store as _store_module
from pareto.memory.frozen_menu import build_frozen_menu_record, save_frozen_menu_record

PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "pages" / "3_variance_panel.py"


def _frozen_estimand():
    proposal = TACProposal(
        estimand_type="ATT",
        treatment="expanded",
        treatment_coding="1 = expanded state x post-expansion year, 0 = otherwise",
        outcome="uninsured_rate",
        outcome_unit="percentage points",
        population="US states",
        time_scope="2010-2020",
        expected_sign="negative",
        identification_assumption="parallel_trends",
        h0="ATT = 0",
        h1="ATT < 0",
        implied_result_translation="Lower uninsured rates in expansion states.",
        confirmation_question="Freeze?",
    )
    return freeze_estimand(proposal, approved=True)


def _results_file(tmp_path: Path) -> Path:
    results = [
        EstimationResult(
            spec_id="s1",
            estimator="OLS",
            coefficient=0.42,
            ci_low=0.21,
            ci_high=0.63,
            p_value=0.01,
            n_obs=10,
        ).model_dump()
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _panel_with_explicit_cohort() -> pd.DataFrame:
    rows = []
    cohorts = [2014] * 8 + [2015] * 8 + [pd.NA] * 8
    for unit, cohort in enumerate(cohorts):
        never = pd.isna(cohort)
        unit_fe = unit * 0.2
        for year in range(2011, 2019):
            year_fe = (year - 2011) * 0.1
            event_time = None if pd.isna(cohort) else year - int(cohort)
            effect = 0.0 if event_time is None or event_time < 0 else 0.5 + 0.1 * event_time
            rows.append(
                {
                    "unit": f"u{unit:02d}",
                    "year": year,
                    "cohort": cohort,
                    "never_treated": never,
                    "uninsured_rate": unit_fe + year_fe + effect,
                    "expanded": int(not never),
                    "weight": 1.0 + (unit % 3),
                }
            )
    return pd.DataFrame(rows)


def _panel_missing_cohort_columns() -> pd.DataFrame:
    rows = []
    for unit in range(8):
        for year in range(2011, 2019):
            rows.append(
                {
                    "unit": f"u{unit:02d}",
                    "year": year,
                    "uninsured_rate": unit * 0.2 + (year - 2011) * 0.1,
                    "expanded": unit % 2,
                }
            )
    return pd.DataFrame(rows)


def _load_variance_panel(app: AppTest, results_path: Path) -> None:
    app.run()
    app.text_input[0].set_value(str(results_path))
    app.run()


def _failed_results_file(tmp_path: Path) -> Path:
    """Katsayı üretmemiş tek spesifikasyon: spec curve çizilemeyen durum."""
    results = [
        EstimationResult(
            spec_id="s1",
            estimator="OLS",
            status="failed",
            error="singular matrix",
        ).model_dump()
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_variance_panel_explains_missing_spec_curve(tmp_path: Path) -> None:
    """Hiçbir spesifikasyon katsayı üretmediğinde spec curve bölümü tamamen
    kayboluyordu; panel sanki o bölüm hiç yokmuş gibi görünüyordu.

    Kırılganlık teşhisi ürünün ana iddiası, bu yüzden grafiğin YOKLUĞU da
    açıklanmalı. Mesaj kaldırılırsa test düşer.
    """
    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, _failed_results_file(tmp_path))

    assert not app.exception
    assert any("spec curve çizilemiyor" in warning.value for warning in app.warning)


def test_variance_panel_reports_unreadable_results_file(tmp_path: Path) -> None:
    """Multiverse yarıda kesilirse results.json bozuk kalıyor ve sayfa ham
    traceback'e düşüyordu.

    Jüri demosunda görülebilecek en kötü hata yolu buydu; yönlendirici bir
    mesaja bağlanması gerekiyor.
    """
    path = tmp_path / "results.json"
    path.write_text('[{"spec_id": "s1", "estim', encoding="utf-8")

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, path)

    assert not app.exception
    assert any("Sonuç dosyası okunamadı" in error.value for error in app.error)


def test_variance_panel_prompts_for_explicit_selection_when_columns_missing(tmp_path: Path) -> None:
    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = _frozen_estimand()

    _load_variance_panel(app, _results_file(tmp_path))

    labels = {selectbox.label for selectbox in app.selectbox}
    assert "Kohort kolonu" in labels
    assert "Never-treated kolonu" in labels
    assert any(
        "Pre-trend hesabı için cohort ve never-treated kolonlarını seçin." in info.value
        for info in app.info
    )


def test_variance_panel_runs_pretrend_after_explicit_columns(tmp_path: Path) -> None:
    pytest.importorskip("pyfixest")

    app = AppTest.from_file(PAGE_PATH, default_timeout=20)
    app.session_state["clean_df"] = _panel_with_explicit_cohort()
    app.session_state["frozen_estimand"] = _frozen_estimand()

    _load_variance_panel(app, _results_file(tmp_path))

    assert not any(selectbox.label == "Kohort kolonu" for selectbox in app.selectbox)
    button = next(
        button for button in app.button if button.label == "Bu kolonlarla pre-trend hesapla"
    )
    button.click()
    app.run()

    assert app.session_state["_event_study_cache"]["status"] == "ok"
    assert app.session_state["_event_study_cache_key"][0].endswith("results.json")


def test_variance_panel_invalidates_cached_pretrend_when_treated_cohort_changes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyfixest")

    app = AppTest.from_file(PAGE_PATH, default_timeout=20)
    app.session_state["clean_df"] = _panel_with_explicit_cohort()
    app.session_state["frozen_estimand"] = _frozen_estimand()

    _load_variance_panel(app, _results_file(tmp_path))

    app.multiselect[0].set_value([2014])
    next(
        button for button in app.button if button.label == "Bu kolonlarla pre-trend hesapla"
    ).click()
    app.run()

    assert app.session_state["_event_study_cache"]["status"] == "ok"

    app.multiselect[0].set_value([2015])
    app.run()

    assert any("Hesaplamak için butona basın." in info.value for info in app.info)
    assert len(app.get("plotly_chart")) == 1


def test_variance_panel_skips_pretrend_when_required_columns_are_missing(tmp_path: Path) -> None:
    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = pd.DataFrame({"unit": ["u1"], "year": [2014]})

    _load_variance_panel(app, _results_file(tmp_path))

    assert any("gerekli kolonlar eksik" in info.value for info in app.info)


def test_variance_panel_warns_when_pretrend_estimation_fails(tmp_path: Path) -> None:
    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    panel = _panel_with_explicit_cohort()
    panel["never_treated"] = False
    app.session_state["clean_df"] = panel
    app.session_state["frozen_estimand"] = _frozen_estimand()

    _load_variance_panel(app, _results_file(tmp_path))
    button = next(
        button for button in app.button if button.label == "Bu kolonlarla pre-trend hesapla"
    )
    button.click()
    app.run()

    assert any("Pre-trend görseli hazırlanamadı" in warning.value for warning in app.warning)


def _persist_frozen_menu_record(run_id: str, *, estimand_hash: str, menu_hash: str) -> None:
    """Kaydı analiz sayfasının kullandığı üreticiyle yazar.

    Şekil test tarafında elle kurulursa anahtar ya da alan adı yazan tarafta
    değiştiğinde bu testler yeşil kalır ve panel sessizce "kayıt bulunamadı"
    demeye başlar; üretici paylaşıldığı sürece bu mümkün değil.
    """
    record = build_frozen_menu_record(
        estimand_hash=estimand_hash,
        menu_hash=menu_hash,
        spec_count=4,
        run_id=run_id,
        estimand={"outcome": "uninsured_rate"},
        menu={"control_sets": [[]]},
    )
    save_frozen_menu_record(run_id, record)


def _isolate_store_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`ProjectStore` gerçek `runs/store`'a değil, `tmp_path`'e yazsın.

    NEDEN: `ParetoSettings` `@dataclass(frozen=True)` — `SETTINGS.store_dir = ...`
    ataması reddedilir. `store.py` `from ..config import SETTINGS` ile modül
    seviyesinde bir referans tuttuğu için, o modüldeki isim `dataclasses.replace`
    ile üretilen izole bir kopyaya monkeypatch edilir.
    """
    isolated = replace(_BASE_SETTINGS, store_dir=str(tmp_path / "store"))
    monkeypatch.setattr(_store_module, "SETTINGS", isolated)


def test_variance_panel_shows_matching_provenance_when_hashes_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_store_dir(monkeypatch, tmp_path)

    frozen_estimand = _frozen_estimand()
    run_id = "run-match-001"
    _persist_frozen_menu_record(
        run_id, estimand_hash=frozen_estimand.freeze_hash, menu_hash="deadbeefdeadbeef"
    )

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = frozen_estimand
    app.session_state["multiverse_run_id"] = run_id

    _load_variance_panel(app, _results_file(tmp_path))

    assert any("eşleşiyor" in success.value for success in app.success)


def test_variance_panel_warns_when_provenance_hash_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_store_dir(monkeypatch, tmp_path)

    run_id = "run-mismatch-001"
    _persist_frozen_menu_record(
        run_id, estimand_hash="stale0000000000", menu_hash="deadbeefdeadbeef"
    )

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = _frozen_estimand()  # farklı hash üretir
    app.session_state["multiverse_run_id"] = run_id

    _load_variance_panel(app, _results_file(tmp_path))

    assert any("eşleşmiyor" in warning.value for warning in app.warning)


def test_variance_panel_shows_no_record_message_when_run_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_store_dir(monkeypatch, tmp_path)

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = _frozen_estimand()
    app.session_state["multiverse_run_id"] = "never-persisted-run"

    _load_variance_panel(app, _results_file(tmp_path))

    assert any("kaydı bulunamadı" in caption.value for caption in app.caption)


def test_variance_panel_provenance_follows_inspected_run_not_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance, oturumdaki koşuyu değil bakılan dosyanın koşusunu anlatmalı.

    Sonuç yolu serbest metin: kullanıcı eski bir koşunun `results.json`'ını
    açtığında karşılaştırma o koşuya ait olmalı. Oturuma öncelik verilirse
    gösterge tam da uyarması gereken anda susar ve üstüne yanlış koşu için
    "eşleşiyor" güvencesi verir.
    """
    _isolate_store_dir(monkeypatch, tmp_path)

    frozen_estimand = _frozen_estimand()
    inspected_run_id = "run-on-disk-001"
    session_run_id = "run-session-002"

    # Bakılan koşu oturumdaki estimand ile ÜRETİLMEMİŞ; oturumdaki koşu üretilmiş.
    _persist_frozen_menu_record(
        inspected_run_id, estimand_hash="stale0000000000", menu_hash="deadbeefdeadbeef"
    )
    _persist_frozen_menu_record(
        session_run_id, estimand_hash=frozen_estimand.freeze_hash, menu_hash="deadbeefdeadbeef"
    )

    results_path = _results_file(tmp_path)
    results_path.with_name("run_id.txt").write_text(inspected_run_id, encoding="utf-8")

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = frozen_estimand
    app.session_state["multiverse_run_id"] = session_run_id

    _load_variance_panel(app, results_path)

    assert any("eşleşmiyor" in warning.value for warning in app.warning)
    assert not any("estimand" in success.value for success in app.success)


def _write_results(tmp_path: Path, results: list[EstimationResult]) -> Path:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False), encoding="utf-8"
    )
    return path


def _verdict_html(app: AppTest) -> str:
    """Karar kartının gövdesi.

    `st.html` gövdesi proto'da ham duruyor (sanitize tarayıcıda oluyor), bu yüzden
    kartın ne söylediği testten okunabilir.
    """
    bodies = [element.proto.body for element in app.get("html")]
    return next(body for body in bodies if "pa-vp-verdict" in body)


def test_verdict_shows_both_band_conditions_against_their_own_thresholds(tmp_path: Path) -> None:
    """Bant kuralı bir VE bağlacı: işaret-uyumu ≥%95 VE anlamlılık ≥%70.

    Karar tek bir sayıya ya da tek bir çubuğa indirilirse, işaret-uyumu eşiği geçmişken
    KARIŞIK yazan ekran kullanıcıya hata gibi görünür. Kartın iki koşulu da kendi eşiğiyle
    birlikte göstermesi, "neden bu bant" sorusunun cevabı.
    """
    results = [
        EstimationResult(
            spec_id="anlamli_1",
            estimator="OLS",
            coefficient=0.4,
            ci_low=0.2,
            ci_high=0.6,
            p_value=0.01,
            n_obs=10,
        ),
        EstimationResult(
            spec_id="anlamli_2",
            estimator="OLS",
            coefficient=0.3,
            ci_low=0.1,
            ci_high=0.5,
            p_value=0.02,
            n_obs=10,
        ),
        EstimationResult(
            spec_id="anlamsiz",
            estimator="OLS",
            coefficient=0.2,
            ci_low=-0.1,
            ci_high=0.5,
            p_value=0.4,
            n_obs=10,
        ),
    ]

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, _write_results(tmp_path, results))
    body = _verdict_html(app)

    # İşaret-uyumu %100 (üçü de pozitif), anlamlılık %67 — yani biri tutuyor, biri tutmuyor.
    assert "KARIŞIK" in body and "mixed" in body
    assert "<b>%100</b>" in body
    assert "<b>%67</b>" in body

    # Çentikler `variance.py`deki eşik sabitlerinden gelmeli; ekrana elle yazılmış bir eşik,
    # kural yeniden kalibre edildiğinde sessizce yanlış yeri göstermeye başlar.
    assert f"left:{SIGN_FRAGILE * 100:.4g}%" in body
    assert f"left:{SIGN_ROBUST * 100:.4g}%" in body
    assert f"left:{SIG_ROBUST * 100:.4g}%" in body

    # Tutmayan koşul sönük dolguyla ayrılıyor: iki çubuğu aynı renkte göstermek
    # "ikisi de tamam" der ve kuralı yanlış anlatır.
    assert "rgba(250, 250, 250, 0.35)" in body


def test_verdict_reports_zero_significance_rate_as_zero_not_as_missing(tmp_path: Path) -> None:
    """%0 anlamlılık panelin verebileceği en alarm verici sonuç, "veri yok" değil.

    Eski gösterim doğruluk sınıyordu (`if rate`), bu yüzden gerçek bir %0 ekranda "—"
    olarak, yani ölçülememiş gibi görünüyordu.
    """
    results = [
        EstimationResult(
            spec_id=f"s{i}",
            estimator="OLS",
            coefficient=0.2,
            ci_low=-0.1,
            ci_high=0.5,
            p_value=0.4,
            n_obs=10,
        )
        for i in range(2)
    ]

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, _write_results(tmp_path, results))

    assert "<b>%0</b>" in _verdict_html(app)


def test_spec_curve_legend_names_only_the_colors_actually_plotted(tmp_path: Path) -> None:
    """Lejant grafikteki renkleri açıklar, olası tüm renkleri değil.

    Plotly nokta başına renk verilen tek izde lejant çizemiyor, bu yüzden lejant elle
    kuruluyor. Sabit dört girdi yazılırsa tek yönlü bir eğride "ters yön de var"
    izlenimi doğar; kullanıcı olmayan bir kova arar.
    """
    results = [
        EstimationResult(
            spec_id=f"s{i}",
            estimator="OLS",
            coefficient=0.4,
            ci_low=0.2,
            ci_high=0.6,
            p_value=0.01,
            n_obs=10,
        )
        for i in range(2)
    ]

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, _write_results(tmp_path, results))
    legend = next(
        element.proto.body for element in app.get("html") if "pa-vp-legend" in element.proto.body
    )

    assert "anlamlı, pozitif" in legend
    assert "anlamlı, negatif" not in legend


def test_evidence_grid_layout_renders_every_panel_exactly_once(tmp_path: Path) -> None:
    """Izgara düzeni sekmelerin yerine geçer, üstüne gelmez.

    Bloklar iki düzende de bir kez çiziliyor; ikinci kez çizilseler Streamlit yinelenen
    widget key hatası verir ve sayfa komple düşer. Testin koruduğu şey bu: düzen anahtarı
    kabı değiştirir, içeriği çoğaltmaz.
    """
    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = _frozen_estimand()
    _load_variance_panel(app, _results_file(tmp_path))

    assert [tab.label for tab in app.tabs] == ["Eksen atfı", "Ön-trend", "Efektif N", "Makbuzlar"]

    # Sidebar'da gizlilik modu da bir segmented_control: sırayla değil etiketle seçiliyor.
    layout_switch = next(
        control for control in app.segmented_control if control.label == "Kanıt düzeni"
    )
    layout_switch.set_value("Izgara")
    app.run()

    assert not app.exception
    assert not app.tabs
    bodies = [element.proto.body for element in app.get("html")]
    for label in ("Eksen atfı", "Ön-trend", "Efektif N", "Makbuzlar"):
        assert sum(label in body for body in bodies) == 1, label
    # Efektif N + makbuzlar tabloları; specs.json olmadığı için eksen atfı tablo çizmiyor.
    assert len(app.dataframe) == 2


def test_receipts_table_hides_nested_fields_and_says_that_it_did(tmp_path: Path) -> None:
    """Makbuz tablosu okunabilir kalmalı, ama sakladığını saklamamalı.

    `event_study` ve `group_time_atts` hücreye sığmayan iç içe yapılar; tabloda kaldıklarında
    katsayı/CI/hata sütunlarını eziyorlar. Gizlenmeleri bir bilgi kaybı olduğu için sessiz
    olamaz: makbuzun iddiası eksiksizlik.
    """
    results = [
        EstimationResult(
            spec_id="s1",
            estimator="OLS",
            coefficient=0.4,
            ci_low=0.2,
            ci_high=0.6,
            p_value=0.01,
            n_obs=10,
            event_study={"t_minus_1": 0.0, "t_0": 0.4},
        )
    ]

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    _load_variance_panel(app, _write_results(tmp_path, results))

    receipts = app.dataframe[-1]
    shown = list(receipts.proto.column_order)
    assert "event_study" not in shown and "group_time_atts" not in shown
    # Makbuzun asıl taşıdığı alanlar tabloda kalmalı; filtre genişlerse tablo boşalır.
    for column in ("spec_id", "estimator", "coefficient", "ci_low", "ci_high", "status", "error"):
        assert column in shown, column
    assert any("event_study" in caption.value for caption in app.caption)


def test_spec_curve_colors_points_by_significance_and_direction(tmp_path: Path) -> None:
    """Spec-curve'ün tek okunur sinyali renk: gri anlamsız, yeşil/kırmızı anlamlı yön.

    Renk düşerse eğri hâlâ çizilir ama "hangi spesifikasyon anlamlı ve hangi yönde"
    sorusu grafikten okunamaz hale gelir; panelin sattığı şey tam olarak budur.
    """
    results = [
        EstimationResult(
            spec_id="anlamli_negatif",
            estimator="OLS",
            coefficient=-0.4,
            ci_low=-0.6,
            ci_high=-0.2,
            p_value=0.001,
            n_obs=10,
        ),
        EstimationResult(
            spec_id="anlamsiz",
            estimator="OLS",
            coefficient=0.1,
            ci_low=-0.3,
            ci_high=0.5,
            p_value=0.7,
            n_obs=10,
        ),
        EstimationResult(
            spec_id="anlamli_pozitif",
            estimator="OLS",
            coefficient=0.4,
            ci_low=0.2,
            ci_high=0.6,
            p_value=0.01,
            n_obs=10,
        ),
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False), encoding="utf-8"
    )

    app = AppTest.from_file(PAGE_PATH, default_timeout=10)
    app.session_state["clean_df"] = _panel_missing_cohort_columns()
    app.session_state["frozen_estimand"] = _frozen_estimand()

    _load_variance_panel(app, results_path)

    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    trace = figure["data"][0]
    # Eğri katsayıya göre sıralı çizilir; renk ile spec_id aynı noktayı göstermeli.
    colors_by_spec = dict(zip(trace["text"], trace["marker"]["color"], strict=True))

    # Ekrana çizilen kopya koyu tema paletinde (`_for_dark_ui`): uygulama teması koyuya
    # sabit, beyaz zemine göre seçilmiş `seagreen`/`indianred` orada okunmuyordu.
    assert colors_by_spec["anlamli_negatif"] == "#f87171"
    assert colors_by_spec["anlamli_pozitif"] == "#4ade80"
    assert colors_by_spec["anlamsiz"] == "#9ca3af"
    # Üç kova birbirinden ayrı kalmalı; eşleme tablosu çökerse renk sinyali de çöker.
    assert len(set(colors_by_spec.values())) == 3

    # Hata çubukları güven aralığını taşıyor. Renk verilmezse Plotly marker rengini miras
    # almaya çalışır; marker rengi nokta başına bir dizi olduğu için bu başarısız olup
    # varsayılan koyu griye (#444) düşüyor ve koyu zeminde çubuklar gözden kayboluyor —
    # nokta hâlâ görünür ama belirsizliği okunamaz hale gelir.
    assert trace["error_y"]["color"] == "rgba(250, 250, 250, 0.55)"
