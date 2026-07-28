"""3 · Varyans Paneli — ürünün ruhu.

Prototip `p4_dashboard/app.py:54-88` spec-curve çizimi (CI barları, anlamlılık
renklendirmesi) buraya migre edildi; artık tipli `EstimationResult` okur ve
3-bant robust/fragile kuralını (deterministik) gösterir. Matched-pair/ANOVA teşhisi ve
LLM narrative Sprint-2 (variance.diagnose_axes + router JUDGE narrative).

Girdi: runner çıktısı `runs/<run_id>/results.json`.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pareto.analysis.event_study import estimate_pretrend_event_study
from pareto.analysis.event_study_columns import (
    build_event_study_payload,
)
from pareto.analysis.event_study_columns import (
    event_study_cache_key as _event_study_cache_key,
)
from pareto.analysis.variance import ROBUST_RULE_TEXT, diagnose_axes, summarize
from pareto.config import SETTINGS
from pareto.contracts import EstimationResult
from pareto.llm.cache import CannedModeCacheMissError
from pareto.llm.narrative import generate_narrative
from pareto.memory.frozen_menu import load_frozen_menu_record
from pareto.repro import (
    ARTIFACT_LABELS,
    ReproInputs,
    ReproPackageError,
    build_reproduction_package,
    figure_html,
    missing_artifacts,
    package_key,
)
from pareto.spec import Specification
from pareto.streamlit_ui import render_compact_sidebar

with st.sidebar:
    render_compact_sidebar()

st.title("📊 3 · Varyans Paneli")
st.caption("Tek kesin cevap yok; savunulabilir seçimler menüsü ve her birinin sonucu.")

# local-only: dosya sistemi erişimi güvenilir ortamda varsayılır.
results_path = st.text_input(
    "Sonuç dosyası",
    value=st.session_state.get("multiverse_results_path", "runs/latest/results.json"),
)
if not Path(results_path).exists():
    st.warning(f"Sonuç dosyası yok: {results_path}. Önce multiverse runner koş.")
    st.stop()

raw = json.loads(Path(results_path).read_text(encoding="utf-8"))
results = [EstimationResult(**r) for r in raw]
summary = summarize(results)

specs_path = Path(results_path).with_name("specs.json")
if specs_path.exists():
    specs = [Specification(**s) for s in json.loads(specs_path.read_text(encoding="utf-8"))]
else:
    specs = []


def _resolve_run_id(results_file: Path) -> str:
    """Sonuç dizininin gerçek run_id'si.

    Öncelik dizinin yanındaki `run_id.txt`tedir: `runs/latest` bir aynadır, dizin
    adı "latest"tir ve oturumdaki run_id kullanıcının elle girdiği eski bir koşuya
    ait olabilir. run_id yanlış çözülürse donmuş menü bulunamaz ve METHODS.md
    estimand bölümünü tamamen kaybeder.
    """
    marker = results_file.with_name("run_id.txt")
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    return str(st.session_state.get("multiverse_run_id") or results_file.parent.name)


# Panelde çizilen figürler reprodüksiyon paketine de girer: dosya adı -> HTML gövdesi.
package_figures: dict[str, str] = {}


def _capture_figure(name: str, figure: go.Figure) -> None:
    """Figürü paket için saklar. Render kuralları `pareto.repro.figure_html`te."""
    package_figures[name] = figure_html(name, figure)


def _render_run_provenance(results_path: str) -> None:
    # Kimlik bakılan dosyadan çözülür, oturumdan değil: kullanıcı eski bir
    # koşunun sonuç yolunu elle yazdığında aşağıdaki hash karşılaştırması o
    # koşuya ait olmalı. Oturuma öncelik verilirse gösterge tam da uyarması
    # gereken durumda yanlış koşuyu doğruluyor.
    run_id = _resolve_run_id(Path(results_path))
    record = load_frozen_menu_record(run_id)

    st.subheader("Run provenance")
    if record is None:
        st.caption(
            "Bu run için ProjectStore'da dondurulmuş estimand/menu kaydı bulunamadı "
            "(run doğrudan CLI/worker ile üretilmiş olabilir)."
        )
        return

    stored_estimand_hash = record.get("estimand_hash")
    stored_menu_hash = record.get("menu_hash")
    stored_spec_count = record.get("spec_count")
    st.caption(
        f"Kayıtlı estimand_hash=`{stored_estimand_hash}` · "
        f"menu_hash=`{stored_menu_hash}` · spec_count={stored_spec_count}"
    )

    current_estimand = st.session_state.get("frozen_estimand")
    current_estimand_hash = getattr(current_estimand, "freeze_hash", None)

    if current_estimand_hash and stored_estimand_hash:
        if current_estimand_hash == stored_estimand_hash:
            st.success("✓ Oturumdaki estimand, bu run'ı üreten dondurulmuş estimand ile eşleşiyor.")
        else:
            st.warning(
                "⚠️ Oturumdaki estimand hash'i, bu run'ı üreten dondurulmuş estimand ile "
                "**eşleşmiyor**. Aşağıdaki sonuçlar farklı bir estimand'dan üretilmiş olabilir."
            )
    elif stored_estimand_hash:
        st.info("Oturumda aktif bir estimand yok; run provenance'ı yalnız kayıttan gösteriliyor.")


_render_run_provenance(results_path)


def _plot_coefficient_series(
    *,
    x,
    y,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    hovertemplate: str,
    marker_size: int = 8,
    marker_color: list[str] | None = None,
    connectgaps: bool = False,
    add_zero_line: bool = True,
    zero_line_y: float = 0.0,
    add_vline_x: int | float | None = None,
    height: int = 360,
    trace_mode: str = "markers",
    trace_name: str | None = None,
    error_y: dict | None = None,
) -> go.Figure:
    fig = go.Figure()
    marker: dict[str, Any] = {"size": marker_size}
    if marker_color is not None:
        marker["color"] = marker_color
    trace_kwargs = {
        "x": x,
        "y": y,
        "mode": trace_mode,
        "marker": marker,
        "hovertemplate": hovertemplate,
    }
    if trace_name is not None:
        trace_kwargs["name"] = trace_name
    if error_y is not None:
        trace_kwargs["error_y"] = error_y
    if not connectgaps:
        trace_kwargs["connectgaps"] = False
    fig.add_trace(go.Scatter(**trace_kwargs))
    if add_vline_x is not None:
        fig.add_vline(x=add_vline_x, line_dash="dot", line_color="black", opacity=0.6)
    if add_zero_line:
        fig.add_hline(y=zero_line_y, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="simple_white",
        height=height,
    )
    return fig


@st.cache_data(show_spinner=False)
def _diagnose_axes_cached(results_json: str, specs_json: str) -> dict[str, Any]:
    return diagnose_axes(
        [EstimationResult(**item) for item in json.loads(results_json)],
        [Specification(**item) for item in json.loads(specs_json)],
    )


# --- Özet metrikleri + 3-bant etiket ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam spec", summary["n_total"])
c2.metric("Başarılı", summary["n_ok"])
c3.metric("İşaret-uyumu", f"{summary['sign_agreement']:.0%}" if summary["sign_agreement"] else "—")
c4.metric(
    "Anlamlılık", f"{summary['significance_rate']:.0%}" if summary["significance_rate"] else "—"
)

band = summary.get("band")
band_style = {"robust": "success", "mixed": "warning", "fragile": "error"}.get(str(band), "info")
getattr(st, band_style)(f"**Etiket: {str(band).upper()}** — {ROBUST_RULE_TEXT}")


def _color(r: EstimationResult) -> str:
    if r.status != "ok" or r.p_value is None or r.coefficient is None:
        return "lightgray"
    if r.p_value >= 0.05:
        return "gray"
    return "seagreen" if r.coefficient > 0 else "indianred"


# (sonuç, katsayı) çifti: katsayının None olmadığı filtrede bir kez daraltılır,
# aşağıdaki sıralama ve hata çubuğu hesapları aynı daraltılmış değeri kullanır.
ok = [(r, r.coefficient) for r in results if r.status == "ok" and r.coefficient is not None]
if ok:
    ok.sort(key=lambda pair: pair[1])
    fig = _plot_coefficient_series(
        x=list(range(len(ok))),
        y=[coefficient for _result, coefficient in ok],
        title="Specification Curve (katsayıya göre sıralı)",
        xaxis_title="Spesifikasyonlar",
        yaxis_title="Tahmini etki (katsayı)",
        hovertemplate="%{text}<br>katsayı=%{y:.4f}<extra></extra>",
        marker_size=8,
        marker_color=[_color(r) for r, _coefficient in ok],
        trace_mode="markers",
        add_zero_line=True,
        height=480,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [(r.ci_high - coefficient) if r.ci_high else 0 for r, coefficient in ok],
            "arrayminus": [(coefficient - r.ci_low) if r.ci_low else 0 for r, coefficient in ok],
            "thickness": 1,
            "width": 0,
        },
    )
    fig.data[0].text = [r.spec_id for r, _coefficient in ok]
    st.plotly_chart(fig, use_container_width=True)
    _capture_figure("specification_curve.html", fig)

st.subheader("Eksen atfı paneli")
if specs:
    results_json = json.dumps([r.model_dump() for r in results], ensure_ascii=False)
    specs_json = json.dumps([s.model_dump() for s in specs], ensure_ascii=False)
    diagnosis = _diagnose_axes_cached(results_json, specs_json)

    excluded_count = sum(diagnosis["n_excluded"].values())
    st.caption(f"Kullanılan sonuç: {diagnosis['n_used']} · Hariç tutulan: {excluded_count}")
    matched_pairs = pd.DataFrame.from_dict(diagnosis.get("matched_pairs", {}), orient="index")
    if not matched_pairs.empty:
        st.dataframe(matched_pairs, use_container_width=True)
    anova_r2 = pd.DataFrame([diagnosis.get("anova_partial_r2", {})]).T
    if not anova_r2.empty:
        anova_r2.columns = ["partial_r2"]
        st.dataframe(anova_r2, use_container_width=True)
    if diagnosis.get("warnings"):
        for warning in diagnosis.get("warnings", []):
            st.caption(f"• {warning}")

    narrative_key = json.dumps(
        {"summary": summary, "diagnosis": diagnosis}, ensure_ascii=False, sort_keys=True
    )
    if st.button("LLM narrative oluştur"):
        try:
            with st.spinner("LLM narrative hazırlanıyor..."):
                st.session_state["_variance_narrative"] = generate_narrative(summary, diagnosis)
                st.session_state["_variance_narrative_key"] = narrative_key
        except (ValueError, CannedModeCacheMissError) as exc:
            # Narrative opsiyonel bir ek: canned modda cache'te olmayan bir
            # anlatı istendiğinde akış durmaz, uyarı ile geçilir.
            st.warning(f"Narrative oluşturulamadı: {exc}")

    if st.session_state.get("_variance_narrative_key") == narrative_key:
        narrative = st.session_state.get("_variance_narrative")
        if narrative is not None:
            st.subheader("LLM narrative")
            st.write(narrative.ozet)
            for comment in narrative.eksen_yorumlari:
                st.caption(f"• {comment.axis}: {comment.yorum}")
else:
    st.info("Bu run için specs.json bulunamadı; eksen atfı ve narrative gösterilemiyor.")

st.subheader("Efektif N")
rows = []
for result in results:
    rows.append(
        {
            "spec_id": result.spec_id,
            "effective_n": int(result.n_obs)
            if (result.status == "ok" and result.n_obs is not None)
            else None,
            "status": result.status,
            "error": result.error if result.status != "ok" else None,
        }
    )
if rows:
    df_n = pd.DataFrame(rows)
    st.dataframe(df_n, use_container_width=True)
    n_missing = int(df_n["effective_n"].isna().sum())
    if n_missing:
        st.caption(
            f"{n_missing} spesifikasyon için efektif N üretilemedi "
            "(yukarıdaki 'status'/'error' sütununa bakın)."
        )
else:
    st.info("Efektif N bilgisi bulunamadı.")

st.subheader("Pre-trend event study")
payload = build_event_study_payload(
    st.session_state.get("clean_df"),
    estimand=st.session_state.get("frozen_estimand"),
    analysis_state=st.session_state.get("analysis_state"),
)
event_study = None
if payload is None:
    st.info("Pre-trend görseli için temizlenmiş veri seti yok.")
elif payload["status"] == "skipped":
    st.info(payload["reason"])
else:
    sources = payload["sources"]
    st.caption(
        "Kullanılacak kolonlar → "
        f"outcome: `{payload['outcome_col']}` ({sources.get('outcome_col', 'unknown')}), "
        f"unit: `{payload['unit_col']}` ({sources.get('unit_col', 'unknown')}), "
        f"time: `{payload['time_col']}` ({sources.get('time_col', 'unknown')})"
    )
    column_options = payload["column_options"]
    cohort_col = payload["cohort_col"]
    never_treated_col = payload["never_treated_col"]

    if cohort_col is None:
        cohort_choice = st.selectbox(
            "Kohort kolonu",
            options=["-- seçin --", *column_options],
            key="event_study_cohort_choice",
        )
        cohort_col = None if cohort_choice == "-- seçin --" else cohort_choice
    else:
        st.caption(f"Kohort kolonu: `{cohort_col}` ({sources.get('cohort_col', 'unknown')})")

    if never_treated_col is None:
        never_choice = st.selectbox(
            "Never-treated kolonu",
            options=["-- seçin --", *column_options],
            key="event_study_never_treated_choice",
        )
        never_treated_col = None if never_choice == "-- seçin --" else never_choice
    else:
        never_source = sources.get("never_treated_col", "unknown")
        st.caption(f"Never-treated kolonu: `{never_treated_col}` ({never_source})")

    if sources.get("cohort_col") == "inferred" or sources.get("never_treated_col") == "inferred":
        st.caption(
            "⚠️ `cohort` ve/veya `never_treated` kolonları isimden sezgisel olarak "
            "tahmin edildi; bu yüzden işlemi onaylamak iyi olur."
        )

    if cohort_col is None or never_treated_col is None:
        st.info("Pre-trend hesabı için cohort ve never-treated kolonlarını seçin.")
    else:
        cohort_values = sorted(
            {value for value in st.session_state["clean_df"][cohort_col].dropna().tolist()},
            key=lambda value: str(value),
        )
        treated_cohort_selection = st.multiselect(
            "Treated cohorts",
            options=cohort_values,
            default=cohort_values[:1] if cohort_values else [],
            key="event_study_treated_cohorts",
            help="Committed-baseline diagnostic için açık treated cohort seçin.",
        )
        cache_key = _event_study_cache_key(
            results_path=results_path,
            df=st.session_state["clean_df"],
            payload=payload,
            cohort_col=str(cohort_col),
            never_treated_col=str(never_treated_col),
            treated_cohorts=tuple(treated_cohort_selection),
        )
        cached_key = st.session_state.get("_event_study_cache_key")
        if cached_key == cache_key:
            event_study = st.session_state.get("_event_study_cache")

        if not treated_cohort_selection:
            st.info("Pre-trend hesabı için en az bir treated cohort seçin.")

        # Z3: estimate_pretrend_event_study kendi içindeki her riskli adımı
        # `except Exception` ile sarıp `_failed_result(...)` döndürüyor
        # (event_study.py) — yani pratikte ValueError fırlatmıyor. Sayfadaki
        # try/except ValueError, kütüphanenin zaten koruduğu bir yolu bir daha
        # koruyormuş gibi görünen ölü kod ve test edilemiyordu. Kaldırıldı;
        # artık doğrudan {"status": ..., "error": ...} sözleşmesine güveniliyor.
        if st.button("Bu kolonlarla pre-trend hesapla", disabled=not treated_cohort_selection):
            event_study = estimate_pretrend_event_study(
                st.session_state["clean_df"],
                outcome_col=payload["outcome_col"],
                unit_col=payload["unit_col"],
                time_col=payload["time_col"],
                cohort_col=str(cohort_col),
                never_treated_col=str(never_treated_col),
                controls=payload["controls"],
                treated_cohorts=tuple(treated_cohort_selection),
            )
            st.session_state["_event_study_cache"] = event_study
            st.session_state["_event_study_cache_key"] = cache_key

    if event_study is None:
        st.info("Hesaplamak için butona basın.")
    elif event_study.get("status") == "ok":
        series = pd.DataFrame(event_study.get("series", []))
        if series.empty:
            st.info("Pre-trend serisi boş.")
        else:
            reference_period = int(event_study.get("reference_period", -1))
            if reference_period not in set(series["event_time"]):
                series = pd.concat(
                    [
                        series,
                        pd.DataFrame(
                            [
                                {
                                    "event_time": reference_period,
                                    "coefficient": None,
                                    "ci_low": None,
                                    "ci_high": None,
                                    "p_value": None,
                                    "n_obs": None,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            series = series.sort_values("event_time")
            fig = _plot_coefficient_series(
                x=series["event_time"],
                y=series["coefficient"],
                title="Pre-trend event-study coefficients",
                xaxis_title="Event time",
                yaxis_title="Coefficient",
                hovertemplate="event_time=%{x}<br>estimate=%{y:.3f}<extra></extra>",
                marker_size=8,
                connectgaps=False,
                add_zero_line=True,
                add_vline_x=reference_period,
                trace_mode="lines+markers",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        (row["ci_high"] - row["coefficient"])
                        if row.get("ci_high") is not None and row.get("coefficient") is not None
                        else 0
                        for _, row in series.iterrows()
                    ],
                    "arrayminus": [
                        (row["coefficient"] - row["ci_low"])
                        if row.get("ci_low") is not None and row.get("coefficient") is not None
                        else 0
                        for _, row in series.iterrows()
                    ],
                    "thickness": 1,
                    "width": 0,
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            _capture_figure("pretrend_event_study.html", fig)
        if event_study.get("warnings"):
            for warning in event_study.get("warnings", []):
                st.caption(f"• {warning}")
    elif event_study.get("status") == "failed":
        st.warning(f"Pre-trend görseli hazırlanamadı: {event_study.get('error')}")
    else:
        st.info(event_study.get("reason", "Pre-trend görseli için gerekli veri yok."))

st.subheader("Şeffaflık makbuzları")
st.dataframe(pd.DataFrame([r.model_dump() for r in results]), use_container_width=True)


# --------------------------------------------------------------------------- #
# Reprodüksiyon paketi
# --------------------------------------------------------------------------- #
def _repro_inputs() -> ReproInputs:
    """Koşunun artefakt yollarını oturumdan ve disk düzeninden çözer.

    Temizleme run_id'si ile multiverse run_id'si farklıdır, üstelik varsayılan
    sonuç yolu (`runs/latest`) hiçbir run_id taşımaz; bu yüzden yollar burada
    açıkça çözülür ve pakete hazır olarak verilir.

    Temizleme artefaktları oturumdan gelir ve gösterilen sonuçlarla aynı koşuya ait
    OLMAYABİLİR; sandbox çıktısı da (`reproduced.pkl`) pakete verilir ki paket bu
    bağı kendi denetleyip manifest'e yazabilsin.
    """
    results_file = Path(results_path)
    run_id = _resolve_run_id(results_file)

    def _session_path(key: str) -> Path | None:
        value = st.session_state.get(key)
        return Path(str(value)) if value else None

    repro_dir = _session_path("last_repro_dir")
    return ReproInputs(
        run_id=run_id,
        results_path=results_file,
        specs_path=results_file.with_name("specs.json"),
        panel_path=results_file.with_name("panel.pkl"),
        frozen_menu_path=Path(SETTINGS.store_dir) / run_id / "frozen_menu.json",
        ledger_path=_session_path("last_ledger_path"),
        cleaning_script_path=_session_path("last_audit_path"),
        raw_panel_path=(repro_dir / "raw.pkl") if repro_dir is not None else None,
        cleaned_panel_path=(repro_dir / "reproduced.pkl") if repro_dir is not None else None,
        cleaning_run_id=(repro_dir.name.removesuffix("_repro") if repro_dir is not None else None),
        figures=dict(package_figures),
    )


st.subheader("Reprodüksiyon paketi")
st.caption(
    "Denetim izinin tek dosyalık hâli: karar defteri, temizleme script'i, donmuş "
    "hash'ler, spesifikasyonlar, sonuçlar, figürler ve tek komutluk doğrulama script'i."
)

repro_inputs = _repro_inputs()
repro_missing = missing_artifacts(repro_inputs)
if repro_missing:
    st.warning(
        "Paket şu artefaktlar olmadan kurulacak: "
        + ", ".join(ARTIFACT_LABELS.get(key, key) for key in repro_missing)
        + ". Eksikler paketin MANIFEST dosyasına da yazılır."
    )

# Paket birkaç MB olabilir; her rerun'da yeniden kurulmasın diye anahtarla
# önbelleklenir. Anahtar `pareto.repro`da hesaplanır: temizleme artefaktlarını
# kapsamayan bir anahtar, provenans uyarısını tam da uyarının gerektiği senaryoda
# (A'yı koşup paketleyip sonra B'yi temizlemek) önbellekte yutardı.
repro_key = package_key(repro_inputs)


def _forget_package() -> None:
    for key in (
        "_repro_package",
        "_repro_package_key",
        "_repro_package_provenance",
        "_repro_package_stale",
    ):
        st.session_state.pop(key, None)


if st.button("Reprodüksiyon paketini hazırla"):
    try:
        with st.spinner("Paket hazırlanıyor..."):
            package_bytes = build_reproduction_package(repro_inputs)
        st.session_state["_repro_package"] = package_bytes
        st.session_state["_repro_package_key"] = repro_key
        # Provenans denetimi paketin içinde yapılır; arayüz sonucu manifest'ten
        # okur ki uyarı ile pakete yazılan kayıt tek kaynaktan gelsin.
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
            manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
        st.session_state["_repro_package_provenance"] = manifest.get("provenance") or {}
        st.session_state["_repro_package_stale"] = False
    except ReproPackageError as exc:
        _forget_package()
        st.error(f"Reprodüksiyon paketi kurulamadı: {exc}")

if st.session_state.get("_repro_package_key") == repro_key:
    provenance = st.session_state.get("_repro_package_provenance") or {}
    if provenance.get("cleaning_matches_panel") is False:
        st.error(
            "Pakete giren temizleme izi bu sonuçları üreten veriyle EŞLEŞMİYOR "
            f"(temizleme koşusu: `{provenance.get('cleaning_run_id')}`). Karar defteri "
            "ve temizleme script'i başka bir koşudan geliyor; paket bu hâliyle denetim "
            "izi sayılmaz. Uyarı paketin MANIFEST ve METHODS dosyalarına da yazıldı."
        )
    elif provenance.get("cleaning_matches_panel") is None:
        st.info(
            "Temizleme izi ile analiz paneli arasındaki bağ denetlenemedi; "
            "paket bunu doğrulanmamış olarak kaydediyor."
        )

    st.download_button(
        "Reprodüksiyon paketini indir",
        data=st.session_state["_repro_package"],
        file_name=f"pareto_repro_{repro_inputs.run_id}.zip",
        mime="application/zip",
        type="primary",
    )
elif st.session_state.get("_repro_package") is not None or st.session_state.get(
    "_repro_package_stale"
):
    # Panel içeriği hazırlanan paketten sonra değişti (ör. pre-trend hesaplandı).
    # İndirme butonunu açıklamasız kaldırmak "sessiz hiçlik" olurdu. Bayat paketin
    # birkaç MB'ı oturumda tutulmaz, yerine yalnız uyarıyı ayakta tutan bayrak kalır.
    _forget_package()
    st.session_state["_repro_package_stale"] = True
    st.info("Panel hazırlanan paketten sonra değişti; paketi yeniden hazırlayın.")
else:
    st.info("Paketi indirmeden önce hazırlayın.")
