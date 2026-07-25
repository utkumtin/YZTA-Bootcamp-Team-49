"""3 · Varyans Paneli — ürünün ruhu.

Prototip `p4_dashboard/app.py:54-88` spec-curve çizimi (CI barları, anlamlılık
renklendirmesi) buraya migre edildi; artık tipli `EstimationResult` okur ve
3-bant robust/fragile kuralını (deterministik) gösterir. Matched-pair/ANOVA teşhisi ve
LLM narrative Sprint-2 (variance.diagnose_axes + router JUDGE narrative).

Girdi: runner çıktısı `runs/<run_id>/results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pareto.analysis.event_study import estimate_pretrend_event_study
from pareto.analysis.event_study_columns import (
    event_study_cache_key as _event_study_cache_key,
)
from pareto.analysis.event_study_columns import (
    infer_column as _infer_column,
)
from pareto.analysis.variance import ROBUST_RULE_TEXT, diagnose_axes, summarize
from pareto.contracts import EstimationResult
from pareto.llm.narrative import generate_narrative
from pareto.spec import Specification
from pareto.streamlit_ui import render_compact_sidebar

with st.sidebar:
    render_compact_sidebar()

st.title("📊 3 · Varyans Paneli")
st.caption("Tek kesin cevap yok; savunulabilir seçimler menüsü ve her birinin sonucu.")

results_path = st.text_input("Sonuç dosyası", value="runs/latest/results.json")
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


def _build_event_study_payload() -> dict[str, object] | None:
    df = st.session_state.get("clean_df")
    if df is None:
        return None

    analysis_state = st.session_state.get("analysis_state") or {}
    estimand = st.session_state.get("frozen_estimand")

    outcome_col = None
    outcome_source = "unknown"
    if estimand is not None:
        outcome_col = getattr(getattr(estimand, "estimand", None), "outcome", None)
        if outcome_col:
            outcome_source = "estimand"

    if not outcome_col:
        outcome_col = _infer_column(df, "outcome", "y", "dependent")
        outcome_source = "inferred" if outcome_col else "unknown"

    unit_col = analysis_state.get("unit_col")
    unit_source = "analysis_state" if unit_col else "inferred"
    if not unit_col:
        unit_col = _infer_column(df, "unit", "unit_id", "id")

    time_col = analysis_state.get("time_col")
    time_source = "analysis_state" if time_col else "inferred"
    if not time_col:
        time_col = _infer_column(df, "year", "time", "date", "period")

    cohort_col = _infer_column(df, "cohort", "cohort_id", "treatment_time", "group")
    cohort_source = "inferred" if cohort_col else "unknown"

    never_treated_col = _infer_column(df, "never_treated", "never_treat", "untreated", "control")
    never_treated_source = "inferred" if never_treated_col else "unknown"

    controls = analysis_state.get("controls") or []

    if not outcome_col or not unit_col or not time_col:
        return {
            "status": "skipped",
            "reason": "required columns for pre-trend diagnostic are missing",
        }

    return {
        "status": "needs_selection"
        if not cohort_col or not never_treated_col
        else "pending_columns",
        "outcome_col": outcome_col,
        "unit_col": unit_col,
        "time_col": time_col,
        "cohort_col": cohort_col,
        "never_treated_col": never_treated_col,
        "controls": tuple(controls),
        "column_options": [str(col) for col in df.columns],
        "sources": {
            "outcome_col": outcome_source,
            "unit_col": unit_source,
            "time_col": time_source,
            "cohort_col": cohort_source,
            "never_treated_col": never_treated_source,
        },
    }


def _plot_coefficient_series(
    *,
    x,
    y,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    hovertemplate: str,
    marker_size: int = 8,
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
    trace_kwargs = {
        "x": x,
        "y": y,
        "mode": trace_mode,
        "marker": {"size": marker_size},
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
def _diagnose_axes_cached(results_json: str, specs_json: str) -> dict[str, object]:
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
band_style = {"robust": "success", "mixed": "warning", "fragile": "error"}.get(band, "info")
getattr(st, band_style)(f"**Etiket: {str(band).upper()}** — {ROBUST_RULE_TEXT}")


def _color(r: EstimationResult) -> str:
    if r.status != "ok" or r.p_value is None or r.coefficient is None:
        return "lightgray"
    if r.p_value >= 0.05:
        return "gray"
    return "seagreen" if r.coefficient > 0 else "indianred"


ok = [r for r in results if r.status == "ok" and r.coefficient is not None]
if ok:
    ok.sort(key=lambda r: r.coefficient)
    fig = _plot_coefficient_series(
        x=list(range(len(ok))),
        y=[r.coefficient for r in ok],
        title="Specification Curve (katsayıya göre sıralı)",
        xaxis_title="Spesifikasyonlar",
        yaxis_title="Tahmini etki (katsayı)",
        hovertemplate="%{text}<br>katsayı=%{y:.4f}<extra></extra>",
        marker_size=8,
        trace_mode="markers",
        add_zero_line=True,
        height=480,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [(r.ci_high - r.coefficient) if r.ci_high else 0 for r in ok],
            "arrayminus": [(r.coefficient - r.ci_low) if r.ci_low else 0 for r in ok],
            "thickness": 1,
            "width": 0,
        },
    )
    fig.data[0].text = [r.spec_id for r in ok]
    st.plotly_chart(fig, use_container_width=True)

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
        except ValueError as exc:
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
payload = _build_event_study_payload()
event_study = None
if payload is None:
    st.info("Pre-trend görseli için temizlenmiş veri seti yok.")
elif payload.get("status") == "skipped":
    st.info(payload.get("reason", "Pre-trend görseli için gerekli veri yok."))
elif payload.get("status") in {"pending_columns", "needs_selection"}:
    sources = payload.get("sources", {})
    st.caption(
        "Kullanılacak kolonlar → "
        f"outcome: `{payload['outcome_col']}` ({sources.get('outcome_col', 'unknown')}), "
        f"unit: `{payload['unit_col']}` ({sources.get('unit_col', 'unknown')}), "
        f"time: `{payload['time_col']}` ({sources.get('time_col', 'unknown')})"
    )
    column_options = list(
        payload.get("column_options", [str(col) for col in st.session_state["clean_df"].columns])
    )
    cohort_col = payload.get("cohort_col")
    never_treated_col = payload.get("never_treated_col")

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

        if st.button("Bu kolonlarla pre-trend hesapla"):
            try:
                event_study = estimate_pretrend_event_study(
                    st.session_state["clean_df"],
                    outcome_col=payload["outcome_col"],
                    unit_col=payload["unit_col"],
                    time_col=payload["time_col"],
                    cohort_col=str(cohort_col),
                    never_treated_col=str(never_treated_col),
                    controls=payload["controls"],
                    treated_cohorts=tuple(treated_cohort_selection) or None,
                )
                st.session_state["_event_study_cache"] = event_study
                st.session_state["_event_study_cache_key"] = cache_key
            except Exception as exc:  # noqa: BLE001
                st.session_state["_event_study_cache"] = {"status": "failed", "error": str(exc)}
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
        if event_study.get("warnings"):
            for warning in event_study.get("warnings", []):
                st.caption(f"• {warning}")
    elif event_study.get("status") == "failed":
        st.warning(f"Pre-trend görseli hazırlanamadı: {event_study.get('error')}")
    else:
        st.info(event_study.get("reason", "Pre-trend görseli için gerekli veri yok."))

st.subheader("Şeffaflık makbuzları")
st.dataframe(pd.DataFrame([r.model_dump() for r in results]), use_container_width=True)
