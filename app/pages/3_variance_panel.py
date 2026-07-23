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


def _infer_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in df.columns:
        col_name = str(col).lower()
        if any(token in col_name for token in ("cohort", "never", "treated", "year", "time", "date", "unit", "id")):
            return str(col)
    return None


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

    if not outcome_col or not unit_col or not time_col or not cohort_col or not never_treated_col:
        return {"status": "skipped", "reason": "required columns for pre-trend diagnostic are missing"}

    return {
        "status": "pending_columns",
        "outcome_col": outcome_col,
        "unit_col": unit_col,
        "time_col": time_col,
        "cohort_col": cohort_col,
        "never_treated_col": never_treated_col,
        "controls": tuple(controls),
        "sources": {
            "outcome_col": outcome_source,
            "unit_col": unit_source,
            "time_col": time_source,
            "cohort_col": cohort_source,
            "never_treated_col": never_treated_source,
        },
    }

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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(ok))),
            y=[r.coefficient for r in ok],
            mode="markers",
            marker={"color": [_color(r) for r in ok], "size": 8},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [(r.ci_high - r.coefficient) if r.ci_high else 0 for r in ok],
                "arrayminus": [(r.coefficient - r.ci_low) if r.ci_low else 0 for r in ok],
                "thickness": 1,
                "width": 0,
            },
            text=[r.spec_id for r in ok],
            hovertemplate="%{text}<br>katsayı=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title="Specification Curve (katsayıya göre sıralı)",
        xaxis_title="Spesifikasyonlar",
        yaxis_title="Tahmini etki (katsayı)",
        template="simple_white",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Eksen atfı paneli")
if specs:
    diagnosis = diagnose_axes(results, specs)
    st.json(diagnosis)

    try:
        narrative = generate_narrative(summary, diagnosis)
        st.subheader("LLM narrative")
        st.write(narrative.ozet)
        for comment in narrative.eksen_yorumlari:
            st.caption(f"• {comment.axis}: {comment.yorum}")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Narrative oluşturulamadı: {exc}")
else:
    st.info("Bu run için specs.json bulunamadı; eksen atfı ve narrative gösterilemiyor.")

st.subheader("Efektif N")
rows = []
for result in results:
    error_value = getattr(result, "error", None) or getattr(result, "error_message", None)
    rows.append(
        {
            "spec_id": result.spec_id,
            "effective_n": int(result.n_obs) if (result.status == "ok" and result.n_obs is not None) else None,
            "status": result.status,
            "error": error_value if result.status != "ok" else None,
        }
    )
if rows:
    df_n = pd.DataFrame(rows)
    st.dataframe(df_n, use_container_width=True)
    n_missing = int(df_n["effective_n"].isna().sum())
    if n_missing:
        st.caption(f"{n_missing} spesifikasyon için efektif N üretilemedi (yukarıdaki 'status'/'error' sütununa bakın).")
else:
    st.info("Efektif N bilgisi bulunamadı.")

st.subheader("Pre-trend event study")
payload = _build_event_study_payload()
event_study = None
if payload is None:
    st.info("Pre-trend görseli için temizlenmiş veri seti yok.")
elif payload.get("status") == "skipped":
    st.info(payload.get("reason", "Pre-trend görseli için gerekli veri yok."))
elif payload.get("status") == "pending_columns":
    sources = payload.get("sources", {})
    st.caption(
        "Kullanılacak kolonlar → "
        f"outcome: `{payload['outcome_col']}` ({sources.get('outcome_col', 'unknown')}), "
        f"unit: `{payload['unit_col']}` ({sources.get('unit_col', 'unknown')}), "
        f"time: `{payload['time_col']}` ({sources.get('time_col', 'unknown')})"
    )
    st.caption(
        "Kullanılacak özel kolonlar → "
        f"cohort: `{payload['cohort_col']}` ({sources.get('cohort_col', 'unknown')}), "
        f"never_treated: `{payload['never_treated_col']}` ({sources.get('never_treated_col', 'unknown')})"
    )
    if sources.get("cohort_col") == "inferred" or sources.get("never_treated_col") == "inferred":
        st.caption("⚠️ `cohort` ve/veya `never_treated` kolonları isimden sezgisel olarak tahmin edildi; bu yüzden işlemi onaylamak iyi olur.")
    if st.button("Bu kolonlarla pre-trend hesapla"):
        try:
            event_study = estimate_pretrend_event_study(
                st.session_state["clean_df"],
                outcome_col=payload["outcome_col"],
                unit_col=payload["unit_col"],
                time_col=payload["time_col"],
                cohort_col=payload["cohort_col"],
                never_treated_col=payload["never_treated_col"],
                controls=payload["controls"],
            )
            st.session_state["_event_study_cache"] = event_study
        except Exception as exc:  # noqa: BLE001
            st.session_state["_event_study_cache"] = {"status": "failed", "error": str(exc)}
        st.rerun()

    event_study = st.session_state.get("_event_study_cache")
    if event_study is None:
        st.info("Hesaplamak için butona basın.")
    elif event_study.get("status") == "ok":
        series = pd.DataFrame(event_study.get("series", []))
        if series.empty:
            st.info("Pre-trend serisi boş.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=series["event_time"],
                    y=series["coefficient"],
                    mode="lines+markers",
                    marker={"size": 8},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            (row["ci_high"] - row["coefficient"]) if row.get("ci_high") is not None and row.get("coefficient") is not None else 0
                            for _, row in series.iterrows()
                        ],
                        "arrayminus": [
                            (row["coefficient"] - row["ci_low"]) if row.get("ci_low") is not None and row.get("coefficient") is not None else 0
                            for _, row in series.iterrows()
                        ],
                        "thickness": 1,
                        "width": 0,
                    },
                    hovertemplate="event_time=%{x}<br>estimate=%{y:.3f}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig.update_layout(
                title="Pre-trend event-study coefficients",
                xaxis_title="Event time",
                yaxis_title="Coefficient",
                template="simple_white",
                height=360,
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
