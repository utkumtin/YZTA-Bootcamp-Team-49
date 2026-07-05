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

from pareto.analysis.variance import ROBUST_RULE_TEXT, summarize
from pareto.contracts import EstimationResult

st.title("📊 3 · Varyans Paneli")
st.caption("Tek kesin cevap yok; savunulabilir seçimler menüsü ve her birinin sonucu.")

results_path = st.text_input("Sonuç dosyası", value="runs/latest/results.json")
if not Path(results_path).exists():
    st.warning(f"Sonuç dosyası yok: {results_path}. Önce multiverse runner koş.")
    st.stop()

raw = json.loads(Path(results_path).read_text(encoding="utf-8"))
results = [EstimationResult(**r) for r in raw]
summary = summarize(results)

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

st.subheader("Şeffaflık makbuzları")
st.dataframe(pd.DataFrame([r.model_dump() for r in results]), use_container_width=True)

st.info("Matched-pair + ANOVA eksen-teşhisi ve LLM narrative Sprint-2 (bkz docs/scrum).")
