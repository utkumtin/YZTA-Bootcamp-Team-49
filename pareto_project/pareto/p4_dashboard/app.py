"""Varyans Paneli (P4) -- Pareto'nun ruhu.

Çalıştırma:
    streamlit run pareto/p4_dashboard/app.py -- --results runs/latest_results.json

İçerik:
  1. Specification Curve: her nokta bir spesifikasyon, katsayıya göre sıralı,
     %95 güven aralıklı, anlamlılığa göre renklendirilmiş.
  2. LLM Teşhis Modülü: sonuçların nerede kırıldığını özetler.
  3. Şeffaflık Makbuzları Tablosu: her spec'in tam tarifi + üreten kod.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pareto.llm_client import call_llm  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="runs/latest_results.json")
    # streamlit `--` sonrası argümanları geçirir; bilinmeyenleri yok say
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        row = {**r["spec_dict"], **{k: v for k, v in r.items() if k != "spec_dict"}}
        rows.append(row)
    return pd.DataFrame(rows)


def _significance_color(row: pd.Series) -> str:
    if row["hata_mesaji"] is not None or pd.isna(row["p_value"]):
        return "lightgray"
    if row["p_value"] >= 0.05:
        return "gray"
    return "seagreen" if row["katsayi"] > 0 else "indianred"


def render_specification_curve(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["katsayi"]).sort_values("katsayi").reset_index(drop=True)
    plot_df["rank"] = range(len(plot_df))
    plot_df["color"] = plot_df.apply(_significance_color, axis=1)
    plot_df["ci_low"] = plot_df["katsayi"] - 1.96 * plot_df["standart_hata"]
    plot_df["ci_high"] = plot_df["katsayi"] + 1.96 * plot_df["standart_hata"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["rank"],
            y=plot_df["katsayi"],
            mode="markers",
            marker=dict(color=plot_df["color"], size=8),
            error_y=dict(
                type="data",
                symmetric=False,
                array=plot_df["ci_high"] - plot_df["katsayi"],
                arrayminus=plot_df["katsayi"] - plot_df["ci_low"],
                thickness=1,
                width=0,
            ),
            text=plot_df["spec_id"],
            hovertemplate="%{text}<br>katsayı=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title="Specification Curve",
        xaxis_title="Spesifikasyonlar (katsayıya göre sıralı)",
        yaxis_title="Tahmini Etki (katsayı)",
        template="simple_white",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_diagnosis(df: pd.DataFrame) -> None:
    summary = df[["spec_id", "controls", "cluster_by", "estimator", "katsayi", "p_value", "hata_mesaji"]]
    summary_str = summary.to_json(orient="records", force_ascii=False)

    system_prompt = (
        "Sen bir ekonometri hakemisin. Sana bir çokluevren (multiverse) analizinin "
        "sonuç tablosu verilecek. Sonuçların nerede kırılgan olduğunu, işaret "
        "değiştiren veya anlamsızlaşan spesifikasyonların ORTAK ÖZELLİĞİNİ tespit et. "
        "Kısa, somut, 3-5 cümlelik bir teşhis yaz. Kolon adlarına referans ver."
    )
    with st.spinner("AI teşhis ediyor..."):
        diagnosis = call_llm(system_prompt, summary_str, max_tokens=500)
    st.info(f"**AI Teşhisi**\n\n{diagnosis}")


def render_transparency_table(df: pd.DataFrame, audit_code_dir: str = "runs/audit_trail") -> None:
    st.subheader("Şeffaflık Makbuzları")
    st.caption("Her satır bir spesifikasyonun tam tarifidir. Genişletip kodunu incele.")

    for _, row in df.sort_values("katsayi", na_position="last").iterrows():
        label = f"{row['spec_id']} | {row['estimator']} | katsayı={row['katsayi']}"
        with st.expander(label):
            st.json(
                {
                    "outcome": row["outcome"],
                    "treatment": row["treatment"],
                    "controls": row["controls"],
                    "unit_fe": row.get("unit_fe"),
                    "time_fe": row.get("time_fe"),
                    "pre_period_window": row.get("pre_period_window"),
                    "cluster_by": row.get("cluster_by"),
                    "estimator": row["estimator"],
                    "katsayi": row["katsayi"],
                    "p_value": row["p_value"],
                    "standart_hata": row["standart_hata"],
                    "hata_mesaji": row["hata_mesaji"],
                }
            )
            code_path = Path(audit_code_dir)
            candidate_files = list(code_path.glob("*_cleaning_steps.py")) if code_path.exists() else []
            if candidate_files:
                st.caption("İlgili temizleme denetim izi (en son çalıştırma):")
                st.code(candidate_files[-1].read_text(encoding="utf-8"), language="python")


def main() -> None:
    st.set_page_config(page_title="Pareto — Varyans Paneli", layout="wide")
    st.title("📊 Pareto — Varyans Paneli")
    st.caption(
        "Tek bir kesin doğru cevap yok; işte savunulabilir seçimler menüsü ve "
        "her birinin sonucu."
    )

    args = _parse_args()
    results_path = st.sidebar.text_input("Sonuç dosyası", value=args.results)

    if not Path(results_path).exists():
        st.error(f"Sonuç dosyası bulunamadı: {results_path}")
        st.stop()

    df = load_results(results_path)

    n_ok = df["hata_mesaji"].isna().sum()
    n_fail = len(df) - n_ok
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Spesifikasyon", len(df))
    col2.metric("Başarılı Koşu", int(n_ok))
    col3.metric("Hatalı Koşu", int(n_fail))

    render_specification_curve(df)

    if st.button("AI ile Teşhis Et"):
        render_diagnosis(df)

    render_transparency_table(df)


if __name__ == "__main__":
    main()
