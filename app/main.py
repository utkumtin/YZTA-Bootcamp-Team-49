"""Pareto — Streamlit girişi (Streamlit + Community Cloud).

review sorun: prototip CLI-first'tü. Artık Streamlit uçtan uca. Bu landing sayfası
modu (public/demo · private no-train) ve BYOK anahtar girişini kurar; akış sol
menüdeki sayfalarda: 1_cleaning → 2_analysis → 3_variance_panel.

Çalıştırma:  streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Pareto", page_icon="📊", layout="wide")

st.title("📊 Pareto")
st.caption(
    "Tek bir kesin cevap değil; savunulabilir seçimler üzerinde bir dağılım. "
    "Sözümüz: **savunulabilir sonuç.**"
)

with st.sidebar:
    st.header("Ayarlar")
    mode = st.radio(
        "Gizlilik modu",
        options=["public", "private"],
        help="public = free model + canned demo. private = yalnız no-train uçlar.",
    )
    st.text_input(
        "BYOK — API anahtarı",
        type="password",
        help="Canlı LLM için kendi anahtarın (env / st.secrets). Repo'ya asla yazılmaz.",
    )
    st.caption(f"Aktif mod: **{mode}**")

st.subheader("Akış")
st.markdown(
    "1. **Temizleme** — profil → karar defteri + üretilen kod (human-in-the-loop)\n"
    "2. **Analiz** — estimand/H0-H1 → savunulabilir spec menüsü (dondurulur)\n"
    "3. **Varyans Paneli** — çokluevren sonuçları: spec curve + 3-bant kırılganlık teşhisi"
)

st.info(
    "Sprint-1 iskeleti kuruldu. Temizleme agent'ı, spec-menü üretimi ve canlı LLM "
    "akışı Sprint-2 kapsamında (bkz docs/scrum/sprint-2-plan.md). Varyans paneli, "
    "runner çıktısı `runs/<run_id>/results.json` üzerinden şimdiden çalışır."
)
