"""1 · Temizleme (human-in-the-loop).

SPRINT-2: yükle → profille → karar defteri (vetted transform kararları) → belirsizleri
gatekeeper'da sor → temiz CleanPanel + üretilen kod. Şimdilik profilleme deterministik
olarak çalışır; LLM karar üretimi ve gatekeeper UI Sprint-2 (bkz docs/scrum).
"""

from __future__ import annotations

import streamlit as st

from pareto.profiling import load_raw_file, profile_dataframe

st.title("1 · Temizleme")

uploaded = st.file_uploader("Ham veri (csv/tsv/xlsx/dta)", type=["csv", "tsv", "xlsx", "dta"])
if uploaded is not None:
    df = load_raw_file(uploaded.name) if hasattr(uploaded, "name") else None
    if df is not None:
        st.success(f"{len(df)} satır · {df.shape[1]} kolon")
        st.subheader("Deterministik profil (LLM'e giden özet payload)")
        st.json(profile_dataframe(df))

st.info("Karar defteri + vetted-transform üretimi + gatekeeper onayı: Sprint-2.")
