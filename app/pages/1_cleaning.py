"""1 - Cleaning (human-in-the-loop)."""

from __future__ import annotations

import streamlit as st

from pareto.profiling import load_raw_file, profile_dataframe
from pareto.streamlit_ui import render_compact_sidebar

with st.sidebar:
    render_compact_sidebar()

st.title("1 - Cleaning")

# Oturumda yüklü veri varsa göster (sayfa değişince kaybolmaz)
if st.session_state.get("clean_df") is not None:
    df_saved = st.session_state["clean_df"]
    st.success(f"Oturumda yüklü veri: **{df_saved.shape[0]}** satır × **{df_saved.shape[1]}** kolon")
    with st.expander("Kayıtlı profil", expanded=False):
        st.json(st.session_state.get("clean_profile", {}))
    if st.button("Veriyi oturumdan sil"):
        st.session_state.pop("clean_df", None)
        st.session_state.pop("clean_profile", None)
        st.rerun()

uploaded = st.file_uploader(
    "Raw data (csv/tsv/xlsx/dta)",
    type=["csv", "tsv", "xlsx", "dta"],
    key="cleaning_file_uploader",
)
if uploaded is not None:
    try:
        df = load_raw_file(uploaded)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["clean_df"] = df
        st.session_state["clean_profile"] = profile_dataframe(df)
        st.success(f"Yüklendi: {len(df)} satır × {df.shape[1]} kolon")
        st.subheader("Deterministic profile")
        st.json(st.session_state["clean_profile"])

st.info("Decision ledger + vetted transforms + gatekeeper approval: Sprint-2.")
