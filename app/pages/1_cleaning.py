"""1 - Cleaning (human-in-the-loop)."""

from __future__ import annotations

import streamlit as st

from pareto.profiling import load_raw_file, profile_dataframe

st.title("1 - Cleaning")

uploaded = st.file_uploader("Raw data (csv/tsv/xlsx/dta)", type=["csv", "tsv", "xlsx", "dta"])
if uploaded is not None:
    try:
        df = load_raw_file(uploaded)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["clean_df"] = df
        st.session_state["clean_profile"] = profile_dataframe(df)
        st.success(f"{len(df)} rows x {df.shape[1]} columns")
        st.subheader("Deterministic profile sent to the cleaning agent")
        st.json(st.session_state["clean_profile"])

st.info("Decision ledger + vetted transforms + gatekeeper approval: Sprint-2.")
