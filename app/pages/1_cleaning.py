"""1 - Cleaning (human-in-the-loop)."""

from __future__ import annotations

import json
import uuid

import streamlit as st

from pareto.cleaning.agent import Resolution, ResolvedDecision, generate_ledger, resolve
from pareto.cleaning.codegen import apply_ledger, render_audit_script
from pareto.cleaning.ledger import LedgerEntry, persist_ledger
from pareto.profiling import load_raw_file, profile_dataframe
from pareto.streamlit_ui import render_clean_panel, render_compact_sidebar

with st.sidebar:
    render_compact_sidebar()

st.title("1 - Temizleme")

# --------------------------------------------------------------------------- #
# Veri yükleme (mevcut davranış)
# --------------------------------------------------------------------------- #
if st.session_state.get("clean_df") is not None:
    df_saved = st.session_state["clean_df"]
    st.success(f"Oturumda yüklü veri: **{df_saved.shape[0]}** satır × **{df_saved.shape[1]}** kolon")
    with st.expander("Kayıtlı profil", expanded=False):
        st.json(st.session_state.get("clean_profile", {}))
    if st.button("Veriyi oturumdan sil"):
        for key in ("clean_df", "clean_profile", "clean_df_raw", "ledger", "resolutions", "run_id", "last_script"):
            st.session_state.pop(key, None)
        st.rerun()

uploaded = st.file_uploader(
    "Ham veri dosyası (csv/tsv/xlsx/dta)",
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
        # Yeni dosya yüklendiyse önceki ledger/karar geçmişi geçersiz.
        for key in ("clean_df_raw", "ledger", "resolutions", "run_id", "last_script"):
            st.session_state.pop(key, None)
        st.success(f"Yüklendi: {len(df)} satır × {df.shape[1]} kolon")
        st.subheader("Deterministik profil")
        st.json(st.session_state["clean_profile"])

# --------------------------------------------------------------------------- #
# resolutions -> apply_ledger'ın beklediği düz LedgerEntry listesi
# --------------------------------------------------------------------------- #
def _entries_to_apply(
    entries: list[LedgerEntry], resolutions: dict[int, ResolvedDecision]
) -> list[LedgerEntry]:
    """REJECTED elenir, MODIFIED'de params güncellenir, APPROVED aynen geçer."""
    out: list[LedgerEntry] = []
    for i, entry in enumerate(entries):
        decision = resolutions.get(i)
        if decision is None or decision.resolution == Resolution.REJECTED:
            continue
        if decision.resolution == Resolution.MODIFIED and decision.modified_params is not None:
            out.append(entry.model_copy(update={"params": decision.modified_params}))
        else:
            out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Decision ledger + gatekeeper (Sprint-2)
# --------------------------------------------------------------------------- #
if st.session_state.get("clean_df") is not None:
    st.divider()
    st.header("Karar defteri (decision ledger)")

    if st.button("Temizlik kararlarını üret (JUDGE)", type="primary"):
        try:
            entries = generate_ledger(st.session_state["clean_profile"])
        except ValueError as exc:
            st.error(f"JUDGE karar üretemedi: {exc}")
        else:
            st.session_state["ledger"] = entries
            st.session_state["resolutions"] = {}
            st.session_state["run_id"] = uuid.uuid4().hex
            st.session_state.setdefault("clean_df_raw", st.session_state["clean_df"].copy())
            st.session_state.pop("last_script", None)

            # Belirsizlik bayrağı olmayanlar: tek tık — otomatik onay, insan kapısı yok.
            for i, entry in enumerate(entries):
                if not entry.belirsizlik_bayragi:
                    st.session_state["resolutions"][i] = resolve(entry, auto_approve=True)
            st.rerun()

    entries = st.session_state.get("ledger")
    if entries:
        resolutions = st.session_state.setdefault("resolutions", {})

        auto = [(i, e) for i, e in enumerate(entries) if not e.belirsizlik_bayragi]
        flagged = [(i, e) for i, e in enumerate(entries) if e.belirsizlik_bayragi]

        if auto:
            st.subheader(f"Otomatik onaylanan kararlar ({len(auto)})")
            for i, e in auto:
                st.success(f"**{e.transform_name}** — {e.bulgu}")

        if flagged:
            st.subheader(f"Belirsiz kararlar — onayınız gerekli ({len(flagged)})")
            choices = [Resolution.APPROVED.value, Resolution.MODIFIED.value, Resolution.REJECTED.value]

            for i, entry in flagged:
                resolved_i = resolutions.get(i)
                status = f" — **{resolved_i.resolution.value}**" if resolved_i else " — *bekliyor*"
                with st.expander(f"Karar {i + 1}: {entry.transform_name}{status}", expanded=resolved_i is None):
                    st.write(f"**Bulgu:** {entry.bulgu}")
                    st.write(f"**Gerekçe:** {entry.gerekce}")
                    st.json(entry.params)

                    default_choice = resolved_i.resolution.value if resolved_i else Resolution.APPROVED.value
                    choice = st.radio(
                        "Karar",
                        options=choices,
                        index=choices.index(default_choice),
                        key=f"resolution_choice_{i}",
                        horizontal=True,
                    )

                    params_raw = None
                    if choice == Resolution.MODIFIED.value:
                        default_params = json.dumps(
                            (resolved_i.modified_params if resolved_i and resolved_i.modified_params else entry.params),
                            ensure_ascii=False,
                            indent=2,
                        )
                        params_raw = st.text_area(
                            "Değiştirilmiş params (JSON)", value=default_params, key=f"params_edit_{i}"
                        )

                    if st.button("Bu kararı kaydet", key=f"confirm_{i}"):
                        modified_params = None
                        parse_ok = True
                        if choice == Resolution.MODIFIED.value:
                            try:
                                modified_params = json.loads(params_raw)
                            except json.JSONDecodeError as exc:
                                st.error(f"Geçersiz JSON: {exc}")
                                parse_ok = False
                        if parse_ok:
                            try:
                                resolutions[i] = resolve(
                                    entry,
                                    auto_approve=False,
                                    resolution=Resolution(choice),
                                    modified_params=modified_params,
                                )
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))

        pending = [i for i, _ in flagged if i not in resolutions]
        all_resolved = len(pending) == 0
        if pending:
            st.warning(f"{len(pending)} belirsiz karar çözülmeden ilerlenemez.")

        if st.button("Kararları uygula (codegen + apply)", type="primary", disabled=not all_resolved):
            to_apply = _entries_to_apply(entries, resolutions)
            raw_df = st.session_state["clean_df_raw"]

            cleaned_df, audit_path = apply_ledger(raw_df, to_apply, st.session_state["run_id"])

            st.session_state["clean_df"] = cleaned_df
            st.session_state["clean_profile"] = profile_dataframe(cleaned_df)
            st.session_state["last_script"] = render_audit_script(to_apply)
            st.session_state["last_audit_path"] = str(audit_path)

            ledger_path = persist_ledger(entries, st.session_state["run_id"])
            st.session_state["last_ledger_path"] = str(ledger_path)
            st.rerun()

        if st.session_state.get("last_script"):
            st.divider()
            render_clean_panel(
                df_before=st.session_state["clean_df_raw"],
                df_after=st.session_state["clean_df"],
                script=st.session_state["last_script"],
            )
            st.caption(
                f"Audit script: `{st.session_state.get('last_audit_path', '')}` · "
                f"Karar defteri: `{st.session_state.get('last_ledger_path', '')}`"
            )