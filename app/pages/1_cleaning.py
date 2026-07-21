"""1 - Cleaning (human-in-the-loop)."""

from __future__ import annotations

import json
import uuid

import streamlit as st

from pareto.cleaning.agent import (
    Resolution,
    audit_entries,
    entries_to_apply,
    generate_ledger,
    resolve,
)
from pareto.cleaning.codegen import (
    ReproductionError,
    apply_ledger,
    render_audit_script,
    verify_reproduction,
)
from pareto.cleaning.ledger import persist_ledger
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
    st.success(
        f"Oturumda yüklü veri: **{df_saved.shape[0]}** satır × **{df_saved.shape[1]}** kolon"
    )
    with st.expander("Kayıtlı profil", expanded=False):
        st.json(st.session_state.get("clean_profile", {}))
    if st.button("Veriyi oturumdan sil"):
        for key in (
            "clean_df",
            "clean_profile",
            "clean_df_raw",
            "ledger",
            "resolutions",
            "run_id",
            "last_script",
        ):
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
# Decision ledger + gatekeeper (Sprint-2)
# --------------------------------------------------------------------------- #
if st.session_state.get("clean_df") is not None:
    st.divider()
    st.header("Karar defteri (decision ledger)")

    if st.button("Temizlik kararlarını üret (JUDGE)", type="primary"):
        # Ham veriyi burada sabitle: her üretim-uygulama turu (ilk ya da tekrar)
        # aynı ham dataframe'den başlasın; JUDGE'ın gördüğü profil ile
        # apply_ledger'ın uygulandığı veri hep tutarlı kalsın.
        raw_df = st.session_state.setdefault("clean_df_raw", st.session_state["clean_df"].copy())
        raw_profile = profile_dataframe(raw_df)
        try:
            entries = generate_ledger(raw_profile)
        except (ValueError, OSError) as exc:
            st.error(
                f"JUDGE karar üretemedi: {exc} (API anahtarı eksikse ana sayfada BYOK kaydedin.)"
            )
        else:
            st.session_state["clean_profile"] = raw_profile
            st.session_state["ledger"] = entries
            st.session_state["resolutions"] = {}
            st.session_state["run_id"] = uuid.uuid4().hex
            st.session_state.pop("last_script", None)

            # Belirsizlik bayrağı olmayanlar: tek tık — otomatik onay, insan kapısı yok.
            for i, entry in enumerate(entries):
                if not entry.belirsizlik_bayragi:
                    st.session_state["resolutions"][i] = resolve(entry, auto_approve=True)
            st.rerun()

    entries = st.session_state.get("ledger")
    if entries is not None and len(entries) == 0:
        st.info("JUDGE 0 karar üretti — veri temiz görünüyor.")

    if entries is not None and len(entries) > 0:
        resolutions = st.session_state.setdefault("resolutions", {})
        run_id = st.session_state["run_id"]

        auto = [(i, e) for i, e in enumerate(entries) if not e.belirsizlik_bayragi]
        flagged = [(i, e) for i, e in enumerate(entries) if e.belirsizlik_bayragi]

        if auto:
            st.subheader(f"Otomatik onaylanan kararlar ({len(auto)})")
            for _, e in auto:
                st.success(f"**{e.transform_name}** — {e.bulgu}")

        if flagged:
            st.subheader(f"Belirsiz kararlar — onayınız gerekli ({len(flagged)})")
            choices = [
                Resolution.APPROVED.value,
                Resolution.MODIFIED.value,
                Resolution.REJECTED.value,
            ]

            for i, entry in flagged:
                resolved_i = resolutions.get(i)
                status = f" — **{resolved_i.resolution.value}**" if resolved_i else " — *bekliyor*"
                title = f"Karar {i + 1}: {entry.transform_name}{status}"
                with st.expander(title, expanded=resolved_i is None):
                    st.write(f"**Bulgu:** {entry.bulgu}")
                    st.write(f"**Gerekçe:** {entry.gerekce}")
                    st.json(entry.params)

                    default_choice = (
                        resolved_i.resolution.value if resolved_i else Resolution.APPROVED.value
                    )
                    choice = st.radio(
                        "Karar",
                        options=choices,
                        index=choices.index(default_choice),
                        key=f"resolution_choice_{run_id}_{i}",
                        horizontal=True,
                    )

                    params_raw = None
                    if choice == Resolution.MODIFIED.value:
                        prior_params = resolved_i.modified_params if resolved_i else None
                        default_params = json.dumps(
                            prior_params or entry.params,
                            ensure_ascii=False,
                            indent=2,
                        )
                        params_raw = st.text_area(
                            "Değiştirilmiş params (JSON)",
                            value=default_params,
                            key=f"params_edit_{run_id}_{i}",
                        )

                    if st.button("Bu kararı kaydet", key=f"confirm_{run_id}_{i}"):
                        modified_params = None
                        parse_ok = True
                        if choice == Resolution.MODIFIED.value:
                            try:
                                modified_params = json.loads(params_raw)
                            except json.JSONDecodeError as exc:
                                st.error(f"Geçersiz JSON: {exc}")
                                parse_ok = False
                            else:
                                if not isinstance(modified_params, dict):
                                    st.error(
                                        'params bir JSON objesi (dict) olmalı, ör. {"col": "..."}.'
                                    )
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

        if st.button(
            "Kararları uygula (codegen + apply)", type="primary", disabled=not all_resolved
        ):
            to_apply = entries_to_apply(entries, resolutions)
            raw_df = st.session_state["clean_df_raw"]

            cleaned_df, audit_path = apply_ledger(raw_df, to_apply, st.session_state["run_id"])

            # L4 kapısı: diske yazılan script sandbox'ta koşulur ve çıktısı
            # in-process sonuçla tolerans içinde eşleşmek zorundadır. Eşleşmezse
            # akış burada durur, temiz veri oturuma yazılmaz (sessiz geçme yok).
            try:
                repro_dir = verify_reproduction(
                    raw_df, audit_path, cleaned_df, st.session_state["run_id"]
                )
            except ReproductionError as exc:
                st.error(f"Reprodüksiyon doğrulaması başarısız: {exc}")
                st.stop()
            st.session_state["last_repro_dir"] = str(repro_dir)

            st.session_state["clean_df"] = cleaned_df
            st.session_state["clean_profile"] = profile_dataframe(cleaned_df)
            st.session_state["last_script"] = render_audit_script(to_apply)
            st.session_state["last_audit_path"] = str(audit_path)

            # Diske TÜM kararlar yazılır (REJECTED dahil), her biri insanın verdiği
            # resolution ile damgalanmış olarak — insan kararı da denetim izinin
            # parçasıdır ve audit trail artık uygulanan `to_apply` ile çelişmez.
            audited = audit_entries(entries, resolutions)
            ledger_path = persist_ledger(audited, st.session_state["run_id"])
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
                f"Karar defteri: `{st.session_state.get('last_ledger_path', '')}` · "
                f"L4 repro sandbox: `{st.session_state.get('last_repro_dir', '')}`"
            )
