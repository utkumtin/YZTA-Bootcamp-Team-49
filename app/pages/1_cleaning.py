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
from pareto.cleaning.ledger import LedgerEntry, persist_ledger
from pareto.cleaning.uploads import uploaded_file_identity
from pareto.profiling import load_raw_file, profile_dataframe
from pareto.streamlit_ui import render_clean_panel, render_compact_sidebar

with st.sidebar:
    render_compact_sidebar()

st.title("1 - Temizleme")

# JUDGE turuna bağlı dinamik widget anahtarlarının prefix'leri. Anahtarlar
# `{prefix}{run_id}_{i}` biçiminde üretiliyor (bkz. `st.radio`/`st.text_area`/
# `st.button` çağrıları aşağıda), yani her yeni turda erişilemez hale geliyorlar.
_DYNAMIC_WIDGET_KEY_PREFIXES = ("resolution_choice_", "params_edit_", "confirm_")


def _purge_dynamic_widget_keys() -> None:
    """Erişilemez hale gelmiş JUDGE widget anahtarlarını session'dan siler.

    NEDEN tek yardımcı: temizliği iki akış yapıyor — yeni bir JUDGE turu
    başlarken ve "Veriyi oturumdan sil" ile oturum sıfırlanırken. İki yerde
    ayrı ayrı yazıldığında biri prefix listesine güncellenip diğeri geride
    kalıyordu; tek kaynak bunu imkânsız kılıyor.

    ÖLÇÜM NOTU: Streamlit'in kendisi, son koşuda render edilmeyen widget'ların
    state'ini zaten topluyor. Üç ardışık JUDGE turu ölçüldüğünde bu döngü ister
    çağrılsın ister çağrılmasın session'da yalnız içinde bulunulan turun
    anahtarları kalıyor — yani beklenen birikme gözlenmiyor ve döngünün
    silecek bir şeyi olmuyor. Savunma amaçlı tutuluyor (prefix'lerden biri
    ileride widget'a bağlı olmayan bir anahtara verilirse GC devreye girmez),
    ama bugün ölçülebilir bir etkisi yok; testle sabitlenmemesinin sebebi bu.

    `list(st.session_state)` stub'larda `str | int` dönebildiği için anahtar
    `str(...)` ile daraltılıyor; `.pop` orijinal anahtarla çağrılıyor.
    """
    for state_key in list(st.session_state):
        if str(state_key).startswith(_DYNAMIC_WIDGET_KEY_PREFIXES):
            st.session_state.pop(state_key)


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
            "last_repro_dir",
            "last_audit_path",
            "last_ledger_path",
            "cleaning_uploaded_file_id",
            "cleaning_file_uploader",
            "cleaning_failed_file_id",
            "cleaning_failed_file_error",
        ):
            st.session_state.pop(key, None)
        # Sabit anahtarlar yetmiyor: JUDGE turunun dinamik widget anahtarları da
        # bu yolda temizlenmeli, aksi halde silme sonrası yüklenen yeni dosyada
        # eski turun seçimleri session'da kalıyor.
        _purge_dynamic_widget_keys()
        st.rerun()

uploaded = st.file_uploader(
    "Ham veri dosyası (csv/tsv/xlsx/dta)",
    type=["csv", "tsv", "xlsx", "dta"],
    key="cleaning_file_uploader",
)
if uploaded is not None:
    # Streamlit her widget etkileşiminde script'i yeniden çalıştırır. Aynı
    # UploadedFile için ledger'ı sıfırlamak yerine, sadece gerçekten yeni bir
    # dosya geldiğinde oturumdaki karar akışını baştan başlat.
    # Streamlit 1.32+'da UploadedFile.file_id her zaman doludur (#53/2:
    # eski ad+boyut+hash fallback'i kaldırıldı, artık gerekmiyor).
    uploaded_file_id = uploaded_file_identity(uploaded)
    is_new_file = st.session_state.get("cleaning_uploaded_file_id") != uploaded_file_id
    # #53/1: bu dosya için daha önce bir deneme yapılıp başarısız olduysa,
    # `is_new_file` hâlâ True olur (çünkü başarılı bir yükleme hiç kaydedilmedi)
    # ve her otomatik rerun'da `load_raw_file` yeniden çağrılıp aynı hata banner'ı
    # tekrar tekrar basılırdı. `cleaning_failed_file_id` ile bu dosya için zaten
    # denendiğini ayrıca izliyoruz; retry artık yalnızca kullanıcının açıkça
    # bastığı "Tekrar dene" ile tetiklenir, otomatik rerun'larla değil.
    already_failed_this_file = st.session_state.get("cleaning_failed_file_id") == uploaded_file_id

    if is_new_file and not already_failed_this_file:
        try:
            df = load_raw_file(uploaded)
        except ValueError as exc:
            st.session_state["cleaning_failed_file_id"] = uploaded_file_id
            st.session_state["cleaning_failed_file_error"] = str(exc)
            st.error(str(exc))
        else:
            st.session_state.pop("cleaning_failed_file_id", None)
            st.session_state.pop("cleaning_failed_file_error", None)
            st.session_state["clean_df"] = df
            st.session_state["clean_profile"] = profile_dataframe(df)
            st.session_state["cleaning_uploaded_file_id"] = uploaded_file_id
            # Sadece yeni dosya yüklendiyse önceki ledger/karar geçmişi geçersiz.
            for key in (
                "clean_df_raw",
                "ledger",
                "resolutions",
                "run_id",
                "last_script",
                "last_repro_dir",
                "last_audit_path",
                "last_ledger_path",
            ):
                st.session_state.pop(key, None)
            st.success(f"Yüklendi: {len(df)} satır × {df.shape[1]} kolon")
            st.subheader("Deterministik profil")
            st.json(st.session_state["clean_profile"])
    elif already_failed_this_file:
        st.error(st.session_state.get("cleaning_failed_file_error", "Dosya yüklenemedi."))
        if st.button("Bu dosyayı Tekrar dene"):
            st.session_state.pop("cleaning_failed_file_id", None)
            st.session_state.pop("cleaning_failed_file_error", None)
            st.rerun()

# Uploader'dan dosyayı kaldırmak mevcut temizleme oturumunu korur; kullanıcı
# bunun yerine "Veriyi oturumdan sil" eylemini kullanarak açıkça sıfırlayabilir.

# --------------------------------------------------------------------------- #
# Decision ledger + gatekeeper (Sprint-2)
# --------------------------------------------------------------------------- #
if st.session_state.get("clean_df") is not None:
    st.divider()
    st.header("Karar defteri (decision ledger)")

    # mypy Error 2 (satır 161): bu değişken hem `generate_ledger(...)`
    # (list[LedgerEntry]) hem de `st.session_state.get("ledger")` (Any | None)
    # tarafından atanıyordu; açık anotasyon iki atamayı da tek bir tutarlı
    # tipe bağlıyor.
    entries: list[LedgerEntry] | None

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
            # Önceki JUDGE turunun dinamik widget değerleri artık erişilemez.
            # Uzun oturumlarda bu anahtarların session_state'te birikmesini önle.
            _purge_dynamic_widget_keys()
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
        run_id = str(st.session_state["run_id"])

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
                            # mypy Error 3 (satır 224): `params_raw` yalnızca bu
                            # daldayken set edilir ve `str | None` tipindedir;
                            # `json.loads` `None` kabul etmez.
                            if params_raw is None:
                                st.error("Beklenmeyen durum: parametre alanı boş.")
                                parse_ok = False
                            else:
                                try:
                                    modified_params = json.loads(params_raw)
                                except json.JSONDecodeError as exc:
                                    st.error(f"Geçersiz JSON: {exc}")
                                    parse_ok = False
                                else:
                                    if not isinstance(modified_params, dict):
                                        st.error(
                                            "params bir JSON objesi (dict) olmalı, "
                                            ' ör. {"col": "..."}.'
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
