"""3 · Varyans Paneli — ürünün ruhu.

Prototip `p4_dashboard/app.py:54-88` spec-curve çizimi (CI barları, anlamlılık
renklendirmesi) buraya migre edildi; artık tipli `EstimationResult` okur ve
3-bant robust/fragile kuralını (deterministik) gösterir. Matched-pair/ANOVA teşhisi ve
LLM narrative Sprint-2 (variance.diagnose_axes + router JUDGE narrative).

Girdi: runner çıktısı `runs/<run_id>/results.json`.

Sayfa düzeni üç katmandır ve bu sıra bilinçlidir:
  1. Karar — bant + iki eşik çubuğu. Kullanıcının cevabını aradığı tek şey.
  2. Kanıt — spec curve, sonra dört teşhis bloğu (sekmeli ya da ızgara).
  3. Paket — denetim izinin tek dosyalık hâli.
"""

from __future__ import annotations

import io
import json
import zipfile
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pareto.analysis.event_study import estimate_pretrend_event_study
from pareto.analysis.event_study_columns import (
    build_event_study_payload,
)
from pareto.analysis.event_study_columns import (
    event_study_cache_key as _event_study_cache_key,
)
from pareto.analysis.variance import (
    ROBUST_RULE_TEXT,
    SIG_ROBUST,
    SIGN_FRAGILE,
    SIGN_ROBUST,
    diagnose_axes,
    summarize,
)
from pareto.config import SETTINGS
from pareto.contracts import EstimationResult
from pareto.llm.cache import CannedModeCacheMissError
from pareto.llm.narrative import generate_narrative
from pareto.memory.frozen_menu import load_frozen_menu_record
from pareto.repro import (
    ARTIFACT_LABELS,
    ReproInputs,
    ReproPackageError,
    build_reproduction_package,
    figure_html,
    missing_artifacts,
    package_key,
)
from pareto.spec import Specification
from pareto.streamlit_ui import llm_call_status, render_compact_sidebar, render_page_title

with st.sidebar:
    render_compact_sidebar()


# --------------------------------------------------------------------------- #
# Görsel dil
# --------------------------------------------------------------------------- #
# NEDEN sayfaya özel `pa-vp-` öneki: `_SETTINGS_STYLE_HTML` (streamlit_ui.py) yalnız Ayarlar
# sekmesinde enjekte ediliyor, oradaki `.pa-eyebrow` bu sayfada tanımlı değil. Aynı tipografi
# değerleri burada bir kez daha yazılıyor; ortak bir token dosyasına çıkarmak ayrı bir iş,
# iki blok arasında sürüklenme riski var.
#
# Bant renkleri uygulamada zaten kullanılan üç tondan geliyor (yeşil/amber/kırmızı;
# gizlilik rozetleri ve koyu tema grafik paleti). Panele yeni renk sokulmuyor.
_BAND_COLOR = {"robust": "#4ade80", "mixed": "#fbbf24", "fragile": "#f87171"}
_BAND_LABEL = {
    "robust": ("SAĞLAM", "robust"),
    "mixed": ("KARIŞIK", "mixed"),
    "fragile": ("KIRILGAN", "fragile"),
}
_NO_BAND_COLOR = "rgba(250, 250, 250, 0.35)"


def _tint(hex_color: str, alpha: float) -> str:
    """Bant renginin saydam tonu.

    NEDEN Python'da hesaplanıyor, CSS `color-mix` ile değil: `st.html` gövdesi tarayıcıda
    sanitize ediliyor ve bu sayfa görsel olarak doğrulanamıyor (ortamda tarayıcı yok).
    Düz `rgba(...)` her yerde çalışır, `color-mix` düşerse kenarlık sessizce kaybolurdu.
    """
    if not hex_color.startswith("#"):
        return hex_color
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


_STYLE_HTML = """
<style>
/* Kaynak şeridi: solda sayfa alt başlığı, sağda kaynak menüsü. İki flex çocuğu var,
   space-between ikisini uçlara itiyor. */
.st-key-pa_vp_source {
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}

/* NEDEN Ayarlar panelindeki `.pa-eyebrow`dan (0.64rem) daha büyük: orada eyebrow bir rayın
   üstündeki mikro etiket, burada `st.subheader`ın yerini alan bölüm başlığı. Aynı isim,
   farklı iş; 0.64rem'de başlık olduğu anlaşılmıyordu. */
.pa-vp-eyebrow {
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: rgba(250, 250, 250, 0.68);
    margin: 0 0 12px 2px;
}

/* --- Karar kartı --- */
/* Kenarlık, sol şerit ve zemin rengi bant rengine bağlı, satır içi veriliyor. */
.pa-vp-verdict {
    border: 1px solid;
    border-left-width: 3px;
    border-radius: 12px;
    padding: 16px 20px 14px;
    margin-bottom: 26px;
}
.pa-vp-verdict-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.pa-vp-band {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.015em;
}
/* NEDEN kartın içindeki gri tonları sayfa geneline göre daha açık: kart zemini bant
   renginin %7'lik tonu, yani düz koyu zeminden biraz açık. Sayfada okunan 0.35-0.55
   aralığındaki griler bu zeminde kontrastını kaybediyordu; ikincil metinler beyaza
   yaklaştırıldı. Renk sırası korunuyor: kural notu hâlâ en sönük satır. */

/* Literatürdeki terim: makaleye yazarken hangi kelimeye denk geldiği kaybolmasın diye
   duruyor, ama okuma sırasını bölmeyecek kadar sönük. */
.pa-vp-band-en {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(250, 250, 250, 0.62);
    margin-left: 10px;
}
.pa-vp-run {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.74rem;
    color: rgba(250, 250, 250, 0.68);
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 6px;
    padding: 2px 8px;
}
.pa-vp-facts {
    font-size: 0.86rem;
    color: rgba(250, 250, 250, 0.78);
    margin-top: 6px;
}
.pa-vp-facts b { color: #fafafa; font-weight: 600; }

/* --- Eşik çubukları --- */
.pa-vp-meters {
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.pa-vp-meter { flex: 1 1 260px; min-width: 0; }
.pa-vp-meter-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    font-size: 0.82rem;
    color: rgba(250, 250, 250, 0.78);
    margin-bottom: 6px;
}
.pa-vp-meter-head b {
    color: #fafafa;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.pa-vp-track {
    position: relative;
    height: 6px;
    border-radius: 999px;
    background: rgba(250, 250, 250, 0.1);
}
.pa-vp-fill {
    height: 100%;
    border-radius: 999px;
}
/* Eşik çentiği çubuğun üstüne taşıyor: dolgunun nerede bittiği ile kuralın nerede
   başladığı tek bakışta karşılaştırılabilsin. */
.pa-vp-tick {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 14px;
    border-radius: 1px;
    background: rgba(250, 250, 250, 0.55);
}
.pa-vp-meter-foot {
    font-size: 0.74rem;
    color: rgba(250, 250, 250, 0.62);
    margin-top: 8px;
}
.pa-vp-rule {
    font-size: 0.75rem;
    line-height: 1.5;
    color: rgba(250, 250, 250, 0.58);
    border-top: 1px solid rgba(250, 250, 250, 0.12);
    margin-top: 16px;
    padding-top: 10px;
}

/* --- Spec curve lejantı --- */
/* Lejant bir açıklama değil, grafiğin okunma anahtarı: gövde metniyle aynı okunurlukta. */
.pa-vp-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 0.8rem;
    color: rgba(250, 250, 250, 0.75);
    margin: 2px 0 26px 2px;
}
.pa-vp-legend span { display: inline-flex; align-items: center; gap: 7px; }
.pa-vp-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    display: inline-block;
}

/* Kanıt başlığı ile düzen anahtarı aynı satırda; anahtar sağa yaslı. */
.st-key-pa_vp_evidence_head {
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.st-key-evidence_layout button { font-size: 0.78rem; }
</style>
"""

st.html(_STYLE_HTML)


def _pct(value: float | None) -> str:
    """Oranı yüzde olarak yazar.

    NEDEN `is None`: eski hâli doğruluk (`if value`) sınıyordu ve gerçek bir %0
    anlamlılık oranı "—" olarak, yani veri yokmuş gibi görünüyordu. Panelin en
    alarm verici sonucu tam olarak %0 anlamlılık; onu boşluk gibi göstermek
    ürünün iddiasını tersine çeviriyordu.
    """
    return "—" if value is None else f"%{value * 100:.0f}"


render_page_title("normal-curve", "Varyans Paneli")

# --------------------------------------------------------------------------- #
# Kaynak şeridi
# --------------------------------------------------------------------------- #
# Sonuç yolu bir tesisat kontrolü, sayfanın konusu değil: menüye alınıyor. Sayfanın
# en üstünde tam genişlikte bir metin kutusu olarak durduğunda, cevabı aramaya gelen
# kullanıcının gördüğü ilk şey oluyordu.
# local-only: dosya sistemi erişimi güvenilir ortamda varsayılır.
with st.container(horizontal=True, vertical_alignment="center", key="pa_vp_source"):
    st.caption("Tek kesin cevap yok; savunulabilir seçimler menüsü ve her birinin sonucu.")
    _source_menu = st.popover("Kaynak", icon=":material/database:")

with _source_menu:
    results_path = st.text_input(
        "Sonuç dosyası",
        value=st.session_state.get("multiverse_results_path", ""),
    )

if not results_path:
    st.info("Henüz bir multiverse koşusu yok. Önce **Analiz** sayfasından bir koşu başlat.")
    st.stop()

if not Path(results_path).exists():
    st.warning(
        f"Sonuç dosyası yok: {results_path}. Önce multiverse runner koş, ya da "
        "sağ üstteki Kaynak menüsünden başka bir dosya göster."
    )
    st.stop()

# NEDEN sarmalama: multiverse yarıda kesilirse results.json eksik/bozuk yazılıyor. Sarmalama
# olmadan bu durum kullanıcıya ham traceback olarak düşüyor, ne olduğu ve ne yapılacağı
# anlaşılmıyor.
try:
    raw = json.loads(Path(results_path).read_text(encoding="utf-8"))
    results = [EstimationResult(**r) for r in raw]
except (OSError, ValueError) as exc:
    st.error(
        f"Sonuç dosyası okunamadı: {results_path} ({exc}). Dosya yarıda kalmış olabilir; "
        "multiverse'i yeniden koşun."
    )
    st.stop()

summary = summarize(results)

specs_path = Path(results_path).with_name("specs.json")
if specs_path.exists():
    specs = [Specification(**s) for s in json.loads(specs_path.read_text(encoding="utf-8"))]
else:
    specs = []


def _resolve_run_id(results_file: Path) -> str:
    """Sonuç dizininin gerçek run_id'si.

    Öncelik dizinin yanındaki `run_id.txt`tedir: `runs/latest` bir aynadır, dizin
    adı "latest"tir ve oturumdaki run_id kullanıcının elle girdiği eski bir koşuya
    ait olabilir. run_id yanlış çözülürse donmuş menü bulunamaz ve METHODS.md
    estimand bölümünü tamamen kaybeder.
    """
    marker = results_file.with_name("run_id.txt")
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    return str(st.session_state.get("multiverse_run_id") or results_file.parent.name)


run_id = _resolve_run_id(Path(results_path))


# Panelde çizilen figürler reprodüksiyon paketine de girer: dosya adı -> HTML gövdesi.
package_figures: dict[str, str] = {}


def _capture_figure(name: str, figure: go.Figure) -> None:
    """Figürü paket için saklar. Render kuralları `pareto.repro.figure_html`te."""
    package_figures[name] = figure_html(name, figure)


# Aynı figür iki farklı zeminde yaşıyor: uygulama teması `.streamlit/config.toml`da koyuya
# sabitlendi, ama `_capture_figure` ile pakete giren HTML tarayıcıda tek başına, büyük
# olasılıkla beyaz sayfada açılıyor. Bu yüzden figür `simple_white` ile üretilip pakete
# öyle giriyor; ekrana çizilen kopya `_for_dark_ui` ile koyulaştırılıyor. Sıra önemli değil,
# kopya alındığı için orijinal figür bozulmuyor.
#
# Marker renkleri (`_color`) beyaz zemine göre seçilmiş; koyuda anlam sırası korunacak
# şekilde yeniden eşleniyor: başarısız en sönük, anlamsız orta, anlamlı renkli.
_DARK_MARKER_COLOR = {
    "lightgray": "#4b5563",
    "gray": "#9ca3af",
    "seagreen": "#4ade80",
    "indianred": "#f87171",
}
# Sıfır çizgisi ve referans dönem çizgisi `line_color="black"` ile ekleniyor — koyu zeminde
# görünmez. İkisi de layout shape'i olduğu için tek `update_shapes` çağrısı ikisini de kapsar.
_DARK_LINE_COLOR = "rgba(250, 250, 250, 0.45)"
# Hata çubukları (güven aralığı) `error_y.color` verilmediğinde marker rengini miras alır;
# ama marker rengi burada nokta başına bir DİZİ (bkz. `_color`) ve Plotly diziyi hata
# çubuğuna eşleyemiyor, varsayılan koyu griye (#444) düşüyor — koyu zeminde kayboluyor.
# `error_y.color` skaler olmak zorunda, bu yüzden noktaların rengini takip edemiyor; nötr
# ve okunur bir ton seçiliyor. Kılavuz çizgilerinden biraz daha güçlü: bu bir veri katmanı.
_DARK_ERROR_BAR_COLOR = "rgba(250, 250, 250, 0.55)"


def _for_dark_ui(figure: go.Figure) -> go.Figure:
    """Figürün ekran kopyasını verir; orijinal (paket kopyası) değişmez.

    Başlık ekran kopyasında düşürülüyor: sayfada her grafiğin üstünde zaten bir bölüm
    başlığı var, figürün kendi başlığı onu tekrarlıyordu. Pakete giren kopyada başlık
    KALIYOR, çünkü o HTML tek başına, bağlamsız açılıyor.
    """
    dark = go.Figure(figure)
    dark.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=None,
        # Plotly varsayılan boşlukları başlıklı bir figüre göre (t=100, l=80): başlık
        # düşünce üstte ölü boşluk, ızgara düzeninde yarım genişlikteki grafikte ise solda
        # çizim alanının ~%15'i kadar boş gutter kalıyordu. Daraltılıyor, `automargin` ise
        # eksen başlığı/etiketleri sığmazsa boşluğu geri büyütüyor, yani kırpılma olmuyor.
        margin={"t": 30, "l": 45, "r": 20, "b": 45},
        xaxis={"automargin": True},
        yaxis={"automargin": True},
    )
    dark.update_shapes(line_color=_DARK_LINE_COLOR)
    for trace in dark.data:
        color = trace.marker.color
        if isinstance(color, str):
            trace.marker.color = _DARK_MARKER_COLOR.get(color, color)
        elif color is not None:
            trace.marker.color = [_DARK_MARKER_COLOR.get(c, c) for c in color]
        # Yalnız renk verilmemişse dokunuluyor: çağıran bilinçli bir renk seçtiyse korunur.
        # Hata çubuğu olmayan izde `error_y` yine de var ama çizilmiyor; atama no-op.
        if trace.error_y.color is None:
            trace.error_y.color = _DARK_ERROR_BAR_COLOR
    return dark


def _eyebrow(text: str) -> None:
    st.html(f'<div class="pa-vp-eyebrow">{escape(text)}</div>')


def _render_run_provenance(*, detail) -> None:
    """Run künyesini `detail` kabına yazar, uyuşmazlığı sayfa seviyesine çıkarır.

    Kimlik bakılan dosyadan çözülür, oturumdan değil: kullanıcı eski bir koşunun
    sonuç yolunu elle yazdığında aşağıdaki hash karşılaştırması o koşuya ait olmalı.
    Oturuma öncelik verilirse gösterge tam da uyarması gereken durumda yanlış koşuyu
    doğruluyor.

    NEDEN uyuşmazlık menünün DIŞINDA: künye doğru olduğunda kimse bakmaz, yanlış
    olduğunda her şeyi geçersiz kılar. Doğrulama sessiz kalıp menüde beklerken,
    uyuşmazlık kullanıcının açması gerekmeyen bir yere, sayfaya düşüyor.
    """
    record = load_frozen_menu_record(run_id)

    detail.caption(f"Run: `{run_id}`")
    if record is None:
        detail.caption(
            "Bu run için ProjectStore'da dondurulmuş estimand/menu kaydı bulunamadı "
            "(run doğrudan CLI/worker ile üretilmiş olabilir)."
        )
        return

    stored_estimand_hash = record.get("estimand_hash")
    stored_menu_hash = record.get("menu_hash")
    stored_spec_count = record.get("spec_count")
    detail.caption(
        f"Kayıtlı estimand_hash=`{stored_estimand_hash}` · "
        f"menu_hash=`{stored_menu_hash}` · spec_count={stored_spec_count}"
    )

    current_estimand = st.session_state.get("frozen_estimand")
    current_estimand_hash = getattr(current_estimand, "freeze_hash", None)

    if current_estimand_hash and stored_estimand_hash:
        if current_estimand_hash == stored_estimand_hash:
            detail.success(
                "Oturumdaki estimand, bu run'ı üreten dondurulmuş estimand ile eşleşiyor."
            )
        else:
            st.warning(
                "Oturumdaki estimand hash'i, bu run'ı üreten dondurulmuş estimand ile "
                "**eşleşmiyor**. Aşağıdaki sonuçlar farklı bir estimand'dan üretilmiş olabilir."
            )
    elif stored_estimand_hash:
        detail.info("Oturumda aktif bir estimand yok; run künyesi yalnız kayıttan gösteriliyor.")


_render_run_provenance(detail=_source_menu)


# --------------------------------------------------------------------------- #
# Karar
# --------------------------------------------------------------------------- #
def _meter_html(
    *,
    label: str,
    value: float | None,
    ticks: tuple[float, ...],
    foot: str,
    color: str,
) -> str:
    """Tek koşullu bir eşik çubuğu.

    Çentikler `variance.py`deki eşik sabitlerinden geliyor: kural yeniden kalibre
    edilirse çubuk da onunla birlikte kayar, elle güncellenecek ikinci bir yer yok.
    """
    filled = 0.0 if value is None else max(0.0, min(1.0, value)) * 100
    tick_html = "".join(
        f'<div class="pa-vp-tick" style="left:{t * 100:.4g}%"></div>' for t in ticks
    )
    return (
        '<div class="pa-vp-meter">'
        f'<div class="pa-vp-meter-head"><span>{escape(label)}</span><b>{_pct(value)}</b></div>'
        '<div class="pa-vp-track">'
        f'<div class="pa-vp-fill" style="width:{filled:.4g}%;background:{color}"></div>'
        f"{tick_html}</div>"
        f'<div class="pa-vp-meter-foot">{escape(foot)}</div>'
        "</div>"
    )


def _verdict_html(summary: dict[str, Any]) -> str:
    """Bant kararı + iki eşik çubuğu.

    NEDEN iki ayrı çubuk: kural bir VE bağlacı (işaret-uyumu ≥%95 VE anlamlılık ≥%70).
    Tek çubukta işaret-uyumu eşiği geçmişken etiket KARIŞIK yazar ve ekranda hata gibi
    okunur; iki çubuk hangi koşulun tutmadığını doğrudan gösterir.

    NEDEN iki çubuğun renk kuralı farklı: işaret-uyumu tek başına KIRILGAN'a düşürebilen
    tek ölçü (`SIGN_FRAGILE` altı), anlamlılığın böyle bir yetkisi yok. Renkler bu
    asimetriyi taşıyor, ikisini aynı göstermek kuralı yanlış anlatırdı.
    """
    band = str(summary.get("band") or "")
    color = _BAND_COLOR.get(band, _NO_BAND_COLOR)
    label_tr, label_en = _BAND_LABEL.get(band, ("KARAR YOK", ""))
    sign = summary.get("sign_agreement")
    significance = summary.get("significance_rate")

    facts = [f"<b>{summary['n_total']}</b> spesifikasyon", f"<b>{summary['n_ok']}</b> sonuç"]
    if summary.get("point_min") is not None:
        facts.append(f"etki aralığı <b>{summary['point_min']:.3g} … {summary['point_max']:.3g}</b>")

    if band:
        sign_color = (
            _BAND_COLOR["robust"]
            if sign is not None and sign >= SIGN_ROBUST
            else _BAND_COLOR["fragile"]
            if sign is not None and sign < SIGN_FRAGILE
            else _BAND_COLOR["mixed"]
        )
        body = (
            '<div class="pa-vp-meters">'
            + _meter_html(
                label="İşaret-uyumu",
                value=sign,
                ticks=(SIGN_FRAGILE, SIGN_ROBUST),
                foot=f"{_pct(SIGN_FRAGILE)} altı kırılgan · {_pct(SIGN_ROBUST)} ve üstü sağlam",
                color=sign_color,
            )
            + _meter_html(
                label="Anlamlılık",
                value=significance,
                ticks=(SIG_ROBUST,),
                foot=f"{_pct(SIG_ROBUST)} ve üstü sağlam",
                color=(
                    _BAND_COLOR["robust"]
                    if significance is not None and significance >= SIG_ROBUST
                    else _NO_BAND_COLOR
                ),
            )
            + "</div>"
        )
    else:
        body = (
            '<div class="pa-vp-facts" style="margin-top:14px">'
            "Hiçbir spesifikasyon katsayı üretmedi, bu yüzden bant kuralı uygulanamıyor."
            "</div>"
        )

    return (
        f'<div class="pa-vp-verdict" style="border-color:{_tint(color, 0.3)};'
        f'border-left-color:{color};background:{_tint(color, 0.07)}">'
        '<div class="pa-vp-verdict-head"><div>'
        f'<span class="pa-vp-band" style="color:{color}">{label_tr}</span>'
        f'<span class="pa-vp-band-en">{label_en}</span>'
        "</div>"
        f'<span class="pa-vp-run">{escape(run_id)}</span></div>'
        f'<div class="pa-vp-facts">{" · ".join(facts)}</div>'
        f"{body}"
        f'<div class="pa-vp-rule">{escape(ROBUST_RULE_TEXT)}</div>'
        "</div>"
    )


_eyebrow("Karar")
st.html(_verdict_html(summary))


# --------------------------------------------------------------------------- #
# Spesifikasyon eğrisi
# --------------------------------------------------------------------------- #
def _plot_coefficient_series(
    *,
    x,
    y,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    hovertemplate: str,
    marker_size: int = 8,
    marker_color: list[str] | None = None,
    connectgaps: bool = False,
    add_zero_line: bool = True,
    zero_line_y: float = 0.0,
    add_vline_x: int | float | None = None,
    height: int = 360,
    trace_mode: str = "markers",
    trace_name: str | None = None,
    error_y: dict | None = None,
) -> go.Figure:
    fig = go.Figure()
    marker: dict[str, Any] = {"size": marker_size}
    if marker_color is not None:
        marker["color"] = marker_color
    trace_kwargs = {
        "x": x,
        "y": y,
        "mode": trace_mode,
        "marker": marker,
        "hovertemplate": hovertemplate,
    }
    if trace_name is not None:
        trace_kwargs["name"] = trace_name
    if error_y is not None:
        trace_kwargs["error_y"] = error_y
    if not connectgaps:
        trace_kwargs["connectgaps"] = False
    fig.add_trace(go.Scatter(**trace_kwargs))
    if add_vline_x is not None:
        fig.add_vline(x=add_vline_x, line_dash="dot", line_color="black", opacity=0.6)
    if add_zero_line:
        fig.add_hline(y=zero_line_y, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="simple_white",
        height=height,
    )
    return fig


def _color(r: EstimationResult) -> str:
    if r.status != "ok" or r.p_value is None or r.coefficient is None:
        return "lightgray"
    if r.p_value >= 0.05:
        return "gray"
    return "seagreen" if r.coefficient > 0 else "indianred"


# Grafiğin tek okunur sinyali renk ve Plotly tarafında bir lejant yok: nokta başına renk
# verilen tek izde Plotly tek bir lejant girdisi çizer, hangi rengin ne demek olduğunu
# söyleyemez. Lejant bu yüzden elle kuruluyor.
_LEGEND_TEXT = {
    "seagreen": "anlamlı, pozitif",
    "indianred": "anlamlı, negatif",
    "gray": "anlamsız (p ≥ 0,05)",
    "lightgray": "p-değeri yok",
}


def _legend_html(colors: list[str]) -> str:
    """Yalnız grafikte gerçekten geçen renkleri açıklar.

    Boş bir lejant girdisi kullanıcıya olmayan bir kova gösterir; kovaların hepsini
    sabit yazmak, tek yönlü bir eğride "diğer yön de var" izlenimi verir.
    """
    present = [key for key in _LEGEND_TEXT if key in colors]
    chips = "".join(
        f'<span><i class="pa-vp-dot" style="background:{_DARK_MARKER_COLOR[key]}"></i>'
        f"{escape(_LEGEND_TEXT[key])}</span>"
        for key in present
    )
    return f'<div class="pa-vp-legend">{chips}</div>'


# (sonuç, katsayı) çifti: katsayının None olmadığı filtrede bir kez daraltılır,
# aşağıdaki sıralama ve hata çubuğu hesapları aynı daraltılmış değeri kullanır.
ok = [(r, r.coefficient) for r in results if r.status == "ok" and r.coefficient is not None]
if ok:
    _eyebrow("Spesifikasyon eğrisi · katsayıya göre sıralı")
    spec_colors = [_color(r) for r, _coefficient in ok]
    fig = _plot_coefficient_series(
        x=list(range(len(ok))),
        y=[coefficient for _result, coefficient in ok],
        title="Specification Curve (katsayıya göre sıralı)",
        xaxis_title="Spesifikasyonlar",
        yaxis_title="Tahmini etki (katsayı)",
        hovertemplate="%{text}<br>katsayı=%{y:.4f}<extra></extra>",
        marker_size=8,
        marker_color=spec_colors,
        trace_mode="markers",
        add_zero_line=True,
        height=460,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [(r.ci_high - coefficient) if r.ci_high else 0 for r, coefficient in ok],
            "arrayminus": [(coefficient - r.ci_low) if r.ci_low else 0 for r, coefficient in ok],
            "thickness": 1,
            "width": 0,
        },
    )
    fig.data[0].text = [r.spec_id for r, _coefficient in ok]
    st.plotly_chart(_for_dark_ui(fig), use_container_width=True)
    st.html(_legend_html(spec_colors))
    _capture_figure("specification_curve.html", fig)
else:
    st.warning(
        "Katsayı üreten spesifikasyon yok, bu yüzden spec curve çizilemiyor. Aşağıdaki "
        "şeffaflık makbuzları tablosunda her spesifikasyonun hata nedeni yazıyor."
    )


# --------------------------------------------------------------------------- #
# Kanıt blokları
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _diagnose_axes_cached(results_json: str, specs_json: str) -> dict[str, Any]:
    return diagnose_axes(
        [EstimationResult(**item) for item in json.loads(results_json)],
        [Specification(**item) for item in json.loads(specs_json)],
    )


def _render_axis_attribution() -> None:
    if not specs:
        st.info("Bu run için specs.json bulunamadı; eksen atfı ve narrative gösterilemiyor.")
        return

    results_json = json.dumps([r.model_dump() for r in results], ensure_ascii=False)
    specs_json = json.dumps([s.model_dump() for s in specs], ensure_ascii=False)
    diagnosis = _diagnose_axes_cached(results_json, specs_json)

    excluded_count = sum(diagnosis["n_excluded"].values())
    st.caption(f"Kullanılan sonuç: {diagnosis['n_used']} · Hariç tutulan: {excluded_count}")
    matched_pairs = pd.DataFrame.from_dict(diagnosis.get("matched_pairs", {}), orient="index")
    if not matched_pairs.empty:
        st.dataframe(matched_pairs, use_container_width=True)
    else:
        st.caption("Eşleşen spesifikasyon çifti yok; eksen başına etki farkı hesaplanamadı.")
    anova_r2 = pd.DataFrame([diagnosis.get("anova_partial_r2", {})]).T
    if not anova_r2.empty:
        anova_r2.columns = ["partial_r2"]
        st.dataframe(anova_r2, use_container_width=True)
    else:
        st.caption("Partial R² tablosu boş; varyansı eksenlere dağıtacak kadar sonuç yok.")
    if diagnosis.get("warnings"):
        for warning in diagnosis.get("warnings", []):
            st.caption(f"• {warning}")

    narrative_key = json.dumps(
        {"summary": summary, "diagnosis": diagnosis}, ensure_ascii=False, sort_keys=True
    )
    if st.button("LLM narrative oluştur"):
        try:
            with llm_call_status("JUDGE varyans anlatısını hazırlıyor…"):
                st.session_state["_variance_narrative"] = generate_narrative(summary, diagnosis)
                st.session_state["_variance_narrative_key"] = narrative_key
        except (ValueError, CannedModeCacheMissError) as exc:
            # Narrative opsiyonel bir ek: canned modda cache'te olmayan bir
            # anlatı istendiğinde akış durmaz, uyarı ile geçilir.
            st.warning(f"Narrative oluşturulamadı: {exc}")

    if st.session_state.get("_variance_narrative_key") == narrative_key:
        narrative = st.session_state.get("_variance_narrative")
        if narrative is not None:
            _eyebrow("LLM narrative")
            st.write(narrative.ozet)
            for comment in narrative.eksen_yorumlari:
                st.caption(f"• {comment.axis}: {comment.yorum}")


def _render_effective_n() -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "spec_id": result.spec_id,
                "effective_n": int(result.n_obs)
                if (result.status == "ok" and result.n_obs is not None)
                else None,
                "status": result.status,
                "error": result.error if result.status != "ok" else None,
            }
        )
    if not rows:
        st.info("Efektif N bilgisi bulunamadı.")
        return
    df_n = pd.DataFrame(rows)
    st.dataframe(df_n, use_container_width=True)
    n_missing = int(df_n["effective_n"].isna().sum())
    if n_missing:
        st.caption(
            f"{n_missing} spesifikasyon için efektif N üretilemedi "
            "(yukarıdaki 'status'/'error' sütununa bakın)."
        )


def _render_pretrend() -> None:
    payload = build_event_study_payload(
        st.session_state.get("clean_df"),
        estimand=st.session_state.get("frozen_estimand"),
        analysis_state=st.session_state.get("analysis_state"),
    )
    event_study = None
    if payload is None:
        st.info("Pre-trend görseli için temizlenmiş veri seti yok.")
        return
    if payload["status"] == "skipped":
        st.info(payload["reason"])
        return

    sources = payload["sources"]
    st.caption(
        "Kullanılacak kolonlar → "
        f"outcome: `{payload['outcome_col']}` ({sources.get('outcome_col', 'unknown')}), "
        f"unit: `{payload['unit_col']}` ({sources.get('unit_col', 'unknown')}), "
        f"time: `{payload['time_col']}` ({sources.get('time_col', 'unknown')})"
    )
    column_options = payload["column_options"]
    cohort_col = payload["cohort_col"]
    never_treated_col = payload["never_treated_col"]

    if cohort_col is None:
        cohort_choice = st.selectbox(
            "Kohort kolonu",
            options=["-- seçin --", *column_options],
            key="event_study_cohort_choice",
        )
        cohort_col = None if cohort_choice == "-- seçin --" else cohort_choice
    else:
        st.caption(f"Kohort kolonu: `{cohort_col}` ({sources.get('cohort_col', 'unknown')})")

    if never_treated_col is None:
        never_choice = st.selectbox(
            "Never-treated kolonu",
            options=["-- seçin --", *column_options],
            key="event_study_never_treated_choice",
        )
        never_treated_col = None if never_choice == "-- seçin --" else never_choice
    else:
        never_source = sources.get("never_treated_col", "unknown")
        st.caption(f"Never-treated kolonu: `{never_treated_col}` ({never_source})")

    if sources.get("cohort_col") == "inferred" or sources.get("never_treated_col") == "inferred":
        st.caption(
            "⚠️ `cohort` ve/veya `never_treated` kolonları isimden sezgisel olarak "
            "tahmin edildi; bu yüzden işlemi onaylamak iyi olur."
        )

    if cohort_col is None or never_treated_col is None:
        st.info("Pre-trend hesabı için cohort ve never-treated kolonlarını seçin.")
    else:
        cohort_values = sorted(
            {value for value in st.session_state["clean_df"][cohort_col].dropna().tolist()},
            key=lambda value: str(value),
        )
        treated_cohort_selection = st.multiselect(
            "Treated kohortlar",
            options=cohort_values,
            default=cohort_values[:1] if cohort_values else [],
            key="event_study_treated_cohorts",
            help="Committed-baseline diagnostic için açık treated cohort seçin.",
        )
        cache_key = _event_study_cache_key(
            results_path=results_path,
            df=st.session_state["clean_df"],
            payload=payload,
            cohort_col=str(cohort_col),
            never_treated_col=str(never_treated_col),
            treated_cohorts=tuple(treated_cohort_selection),
        )
        cached_key = st.session_state.get("_event_study_cache_key")
        if cached_key == cache_key:
            event_study = st.session_state.get("_event_study_cache")

        if not treated_cohort_selection:
            st.info("Pre-trend hesabı için en az bir treated cohort seçin.")

        # Z3: estimate_pretrend_event_study kendi içindeki her riskli adımı
        # `except Exception` ile sarıp `_failed_result(...)` döndürüyor
        # (event_study.py) — yani pratikte ValueError fırlatmıyor. Sayfadaki
        # try/except ValueError, kütüphanenin zaten koruduğu bir yolu bir daha
        # koruyormuş gibi görünen ölü kod ve test edilemiyordu. Kaldırıldı;
        # artık doğrudan {"status": ..., "error": ...} sözleşmesine güveniliyor.
        if st.button("Bu kolonlarla pre-trend hesapla", disabled=not treated_cohort_selection):
            event_study = estimate_pretrend_event_study(
                st.session_state["clean_df"],
                outcome_col=payload["outcome_col"],
                unit_col=payload["unit_col"],
                time_col=payload["time_col"],
                cohort_col=str(cohort_col),
                never_treated_col=str(never_treated_col),
                controls=payload["controls"],
                treated_cohorts=tuple(treated_cohort_selection),
            )
            st.session_state["_event_study_cache"] = event_study
            st.session_state["_event_study_cache_key"] = cache_key

    if event_study is None:
        st.info("Hesaplamak için butona basın.")
    elif event_study.get("status") == "ok":
        series = pd.DataFrame(event_study.get("series", []))
        if series.empty:
            st.info("Pre-trend serisi boş.")
        else:
            reference_period = int(event_study.get("reference_period", -1))
            if reference_period not in set(series["event_time"]):
                series = pd.concat(
                    [
                        series,
                        pd.DataFrame(
                            [
                                {
                                    "event_time": reference_period,
                                    "coefficient": None,
                                    "ci_low": None,
                                    "ci_high": None,
                                    "p_value": None,
                                    "n_obs": None,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            series = series.sort_values("event_time")
            fig = _plot_coefficient_series(
                x=series["event_time"],
                y=series["coefficient"],
                title="Pre-trend event-study coefficients",
                xaxis_title="Event time",
                yaxis_title="Coefficient",
                hovertemplate="event_time=%{x}<br>estimate=%{y:.3f}<extra></extra>",
                marker_size=8,
                connectgaps=False,
                add_zero_line=True,
                add_vline_x=reference_period,
                trace_mode="lines+markers",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        (row["ci_high"] - row["coefficient"])
                        if row.get("ci_high") is not None and row.get("coefficient") is not None
                        else 0
                        for _, row in series.iterrows()
                    ],
                    "arrayminus": [
                        (row["coefficient"] - row["ci_low"])
                        if row.get("ci_low") is not None and row.get("coefficient") is not None
                        else 0
                        for _, row in series.iterrows()
                    ],
                    "thickness": 1,
                    "width": 0,
                },
            )
            st.plotly_chart(_for_dark_ui(fig), use_container_width=True)
            _capture_figure("pretrend_event_study.html", fig)
        if event_study.get("warnings"):
            for warning in event_study.get("warnings", []):
                st.caption(f"• {warning}")
    elif event_study.get("status") == "failed":
        st.warning(f"Pre-trend görseli hazırlanamadı: {event_study.get('error')}")
    else:
        st.info(event_study.get("reason", "Pre-trend görseli için gerekli veri yok."))


# `EstimationResult`ün iki alanı (`event_study`, `group_time_atts`) tablo hücresine sığmayan
# iç içe yapılar: ham dökümde iki sütunu kaplayıp geri kalan on sütunu eziyorlardı, üstelik
# yarım genişlikteki ızgara hücresinde tablo tamamen okunmaz hale geliyordu. Tablodan
# çıkarılıyorlar, ama kaybolduklarını söylemeden değil (aşağıdaki not).
_NESTED_RECEIPT_FIELDS = ("event_study", "group_time_atts")


def _render_receipts() -> None:
    frame = pd.DataFrame([r.model_dump() for r in results])
    flat_columns = [column for column in frame.columns if column not in _NESTED_RECEIPT_FIELDS]
    st.dataframe(frame, use_container_width=True, column_order=flat_columns)
    dropped = [
        field
        for field in _NESTED_RECEIPT_FIELDS
        if field in frame.columns and frame[field].notna().any()
    ]
    if dropped:
        st.caption(
            f"Tabloya sığmayan iç içe teşhis alanları gizlendi ({', '.join(dropped)}); "
            "ikisi de results.json'da ve reprodüksiyon paketinde tam hâliyle duruyor."
        )


_EVIDENCE_PANELS: tuple[tuple[str, Any], ...] = (
    ("Eksen atfı", _render_axis_attribution),
    ("Ön-trend", _render_pretrend),
    ("Efektif N", _render_effective_n),
    ("Makbuzlar", _render_receipts),
)

# NEDEN iki düzen: sekmeler sayfayı tek ekran yüksekliğinde tutuyor ama aynı anda tek
# bloğa bakılabiliyor; ızgara dördünü birden açıyor. Bloklar her iki düzende de BİR KEZ
# çiziliyor, yalnız içine çizildikleri kap değişiyor — iki kez çizmek widget key'lerini
# çakıştırırdı.
with st.container(horizontal=True, vertical_alignment="center", key="pa_vp_evidence_head"):
    _eyebrow("Kanıt")
    _layout = st.segmented_control(
        "Kanıt düzeni",
        options=["Sekmeli", "Izgara"],
        default="Sekmeli",
        key="evidence_layout",
        label_visibility="collapsed",
        # Seçili düğmeye tekrar basınca seçim boşalıyor; boş seçimde düzen sekmeliye
        # dönerdi ama anahtar hiçbir şey seçili değilmiş gibi görünürdü.
        required=True,
    )

if _layout == "Izgara":
    # Izgarada blok başlığı sekme etiketinden gelmiyor, her hücrenin içine yazılıyor.
    _slots = []
    for _row_start in (0, 2):
        _row = st.columns(2, gap="large")
        for _offset in (0, 1):
            _cell = _row[_offset].container(border=True)
            with _cell:
                _eyebrow(_EVIDENCE_PANELS[_row_start + _offset][0])
            _slots.append(_cell)
else:
    _slots = list(st.tabs([label for label, _render in _EVIDENCE_PANELS]))

for _slot, (_label, _render) in zip(_slots, _EVIDENCE_PANELS, strict=True):
    with _slot:
        _render()


# --------------------------------------------------------------------------- #
# Reprodüksiyon paketi
# --------------------------------------------------------------------------- #
# SIRA KISITI: bu bölüm sayfanın SONUNDA kalmalı. `_repro_inputs()` `package_figures`in
# o anki hâlini kopyalıyor ve sözlüğü yalnız yukarıdaki `_capture_figure` çağrıları
# dolduruyor. Paket bölümü kanıt bloklarının üstüne alınırsa indirilen zip pre-trend
# figürünü sessizce kaybeder: hata çıkmaz, dosya eksik olur.
def _repro_inputs() -> ReproInputs:
    """Koşunun artefakt yollarını oturumdan ve disk düzeninden çözer.

    Temizleme run_id'si ile multiverse run_id'si farklıdır, üstelik varsayılan
    sonuç yolu (`runs/latest`) hiçbir run_id taşımaz; bu yüzden yollar burada
    açıkça çözülür ve pakete hazır olarak verilir.

    Temizleme artefaktları oturumdan gelir ve gösterilen sonuçlarla aynı koşuya ait
    OLMAYABİLİR; sandbox çıktısı da (`reproduced.pkl`) pakete verilir ki paket bu
    bağı kendi denetleyip manifest'e yazabilsin.
    """
    results_file = Path(results_path)

    def _session_path(key: str) -> Path | None:
        value = st.session_state.get(key)
        return Path(str(value)) if value else None

    repro_dir = _session_path("last_repro_dir")
    return ReproInputs(
        run_id=run_id,
        results_path=results_file,
        specs_path=results_file.with_name("specs.json"),
        panel_path=results_file.with_name("panel.pkl"),
        frozen_menu_path=Path(SETTINGS.store_dir) / run_id / "frozen_menu.json",
        ledger_path=_session_path("last_ledger_path"),
        cleaning_script_path=_session_path("last_audit_path"),
        raw_panel_path=(repro_dir / "raw.pkl") if repro_dir is not None else None,
        cleaned_panel_path=(repro_dir / "reproduced.pkl") if repro_dir is not None else None,
        cleaning_run_id=(repro_dir.name.removesuffix("_repro") if repro_dir is not None else None),
        figures=dict(package_figures),
    )


_eyebrow("Paket")
with st.container(border=True):
    st.caption(
        "Denetim izinin tek dosyalık hâli: karar defteri, temizleme script'i, donmuş "
        "hash'ler, spesifikasyonlar, sonuçlar, figürler ve tek komutluk doğrulama script'i."
    )

    repro_inputs = _repro_inputs()
    repro_missing = missing_artifacts(repro_inputs)
    if repro_missing:
        st.warning(
            "Paket şu artefaktlar olmadan kurulacak: "
            + ", ".join(ARTIFACT_LABELS.get(key, key) for key in repro_missing)
            + ". Eksikler paketin MANIFEST dosyasına da yazılır."
        )

    # Paket birkaç MB olabilir; her rerun'da yeniden kurulmasın diye anahtarla
    # önbelleklenir. Anahtar `pareto.repro`da hesaplanır: temizleme artefaktlarını
    # kapsamayan bir anahtar, provenans uyarısını tam da uyarının gerektiği senaryoda
    # (A'yı koşup paketleyip sonra B'yi temizlemek) önbellekte yutardı.
    repro_key = package_key(repro_inputs)

    def _forget_package() -> None:
        for key in (
            "_repro_package",
            "_repro_package_key",
            "_repro_package_provenance",
            "_repro_package_stale",
        ):
            st.session_state.pop(key, None)

    if st.button("Reprodüksiyon paketini hazırla"):
        try:
            with st.spinner("Paket hazırlanıyor..."):
                package_bytes = build_reproduction_package(repro_inputs)
            st.session_state["_repro_package"] = package_bytes
            st.session_state["_repro_package_key"] = repro_key
            # Provenans denetimi paketin içinde yapılır; arayüz sonucu manifest'ten
            # okur ki uyarı ile pakete yazılan kayıt tek kaynaktan gelsin.
            with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
                manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
            st.session_state["_repro_package_provenance"] = manifest.get("provenance") or {}
            st.session_state["_repro_package_stale"] = False
        except ReproPackageError as exc:
            _forget_package()
            st.error(f"Reprodüksiyon paketi kurulamadı: {exc}")

    if st.session_state.get("_repro_package_key") == repro_key:
        provenance = st.session_state.get("_repro_package_provenance") or {}
        if provenance.get("cleaning_matches_panel") is False:
            st.error(
                "Pakete giren temizleme izi bu sonuçları üreten veriyle EŞLEŞMİYOR "
                f"(temizleme koşusu: `{provenance.get('cleaning_run_id')}`). Karar defteri "
                "ve temizleme script'i başka bir koşudan geliyor; paket bu hâliyle denetim "
                "izi sayılmaz. Uyarı paketin MANIFEST ve METHODS dosyalarına da yazıldı."
            )
        elif provenance.get("cleaning_matches_panel") is None:
            st.info(
                "Temizleme izi ile analiz paneli arasındaki bağ denetlenemedi; "
                "paket bunu doğrulanmamış olarak kaydediyor."
            )

        st.download_button(
            "Reprodüksiyon paketini indir",
            data=st.session_state["_repro_package"],
            file_name=f"pareto_repro_{repro_inputs.run_id}.zip",
            mime="application/zip",
            type="primary",
        )
    elif st.session_state.get("_repro_package") is not None or st.session_state.get(
        "_repro_package_stale"
    ):
        # Panel içeriği hazırlanan paketten sonra değişti (ör. pre-trend hesaplandı).
        # İndirme butonunu açıklamasız kaldırmak "sessiz hiçlik" olurdu. Bayat paketin
        # birkaç MB'ı oturumda tutulmaz, yerine yalnız uyarıyı ayakta tutan bayrak kalır.
        _forget_package()
        st.session_state["_repro_package_stale"] = True
        st.info("Panel hazırlanan paketten sonra değişti; paketi yeniden hazırlayın.")
    else:
        st.info("Paketi indirmeden önce hazırlayın.")
