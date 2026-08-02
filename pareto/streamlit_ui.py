"""Streamlit ortak UI — Ayarlar paneli (yalnız ana sayfa) + kompakt sidebar + oturum durumu."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import (
    ModelRole,
    PrivacyMode,
    get_effective_privacy_mode,
    resolve_api_key,
    resolve_setting,
)
from .llm.providers import (
    JUDGE_PRIVATE_SLOT,
    JUDGE_SLOT,
    ModelOption,
    ModelSlot,
    chain_for,
    judge_slots_for,
    option_ids,
)
from .llm.router import is_canned_mode

_LOGO_PATH = Path(__file__).resolve().parent.parent / "app" / "assets" / "pareto_logo.svg"
_ICONS_DIR = Path(__file__).resolve().parent.parent / "app" / "assets" / "icons"

BYOK_WIDGET_KEYS: dict[str, str] = {
    "GEMINI_API_KEY": "byok_gemini_input",
    "GEMINI_PAID_API_KEY": "byok_gemini_paid_input",
    "GROQ_API_KEY": "byok_groq_input",
    "OPENROUTER_API_KEY": "byok_openrouter_input",
    "OPENAI_API_KEY": "byok_openai_input",
}

_PROVIDER_LABELS: dict[str, str] = {
    "GEMINI_API_KEY": "Gemini",
    "GEMINI_PAID_API_KEY": "Gemini (private)",
    "GROQ_API_KEY": "Groq",
    "OPENROUTER_API_KEY": "OpenRouter",
    "OPENAI_API_KEY": "OpenAI",
}

# no-train/DPA garantisi hesap-seviyesinde bir özelliktir, uygulama bunu doğrulayamaz
# (bkz. JUDGE_GROQ_PRIVATE_SLOT'taki aynı varsayım, providers.py). BYOK ile kendi
# anahtarını giren kullanıcı bu sorumluluğu üstlenir — widget'ın yanında hatırlatılır.
_PRIVATE_KEY_WARNING = (
    "Bu anahtarın gerçekten no-train/DPA kapsamındaki bir Gemini projesine ait "
    "olduğunu doğrulamak size aittir — ücretsiz katman bir anahtar girerseniz "
    "PRIVATE modun no-train garantisi sessizce bozulur."
)

# pydantic-ai provider prefix -> kullanıcının tanıdığı ad. Ayarlar sekmesi eskiden ham
# prefix'i ("google") gösteriyordu, sidebar ise "Gemini" diyordu; aynı şey için iki
# kelime. Oturumda saklanan değer prefix olarak KALIR (providers.py sözleşmesi),
# yalnız görünen ad değişir.
_PROVIDER_DISPLAY: dict[str, str] = {
    "google": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "demo_sonnet_5": "Demo (Sonnet 5)",
}

_THINKING_LABELS: dict[str, str] = {
    "off": "Kapalı",
    "low": "Düşük",
    "medium": "Orta",
    "high": "Yüksek",
}

# resolve_api_key'in kaynak kodu -> hangi anahtarın fiilen kullanılacağını söyleyen etiket.
_KEY_SOURCE_LABELS: dict[str, str] = {
    "byok": "kendi anahtarınız",
    "env": ".env",
    "secrets": "secrets",
}

_MODE_LABELS: dict[str, str] = {
    "public": ":material/public: Herkese açık",
    "private": ":material/lock: Özel",
}
# Etiket içeriği KASITLI OLARAK seçime göre değişmiyor (renk direktifi yok) — bir ara
# denemede format_func'ın çıktısı seçime göre değişince (yeşil/amber vs gri metin)
# BaseWeb'in kendi "hangi buton aktif" takibi (`kind=segmented_controlActive`) kalıcı
# olarak bozuldu: her iki buton da bir tıklamadan sonra `kind="segmented_control"`da
# takılı kaldı (Playwright ile doğrulandı — Python tarafı/session_state doğruydu, yalnız
# BaseWeb'in native seçili-stil takibi kırıldı). İçerik render'lar arası SABİT kaldığında
# bu sorun yok. Renk artık aşağıdaki JS'te, `kind` attribute'una bakarak elle uygulanıyor.

# Gerçek DOM (Playwright ile doğrulandı — `role="radio"`/`aria-checked` YOK, o varsayım
# yanlıştı): seçili buton `kind="segmented_controlActive"`, diğeri `kind="segmented_control"`.
# Streamlit'in kendi `st.tabs` alt-çizgisi aynı deseni kullanıyor: `data-baseweb="tab-highlight"`
# adında, JS ile ölçülüp `left/width` + `transition:all` ile kayan ayrı bir overlay div.
# `segmented_control`'ün DOM'unda böyle bir overlay yok — burada aynı tekniği elle kuruyoruz:
# native kırmızı (primaryColor) seçili-buton stilini nötrlüyoruz, ve JS ile aktif butonun
# konumunu ölçüp yeşil/amber bir çerçeve overlay'ini oraya kayarak taşıyoruz. MutationObserver
# `kind`/`data-testid` değişimini izliyor ki Streamlit her rerun'da butonları güncellediğinde
# (React reconciliation ile aynı DOM node'ları yeniden kullanıyor — tab-highlight'ın da
# dayandığı varsayım) overlay senkronize kalsın.
# Raw string: gövdedeki JS regex'i Python kaçış dizisi olarak yorumlanmasın.
_MODE_HIGHLIGHT_HTML = r"""
<style>
/* Native "seçili" arka planı (`kind=segmented_controlActive`) açık temaya sabit bir gri
   koyuyor — koyu temada okunaksız beyaza dönüşüyor (Streamlit tema renklerini CSS custom
   property olarak dışa açmıyor, Emotion render-time'da hesaplıyor). Transparent = sidebar'ın
   kendi arka planı her zaman aynen görünür, hangi tema olursa olsun. */
.st-key-privacy_mode button[kind="segmented_controlActive"] {
    border-color: rgba(49, 51, 63, 0.2) !important;
    background-color: transparent !important;
    transition: border-color 300ms ease, background-color 300ms ease;
}
/* Native köşe yuvarlama yalnız "seçili" butonda uygulanıyor — seçili olmayan uç buton
   (grubun sol/sağ kenarındaki) her zaman border-radius:0 alıyor, pill şeklini bozuyor.
   Pozisyon sabit (public=ilk, private=son) olduğu için kenar yuvarlamayı burada zorluyoruz. */
.st-key-privacy_mode [data-baseweb="button-group"] button:first-of-type {
    border-radius: 8px 0 0 8px !important;
}
.st-key-privacy_mode [data-baseweb="button-group"] button:last-of-type {
    border-radius: 0 8px 8px 0 !important;
}
</style>
<script>
(function() {
    function findGroup() {
        return document.querySelector('.st-key-privacy_mode [data-baseweb="button-group"]');
    }

    function parseRgb(str) {
        var m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(str || '');
        return m ? [parseInt(m[1], 10), parseInt(m[2], 10), parseInt(m[3], 10)] : null;
    }

    // Etiket rengi tamamen JS tarafından yönetiliyor (Streamlit'in `:color[...]` markdown
    // direktifiyle DEĞİL) — bir önceki denemede format_func'ın çıktısı seçime göre
    // değişince (yeşil/amber vs gri metin) BaseWeb'in kendi "hangi buton aktif" takibi
    // (`kind=segmented_controlActive`) kalıcı olarak bozuluyordu; içerik render'lar arası
    // sabit kalınca bu sorun kayboluyor. Renk geçişini manuel requestAnimationFrame ile
    // interpolasyon yaparak sağlıyoruz — Streamlit'in yönettiği node'lar rerun'da baştan
    // kurulduğu için düz bir CSS transition, "önceki değer" hiç boyanmadığından çalışmıyor.
    //
    // Durum, DOM ELEMENTİ değil BUTON POZİSYONU (index) ile takip ediliyor: buton "seçimden
    // düşünce" (yeşil→gri) Streamlit onu native/renksiz içerikle BAŞTAN kuruyor — taze node
    // zaten gri DOĞUYOR, element-bazlı bir "from" okuması "zaten hedefte" sanıp atlıyor,
    // yeşilden griye geçiş hiç görünmüyor (animasyon yalnız gri→renkli yönünde çalışıyordu).
    // Index-bazlı takip, node değişse de "şu an mantıken hangi renkteyiz"i hatırlıyor ve her
    // `syncHighlight` çağrısında CANLI node'a o ara-değeri anında uyguluyor.
    var modeColorState = window.__paretoModeColorState || (window.__paretoModeColorState = {});

    function tweenTo(idx, label, toRgb) {
        var tracked = modeColorState[idx];
        if (tracked && tracked.target[0] === toRgb[0] && tracked.target[1] === toRgb[1] && tracked.target[2] === toRgb[2]) {
            label.style.color = 'rgb(' + tracked.current.join(',') + ')';
            return;
        }
        if (tracked && tracked.frameId) cancelAnimationFrame(tracked.frameId);
        var from = tracked ? tracked.current : parseRgb(getComputedStyle(label).color);
        if (!from || (from[0] === toRgb[0] && from[1] === toRgb[1] && from[2] === toRgb[2])) {
            label.style.color = 'rgb(' + toRgb.join(',') + ')';
            modeColorState[idx] = {current: toRgb, target: toRgb, frameId: null};
            return;
        }
        var duration = 300;
        var start = null;
        function step(ts) {
            if (start === null) start = ts;
            var t = Math.min((ts - start) / duration, 1);
            var current = [
                Math.round(from[0] + (toRgb[0] - from[0]) * t),
                Math.round(from[1] + (toRgb[1] - from[1]) * t),
                Math.round(from[2] + (toRgb[2] - from[2]) * t),
            ];
            label.style.color = 'rgb(' + current.join(',') + ')';
            modeColorState[idx] = {current: current, target: toRgb, frameId: t < 1 ? requestAnimationFrame(step) : null};
        }
        modeColorState[idx] = {current: from, target: toRgb, frameId: requestAnimationFrame(step)};
    }

    function syncHighlight(group) {
        var buttons = Array.prototype.slice.call(group.querySelectorAll('button'));
        var activeIdx = buttons.findIndex(function(b) { return b.getAttribute('kind') === 'segmented_controlActive'; });
        if (activeIdx === -1) return;
        var active = buttons[activeIdx];
        var colors = [[22, 163, 74], [217, 119, 6]];  // yeşil, amber
        var radii = ['8px 0 0 8px', '0 8px 8px 0'];

        group.style.position = 'relative';
        var hl = group.querySelector('#pareto-mode-highlight');
        if (!hl) {
            hl = document.createElement('div');
            hl.id = 'pareto-mode-highlight';
            hl.style.position = 'absolute';
            hl.style.pointerEvents = 'none';
            hl.style.boxSizing = 'border-box';
            hl.style.borderStyle = 'solid';
            hl.style.borderWidth = '2px';
            hl.style.transition = 'left 300ms ease, top 300ms ease, width 300ms ease, height 300ms ease, border-color 300ms ease';
            hl.style.zIndex = '2';
            group.appendChild(hl);
        }
        hl.style.left = active.offsetLeft + 'px';
        hl.style.top = active.offsetTop + 'px';
        hl.style.width = active.offsetWidth + 'px';
        hl.style.height = active.offsetHeight + 'px';
        hl.style.borderColor = 'rgb(' + (colors[activeIdx] || colors[0]).join(',') + ')';
        hl.style.borderRadius = radii[activeIdx] || radii[0];

        if (!window.__paretoModeDefaultColor) {
            var untouched = buttons[1 - activeIdx].querySelector('span[data-has-shortcut]');
            if (untouched) window.__paretoModeDefaultColor = parseRgb(getComputedStyle(untouched).color);
        }
        var defaultColor = window.__paretoModeDefaultColor || [128, 128, 128];
        buttons.forEach(function(btn, i) {
            var label = btn.querySelector('span[data-has-shortcut]');
            if (!label) return;
            tweenTo(i, label, i === activeIdx ? (colors[i] || colors[0]) : defaultColor);
        });
    }

    // Streamlit her rerun'da buton grubunun DOM node'unu BAŞTAN kuruyor (yerinde patch
    // değil, tam değişim) — bu yüzden gruba bağlı tekil bir MutationObserver bir sonraki
    // rerun'da kopan bir node'u izlemeye devam eder, hiçbir şey görmez. Kalıcı bir
    // document.body gözlemcisi tutup her DOM değişiminde "şu an geçerli grup node'u
    // hâlâ bizim izlediğimiz mi?" diye ucuz bir referans kıyası yapıyoruz; değiştiyse
    // overlay'i yeni node'a yeniden bağlıyoruz.
    function ensureAttached() {
        var group = findGroup();
        if (!group) return;
        if (group !== window.__paretoObservedGroup) {
            var observer = new MutationObserver(function() { syncHighlight(group); });
            observer.observe(group, {attributes: true, attributeFilter: ['kind', 'data-testid'], childList: true, subtree: true});
            window.__paretoObservedGroup = group;
        }
        syncHighlight(group);
    }

    if (!window.__paretoBodyObserverAttached) {
        var bodyObserver = new MutationObserver(function() { ensureAttached(); });
        bodyObserver.observe(document.body, {childList: true, subtree: true});
        window.__paretoBodyObserverAttached = true;
    }
    ensureAttached();
})();
</script>
"""


# Ana sayfada H1 yok; st.tabs orada üstteki tek başlık hiyerarşisi, bu yüzden native
# görünümünü (küçük punto, ince alt çizgi) bir navbar'a yükseltiyoruz: daha ağır tipografi,
# sekmeler arası geniş boşluk, marka indigosuyla kalın/yuvarlak kayan alt çizgi — kayma
# animasyonunun kendisi zaten Streamlit'in `tab-highlight` mekanizmasında var, yalnız
# yeniden renklendiriyoruz.
# `key="main_tab"` sayesinde `.st-key-main_tab` ile scope ediliyor, başka st.tabs'e sızmıyor.
# Metin rengi açık/koyu temaya göre değişiyor (logo SVG'deki `prefers-color-scheme` deseniyle
# aynı) — tema TABANI config.toml'da sabitlenmiyor, istemcinin OS/tarayıcı tercihi takip
# ediliyor; sabit "neredeyse beyaz" bir renk açık temada görünmez olurdu.
#
# Sayfanın varsayılan üst boşluğu (`.block-container`'da 96px) eskiden H1'in altına yer
# açıyordu; title kaldırılınca ölü boşluk olarak kaldı. `stHeader` üstte `position:absolute`,
# 60px yüksekliğinde (içeriği aşağı itmiyor, üzerine biniyor) — bu yüzden 60px'in altına
# inilemez, üstüne binilirse tab bar "Deploy" araç çubuğunun arkasında kalır. 72px = 60px
# header + 12px nefes payı.
_MAIN_NAV_HTML = """
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 72px;
}
.st-key-main_tab [data-baseweb="tab-list"] {
    gap: 28px;
    padding: 6px 2px 14px;
}
.st-key-main_tab button[data-baseweb="tab"] {
    height: auto;
    padding: 8px 2px;
    font-size: 1.05rem;
    font-weight: 500;
    color: rgba(49, 51, 63, 0.55);
    transition: color 200ms ease;
}
.st-key-main_tab button[data-baseweb="tab"]:hover {
    color: rgba(49, 51, 63, 0.85);
}
.st-key-main_tab button[aria-selected="true"] {
    color: #31333f;
    font-weight: 600;
}
.st-key-main_tab [data-baseweb="tab-highlight"] {
    background-color: #818cf8;
    height: 3px;
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(129, 140, 248, 0.45);
}
.st-key-main_tab [data-baseweb="tab-border"] {
    background-color: rgba(49, 51, 63, 0.1);
}
@media (prefers-color-scheme: dark) {
    .st-key-main_tab button[data-baseweb="tab"] {
        color: rgba(250, 250, 250, 0.55);
    }
    .st-key-main_tab button[data-baseweb="tab"]:hover {
        color: rgba(250, 250, 250, 0.85);
    }
    .st-key-main_tab button[aria-selected="true"] {
        color: #fafafa;
    }
    .st-key-main_tab [data-baseweb="tab-border"] {
        background-color: rgba(250, 250, 250, 0.06);
    }
}
</style>
"""


def render_main_nav_style() -> None:
    """Ana sayfadaki st.tabs'i (key="main_tab") navbar gibi göstermek için stil enjekte eder."""
    st.html(_MAIN_NAV_HTML)


# Sayfa başlığı = SVG ikon + H1, tek satırda. İkon `st.title`'ın İÇİNE konamıyor: markdown
# sanitizer `<svg>` etiketini düşürüyor, `st.html` ise ayrı bir blok elemanı üretiyor. Bu yüzden
# ikisi `horizontal=True` bir container'da yan yana diziliyor ve H1 gerçek `st.title` olarak
# kalıyor — tipografiyi, tema uyumunu ve anchor linkini elle taklit etmeye çalışmıyoruz.
# NEDEN scope: `key` verilen container `.st-key-...` sınıfı üretiyor; stil yalnız oraya iniyor,
# sayfadaki başka yatay container'lara sızmıyor.
_PAGE_TITLE_KEY_PREFIX = "pa_page_title"

# Ölçülen değerler (Streamlit 1.58, layout="wide"): H1 44px / 700 / 52.8px satır yüksekliği,
# padding yok. İkon 38px kutu, çizim 24'lük viewBox'ın ~19'unu dolduruyor: görünen glif ~30px,
# yani H1'in cap-height'ı. 1.5px kontur, 700 ağırlıklı metnin yanında aynı optik ağırlıkta.
#
# NEDEN elle renk: SVG `currentColor` kullanıyor, ama miras aldığı renk gövde metni rengi;
# başlıkla eşleşmesi için burada sabitleniyor, koyu tema kuralı da bu yüzden gerekiyor.
#
# NEDEN `class*=` seçici: container key'i ikon adıyla tekilleştiriliyor (aynı sayfada iki
# başlık olursa Streamlit yinelenen key hatası veriyor), bu yüzden tam eşleşme kullanılamaz.
_PAGE_TITLE_HTML = """
<style>
[class*="st-key-pa_page_title"] {
    gap: 0.65rem;
    margin-bottom: 0.4rem;
}
[class*="st-key-pa_page_title"] svg {
    display: block;
    width: 38px;
    height: 38px;
    color: #31333f;
}
@media (prefers-color-scheme: dark) {
    [class*="st-key-pa_page_title"] svg {
        color: #fafafa;
    }
}
</style>
"""


def render_page_title(icon_name: str, text: str) -> None:
    """Sayfa H1'i, solunda `app/assets/icons/` altındaki çizgi stilinde SVG ikonla.

    NEDEN `st.markdown`: `st.html` içeriği sanitize ederken `<svg>` etiketini tamamen
    düşürüyor, ikon sessizce kaybolur. Inline SVG'yi ayakta tutan tek yol bu.
    """
    svg = (_ICONS_DIR / f"{icon_name}.svg").read_text(encoding="utf-8")
    st.html(_PAGE_TITLE_HTML)
    key = f"{_PAGE_TITLE_KEY_PREFIX}_{icon_name}"
    with st.container(horizontal=True, vertical_alignment="center", key=key):
        st.markdown(svg, unsafe_allow_html=True, width="content")
        st.title(text)


@contextmanager
def llm_call_status(label: str, *, done_label: str | None = None) -> Iterator[Any]:
    """LLM isteği süresince durum göstergesi; bitince tamamlandı/hata durumuna geçer.

    NEDEN `st.status`, `st.spinner` değil: spinner yalnız "bekleniyor" halini biliyor,
    yanıt geldiğinde iz bırakmadan kayboluyor. Serbest katman gecikmelerinde kullanıcının
    sorduğu soru "uygulama dondu mu" olduğu için istek gönderildi → tamamlandı/hata
    geçişini gösterebilen primitif gerekiyor.

    `st.rerun()` ve `st.stop()` bilinçli olarak yakalanmıyor: ikisi de
    `ScriptControlException` (BaseException) türevi, `except Exception` bunları görmez ve
    akış kontrolü hata gibi işaretlenmez. Cache isabetinde çağrı anında döner; gösterge
    yanıp sönmez, doğrudan tamamlandı durumunda çizilir.
    """
    with st.status(label, expanded=False) as status:
        try:
            yield status
        except Exception:
            status.update(label=f"{label} — başarısız", state="error")
            raise
        status.update(label=done_label or f"{label} — tamamlandı", state="complete")


def _render_canned_mode_banner() -> None:
    """Canned mod bildirimi (O1 + O4).

    Eskiden `app/main.py`de yalnız `GEMINI_API_KEY`'in varlığına bakıyordu; bu, kullanıcı
    yalnız Groq veya OpenRouter anahtarı girdiğinde uygulama canlı çalıştığı halde ekranda
    "Canned Mod Aktif" yazması gibi yanlış bir duruma yol açıyordu. `is_canned_mode`
    (bkz. `llm/router.py`) tek bir sağlayıcıya değil, aktif rolün TÜM zincirine bakar —
    burada da o tek doğru kaynak kullanılıyor.

    JUDGE rolü kontrol edilir: Ayarlar sekmesindeki "Etkin rota" şeridi de JUDGE zincirini
    gösteriyor, kullanıcının orada gördüğü rota ile buradaki uyarı aynı kaynağa dayanmalı.

    Sidebar'da (bu fonksiyon `render_compact_sidebar()` içinden çağrılır) yaşadığı için
    yalnız ana sayfada değil, her sayfada görünür.
    """
    if not is_canned_mode(ModelRole.JUDGE):
        return
    # Hangi golden-path cache'in oynatıldığını söylemek için zincirin başını
    # (gerçek anahtar aranmadan) çözüyoruz — `is_canned_mode` yalnız bool döner.
    head_provider = chain_for(ModelRole.JUDGE, get_effective_privacy_mode())[0].provider
    if head_provider == "demo_sonnet_5":
        st.warning(
            "Canned Mod Aktif: Gösterilen sonuçlar Claude Sonnet 5 (high effort) ile "
            "önceden üretilmiş, committed bir demo çalışmasından geliyor. Kendi "
            "verinizle analiz yapmak ve modeli canlı kullanmak için **Ayarlar** "
            "sekmesinden kendi API anahtarınızı (BYOK) girin.",
            icon=":material/smart_toy:",
        )
        return
    st.warning(
        "Canned Mod Aktif: Kendi verinizle analiz yapmak ve modeli canlı kullanmak için "
        "**Ayarlar** sekmesinden kendi API anahtarınızı (BYOK) girin.",
        icon=":material/smart_toy:",
    )


def _sync_private_cache(privacy: PrivacyMode) -> None:
    """Private cache'in yaşam döngüsünü sidebar'a bağlar.

    Sidebar her sayfada koştuğu için temizlemenin doğal yeri burası. İki iş yapar:
    bayat oturum dizinlerini süpürür (tarayıcısını kapatıp gidenler) ve kullanıcı
    PRIVATE'tan PUBLIC'e döndüğünde o oturumun dizinini hemen siler.

    SINIR: Streamlit'in public bir "oturum bitti" kancası yok, bu yüzden "sekme
    kapandığı anda silinir" garantisi verilemez. Üçüncü ayak süreç kapanışındaki
    `atexit` (bkz. `pareto/llm/cache.py`). Süpürme ucuz: dizin yoksa hemen döner.
    """
    from .llm.cache import current_session_id, purge_private_cache, sweep_stale_private_cache

    sweep_stale_private_cache()
    if privacy is PrivacyMode.PUBLIC and st.session_state.get("_private_cache_used"):
        purge_private_cache(current_session_id())
        st.session_state["_private_cache_used"] = False
    elif privacy is PrivacyMode.PRIVATE:
        # Sonraki PUBLIC dönüşünde neyin silineceğini bilmek için işaretlenir;
        # hiç private'a girmemiş oturumda boşuna silme çağrısı yapılmaz.
        st.session_state["_private_cache_used"] = True


def render_compact_sidebar() -> str:
    """Her sayfada: marka (logo, sayfa navigasyonunun üstünde) + gizlilik modu + oturum özeti."""
    st.logo(str(_LOGO_PATH), size="medium")
    st.header("Oturum")
    mode = st.segmented_control(
        "Gizlilik modu",
        options=["public", "private"],
        default="public",
        required=True,
        key="privacy_mode",
        format_func=lambda v: _MODE_LABELS.get(v, v),
        help="public = free model + canned demo. private = yalnız no-train uçlar.",
    )
    st.html(_MODE_HIGHLIGHT_HTML, unsafe_allow_javascript=True)

    privacy = PrivacyMode.PRIVATE if str(mode) == PrivacyMode.PRIVATE.value else PrivacyMode.PUBLIC
    _sync_private_cache(privacy)

    _render_canned_mode_banner()
    # Oturum boşken de gösterilir (pills erken dönüyor): "hangi modelle
    # çalışıyorum" sorusu veri yüklenmeden önce de geçerli.
    _render_active_model_pill(privacy)
    _render_session_pills()
    _render_api_key_status()

    return str(mode)


def render_settings_panel() -> None:
    """Ayarlar sekmesi: etkin rota şeridi + sağlayıcı rayı / seçili sağlayıcı detayı.

    Gizlilik modu kontrolü sidebar'da KALIR (bkz. `render_compact_sidebar`) — burada
    yalnız etkin rotanın bir parçası olarak gösterilir, kontrol edilmez. Rota şeridi
    türetilmiş durumdur: hangi ucun çalışacağını tek satırda okutur.

    Sağlayıcı rayı ile detay ayrı sütunlarda ama TEK akıştır: anahtar, model ve
    düşünme derinliği aynı sağlayıcıya ait olduğu için bir arada durur (eskiden
    anahtarlar ayrı bir expander'da, model seçimi ayrı bir başlıktaydı).
    """
    st.html(_SETTINGS_STYLE_HTML)

    raw_mode = str(st.session_state.get("privacy_mode", PrivacyMode.PUBLIC.value))
    privacy = PrivacyMode.PRIVATE if raw_mode == PrivacyMode.PRIVATE.value else PrivacyMode.PUBLIC
    slots_by_provider = judge_slots_for(privacy)
    default_provider = (
        JUDGE_PRIVATE_SLOT.provider if privacy is PrivacyMode.PRIVATE else JUDGE_SLOT.provider
    )

    # Oturumdaki seçim bu katmanda geçerli değilse widget RENDER OLMADAN temizlenir:
    # `st.radio` key'i session_state'te varsa `index=` yok sayılır ve değer options'ta
    # yoksa exception atar (providers.py::_session_provider_choice ile aynı savunma).
    stored = str(st.session_state.get("judge_provider_choice", ""))
    if stored and stored not in slots_by_provider:
        st.session_state["judge_provider_choice"] = default_provider

    provider, _slot = active_judge_route(privacy)
    _render_route_strip(privacy, provider, slots_by_provider[provider])
    _render_privacy_note(privacy)

    col_rail, col_detail = st.columns([1, 1.9], gap="medium")
    with col_rail:
        _rail_label("Sağlayıcı", kind="pa-eyebrow")
        with st.container(border=True):
            provider = _render_provider_rail(slots_by_provider, default_provider)

    slot = slots_by_provider[provider]
    _restore_hidden_byok_toggles(slot.api_key_env)

    with col_detail:
        # Sol kolon KATEGORİ ("Sağlayıcı"), sağ kolon o kategorinin SEÇİLİ ÜYESİ.
        # İkisi aynı satırda duruyor ama aynı şey değil, bu yüzden tipografileri de
        # ayrı: sol versal/eyebrow, sağ cümle düzeni bir ad. Yükseklikleri CSS'te
        # eşitleniyor ki iki kart aynı hizadan başlasın.
        _rail_label(_PROVIDER_DISPLAY.get(provider, provider), kind="pa-detail-name")
        with st.container(border=True):
            _render_key_section(slot)
            _render_model_choice(slot, privacy, provider)

    _render_clear_all()


def _rail_label(text: str, *, kind: str) -> None:
    st.html(f'<div class="{kind}">{escape(text)}</div>')


def active_judge_route(privacy: PrivacyMode) -> tuple[str, ModelSlot]:
    """Bu gizlilik modunda çalışacak JUDGE sağlayıcısı ve slotu.

    Salt okunur: oturumdaki seçim bu katmanda geçerli değilse sessizce defaulta
    düşer ama session_state'i DÜZELTMEZ. Düzeltme yalnız Ayarlar panelinde, widget
    render edilmeden hemen önce yapılmalı; sidebar her sayfada çalıştığı için
    buradan yazmak kullanıcının seçimini o farkında olmadan değiştirirdi.
    """
    slots_by_provider = judge_slots_for(privacy)
    default_provider = (
        JUDGE_PRIVATE_SLOT.provider if privacy is PrivacyMode.PRIVATE else JUDGE_SLOT.provider
    )
    stored = str(st.session_state.get("judge_provider_choice", ""))
    provider = stored if stored in slots_by_provider else default_provider
    return provider, slots_by_provider[provider]


def _render_active_model_pill(privacy: PrivacyMode) -> None:
    """Sidebar'da aktif modeli göster.

    Model artık kullanıcı seçimine açık, ama seçim yalnız ana sayfadaki Ayarlar
    panelinde görünüyordu; temizleme/analiz/varyans sayfalarında "hangi modelle
    çalışıyorum" sorusunun cevabı yoktu. Kayıt tarafıyla (repro manifesti,
    dondurma kaydı) aynı bilginin ekrandaki karşılığı bu.
    """
    provider, slot = active_judge_route(privacy)
    label = _PROVIDER_DISPLAY.get(provider, provider)
    st.caption(f"Model: {label} · `{_resolved_model_id(slot)}`")


def _render_route_strip(privacy: PrivacyMode, provider: str, slot: ModelSlot) -> None:
    """İmza öğesi: çalışacak ucun tek satırlık okunuşu (kontrol değil, durum).

    Değerler oturumdan okunur; widget'lar bu şeridin ALTINDA render olsa da Streamlit
    her değişimde script'i baştan çalıştırdığı için şerit her zaman güncel görünür.
    """
    is_private = privacy is PrivacyMode.PRIVATE
    parts = [
        f'<span class="{"pa-mode-private" if is_private else "pa-mode-public"}">'
        f"{escape('Özel' if is_private else 'Herkese açık')}</span>",
        f'<span class="pa-route-val">{escape(_PROVIDER_DISPLAY.get(provider, provider))}</span>',
        f'<span class="pa-route-val pa-mono">{escape(_resolved_model_id(slot))}</span>',
    ]
    if slot.thinking_options:
        depth = _THINKING_LABELS.get(_resolved_thinking(slot), "").lower()
        parts.append(f'<span class="pa-route-val">düşünme: {escape(depth)}</span>')
    arrow = '<span class="pa-route-arrow">→</span>'
    st.html(
        '<div class="pa-route"><span class="pa-route-eyebrow">Etkin rota</span>'
        f"{arrow.join(parts)}</div>"
    )


_PRIVACY_DOC_URL = "https://github.com/utkumtin/YZTA-Bootcamp-Team-49/blob/main/PRIVACY.md"

# Gizlilik notunun metni. Her modda geçerli olan kısım ortak; ilk cümle moda özgü,
# çünkü kullanıcıya asıl lazım olan şey seçtiği modun ne vaat ettiği. Ortak kısımda
# "ham satırlar gönderilmez" iddiasının yanına gerçek veri değerlerinin nerede
# göründüğü de yazılıyor: eksik anlatılan bir garanti, olmayan bir garantiden daha
# yanıltıcı olur.
_PRIVACY_NOTE_BY_MODE: dict[PrivacyMode, str] = {
    PrivacyMode.PRIVATE: (
        "Özel modda istek yalnız eğitim yapmayan uçlara kurulur. Uygun anahtar yoksa "
        "uygulama ücretsiz uca düşmez, açık hata verir."
    ),
    PrivacyMode.PUBLIC: (
        "Herkese açık modda ücretsiz uçlar da kullanılabilir; sağlayıcı gönderilen özeti "
        "model eğitiminde kullanabilir. Yayımlanmamış veya hassas veri için özel moda geçin."
    ),
}

_PRIVACY_NOTE_COMMON: tuple[str, ...] = (
    "Dosyanızdan modele yalnız kolon düzeyinde özet gider: veri tipi, eksik oranı, benzersiz "
    "değer sayısı, sayısal kolonlarda en küçük, en büyük, ortalama ve standart sapma, "
    "kategorik kolonlarda en sık görülen beş değer. Ham satırlar gönderilmez.",
    "En sık görülen değerler ile en küçük ve en büyük değerler verinizden birebir alınır; "
    "bu yüzden kişiyi tanımlayabilecek mikro veri yüklemeyin.",
    "Araştırma sorunuz ve Sokratik beyanınız gibi kendi yazdığınız metinler modele olduğu "
    "gibi gider; sonraki adımlarda gönderilen diğer her şey deterministik hesap çıktısıdır.",
    "Girdiğiniz anahtarlar yalnız oturum belleğinde tutulur, uygulama bunları diske yazmaz. "
    "Karar defteri, üretilen kod ve sonuçlar yerel çalışma dizinine yazılır, dışarıya "
    "gönderilmez.",
)


def _privacy_note_lines(privacy: PrivacyMode) -> tuple[str, ...]:
    """Seçili moda göre gizlilik notunun satırları.

    Metin render'dan ayrı duruyor: notun modla birlikte gerçekten değişip değişmediği
    (özel modda zorlama, herkese açık modda eğitim uyarısı) Streamlit çalıştırmadan
    test edilebilsin.
    """
    return (_PRIVACY_NOTE_BY_MODE[privacy], *_PRIVACY_NOTE_COMMON)


def _render_privacy_note(privacy: PrivacyMode) -> None:
    """Rota şeridinin altındaki kapalı gizlilik notu.

    Yeri bilinçli: kullanıcı hangi ucun çalışacağını şeritte görüyor, not da o ucun
    veriye ne yaptığını söylüyor. Kapalı açılır kutu, ayar akışını bölmeden okunabilir
    kalmasını sağlıyor.
    """
    with st.expander(
        "Bu modda verinize ne oluyor?",
        icon=":material/shield:",
        key="privacy_note_expander",  # key olmadan her rerun'da (mod değişimi dahil) kapanır
    ):
        for line in _privacy_note_lines(privacy):
            st.markdown(f"- {line}")
        # Notun kendisi kısa tutuluyor; sağlayıcı taahhütleri ve neyin GARANTİ EDİLMEDİĞİ
        # tam metinde. Bağlantı `_privacy_note_lines` dışında, çünkü moda göre değişmiyor.
        st.caption(f"Tam metin: [PRIVACY.md]({_PRIVACY_DOC_URL})")


def _sync_judge_provider_choice() -> None:
    """`judge_provider_radio` widget değerini kalıcı `judge_provider_choice`ya kopyalar.

    Ayarlar sekmesi yalnız ana sayfada (`home.py`) render olur; Temizleme/Analiz gibi
    diğer sayfalarda bu radio hiç çizilmiyor. Streamlit, bir widget bir run'da
    render olmazsa o widget'ın session_state kaydını temizliyor
    (`session_state.py: remove_stale_widgets`) — yani seçim widget'ın KENDİ key'inde
    tutulsaydı sayfa değişince silinirdi. Bu yüzden radio'nun key'i (`judge_provider_radio`)
    bilinçli olarak ayrı ve tek kullanımlık: gerçek/kalıcı değer `judge_provider_choice`da
    (düz bir session_state girdisi, hiçbir widget'a bağlı değil) durur ve her sayfada
    `_session_provider_choice`/`active_judge_route` oradan okur.
    """
    st.session_state["judge_provider_choice"] = st.session_state["judge_provider_radio"]


def _render_provider_rail(slots_by_provider: dict[str, ModelSlot], default_provider: str) -> str:
    """Sağlayıcı listesi: her satırda ad + o sağlayıcının anahtar durumu.

    `st.radio` seçildi (buton değil): satır seçimi doğal olarak radio semantiğidir,
    klavye gezinme ve `aria-checked` bedavaya gelir. Native nokta CSS ile gizlenip
    yerine sol çubuk + zemin konuyor, odak halkası ayrıca tanımlanıyor.

    `format_func` çıktısı SEÇİME göre değil yalnız ANAHTAR DURUMUNA göre değişir.
    (Bkz. dosya başındaki `segmented_control` notu: içeriği seçime bağlamak BaseWeb'in
    aktif-buton takibini bozuyordu. Radio native `input:checked` kullandığı için o
    mekanizmaya dayanmıyor, ama aynı kuralı burada da bilerek koruyoruz.)
    """
    names = list(slots_by_provider)
    current = str(st.session_state.get("judge_provider_choice", "")) or default_provider
    if current not in names:
        current = default_provider

    def _row(name: str) -> str:
        ready = bool(resolve_api_key(slots_by_provider[name].api_key_env)[0])
        badge = ":green-badge[hazır]" if ready else ":gray-badge[anahtar yok]"
        return f"**{_PROVIDER_DISPLAY.get(name, name)}** {badge}"

    selected = st.radio(
        "Sağlayıcı",
        options=names,
        index=names.index(current),
        key="judge_provider_radio",
        format_func=_row,
        label_visibility="collapsed",
        on_change=_sync_judge_provider_choice,
        # Sayfadaki tek yer burası: bu seçimin NEYİ sürdüğünü söylüyor. Rota şeridi
        # zinciri gösteriyor ama zincirden neyin geçtiğini söylemiyor.
        help=(
            "Estimand, spec menüsü, temizleme önerisi ve varyans anlatısı bu "
            "sağlayıcıyla üretilir. Hem **public** hem **private** modda geçerli."
        ),
    )
    return str(selected)


def _restore_hidden_byok_toggles(active_env: str) -> None:
    """Render EDİLMEYEN sağlayıcıların BYOK switch'ini session_state'te canlı tutar.

    Streamlit, bir rerun'da render edilmeyen widget'ın session_state kaydını siler.
    `config.py::_from_session_byok` eksik kaydı AÇIK sayıyor (geriye dönük uyumluluk,
    bkz. tests/test_config_byok.py). Master-detail'de aynı anda tek sağlayıcının
    toggle'ı render olduğu için bu ikisi birleşince şu olurdu: kullanıcı Groq'u kapatır,
    Gemini'ye geçer, Groq kaydı silinir, bir sonraki `resolve_api_key("GROQ_API_KEY")`
    çağrısı anahtarı SESSİZCE yeniden kullanmaya başlar. Gölge sözlük (`byok_enabled_state`,
    widget'a bağlı DEĞİL, o yüzden silinmiyor) değeri geri yazar.

    Aktif sağlayıcı bilerek atlanır: onun widget'ı bu çalıştırmada zaten render olacak,
    üstüne yazmak kullanıcının az önceki tıklamasını eski değerle ezerdi.

    KAPSAM: yalnız Ayarlar paneli render olduğu çalıştırmalarda, yani ana sayfada.
    Diğer sayfalarda (cleaning/analysis/variance_panel) hiçbir toggle render olmadığı
    için aynı GC → default-AÇIK açığı orada duruyor. Bu değişiklikten ÖNCE de öyleydi
    (üç toggle da sayfa dışında yoktu); ayrı bir iş olarak ele alınmalı.
    """
    mirror = st.session_state.setdefault("byok_enabled_state", {})
    for env_name in BYOK_WIDGET_KEYS:
        if env_name != active_env:
            st.session_state[f"byok_enabled_{env_name}"] = bool(mirror.get(env_name, True))


def _render_key_section(slot: ModelSlot) -> None:
    """Seçili sağlayıcının anahtarı: durum her zaman görünür, giriş alanı katlanmış.

    Durum, slotun KENDİ `api_key_env`'inden okunur. Bu önemli: private moddaki Gemini
    slotu `GEMINI_PAID_API_KEY` istiyor, eski panel ise her durumda `GEMINI_API_KEY`e
    bakıp "algılandı ✓" diyordu — kimlik doğrulaması başarısız olacak bir uç için
    yanlış güvenlik hissi veriyordu.

    Toggle `st.form` DIŞINDA: etkisi anında olmalı, forma sokulan widget yalnız
    submit'te işlenir. (Denendi ve vazgeçildi: form kapandıktan sonra container'ı
    yeniden açıp toggle'ı görsel olarak kartın içine enjekte etmek, widget state'i ile
    görünen switch konumunu birbirinden koparıyordu — toggle kapalı görünürken
    session_state True kalıyor, yani anahtar sessizce kullanılmaya devam ediyordu.)
    """
    env_name = slot.api_key_env
    key, source = resolve_api_key(env_name)
    status = _KEY_SOURCE_LABELS.get(source, "anahtar yok") if key else "anahtar yok"
    with st.expander(
        f"API anahtarı · {status}",
        icon=":material/key:",
        key="byok_expander",  # key olmadan her rerun'da (kaydet dahil) kapanır
    ):
        widget_key = BYOK_WIDGET_KEYS.get(env_name)
        if widget_key is None:
            st.markdown(f"Yalnız `.env` üzerinden ayarlanır: `{env_name}`")
            return

        if env_name == "GEMINI_PAID_API_KEY":
            st.caption(_PRIVATE_KEY_WARNING)

        with st.form(f"byok_form_{env_name}", clear_on_submit=False, border=False):
            st.text_input(
                "API anahtarı",
                type="password",
                key=widget_key,
                label_visibility="collapsed",
                placeholder="Anahtarı yapıştırın",
            )
            if st.form_submit_button("Kaydet", type="primary"):
                _save_byok_key(env_name, widget_key)

        has_byok = bool(str(st.session_state.get("byok_keys", {}).get(env_name, "")).strip())
        st.toggle(
            "Kendi anahtarımı kullan",
            key=f"byok_enabled_{env_name}",
            disabled=not has_byok,
            help="Kapatınca .env veya secrets anahtarına döner. Kayıtlı anahtar silinmez.",
        )
        st.session_state.setdefault("byok_enabled_state", {})[env_name] = bool(
            st.session_state.get(f"byok_enabled_{env_name}", True)
        )


def _save_byok_key(env_name: str, widget_key: str) -> None:
    """Anahtarı yalnız `st.session_state["byok_keys"]`'e yazar (os.environ'a YAZILMAZ).

    Hangi anahtarın fiilen kullanılacağı `byok_enabled_{env_name}` switch'ine göre
    `config.py::resolve_api_key` içinde çözülür.
    """
    raw = str(st.session_state.get(widget_key, "")).strip()
    if not raw:
        st.warning("Anahtar alanı boş.")
        return
    byok_keys = st.session_state.setdefault("byok_keys", {})
    if env_name not in byok_keys:  # ilk kayıtta switch'i aç
        st.session_state[f"byok_enabled_{env_name}"] = True
        st.session_state.setdefault("byok_enabled_state", {})[env_name] = True
    byok_keys[env_name] = raw
    st.session_state.pop(widget_key, None)  # sır input'ta asılı kalmasın
    # toast rerun'ı aşar (st.success aşmaz); rerun sidebar durum satırını tazeliyor.
    st.toast("Anahtar kaydedildi.", icon=":material/check:")
    st.rerun()


def _render_clear_all() -> None:
    if not st.session_state.get("byok_keys"):
        return
    if st.button("Kayıtlı anahtarları sil", key="byok_clear_all", type="tertiary"):
        for env_name in BYOK_WIDGET_KEYS:
            st.session_state.pop(f"byok_enabled_{env_name}", None)
        st.session_state["byok_keys"] = {}
        st.session_state["byok_enabled_state"] = {}
        for widget_key in BYOK_WIDGET_KEYS.values():
            st.session_state.pop(widget_key, None)
        st.rerun()


def _model_options(slot: ModelSlot) -> tuple[list[ModelOption], list[str], str]:
    """Slotun UI seçenekleri + `.env` pini. Pin küratörlü listede yoksa başa eklenir."""
    pinned = resolve_setting(slot.model_env, slot.default_model)
    options = list(slot.options)
    if pinned not in option_ids(slot):  # operatörün .env pini listede yoksa da görünsün
        options.insert(0, ModelOption(model_id=pinned))
    return options, [o.model_id for o in options], pinned


def _resolved_model_id(slot: ModelSlot) -> str:
    """Rota şeridi için: oturumdaki seçim, geçersizse `.env` pini."""
    _options, ids, pinned = _model_options(slot)
    stored = str(st.session_state.get(f"model_choice_{slot.key}", "")).strip()
    return stored if stored in ids else pinned


def _resolved_thinking(slot: ModelSlot) -> str:
    stored = str(st.session_state.get(f"thinking_choice_{slot.key}", "")).strip()
    return stored if stored in slot.thinking_options else slot.default_thinking


def _render_model_choice(slot: ModelSlot, privacy: PrivacyMode, provider: str) -> None:
    """Seçili slotun küratörlü model + düşünme derinliği seçimi.

    Hem PUBLIC hem PRIVATE modda geçerlidir (bkz. `llm/providers.py: chain_for`).
    Seçim `os.environ`'a YAZILMAZ: widget değerleri yalnız oturumda kalır,
    `provider`/`api_key_env`/`no_train` her zaman kodda pinli kalır — kullanıcı
    yalnız küratörlü slot kümesi içinden seçer, serbest kombinasyon üretemez.

    Düşünme derinliği selectbox değil `segmented_control`: sıralı bir ölçek (kapalı →
    yüksek), dört seçeneğin tamamı tek bakışta görünüyor ve seçim tek tıkla oluyor.
    `format_func` çıktısı statik bir etiket eşlemesi — dosya başındaki BaseWeb notunda
    kırılan şey içeriğin SEÇİME göre değişmesiydi, burada öyle bir bağ yok.
    """
    options, ids, pinned = _model_options(slot)
    # Oturumdaki değer bu slotun listesinde yoksa widget render olmadan temizlenir
    # (`key` session'da varken options'ta olmayan bir değer exception atar).
    if str(st.session_state.get(f"model_choice_{slot.key}", pinned)) not in ids:
        st.session_state.pop(f"model_choice_{slot.key}", None)

    def _label(model_id: str) -> str:
        opt = next(o for o in options if o.model_id == model_id)
        return f"{model_id} · {opt.performance_note}" if opt.performance_note else model_id

    st.selectbox(
        "Model",
        options=ids,
        index=ids.index(pinned),
        key=f"model_choice_{slot.key}",
        format_func=_label,
        filter_mode=None,
        help=f"Kalıcı pin `{slot.model_env}` (.env). Buradaki seçim yalnız oturumda geçerli.",
    )

    if slot.thinking_options:
        thinking_ids = list(slot.thinking_options)
        if str(st.session_state.get(f"thinking_choice_{slot.key}", "")) not in thinking_ids:
            st.session_state.pop(f"thinking_choice_{slot.key}", None)
        st.segmented_control(
            "Düşünme derinliği",
            options=thinking_ids,
            default=slot.default_thinking,
            required=True,
            key=f"thinking_choice_{slot.key}",
            format_func=lambda v: _THINKING_LABELS.get(v, v),
            help="Yanıt vermeden önce ne kadar düşünsün. Arttıkça gecikme ve maliyet artar.",
        )

    chosen_id = str(st.session_state.get(f"model_choice_{slot.key}", pinned))
    chosen = next((o for o in options if o.model_id == chosen_id), options[0])
    if chosen.input_cost_note or chosen.output_cost_note:
        st.caption(
            f"Giriş {chosen.input_cost_note or '·'} · çıkış {chosen.output_cost_note or '·'}"
        )

    if privacy is PrivacyMode.PRIVATE and provider == "groq":
        st.warning(
            "Groq'ta no-train (Zero Data Retention) garantisi bu uygulama tarafından "
            "zorlanamaz; bir hesap ayarıdır. Private moda geçmeden önce "
            "[buradan](https://console.groq.com/settings/data-controls) etkinleştirin. "
            "Bu ayarın açık kalmasının sorumluluğu size aittir."
        )


# Ayarlar sekmesinin tek stil bloğu. Renkler uygulamada zaten kararlaştırılmış olanlardan
# türetildi (indigo #818cf8 = `_MAIN_NAV_HTML`'deki tab vurgusu; yeşil/amber =
# `_MODE_HIGHLIGHT_HTML`'deki gizlilik modu renkleri) — sekmeye özel yeni palet YOK.
#
# BaseWeb'in Select ControlContainer'ı `cursor:text` set ediyor ve bu CSS miras yoluyla tüm
# alt div'lere iniyor — asıl görünen "seçili değer" metni bir <input> DEĞİL, sıradan bir div
# (gerçek <input> boş/görünmez, yalnız erişilebilirlik için duruyor). Playwright'ın
# `elementFromPoint` ile doğrulandı: fare, hem kapalıyken hem açıkken bu metin div'inin
# üzerine düşüyor, dolayısıyla yalnız `input`'u hedeflemek yetersiz — tüm alt ağaca
# `cursor:pointer` basmak gerekiyor. Açılır liste (`stSelectboxVirtualDropdown`) BaseWeb'in
# Layer komponentiyle document.body'ye portal olarak taşınıyor, yani select'in .st-key-*
# sarmalayıcısının DIŞINDA render oluyor — bu yüzden dropdown kuralı scope edilemiyor,
# uygulama genelinde tanımlanıyor. (Sekmede geriye tek selectbox kaldığı için mono
# tipografiyi de aynı global kurala koymak güvenli: düşünme derinliği artık
# segmented_control, sağlayıcı ise radio.)
#
# Tıklayınca görünen yanıp sönen metin imleci (caret) ayrı bir sorun — `cursor` CSS'i mouse
# ikonunu belirler, caret ise gerçek <input> focus alınca tarayıcının kendi native davranışı;
# `cursor:pointer` onu gizlemez. `caret-color:transparent` ile input hâlâ odaklanabilir/klavyeyle
# gezilebilir kalıyor, yalnız görsel yanıp-sönen çizgi kayboluyor.
_SETTINGS_STYLE_HTML = """
<style>
/* --- Etkin rota şeridi --- */
.pa-route {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin: 0 0 22px;
    padding: 12px 16px;
    border: 1px solid rgba(129, 140, 248, 0.28);
    border-radius: 10px;
    background: rgba(129, 140, 248, 0.06);
    font-size: 0.84rem;
    line-height: 1.25;
}
.pa-route-val { color: #31333f; font-weight: 500; }
.pa-route-arrow { color: rgba(49, 51, 63, 0.3); }
.pa-mode-public { color: #15803d; font-weight: 600; }
.pa-mode-private { color: #b45309; font-weight: 600; }
/* Model ID'si bir makine adı, prose değil — tanımlayıcılar mono okunsun. */
.pa-mono {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.8rem;
}
.pa-route-eyebrow, .pa-eyebrow {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: rgba(49, 51, 63, 0.45);
}
.pa-route-eyebrow { margin-right: 4px; }
/* Kategori etiketi (sol) ile seçili üyenin adı (sağ) aynı satırda ama farklı şeyler:
   biri versal mikro-etiket, diğeri cümle düzeninde bir ad. Ortak satır yüksekliği
   iki kartın aynı hizadan başlamasını sağlıyor. */
.pa-eyebrow, .pa-detail-name {
    height: 18px;
    line-height: 18px;
    margin: 0 0 8px 2px;
}
.pa-detail-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #31333f;
}

/* --- Sağlayıcı rayı: radio -> tıklanabilir satır listesi --- */
.st-key-judge_provider_radio { width: 100% !important; }
.st-key-judge_provider_radio [role="radiogroup"] { gap: 2px; width: 100%; }
.st-key-judge_provider_radio label[data-baseweb="radio"] {
    width: 100%;
    margin: 0;
    padding: 9px 12px;
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: background-color 150ms ease, border-left-color 150ms ease;
}
/* Native nokta gizleniyor; seçili durum sol çubuk + zeminle veriliyor. */
.st-key-judge_provider_radio label[data-baseweb="radio"] > div:first-of-type { display: none; }
.st-key-judge_provider_radio label[data-baseweb="radio"] > div:last-of-type { width: 100%; }
.st-key-judge_provider_radio label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] {
    width: 100%;
}
.st-key-judge_provider_radio label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    margin: 0;
}
.st-key-judge_provider_radio label[data-baseweb="radio"]:hover {
    background: rgba(129, 140, 248, 0.09);
}
.st-key-judge_provider_radio label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(129, 140, 248, 0.14);
    border-left-color: #818cf8;
}
/* Nokta gizlenince klavye kullanıcısı için tek görünür ipucu odak halkası kalıyor. */
.st-key-judge_provider_radio label[data-baseweb="radio"]:has(input:focus-visible) {
    outline: 2px solid #818cf8;
    outline-offset: 1px;
}

/* --- Model seçimi: imleç düzeltmesi + mono tipografi --- */
div[class*="st-key-model_choice_"] div[data-baseweb="select"],
div[class*="st-key-model_choice_"] div[data-baseweb="select"] * {
    cursor: pointer !important;
}
div[class*="st-key-model_choice_"] div[data-baseweb="select"] input {
    caret-color: transparent !important;
}
div[class*="st-key-model_choice_"] div[data-baseweb="select"],
ul[data-testid="stSelectboxVirtualDropdown"] li {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.86rem;
}
ul[data-testid="stSelectboxVirtualDropdown"] li { cursor: pointer; }

/* --- Düşünme derinliği: native seçili rengi Streamlit'in primaryColor kırmızısı, bir
   ayar ölçeğinde hata durumu gibi okunuyor. Sekmenin indigo'suna çekiliyor. Yalnız
   renk override'ı: `kind` attribute'una dokunulmuyor, dolayısıyla BaseWeb'in kendi
   aktif-buton takibi (bkz. dosya başındaki not) bozulmuyor. --- */
div[class*="st-key-thinking_choice_"] button[kind="segmented_controlActive"] {
    border-color: #818cf8 !important;
    background-color: rgba(129, 140, 248, 0.14) !important;
}
div[class*="st-key-thinking_choice_"] button[kind="segmented_controlActive"],
div[class*="st-key-thinking_choice_"] button[kind="segmented_controlActive"] span {
    color: #4f46e5 !important;
}
div[class*="st-key-thinking_choice_"] button { cursor: pointer; }

/* --- Kaydet: `type="primary"` Streamlit'in primaryColor kırmızısını alıyor. `_MAIN_NAV_HTML`
   sekme vurgusunu zaten bu kırmızıdan indigo'ya çekmişti; aynı marka rengini burada da
   sürdürüyoruz. Scope BYOK formunun submit butonu — uygulamanın geri kalanındaki primary
   butonlara dokunulmuyor. --- */
div[class*="st-key-FormSubmitter-byok_form_"] button[kind="primaryFormSubmit"] {
    background-color: #4f46e5;
    border-color: #4f46e5;
    color: #ffffff;
}
div[class*="st-key-FormSubmitter-byok_form_"] button[kind="primaryFormSubmit"]:hover {
    background-color: #4338ca;
    border-color: #4338ca;
}

@media (prefers-color-scheme: dark) {
    .pa-route {
        border-color: rgba(129, 140, 248, 0.32);
        background: rgba(129, 140, 248, 0.09);
    }
    .pa-route-val, .pa-detail-name { color: #fafafa; }
    .pa-route-arrow { color: rgba(250, 250, 250, 0.32); }
    .pa-route-eyebrow, .pa-eyebrow { color: rgba(250, 250, 250, 0.45); }
    .pa-mode-public { color: #4ade80; }
    .pa-mode-private { color: #fbbf24; }
    div[class*="st-key-thinking_choice_"] button[kind="segmented_controlActive"],
    div[class*="st-key-thinking_choice_"] button[kind="segmented_controlActive"] span {
        color: #c7d2fe !important;
    }
    div[class*="st-key-FormSubmitter-byok_form_"] button[kind="primaryFormSubmit"] {
        background-color: #6366f1;
        border-color: #6366f1;
    }
    div[class*="st-key-FormSubmitter-byok_form_"] button[kind="primaryFormSubmit"]:hover {
        background-color: #818cf8;
        border-color: #818cf8;
    }
}
</style>
"""


def render_session_overview() -> None:
    """Ana sayfada: yüklü veri / estimand / spec özeti."""
    st.subheader("Oturum durumu")
    cols = st.columns(3)
    with cols[0]:
        df = st.session_state.get("clean_df")
        if df is not None:
            st.success(f"Veri: **{df.shape[0]}** satır × **{df.shape[1]}** kolon")
        else:
            st.info("Veri: henüz yüklenmedi")
    with cols[1]:
        frozen = st.session_state.get("frozen_estimand")
        if frozen is not None:
            st.success(f"Estimand: donduruldu (`{frozen.freeze_hash[:8]}…`)")
        else:
            st.info("Estimand: yok")
    with cols[2]:
        specs = st.session_state.get("analysis_specs")
        if specs:
            st.success(f"Spesifikasyon: **{len(specs)}** adet")
        else:
            st.info("Spesifikasyon: yok")


def _render_session_pills() -> None:
    df = st.session_state.get("clean_df")
    frozen = st.session_state.get("frozen_estimand")
    specs = st.session_state.get("analysis_specs")

    if df is None and frozen is None and not specs:
        # NEDEN: üçü de yokken burası tamamen boş kalıyordu; sidebar'daki boşluk
        # "oturum boş" mu "bir şey kırıldı" mı anlaşılmıyordu.
        st.caption("Oturum boş; Temizleme sayfasından veri yükleyin.")
        return

    if df is not None:
        st.caption(f"Veri: {df.shape[0]}×{df.shape[1]}")
    if frozen is not None:
        st.caption(f"Estimand: `{frozen.freeze_hash[:8]}…`")
    if specs:
        st.caption(f"Specs: {len(specs)}")


def _render_api_key_status() -> None:
    """Sağlayıcı başına 1 kısa satır — kaynak/karakter sayısı gösterilmez."""
    for env_name, label in _PROVIDER_LABELS.items():
        key, _source = resolve_api_key(env_name)
        st.caption(f"{label}: **algılandı** ✓" if key else f"{label}: **yok**")


def render_clean_panel(df_before: pd.DataFrame, df_after: pd.DataFrame, script: str) -> None:
    """Uygulanan temizlik özeti: satır/kolon/eksik değişimi + üretilen script."""
    st.subheader("CleanPanel — uygulanan temizlik")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Satır", df_after.shape[0], delta=df_after.shape[0] - df_before.shape[0])
    with c2:
        st.metric("Kolon", df_after.shape[1], delta=df_after.shape[1] - df_before.shape[1])
    with c3:
        missing_before = int(df_before.isna().sum().sum())
        missing_after = int(df_after.isna().sum().sum())
        st.metric("Toplam eksik", missing_after, delta=missing_after - missing_before)
    with st.expander("Üretilen script (reprodüksiyon)", expanded=False):
        st.code(script, language="python")
