"""Ayarlar/sidebar widget'larının çok-sayfa (`st.navigation`) geçişinde davranışı.

`st.Page` her sayfaya ayrı bir `active_script_hash` atıyor ve Streamlit bunu, açıkça
verilen `key=` aynı string olsa bile widget'ın iç element ID'sine karıştırıyor
(`streamlit/elements/lib/utils.py:compute_and_register_element_id`). Bu yüzden bir
widget'ın KENDİ key'i "sayfa değişince silinen" bir hata sınıfına açık: hem widget'ın
yalnız bir sayfada render olması (13a/13b) hem de her sayfada render olsa bile farklı
sayfalarda farklı element ID üretmesi (gizlilik modu) aynı sonuca (sessiz sıfırlanma)
yol açabiliyor. Testler gerçek `app/main.py` girişi + `AppTest.switch_page` üzerinden
koşuyor çünkü tek sayfalık `AppTest.from_file(page)` bu geçiş mekanizmasını hiç
tetiklemiyor — bug yalnız `st.navigation` üzerinden gerçek sayfa değişiminde ortaya
çıkıyor.

Not: AppTest widget elemanlarında `.options`/`.index` FORMATLANMIŞ (format_func'tan
geçmiş) string'leri tutar, ama `.value`/`.set_value()` HAM (session_state'teki) değeri
kullanır — bu yüzden aşağıdaki testler seçim yaparken hep ham model/thinking/provider
ID'lerini geçirir, `.options`'ı yalnız görünürlük kontrolü (ör. "Demo" hiç yok mu) için
okur.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

MAIN_PATH = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _home() -> AppTest:
    app = AppTest.from_file(str(MAIN_PATH), default_timeout=30)
    app.run()
    return app


def test_provider_rail_never_lists_demo_option() -> None:
    """ "Demo (Sonnet 5)" BYOK ile seçilebilir bir sağlayıcı değil.

    Yerel `claude` CLI oturumuna bağlı, yalnız Genel Bakış'taki "Demo moduna gir"
    butonuyla etkinleştiriliyor (`app/home.py`). Ayarlar'ın "Sağlayıcı" rayında
    seçilebilir bir satır olarak görünmesi BYOK'un "kendi anahtarınla kendi
    seçtiğin sağlayıcıyı kullan" vaadiyle çelişiyordu.
    """
    app = _home()
    radios = [r for r in app.radio if r.key == "judge_provider_radio"]
    assert len(radios) == 1
    assert not any("Demo" in option for option in radios[0].options)


def test_privacy_mode_survives_page_navigation() -> None:
    """Özel moda geçip sayfa değiştirince Herkese açık'a dönmemeli.

    Kök neden `judge_provider_choice`/13a'dan farklı: `render_compact_sidebar` her
    sayfada çalışıyor, ama `st.Page` geçişinde aynı `key="privacy_mode"` bile farklı
    `active_script_hash` yüzünden farklı bir element ID üretiyor, bu yüzden widget
    hâlâ "stale" sayılıp sıfırlanıyordu (ampirik olarak `git stash` ile önce-fix
    davranışı doğrulandı: sayfa değişince "public"e dönüyordu).
    """
    app = _home()
    controls = [c for c in app.segmented_control if c.label == "Gizlilik modu"]
    assert len(controls) == 1
    controls[0].set_value("private").run()
    assert app.session_state["privacy_mode"] == "private"

    app.switch_page("pages/1_cleaning.py")
    app.run()
    assert app.session_state["privacy_mode"] == "private"

    app.switch_page("home.py")
    app.run()
    assert app.session_state["privacy_mode"] == "private"
    controls_after = [c for c in app.segmented_control if c.label == "Gizlilik modu"]
    assert controls_after[0].value == "private"


def test_demo_mode_locks_the_privacy_control() -> None:
    """Demo modunda gizlilik kontrolü kilitli olmalı.

    Özel moda geçmek cache dizinini oturuma özel `llm_cache_private_dir`e çeviriyor
    (llm/cache.py `wrap_with_cache`); commit'li golden-path yanıtları orada olmadığı
    için demo, bir sonraki JUDGE çağrısında `CannedModeCacheMissError` ile duruyordu.
    Kontrol demoya giriş sayfasında, iki tık uzakta: anahtarsız bir ziyaretçinin
    tanıtım turunu yanlışlıkla kırabilmesi ürünün vitrinini riske atıyordu.
    """
    app = _home()
    privacy = next(c for c in app.segmented_control if c.label == "Gizlilik modu")
    assert not privacy.disabled, "demo dışında kontrol açık kalmalı"

    next(b for b in app.button if "Demo moduna gir" in b.label).click().run()

    assert app.session_state["demo_mode"] is True
    privacy_in_demo = next(c for c in app.segmented_control if c.label == "Gizlilik modu")
    assert privacy_in_demo.disabled, "demo modunda gizlilik kontrolü kilitli olmalı"
    assert app.session_state["privacy_mode"] == "public"


def test_thinking_choice_survives_page_navigation() -> None:
    """planned-issues.md madde 13b: düşünme derinliği seçimi Ayarlar dışına çıkınca
    sıfırlanmamalı.

    `_render_model_choice` yalnız Ayarlar panelinde (ana sayfa) render oluyor; diğer
    sayfalarda hiç çizilmiyor, bu yüzden widget'ın kendi key'i (13a'daki
    `judge_provider_choice` ile birebir aynı sebep) diğer sayfalarda "stale" sayılıp
    siliniyordu (ampirik olarak `git stash` ile doğrulandı: sayfa değişince
    `slot.default_thinking`e, yani "medium"e dönüyordu).
    """
    app = _home()
    thinking = next(c for c in app.segmented_control if c.key == "thinking_choice_radio_judge")
    assert thinking.value == "medium"  # JUDGE_SLOT.default_thinking pini
    thinking.set_value("off").run()
    assert app.session_state["thinking_choice_judge"] == "off"

    app.switch_page("pages/1_cleaning.py")
    app.run()
    app.switch_page("home.py")
    app.run()

    assert app.session_state["thinking_choice_judge"] == "off"
    thinking_after = next(
        c for c in app.segmented_control if c.key == "thinking_choice_radio_judge"
    )
    assert thinking_after.value == "off"


def test_model_choice_survives_page_navigation() -> None:
    """Aynı 13b bulgusu, model seçimi için — OpenAI slotu (3 küratörlü model) kullanılıyor
    çünkü varsayılan Gemini slotunda tek seçenek var, varsayılan dışı bir seçim mümkün
    değil.
    """
    app = _home()
    provider_radio = next(r for r in app.radio if r.key == "judge_provider_radio")
    provider_radio.set_value("openai").run()

    model = next(s for s in app.selectbox if s.key == "model_choice_radio_judge_openai")
    non_default_model = next(v for v in ("gpt-5.6-sol", "gpt-5.6-luna") if v != model.value)
    model.set_value(non_default_model).run()
    assert app.session_state["model_choice_judge_openai"] == non_default_model

    app.switch_page("pages/1_cleaning.py")
    app.run()
    app.switch_page("home.py")
    app.run()

    assert app.session_state["judge_provider_choice"] == "openai"
    assert app.session_state["model_choice_judge_openai"] == non_default_model
    model_after = next(s for s in app.selectbox if s.key == "model_choice_radio_judge_openai")
    assert model_after.value == non_default_model


def test_demo_mode_button_still_activates_demo_route() -> None:
    """ "Demo" satırının rayda gizlenmesi, "Demo moduna gir" butonunun kendisinin hâlâ
    çalışmasını (sidebar'daki aktif model pilinin güncellenmesini) gerektiriyor —
    bkz. `app/home.py: _render_demo_mode_entry`.
    """
    app = _home()
    buttons = [b for b in app.button if b.label == "Demo moduna gir"]
    assert len(buttons) == 1
    buttons[0].click().run()

    assert app.session_state["judge_provider_choice"] == "demo_sonnet_5"
    captions = [c.value for c in app.caption if c.value.startswith("Model:")]
    assert any("Demo (Sonnet 5)" in value for value in captions)

    radios = [r for r in app.radio if r.key == "judge_provider_radio"]
    assert not any("Demo" in option for option in radios[0].options)
