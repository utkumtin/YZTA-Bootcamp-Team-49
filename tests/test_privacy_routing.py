"""Gizlilik zorlaması: özel modda yönlendirme ve veri minimizasyonu.

Buradaki testlerin tek konusu şu: ürünün "özel modda veriniz eğitim yapan bir uca
gitmez" iddiası gerçekten zorlanıyor mu, yoksa yalnız kağıt üstünde mi duruyor.
Router'ın kendi konuları (önbellek, failover, model seçimi) ayrı dosyada kalır.

Testler ağa çıkmaz: model nesneleri sahte anahtarla kurulur, istek gönderilmez.
Zorlamanın doğrulandığı yer tam da burasıdır, çünkü ihlal edilse bile ilk hata
istek anında değil, nesnenin kurulduğu anda görünür olmalıdır.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pareto.cleaning.agent import _build_judge_prompt
from pareto.config import SETTINGS, ModelRole, PrivacyMode, load_dotenv_file, resolve_api_key
from pareto.llm import providers as providers_module
from pareto.llm import router as router_module
from pareto.llm.providers import (
    _PRIVATE_JUDGE_SLOTS,
    _PRIVATE_MECHANICAL_SLOTS,
    JUDGE_OPENROUTER_PRIVATE_SLOT,
    JUDGE_PRIVATE_SLOT,
    JUDGE_SLOT,
    MECH_GEMINI_SLOT,
    _resolve,
    chain_for,
    judge_slots_for,
)
from pareto.llm.router import _get_effective_privacy_mode, _resolve_model
from pareto.profiling import profile_dataframe
from pareto.streamlit_ui import _privacy_note_lines

# Geliştirici makinesinde anahtarlar .env'de durabilir; testler bunları bilinçli
# olarak siler ya da sahtesini koyar, dolayısıyla önce yüklenmeleri gerekir.
load_dotenv_file()

# Anahtarın varlığına duyarlı testlerde temizlenecek değişkenler (alias dahil).
_KEY_ENVS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_PAID_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture
def anahtarsiz_ortam(monkeypatch):
    """Tüm sağlayıcı anahtarlarını siler; test yalnız ihtiyacı olanı geri koyar."""
    for env in _KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Özel modda istek yalnız eğitim yapmayan uca kurulabilir
# ---------------------------------------------------------------------------


def test_ozel_modda_ucretsiz_gemini_anahtariyla_istek_kurulmaz(anahtarsiz_ortam):
    """Elde yalnız ücretsiz katman anahtarı varken özel mod sessizce o uca düşmemeli.

    Sessiz düşüş olsaydı kullanıcı, verisi eğitim yapan bir uca giderken kilit
    simgesine bakıp güvende olduğunu sanırdı. Beklenen davranış: açık hata.
    """
    anahtarsiz_ortam.setenv("GEMINI_API_KEY", "ucretsiz-anahtar")
    anahtarsiz_ortam.setattr("streamlit.session_state", {"privacy_mode": "private"})

    with pytest.raises(OSError, match="eksik anahtarlar"):
        _resolve_model(ModelRole.JUDGE)


def test_ucretsiz_anahtar_ozel_gemini_slotuna_devredilemez(anahtarsiz_ortam):
    """Ücretsiz ve ücretli Gemini uçları ayrı değişken okur; alias köprüsü kurulmamalı.

    Ücretsiz anahtarın ücretli slota geçebilmesi, kod tarafında hiçbir kural
    değişmeden gizlilik garantisini boşa çıkarırdı.
    """
    anahtarsiz_ortam.setenv("GEMINI_API_KEY", "ucretsiz-anahtar")
    anahtarsiz_ortam.setenv("GOOGLE_API_KEY", "ucretsiz-anahtar")

    assert resolve_api_key(JUDGE_PRIVATE_SLOT.api_key_env) == ("", "none")
    assert resolve_api_key(JUDGE_SLOT.api_key_env)[0] == "ucretsiz-anahtar"


def test_ozel_modda_yalniz_no_train_anahtarlari_okunur(anahtarsiz_ortam):
    """Model nesnesi kurulurken hangi anahtarın istendiğini doğrudan ölçer.

    Zincirin `no_train` alanını okumak yetmez: asıl soru, kurulan nesnenin hangi
    hesabın anahtarıyla konuşacağıdır. Her yargı sağlayıcısı için ayrı ayrı bakılır,
    çünkü sağlayıcı seçimi kullanıcıya açıktır. Gezilen liste ÖZEL moddan alınır:
    herkese açık moddan türeyen `JUDGE_PROVIDER_CHOICES` üstünden dönmek, iki küme
    ayrıştığı gün farkı sessizce kapsam dışı bırakırdı.
    """
    istenen: list[str] = []

    def _sahte_anahtar(env: str) -> str:
        istenen.append(env)
        return "test-anahtar"

    anahtarsiz_ortam.setattr(router_module, "get_api_key", _sahte_anahtar)
    izinli = {s.api_key_env for s in _PRIVATE_JUDGE_SLOTS + _PRIVATE_MECHANICAL_SLOTS}

    for provider in judge_slots_for(PrivacyMode.PRIVATE):
        istenen.clear()
        anahtarsiz_ortam.setattr(
            "streamlit.session_state",
            {"privacy_mode": "private", "judge_provider_choice": provider},
        )

        _resolve_model(ModelRole.JUDGE)
        _resolve_model(ModelRole.MECHANICAL)

        assert istenen, f"{provider}: hiç anahtar istenmedi, test bir şeyi ölçmüyor"
        assert set(istenen) <= izinli, (
            f"{provider}: özel mod dışı anahtar okundu (örn. ücretsiz katman): "
            f"{sorted(set(istenen) - izinli)}"
        )


def test_ozel_modda_her_saglayicida_no_train_zorunlu(anahtarsiz_ortam):
    """Sağlayıcı seçimi kullanıcıya açık, ama eğitim niteliği seçime açık değil."""
    for provider in judge_slots_for(PrivacyMode.PRIVATE):
        anahtarsiz_ortam.setattr("streamlit.session_state", {"judge_provider_choice": provider})

        chain = chain_for(ModelRole.JUDGE, PrivacyMode.PRIVATE)

        assert chain[0].no_train is True, f"{provider}: özel modda no_train=False uç seçildi"


def test_env_override_ozel_moddaki_no_train_garantisini_bozmaz(anahtarsiz_ortam):
    """Model kimliği ortamdan değiştirilebilir; gizlilik niteliği değiştirilemez."""
    for slot in _PRIVATE_JUDGE_SLOTS + _PRIVATE_MECHANICAL_SLOTS:
        anahtarsiz_ortam.setenv(slot.model_env, "baska-bir-model")

    for role in (ModelRole.JUDGE, ModelRole.MECHANICAL):
        chain = chain_for(role, PrivacyMode.PRIVATE)

        assert chain, "özel mod zinciri boş olamaz"
        assert all(uc.no_train for uc in chain)


# ---------------------------------------------------------------------------
# Zorlamanın kendisi dekoratif değil
# ---------------------------------------------------------------------------


def test_no_train_kontrolu_yargi_rolunde_gercekten_ates_eder(monkeypatch):
    """Kontrol yalnız listeler zaten temiz olduğu için mi geçiyor, yoksa çalışıyor mu?

    Listeye eğitim yapan bir uç yerleştirilir; çağrı hata yükseltmezse kontrol
    dekoratiftir ve iddia edilen garanti maskelenmiş demektir.
    """
    monkeypatch.setattr("streamlit.session_state", {})
    monkeypatch.setattr(
        providers_module,
        "_PRIVATE_JUDGE_SLOTS_BY_PROVIDER",
        {JUDGE_SLOT.provider: JUDGE_SLOT},
    )

    with pytest.raises(RuntimeError, match="no-train"):
        chain_for(ModelRole.JUDGE, PrivacyMode.PRIVATE)


def test_no_train_kontrolu_mekanik_rolde_gercekten_ates_eder(monkeypatch):
    monkeypatch.setattr("streamlit.session_state", {})
    monkeypatch.setattr(providers_module, "_PRIVATE_MECHANICAL_SLOTS", (MECH_GEMINI_SLOT,))

    with pytest.raises(RuntimeError, match="no-train"):
        chain_for(ModelRole.MECHANICAL, PrivacyMode.PRIVATE)


def test_router_gizlilik_modunu_oturumdan_okur(monkeypatch):
    """Zorlama zincirin başında duran bu okumaya dayanır: mod okunmazsa hiçbir kural işlemez."""
    monkeypatch.setattr("streamlit.session_state", {"privacy_mode": "private"})
    assert _get_effective_privacy_mode() is PrivacyMode.PRIVATE

    monkeypatch.setattr("streamlit.session_state", {})
    assert _get_effective_privacy_mode() is SETTINGS.privacy_mode


# ---------------------------------------------------------------------------
# Özel mod seçenek kümesi
# ---------------------------------------------------------------------------


def test_ozel_mod_seceneklerinde_ucretsiz_model_kimligi_yok():
    """OpenRouter'ın `:free` uçları veri saklama taahhüdü vermez, özel modda listelenemez."""
    for slot in _PRIVATE_JUDGE_SLOTS + _PRIVATE_MECHANICAL_SLOTS:
        assert not slot.default_model.endswith(":free"), f"{slot.key}: varsayılan model ücretsiz uç"
        for option in slot.options:
            assert not option.model_id.endswith(":free"), (
                f"{slot.key}: küratörlü listede ücretsiz uç var"
            )


def test_openrouter_ozel_ucunda_zdr_istek_ayarina_kadar_tasinir(anahtarsiz_ortam):
    """OpenRouter'ın no-train niteliği hesap ayarından değil, istek başına zorlamadan gelir.

    Bu ayar istek ayarlarına ulaşmazsa `no_train=True` işareti karşılıksız kalır.
    """
    zdr = {"openrouter_provider": {"zdr": True}}
    assert JUDGE_OPENROUTER_PRIVATE_SLOT.extra_model_settings == zdr
    assert _resolve(JUDGE_OPENROUTER_PRIVATE_SLOT, allow_session=False).extra_model_settings == zdr

    anahtarsiz_ortam.setenv("OPENROUTER_API_KEY", "test-anahtar")
    anahtarsiz_ortam.setattr(
        "streamlit.session_state",
        {"privacy_mode": "private", "judge_provider_choice": "openrouter"},
    )

    _model, extra = _resolve_model(ModelRole.JUDGE)

    assert extra == zdr


# ---------------------------------------------------------------------------
# Veri minimizasyonu — iddia edilen sınır neredeyse orada dursun
# ---------------------------------------------------------------------------


def test_temizleme_yukunde_satir_duzeyinde_veri_yok():
    """TEMİZLEME yükünde satır düzeyinde veri bulunmamalı.

    Sınır bilinçli olarak dar çiziliyor: en sık görülen beş değer ile en küçük ve en
    büyük değerler zaten özetin parçası olduğu için dışarı çıkar. Test bunların
    dışında kalanı kontrol eder, çünkü ürünün verdiği söz "hiçbir değer çıkmaz"
    değil, "ham satırlar çıkmaz". Sözün genişletilmesi bu testi kırar, daraltılması da.

    Kapsam yalnız `_build_judge_prompt`. Sonraki adımların yükleri (estimand, menü,
    varyans anlatısı) dataframe'i hiç görmez; kendi girdileri kullanıcı metni ve
    deterministik hesap çıktısıdır, dolayısıyla "ham satır sızdı mı" burada
    sorulamaz — bkz. PRIVACY.md, "Modele ne gidiyor".
    """
    df = pd.DataFrame(
        {
            "hasta_no": list(range(1000, 1022)),
            "sehir": (
                ["Ankara"] * 6
                + ["İzmir"] * 5
                + ["Bursa"] * 4
                + ["Adana"] * 3
                + ["Konya"] * 2
                + ["Rize", "Sinop"]
            ),
        }
    )

    profil = profile_dataframe(df)
    payload = _build_judge_prompt(profil)

    # Sayısal kolonda sınır ALAN KÜMESİYLE kontrol ediliyor, ara bir hücrenin
    # ("1007") payload'da aranmasıyla değil: o arama, hücrenin ortalama/std
    # basamaklarına rastlamamasına bel bağlar, yani tesadüfen geçer. Kümede
    # `stats` dışında bir alan belirmesi (örn. örnek satırlar) ise tesadüf değildir.
    sayisal = profil["columns"]["hasta_no"]
    assert set(sayisal) == {"dtype", "n_missing", "pct_missing", "n_unique", "stats"}
    assert set(sayisal["stats"]) == {"min", "max", "mean", "std"}

    assert "Rize" not in payload, "en sık beş değerin dışındaki kategori payload'a sızdı"
    assert "Sinop" not in payload
    assert "hasta_no" in payload, "kolon adı özetin parçası, bulunmazsa test bir şey ölçmüyor"


# ---------------------------------------------------------------------------
# Uygulama içi not
# ---------------------------------------------------------------------------


def test_gizlilik_notu_modun_gercegini_soyler():
    """Not iki modda aynı kalırsa kullanıcıya karşılıksız güven verir.

    Özel modda zorlamayı, herkese açık modda eğitim uyarısını söylemek zorunda;
    ham satırların gönderilmediği ise her iki modda geçerli.
    """
    ozel = " ".join(_privacy_note_lines(PrivacyMode.PRIVATE))
    acik = " ".join(_privacy_note_lines(PrivacyMode.PUBLIC))

    assert "eğitim yapmayan" in ozel
    assert "eğitiminde kullanabilir" in acik
    assert "eğitiminde kullanabilir" not in ozel
    for metin in (ozel, acik):
        assert "Ham satırlar gönderilmez" in metin
