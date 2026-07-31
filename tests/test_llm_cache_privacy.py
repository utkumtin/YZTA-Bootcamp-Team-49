from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from pareto.config import SETTINGS, PrivacyMode
from pareto.llm import cache as cache_mod
from pareto.llm.cache import (
    CachedModel,
    purge_private_cache,
    sweep_stale_private_cache,
    wrap_with_cache,
)


def _cache_dir(model: object) -> Path:
    """Sarmalanan modelin yazacağı dizin; sarmalanmadıysa test kurgusu yanlıştır."""
    assert isinstance(model, CachedModel), "cache açıkken model sarmalanmış olmalı"
    return model._cache_dir


@pytest.fixture
def _public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_mod, "get_effective_privacy_mode", lambda: PrivacyMode.PUBLIC)


@pytest.fixture
def _private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_mod, "get_effective_privacy_mode", lambda: PrivacyMode.PRIVATE)


def test_public_mod_hala_commitlenmis_golden_path_dizinini_hedefler(_public):
    """EN PAHALI REGRESYON: PUBLIC yolu `runs/llm_cache`'ten kaymamalı.

    Anahtarsız canned demo, commit'lenmiş 11 golden-path dosyasının runtime
    isteğiyle aynı dizinde bulunmasına bağlı. Dizin kayarsa her istek cache-miss
    olur ve akış `CannedModeCacheMissError` ile düşer — yani jüriye gösterilecek
    anahtarsız demo tamamen ölür. Private mod çalışması bunu bozmadan yapılmalıydı.
    """
    wrapped = _cache_dir(wrap_with_cache(TestModel()))

    assert wrapped == Path(SETTINGS.llm_cache_dir)


def test_private_mod_ayri_dizine_yazar(_private):
    """Özel veriden türeyen prompt ve yanıtlar public cache'e karışmamalı.

    Cache anahtarı model adını içerdiği için kayıtlar zaten çakışmıyordu; sorun
    çakışma değil, özel veriden türeyen içeriğin kalıcı olarak public dizinde
    durmasıydı (privacy by design).
    """
    wrapped = _cache_dir(wrap_with_cache(TestModel()))

    assert Path(SETTINGS.llm_cache_private_dir) in wrapped.parents
    assert wrapped != Path(SETTINGS.llm_cache_dir)


def test_private_dizin_oturuma_ozeldir(_private, monkeypatch: pytest.MonkeyPatch):
    """Her oturum kendi dizinini alır; biri silinince diğeri etkilenmez."""
    monkeypatch.setattr(cache_mod, "current_session_id", lambda: "oturum-a")
    a = _cache_dir(wrap_with_cache(TestModel()))
    monkeypatch.setattr(cache_mod, "current_session_id", lambda: "oturum-b")
    b = _cache_dir(wrap_with_cache(TestModel()))

    assert a != b


def test_private_cache_silinebilir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Oturum sonunda silme sözü tutulmalı: dosyalar gerçekten gitmeli."""
    monkeypatch.setattr(cache_mod, "_private_cache_root", lambda: tmp_path / "priv")
    oturum = tmp_path / "priv" / "oturum-a"
    oturum.mkdir(parents=True)
    (oturum / "yanit.json").write_text("{}", encoding="utf-8")

    purge_private_cache("oturum-a")

    assert not oturum.exists()


def test_bayat_oturumlar_supurulur_taze_olanlar_korunur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Tarayıcısını kapatan kullanıcının dizini başka türlü hiç temizlenmez.

    Streamlit'in "oturum bitti" kancası olmadığı için TTL süpürmesi bu boşluğu
    kapatıyor. Taze dizinin korunması da o kadar önemli: aktif bir oturumun
    cache'ini silmek kotayı boşuna yakar.
    """
    monkeypatch.setattr(cache_mod, "_private_cache_root", lambda: tmp_path / "priv")
    bayat = tmp_path / "priv" / "bayat"
    taze = tmp_path / "priv" / "taze"
    bayat.mkdir(parents=True)
    taze.mkdir(parents=True)
    eski = time.time() - 7200
    import os

    os.utime(bayat, (eski, eski))

    silinen = sweep_stale_private_cache(ttl_seconds=3600)

    assert silinen == 1
    assert not bayat.exists()
    assert taze.exists()


def test_surec_kapanisi_baska_surecin_oturumunu_silmez(
    _private, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`atexit` yalnız BU sürecin yazdığı dizinleri silmeli.

    Kök topluca silinseydi her `pytest` koşusu ya da kısa script, aynı repo üzerinde
    açık duran bir Streamlit oturumunun private cache'ini yok ederdi — kullanıcı
    çalışırken cache'i altından çekilmiş olurdu. Sahiplik kaydı bunu engelliyor.
    """
    monkeypatch.setattr(cache_mod, "_private_cache_root", lambda: tmp_path / "priv")
    # Gerçek koşudan kalan sahiplik kaydı testi kirletmesin; sonunda geri konur.
    onceki = set(cache_mod._OWNED_SESSION_DIRS)
    cache_mod._OWNED_SESSION_DIRS.clear()

    # Bu süreç yalnız "benim" oturumuna yazıyor.
    monkeypatch.setattr(cache_mod, "current_session_id", lambda: "benim")
    wrap_with_cache(TestModel())

    yabanci = tmp_path / "priv" / "baska-surec"
    yabanci.mkdir(parents=True)
    (tmp_path / "priv" / "benim").mkdir(parents=True, exist_ok=True)

    cache_mod._purge_owned_sessions()

    assert not (tmp_path / "priv" / "benim").exists(), "kendi oturumu silinmeli"
    assert yabanci.exists(), "başka sürecin oturumuna dokunulmamalı"

    cache_mod._OWNED_SESSION_DIRS.update(onceki)


def test_cache_kapaliyken_private_modda_da_disk_kullanilmaz(_private, monkeypatch):
    """`PARETO_LLM_CACHE=0` her iki modda da sarmalamayı tamamen kaldırmalı."""
    monkeypatch.setenv("PARETO_LLM_CACHE", "0")
    model = TestModel()

    assert wrap_with_cache(model) is model
