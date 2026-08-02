"""Giriş scripti (`app/main.py`) artık `st.navigation` router'ı.

NEDEN bu dosya: router'ın hiç testi yoktu. `st.Page` yolları düz string; yanlış
yazılan bir yol yalnız uygulama açılırken patlıyor, yani hata deploy'da görülüyordu.
Buradaki tek koşu onu teste taşıyor.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_router_boots_and_renders_default_page() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=25).run()

    assert not app.exception
    # Varsayılan sayfa ana sayfa: oturum özeti + akış anlatısı orada.
    assert {"Oturum durumu", "Akış"} <= {subheader.value for subheader in app.subheader}


def test_registered_page_scripts_exist() -> None:
    """`st.Page` yolları giriş scriptinin dizinine göre çözülüyor; kaymaları yakalar."""
    expected = [
        "home.py",
        "pages/1_cleaning.py",
        "pages/2_analysis.py",
        "pages/3_variance_panel.py",
    ]
    source = APP_PATH.read_text(encoding="utf-8")

    for relative in expected:
        assert f'st.Page("{relative}"' in source
        assert (APP_PATH.parent / relative).exists()
