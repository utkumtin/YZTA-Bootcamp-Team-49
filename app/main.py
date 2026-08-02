"""Pareto — Streamlit girişi (Streamlit + Community Cloud).

Bu dosya yalnız router: sayfa kaydı `st.navigation` ile açıkça yapılıyor, sol menü
başlıkları buradan geliyor. Sayfa içerikleri `home.py` ve `pages/` altında.

NEDEN dosya-adı keşfi değil: otomatik keşifte sol menü etiketleri dosya adlarından
türüyor ve TR arayüzün içinde İngilizce görünüyordu. `st.navigation` çağrıldığı anda
otomatik keşif kapanıyor, etiketler burada tanımlanıyor.

Çalıştırma:  streamlit run app/main.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pareto.config import load_dotenv_file

load_dotenv_file()

# NEDEN PNG: favicon olarak SVG verilince Streamlit'in tarayıcı sekmesine düşürdüğü
# çıktı güvenilir değil. PNG, pareto_mark.svg'den üretiliyor; geometri değişirse
# `magick -background none -density 1200 pareto_mark.svg -resize 512x512 pareto_mark.png`
# ile yeniden üretilmeli. Statik olduğu için koyu temaya uyarlanmıyor, açık tondaki
# marka rengiyle sabit.
_FAVICON = Path(__file__).resolve().parent / "assets" / "pareto_mark.png"

# set_page_config, st.navigation dahil her Streamlit çağrısından önce gelmeli.
st.set_page_config(page_title="Pareto", page_icon=str(_FAVICON), layout="wide")

# Sol menü ikonu emoji ya da material kısayolu olmak zorunda; özel SVG kabul edilmiyor.
# Sayfa başlıklarındaki SVG ikonlar ayrı, `render_page_title` üzerinden geliyor.
_PAGES = [
    st.Page("home.py", title="Pareto", icon=":material/home:", default=True),
    st.Page("pages/1_cleaning.py", title="Temizleme", icon=":material/mop:"),
    st.Page("pages/2_analysis.py", title="Analiz", icon=":material/query_stats:"),
    st.Page("pages/3_variance_panel.py", title="Varyans Paneli", icon=":material/analytics:"),
]

st.navigation(_PAGES).run()
