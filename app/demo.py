"""Demo Mode: anahtarsız/dataset'siz ziyaretçi için sabit bir tanıtım akışı.

Bilinçli olarak `pareto/` DIŞINDA yaşar: çekirdek hiçbir dataset adı/kolonu
bilmeyecek şekilde tasarlandı (bkz. tests/test_core_data_agnostic.py, SCOPE
§11 — jürinin kendi verisini yükleyebilmesi çekirdeğin veri-agnostik kalmasına
bağlı). Bu dosyadaki medicaid'e özgü sabitler tam da o yüzden `app/`de: demo
akışı ve `scripts/generate_canned_cache.py` bir deployment kararıdır, çekirdek
motorun parçası değildir.

Ana sayfadaki "Demo moduna gir" düğmesi, Temizleme sayfasındaki dosya
yükleyiciyi committed `DEMO_DATASET_PATH` ile atlar; Analiz sayfasındaki
Sokratik form da burada tanımlı sabit değerlerle önceden doldurulur. Bu
sabitler `scripts/generate_canned_cache.py` tarafından da AYNEN kullanılır —
demo akışının ürettiği istekler, golden-path cache'ini dolduran üretim
koşusuyla birebir aynı olmalı (bkz. pareto/llm/cache.py: hash anahtarı istek
içeriğine bağlı, bir karakterlik sapma cache-miss demektir).
"""

from __future__ import annotations

from pathlib import Path

from pareto.analysis.hypothesis import SocraticDeclaration

DEMO_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "medicaid" / "demo_panel.csv"

# `demo_panel.csv`, `build_panel("data/medicaid")` çıktısının 2014 kohortu +
# hiç genişletmeyen kontrol grubuna kısıtlanmış ve `treated_post` (1 = 2014
# genişlemesi yapan eyaletteki ilçe x 2014 sonrası yıl) göstergesi eklenmiş
# hâlidir — aynı türetme `scripts/run_e2e_check.py: committed_baseline_sample`
# içinde de yapılır (bkz. data/medicaid/config.yaml: treatment.committed_baseline).
# `treatment_cohort` (yıl ya da NA) doğrudan regresyon değişkeni olarak
# KULLANILAMAZ — TWFE'de zaman sabit etkisiyle çoklu doğrusal bağlantı yaratır;
# demo CSV'sinden bilinçli olarak çıkarıldı.
DEMO_RESEARCH_STORY = (
    "ABD'de Affordable Care Act (ACA) kapsamında bazı eyaletler 2014'te Medicaid'i "
    "genişletirken bazıları hiç genişletmedi. Bu ilçe-yıl panelinde, 2014 "
    "genişlemesinin ilçe düzeyinde sigortasız nüfus oranı üzerindeki etkisini "
    "inceliyoruz: 2014'te genişleyen eyaletlerin ilçeleri ile hiç genişletmeyen "
    "eyaletlerin ilçeleri karşılaştırılıyor. Beklenti: genişleme yapan "
    "eyaletlerin ilçelerinde sigortasızlık oranı, 2014 sonrasında görece daha "
    "çok düşer."
)

DEMO_DECLARATION = SocraticDeclaration(
    conceptual_treatment="Eyaletin 2014'te ACA kapsamında Medicaid'i genişletmesi",
    conceptual_outcome="İlçe düzeyinde sigortasız nüfus oranı",
    expected_sign="negative",
)

# Analiz sayfasındaki "Analiz Yapılandırması" formunun demo modundaki
# varsayılanları — panel kimliği `build_panel("data/medicaid")` manifest'iyle
# eşleşir (bkz. pareto/cleaning/merge.py, PanelManifest.unit_col/time_col).
DEMO_ANALYSIS_STATE = {
    "unit_col": "county_fips",
    "time_col": "year",
    "cluster_by": "state_fips",
    "controls": ["median_hh_income", "poverty_rate", "unemployment_rate"],
}
