"""Panel-merge — çok kaynağı unit×time panele birleştir.

Hero yakıtı (Medicaid): CDC mortalite (ilçe-yıl) + KFF eyalet genişleme tarihleri +
Census kontrolleri → FIPS-vs-isim uyumsuzluğu, tarih formatları, eksik kırsal ilçeler.
Çıktı `CleanPanel` (long df + manifest). Merge deterministik; anahtar-eşleme yargısı
(hangi kolon join-key) temizleme agent'ından gelir.

SPRINT-2: gerçek Medicaid panel-merge implementasyonu (bkz docs/scrum, hero demo).
"""

from __future__ import annotations

import pandas as pd

from ..contracts import CleanPanel


def merge_to_panel(sources: dict[str, pd.DataFrame], config: dict) -> CleanPanel:
    """Birden çok kaynağı config'teki rol/anahtar eşlemesine göre panele birleştirir."""
    raise NotImplementedError("Panel-merge Sprint-2 kapsamında (hero: Medicaid).")
