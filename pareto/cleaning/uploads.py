"""Streamlit yüklemelerini oturumlar arasında güvenle ayırt etme yardımcıları."""

from __future__ import annotations

from typing import Any


def uploaded_file_identity(uploaded: Any) -> str:
    """Yüklenen dosyanın Streamlit-rerun dayanıklı kimliğini döndürür.

    #53/2: Ad+boyut+içerik-hash fallback'i kaldırıldı. Proje `streamlit>=1.32`
    pinliyor (bkz. pyproject.toml / requirements.txt) ve bu sürümde
    `UploadedFile.file_id` her zaman dolu geliyor — fallback dalı hiçbir zaman
    çalışmayan ölü koddu. Minimum sürüm pini bir gün düşürülürse bu fonksiyon
    yeniden gözden geçirilmeli.
    """
    return f"id:{uploaded.file_id}"
