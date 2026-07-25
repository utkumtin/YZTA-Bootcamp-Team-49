"""Streamlit yüklemelerini oturumlar arasında güvenle ayırt etme yardımcıları."""

from __future__ import annotations

import hashlib
from typing import Any


def uploaded_file_identity(uploaded: Any) -> str | tuple[str, int, str]:
    """Yüklenen dosyanın Streamlit-rerun dayanıklı kimliğini döndürür.

    Streamlit'in sağladığı ``file_id`` tercih edilir. Eski sürümlerde aynı ad ve
    boyuta sahip farklı dosyaları ayırt etmek için içerik özeti eklenir.
    """
    file_id = getattr(uploaded, "file_id", None)
    if file_id:
        return file_id

    return (
        uploaded.name,
        uploaded.size,
        hashlib.sha256(uploaded.getvalue()).hexdigest(),
    )
