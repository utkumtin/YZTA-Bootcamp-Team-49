"""Proje-store / Hafıza.

Persistent proje durumu: karar defteri, onaylı (donmuş) hipotez, donmuş spec menüsü,
koşu sonuçları, üretilen kod yolları. Devam ettirilebilirlik + denetim izi. Lokal
artifacts (phone-home yok). Tek-tık reprodüksiyon paketi buradan beslenir.

Disk (json) tabanlı; sqlite'a geçiş gerekirse arayüz sabit kalır. Cross-session
hafıza = upside (bkz docs/scrum).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import SETTINGS


class ProjectStore:
    """Bir kapsamın durumunu diske yaz/oku. Anahtar-değer + JSON.

    `scope_id` bir projeyi ya da tek bir koşuyu adlandırabilir; çağıran taraf
    hangi granülerlikte izole olmak istediğine karar verir. Provenance kayıtları
    koşu başına izole olmak zorunda olduğu için koşu kimliğiyle çağrılır ve alan
    adı bunu artık dürüstçe yansıtıyor.
    """

    def __init__(self, scope_id: str, base_dir: str | Path | None = None) -> None:
        self.scope_id = scope_id
        self.root = Path(base_dir or SETTINGS.store_dir) / scope_id
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def save(self, key: str, value: Any) -> Path:
        """value JSON-serializable olmalı (Pydantic → .model_dump() ile ver)."""
        path = self._path(key)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
