"""LLM yanıt cache'i — temp=0 determinizmine dayanır.

Free-tier RPM limitini azaltmak ve tekrar koşuları hızlandırmak için model
yanıtları diske yazılır. Cache anahtarı; model adı, mesajlar, model ayarları ve
çıktı şemasının hash'idir. Her istekte değişen alanlar (timestamp, run_id gibi)
anahtardan ayıklanır, yoksa aynı istek her seferinde farklı anahtara düşer.

temp=0 dışındaki isteklerde cache atlanır çünkü yanıt deterministik değildir.
`PARETO_LLM_CACHE=0` ortam değişkeni cache'i tamamen kapatır.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_json, to_jsonable_python

from ..config import SETTINGS

try:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.settings import ModelSettings
except ImportError as exc:  # fail-loud, sessiz düşme yok
    raise RuntimeError(
        "pydantic-ai kurulu değil. `uv sync` / `pip install pydantic-ai` gerekli."
    ) from exc

logger = logging.getLogger(__name__)

_CACHE_ENV_FLAG = "PARETO_LLM_CACHE"
_RESPONSE_ADAPTER: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


def cache_enabled() -> bool:
    """Cache açık mı? `PARETO_LLM_CACHE=0` ile kapatılır, varsayılan açık."""
    return os.environ.get(_CACHE_ENV_FLAG, "1") != "0"


def wrap_with_cache(model: Model | str) -> Model | str:
    """Cache açıksa modeli `CachedModel` ile sarar, kapalıysa aynen döndürür."""
    if not cache_enabled():
        return model
    return CachedModel(model, Path(SETTINGS.llm_cache_dir))


# İstek içeriğinden bağımsız, her koşuda değişen alanlar; cache anahtarına giremez.
_VOLATILE_KEYS = frozenset({"timestamp", "run_id", "conversation_id"})


def _scrub_volatile(obj: Any) -> Any:
    """Cache anahtarını bozan oynak alanları rekürsif olarak çıkarır."""
    if isinstance(obj, dict):
        return {k: _scrub_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub_volatile(v) for v in obj]
    return obj


class CachedModel(WrapperModel):
    """temp=0 model yanıtlarını diske cache'leyen sarmalayıcı.

    Deterministik olmayan (temp != 0) istekler sarmalanan modele aynen geçer.
    Bozuk cache dosyası sessizce yutulmaz; uyarı loglanır ve yanıt yeniden
    üretilip dosyanın üstüne yazılır.
    """

    def __init__(self, wrapped: Model | str, cache_dir: Path) -> None:
        # KnownModelName Literal'ı dışındaki "provider:model" stringleri de geçerli
        super().__init__(wrapped)  # type: ignore[arg-type]
        self._cache_dir = cache_dir

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not self._is_deterministic(model_settings):
            return await super().request(messages, model_settings, model_request_parameters)

        path = self._cache_path(messages, model_settings, model_request_parameters)
        cached = self._read(path)
        if cached is not None:
            logger.debug("LLM cache isabet: %s", path.name)
            return cached

        response = await super().request(messages, model_settings, model_request_parameters)
        self._write(path, response)
        return response

    @staticmethod
    def _is_deterministic(model_settings: ModelSettings | None) -> bool:
        """Yalnız temp=0 istekler cache'lenebilir."""
        if not model_settings:
            return False
        return model_settings.get("temperature") == 0

    def _cache_path(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Path:
        payload = to_json(
            _scrub_volatile(
                to_jsonable_python(
                    {
                        "model": self.model_name,
                        "messages": messages,
                        "settings": model_settings,
                        "params": model_request_parameters,
                    },
                    fallback=str,
                )
            )
        )
        key = hashlib.sha256(payload).hexdigest()
        return self._cache_dir / f"{key}.json"

    @staticmethod
    def _read(path: Path) -> ModelResponse | None:
        if not path.exists():
            return None
        try:
            return _RESPONSE_ADAPTER.validate_json(path.read_bytes())
        except Exception as exc:
            logger.warning("Bozuk LLM cache dosyası yeniden üretilecek (%s): %s", path.name, exc)
            return None

    def _write(self, path: Path, response: ModelResponse) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_RESPONSE_ADAPTER.dump_json(response))
