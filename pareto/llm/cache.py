"""LLM yanıt cache'i — temp=0 determinizmine dayanır.

Free-tier RPM limitini azaltmak ve tekrar koşuları hızlandırmak için model
yanıtları diske yazılır. Cache anahtarı; model adı, mesajlar, model ayarları ve
çıktı şemasının hash'idir. Her istekte değişen alanlar (timestamp, run_id gibi)
anahtardan ayıklanır, yoksa aynı istek her seferinde farklı anahtara düşer.

temp=0 dışındaki isteklerde cache atlanır çünkü yanıt deterministik değildir.
`PARETO_LLM_CACHE=0` ortam değişkeni cache'i tamamen kapatır.

Canned mode (gerçek bir API anahtarı bulunamadığında `router._chain_model`
zincirin dummy key ile kurulması) "temiz tarayıcı → canned akış panele kadar
yürür" garantisinin, golden-path cache dosyalarının runtime isteğiyle birebir
aynı hash'e düşmesine bağlı olduğu anlamına gelir. Sistem promptu/model
adı/ayar bir karakter bile değişirse hash değişir, cache miss olur ve dummy key
ile gerçek bir HTTPS isteği atılmaya çalışılırdı — kullanıcı network hatası
değil çirkin bir 401/auth hatası görürdü. `CachedModel` bu durumu
(`canned_mode=True` + cache miss) açıkça algılayıp anlaşılır bir hata fırlatır,
ham sağlayıcı hatasına düşmeden önce.
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


class CannedModeCacheMissError(RuntimeError):
    """Canned mode aktifken (gerçek anahtar yok) cache'te karşılık bulunamadı.

    Bu, golden-path cache dosyalarının runtime isteğiyle (sistem promptu, model adı,
    ayarlar) birebir eşleşmediği anlamına gelir. Gerçek anahtar olmadan sağlayıcıya
    istek atmak yalnız anlamsız bir 401/auth hatasına yol açar; bunun yerine burada
    açıkça durulur.
    """


def cache_enabled() -> bool:
    """Cache açık mı? `PARETO_LLM_CACHE=0` ile kapatılır, varsayılan açık."""
    return os.environ.get(_CACHE_ENV_FLAG, "1") != "0"


def wrap_with_cache(model: Model | str, *, canned_mode: bool = False) -> Model | str:
    """Cache açıksa modeli `CachedModel` ile sarar, kapalıysa aynen döndürür.

    `canned_mode=True`, çağıranın (router._chain_model) zincirdeki hiçbir üye için
    gerçek bir anahtar bulamadığını (hepsinin dummy key ile kurulduğunu) bildirir —
    bkz. `CachedModel` docstring'i.

    Cache kapalıyken (`PARETO_LLM_CACHE=0`) canned mode koruması da devre dışı
    kalır: model sarmalanmadığı için `CannedModeCacheMissError` hiç devreye
    girmez ve dummy key ile gerçek bir ağ isteği denenebilir. Bu nadir ama tam
    olarak engellemek istediğimiz senaryo olduğu için en azından bir uyarı
    logu basılır (O6).
    """
    if not cache_enabled():
        if canned_mode:
            logger.warning(
                "PARETO_LLM_CACHE=0 ve canned_mode=True: cache koruması devre dışı, "
                "dummy anahtarla gerçek bir ağ isteği denenebilir."
            )
        return model
    return CachedModel(model, Path(SETTINGS.llm_cache_dir), canned_mode=canned_mode)


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

    `canned_mode=True` iken (gerçek anahtar yok, yalnız dummy key ile kurulmuş bir
    model sarmalanıyor) cache miss durumunda sarmalanan modele hiç gidilmez — bunun
    yerine `CannedModeCacheMissError` fırlatılır. Aksi halde dummy key ile gerçek bir
    ağ isteği denenir ve kullanıcıya anlamsız bir 401/auth hatası olarak döner.
    """

    def __init__(self, wrapped: Model | str, cache_dir: Path, *, canned_mode: bool = False) -> None:
        # KnownModelName Literal'ı dışındaki "provider:model" stringleri de geçerli
        super().__init__(wrapped)  # type: ignore[arg-type]
        self._cache_dir = cache_dir
        self._canned_mode = canned_mode

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

        if self._canned_mode:
            logger.warning("Canned mode cache miss: %s", path.name)
            raise CannedModeCacheMissError(
                "Canned mode aktif (gerçek bir API anahtarı bulunamadı) ve bu istek için "
                "cache'te karşılık yok. Golden-path cache dosyaları runtime isteğiyle "
                "(sistem promptu, model adı, ayarlar) birebir eşleşmiyor olabilir. "
                "Gerçek bir sağlayıcıya anahtarsız istek atılmayacak — devam etmek için "
                "geçerli bir BYOK anahtarı girin ya da cache'i golden-path isteğiyle "
                "senkronize edin."
            )

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
