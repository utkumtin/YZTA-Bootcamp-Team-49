"""Geçici sağlayıcı hatalarında pinli modeli yeniden deneyen sarmalayıcı.

NEDEN VAR: JUDGE zinciri bilinçli olarak tek üyeli ve pinli (ADR 0004), yani
`FallbackModel` devreye girmiyor ve private modda hiç failover yok. Bu durumda tek
bir 429 ya da geçici 503 tüm analiz adımını düşürüyordu. Serbest katman sağlayıcıları
bu hataları rutin olarak veriyor, dolayısıyla "tek hatada düş" davranışı ürünün en
sık görülen kırılma noktasıydı.

NEDEN model sarmalayıcı, HTTP transport değil: `pydantic_ai.retries`
(`AsyncTenacityTransport`) yalnız httpx tabanlı sağlayıcılarda çalışır. Varsayılan
JUDGE olan Gemini `google-genai` SDK'sı üzerinden gidiyor ve kendi taşımasını kuruyor,
yani transport seviyesindeki bir çözüm tam da en çok kullanılan ucu kapsamazdı.
`WrapperModel` sağlayıcıdan bağımsızdır ve ağ olmadan test edilebilir.

NEDEN yalnız geçici sınıf: şema doğrulama hatalarını pydantic-ai kendisi zaten geri
bildirimle yeniden deniyor; onları burada da denemek aynı işi iki kez yapar. 4xx
(429 hariç) kalıcı hatadır, yeniden denemek yalnız kotayı yakar.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import anyio
import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from ..config import SETTINGS

logger = logging.getLogger(__name__)

# 429 = rate limit (serbest katmanda en sık), 5xx = sağlayıcı tarafı geçici arıza.
# 408/409 sağlayıcılarda timeout/çakışma için kullanılıyor, ikisi de tekrarlanabilir.
_RETRYABLE_STATUS = frozenset({408, 409, 429})


def _is_transient(exc: BaseException) -> bool:
    """Yeniden denenmeye değer mi? Kararsız kalınırsa HAYIR.

    Fazla geniş bir tanım kalıcı bir hatayı (yanlış model adı, geçersiz anahtar)
    gecikmeli olarak aynı hataya çevirir ve kullanıcı sebebi görmeden bekler.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    # Durum kodu taşımayan sağlayıcı hatası: bağlantı kopması, okuma hatası vb.
    return isinstance(exc, ModelAPIError) and not isinstance(exc, ModelHTTPError)


def _delay_for(attempt: int) -> float:
    """Jitter'lı üstel bekleme. `attempt` 1'den başlar."""
    capped = min(SETTINGS.llm_retry_base_delay * (2 ** (attempt - 1)), SETTINGS.llm_retry_max_delay)
    # Tam jitter: eşzamanlı iki oturumun aynı anda tekrar denemesini önler.
    # Kriptografik kullanım değil, yalnız bekleme aralığı dağıtımı (S311).
    return random.uniform(0.0, capped)  # noqa: S311


@dataclass(init=False)
class RetryingModel(WrapperModel):
    """Sarmaladığı modeli geçici hatalarda yeniden dener; zinciri DEĞİŞTİRMEZ.

    Sarmalama sırası `router._resolve_model` içinde önemli: cache en dışta durur,
    böylece cache isabetleri retry yolundan hiç geçmez.
    """

    def __init__(self, wrapped: Any, *, max_attempts: int | None = None) -> None:
        super().__init__(wrapped)
        self.max_attempts = max_attempts if max_attempts is not None else SETTINGS.llm_max_attempts

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if not _is_transient(exc) or attempt == self.max_attempts:
                    raise
                last_exc = exc
                delay = _delay_for(attempt)
                logger.warning(
                    "Geçici sağlayıcı hatası (%s/%s), %.1fs sonra aynı modelde yeniden "
                    "denenecek: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    exc,
                )
                await anyio.sleep(delay)
        # Döngü her koşulda ya döner ya yükseltir; buraya düşmek mantık hatasıdır.
        raise AssertionError("retry döngüsü sessizce bitti") from last_exc
