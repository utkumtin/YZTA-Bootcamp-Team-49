"""Anthropic API üzerinden LLM çağrıları için ince bir sarmalayıcı.

Bu modül, Pareto'nun tüm LLM-tabanlı agent'ları (karar defteri, spec generator,
teşhis modülü) tarafından kullanılır. Tek sorumluluğu vardır: prompt gönder,
metin veya JSON al. İş mantığı (ne sorulacağı) çağıran modüllerde durur.
"""

from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from .config import SETTINGS, get_api_key

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=get_api_key())
    return _client


def call_llm(system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> str:
    """Ham metin cevabı döndürür."""
    client = _get_client()
    response = client.messages.create(
        model=SETTINGS.anthropic_model,
        max_tokens=max_tokens or SETTINGS.llm_max_tokens,
        temperature=SETTINGS.llm_temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def call_llm_json(system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> Any:
    """LLM'den SADECE JSON döndürmesini ister ve parse eder.

    LLM bazen ```json ... ``` bloğu içine sarabiliyor; bu fonksiyon onu temizler.
    Parse başarısız olursa RuntimeError fırlatır (sessizce yanlış veri üretmemek için).
    """
    strict_system = (
        f"{system_prompt}\n\n"
        "ÖNEMLİ: Cevabın SADECE geçerli JSON olmalı. Markdown kod bloğu, açıklama "
        "veya başka hiçbir metin ekleme."
    )
    raw = call_llm(strict_system, user_prompt, max_tokens=max_tokens)
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM geçerli JSON döndürmedi:\n{raw}") from exc
