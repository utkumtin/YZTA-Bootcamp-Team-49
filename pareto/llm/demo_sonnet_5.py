"""JUDGE'ın demo/canned sağlayıcısı: `claude -p` CLI oturumu üzerinden model kurulumu.

İsim bilinçli olarak "anthropic_cli" değil "demo_sonnet_5" — bu sağlayıcı yalnız
S3-05 golden-path demo cache'ini (Sonnet 5, high effort) üretmek için var, genel bir
"Anthropic sağlayıcısı" değil. Diğer sağlayıcılardan (google/groq/openrouter/openai)
farklı: gerçek bir HTTP API anahtarı yok, ağa hiç çıkılmıyor. Yerel `claude` binary'si
(Claude Code OAuth oturumu, `~/.claude/.credentials.json`) subprocess olarak
çağrılıyor — tıpkı `scripts/run_cli_reference.py`'nin JUDGE benchmark'ı için yaptığı
gibi.

Bu yüzden bu sağlayıcının "gerçek anahtar"ı asla yok: `providers.py`daki
`JUDGE_DEMO_SONNET_5_SLOT.api_key_env` (`DEMO_SONNET_5_SESSION`) normal bir ortamda
hiç tanımlı değildir, dolayısıyla `router._chain_model` bu slotu her zaman
canned/dummy yoluna düşürür ve `CachedModel(canned_mode=True)` cache-miss'te bu
modülü hiç çağırmadan `CannedModeCacheMissError` fırlatır (bkz. cache.py). Tek
gerçek çağrı, golden-path cache'ini dolduran bilinçli bir üretim koşusunda olur
(bkz. scripts/generate_canned_cache.py) — orada `DEMO_SONNET_5_SESSION` elle
açılır ve gerçek bir `claude -p` alt süreci çalışır.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# scripts/dump_benchmark_prompts.py ile aynı ambalaj: araç tanımı burada metin
# olarak gidiyor çünkü `claude -p --tools ""` hiçbir araç kabul etmiyor.
_SCHEMA_INSTRUCTION = (
    "Return your answer as a single JSON object that validates against this JSON Schema "
    "(this is the `{tool_name}` output tool{described}):\n\n"
    "{schema}\n\n"
    "Output the JSON object and nothing else: no prose, no explanation, no code fences."
)


def _extract_json(text: str) -> dict[str, Any]:
    """Model çıktısından JSON nesnesini çıkarır. Onarım yapmaz, yalnız ambalajı soyar."""
    candidates = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(f"claude -p çıktısından JSON ayrıştırılamadı: {text[:400]!r}")


def _call_claude(*, model: str, effort: str, system: str, prompt: str, timeout: float) -> str:
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--tools",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt",
        system,
        "--output-format",
        "json",
        prompt,
    ]
    completed = subprocess.run(  # noqa: S603 — sabit komut, kabuk yok
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"claude -p exit {completed.returncode}: {completed.stderr.strip()[:400]}"
        )
    payload = json.loads(completed.stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"claude -p hata döndürdü: {str(payload.get('result'))[:400]}")
    return str(payload.get("result", ""))


def build_model(*, model_id: str, effort: str, timeout: float = 300.0) -> FunctionModel:
    """`model_id`/`effort` için `claude -p` şeklinde çalışan bir FunctionModel kurar.

    `model_name=model_id` bilinçli olarak sabit: `CachedModel._cache_path` bu değeri
    hash anahtarına koyuyor, üretim koşusuyla replay'in aynı anahtara düşmesi buna bağlı.
    """

    def _run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        system = "\n\n".join(p.content for p in parts if p.part_kind == "system-prompt")
        user_texts: list[str] = []
        for p in parts:
            if p.part_kind != "user-prompt":
                continue
            if not isinstance(p.content, str):
                raise RuntimeError(
                    "demo_sonnet_5 yalnız metin promptları destekler (multimodal yok)."
                )
            user_texts.append(p.content)
        user = "\n\n".join(user_texts)
        tools = list(info.output_tools or ())
        if len(tools) != 1:
            raise RuntimeError(
                f"demo_sonnet_5 yalnız tek çıktı aracı bekliyor, bulunan {len(tools)}."
            )
        tool = tools[0]
        described = f", described as: {tool.description}" if tool.description else ""
        instruction = _SCHEMA_INSTRUCTION.format(
            tool_name=tool.name,
            described=described,
            schema=json.dumps(tool.parameters_json_schema, ensure_ascii=False, indent=2),
        )
        prompt = f"{user}\n\n---\n\n{instruction}\n"
        raw = _call_claude(
            model=model_id, effort=effort, system=system, prompt=prompt, timeout=timeout
        )
        args = _extract_json(raw)
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=args)])

    return FunctionModel(_run, model_name=model_id)
