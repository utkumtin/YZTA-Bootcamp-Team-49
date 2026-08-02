"""Benchmark promptlarını vaka başına bir klasöre döker (çağrı yapmaz).

Amaç: bir modeli benchmark'ın ölçüm hattı DIŞINDA (örn. terminal üzerinden bir
Claude Code oturumunda) koşarken, modele benchmark'takiyle AYNI bilgiyi vermek.

Neden döküm, neden dataset kopyası değil: JUDGE görevlerinde model ham CSV
görmüyor. `run_model_benchmark.dataset_inputs` paneli kurup `profile_dataframe`
çıktısını, kolon listesini ve panel config'ini prompt'a gömüyor; modele giden tek
şey bu metin. Dataset klasörünü modele vermek başka bir görevi ölçmek olurdu.

Döküm gerçek üretim yolundan alınır: `router.use_test_model` ile araya bir
`FunctionModel` konur, `TASK_RUNNERS[task](case)` normal şekilde çağrılır ve
model isteği tam oluştuğu anda yakalanıp kesilir. Böylece sistem promptu, user
promptu ve çıktı şeması (pydantic-ai'nin `final_result` aracı) aynen kaydedilir;
prompt burada yeniden yazılmaz.

Kullanım:
    python scripts/dump_benchmark_prompts.py --out runs/benchmark/referans-claude-cli
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_model_benchmark as bench  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from pareto.llm import router  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "benchmark" / "referans-claude-cli"

# Dökümün prompta EKLEDİĞİ tek şey. Benchmark'ta şema, sağlayıcıya bir araç
# tanımı (`final_result`) olarak gider; terminal oturumunda araç yok, o yüzden
# aynı şema metne taşınır. Bu delta meta.json'a da yazılır: raporda "modelin
# gördüğü ekstra bilgi" diye tartışılabilir tek fark budur.
_SCHEMA_INSTRUCTION = (
    "Return your answer as a single JSON object that validates against this JSON Schema "
    "(in the benchmark this is the `{tool_name}` output tool{described}):\n\n"
    "{schema}\n\n"
    "Output the JSON object and nothing else: no prose, no explanation, no code fences."
)


class _PromptCaptured(Exception):
    """Model isteği yakalandı; çağrıyı burada kes."""


def capture_prompt(task: str, case: dict[str, Any]) -> dict[str, Any]:
    """Bir (görev, vaka) için modele giden tam isteği döndürür."""
    box: dict[str, Any] = {}

    def _capture(messages: list[Any], info: AgentInfo) -> Any:
        parts = [part for message in messages for part in message.parts]
        box["system"] = "\n\n".join(p.content for p in parts if p.part_kind == "system-prompt")
        box["user"] = "\n\n".join(p.content for p in parts if p.part_kind == "user-prompt")
        tools = list(info.output_tools or ())
        if len(tools) != 1:
            raise SystemExit(
                f"{task}/{case['case_id']}: beklenen tek çıktı aracı, bulunan {len(tools)}. "
                "Döküm bu varsayıma dayanıyor (bkz. _SCHEMA_INSTRUCTION)."
            )
        box["tool_name"] = tools[0].name
        box["tool_description"] = tools[0].description or ""
        box["schema"] = tools[0].parameters_json_schema
        raise _PromptCaptured

    with router.use_test_model(FunctionModel(_capture)):
        try:
            bench.TASK_RUNNERS[task](case)
        except _PromptCaptured:
            pass

    if not box:
        raise SystemExit(
            f"{task}/{case['case_id']}: model isteği hiç oluşmadı. Görev LLM'e gitmeden mi bitti?"
        )
    return box


def compose_prompt(captured: dict[str, Any]) -> str:
    """Terminal oturumuna yapıştırılacak user-tarafı prompt (sistem promptu ayrı gider)."""
    described = (
        f", described as: {captured['tool_description']}" if captured["tool_description"] else ""
    )
    instruction = _SCHEMA_INSTRUCTION.format(
        tool_name=captured["tool_name"],
        described=described,
        schema=json.dumps(captured["schema"], ensure_ascii=False, indent=2),
    )
    return f"{captured['user']}\n\n---\n\n{instruction}\n"


def dump_case(task: str, case: dict[str, Any], out_dir: Path) -> Path:
    captured = capture_prompt(task, case)
    case_dir = out_dir / task / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "system.txt").write_text(captured["system"], encoding="utf-8")
    (case_dir / "user.txt").write_text(captured["user"], encoding="utf-8")
    (case_dir / "schema.json").write_text(
        json.dumps(captured["schema"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (case_dir / "PROMPT.md").write_text(compose_prompt(captured), encoding="utf-8")
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "case_id": case["case_id"],
                "dataset_dir": case.get("dataset_dir"),
                "tool_name": captured["tool_name"],
                "system_chars": len(captured["system"]),
                "user_chars": len(captured["user"]),
                "prompt_delta": (
                    "Yalnız şema talimatı eklendi (araç tanımı yerine metin). "
                    "Sistem ve user promptu benchmark'takiyle birebir aynı."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return case_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", nargs="+", choices=bench.TASKS, default=list(bench.TASKS))
    args = parser.parse_args(argv)

    prompts_root = (REPO_ROOT / args.out).resolve() / "prompts"
    total = 0
    for task in args.tasks:
        for case in bench.load_gold(task):
            case_dir = dump_case(task, case, prompts_root)
            total += 1
            print(f"  {task}/{case['case_id']:<16} -> {case_dir.relative_to(REPO_ROOT)}")
    print(f"\n{total} vaka döküldü: {prompts_root.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
