"""Elle/CLI koşulmuş model çıktılarını üretim yolundan geçirip puanlar.

Model KENDİ notunu vermez: `results.jsonl`'in `scores` alanı bu script'te,
`run_model_benchmark.py`'nin saf puanlayıcılarından (`score_cleaning`,
`score_estimand`, `score_spec_menu`, `score_narrative`) üretilir. Modelden gelen
tek şey ham çıktı JSON'udur.

Nasıl çalışıyor: modelin JSON'u, benchmark'ın çağırdığı `generate_*`
fonksiyonuna, çıktı aracının (`final_result`) cevabı gibi enjekte edilir. Böylece
pydantic şema zorlaması, guardrail'ler ve fail-loud doğrulayıcılar aynen koşar:
`schema_ok` ve `validator_passed` benchmark'takiyle aynı anlamı taşır.
pydantic-ai şemayı tutturamayıp ikinci kez modele dönerse bu bir şema kusurudur
ve `sema_tutmadi` olarak kaydedilir (gerçek koşuda retry olurdu; burada model
yok, o yüzden tek atış).

ÖLÇÜLMEYEN alanlar bilerek `null`: `latency_s`, `input_tokens`, `output_tokens`,
`requests`, `retries`. Bu çağrılar Claude Code harness'ının içinden geçtiği için
o sayılar sağlayıcı ölçümüyle karşılaştırılamaz (ölçüldü: harness tek kelimelik
promptta bile 12.951 token bağlam yüklüyor). Satırlar `mode: "manual-cli"` ile
işaretlenir ve ücretsiz uçların koşusuyla AYNI raporda toplanmamalıdır.

Kullanım:
    python scripts/score_manual_outputs.py --root runs/benchmark/referans-claude-cli
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
from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from pareto.llm import router  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "runs" / "benchmark" / "referans-claude-cli"


class _SchemaRejected(Exception):
    """Enjekte edilen çıktı şemayı tutturamadı (pydantic-ai ikinci kez modele döndü)."""


def _replay_model(payload: dict[str, Any]) -> FunctionModel:
    state = {"calls": 0}

    def _fn(messages: list[Any], info: AgentInfo) -> ModelResponse:
        state["calls"] += 1
        if state["calls"] > 1:
            raise _SchemaRejected("şema retry istendi")
        tools = list(info.output_tools or ())
        if len(tools) != 1:
            raise SystemExit(f"beklenen tek çıktı aracı, bulunan {len(tools)}")
        return ModelResponse(parts=[ToolCallPart(tool_name=tools[0].name, args=payload)])

    return FunctionModel(_fn)


def score_one(
    task: str, case: dict[str, Any], payload: dict[str, Any] | None, parse_error: str | None
) -> dict[str, Any]:
    """Tek çıktıyı üretim yolundan geçirip puan/hata alanlarını döndürür."""
    if payload is None:
        return {
            "schema_ok": False,
            "validator_passed": False,
            "error": "sema_tutmadi",
            "error_detail": parse_error or "JSON yok",
            "scores": None,
        }
    try:
        with router.use_test_model(_replay_model(payload)):
            scores = bench.TASK_RUNNERS[task](case)
    except Exception as exc:  # noqa: BLE001 — sessiz atlama yok
        kind = "sema_tutmadi" if isinstance(exc, _SchemaRejected) else bench.classify_error(exc)
        return {
            "schema_ok": bench.schema_verdict(kind),
            "validator_passed": False,
            "error": kind,
            "error_detail": str(exc)[:1200],
            "scores": None,
        }
    return {
        "schema_ok": True,
        "validator_passed": True,
        "error": None,
        "error_detail": None,
        "scores": scores,
    }


def _merge_with_existing(results_path: Path, fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yeni satırları var olan `results.jsonl` ile `result_id` üzerinden birleştirir.

    Dosya eskiden `write_text` ile KURULUYORDU: `--models X` ile tek modeli
    yeniden puanlamak diğer modelin satırlarını sessizce siliyordu (bir kez
    yaşandı). Aynı `result_id` taze satırla EZİLİR — yeniden puanlamanın amacı
    zaten budur; ezilmeyen satırlar olduğu gibi korunur.
    """
    if not results_path.exists():
        return fresh

    fresh_ids = {r["result_id"] for r in fresh}
    kept: list[dict[str, Any]] = []
    for lineno, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            existing = json.loads(line)
        except json.JSONDecodeError as exc:
            # Fail-loud: bozuk bir satırı atlamak sessiz veri kaybıdır.
            raise SystemExit(f"{results_path}:{lineno} ayrıştırılamadı: {exc}") from exc
        if "result_id" not in existing:
            raise SystemExit(f"{results_path}:{lineno} `result_id` taşımıyor; birleştirilemez.")
        if existing["result_id"] not in fresh_ids:
            kept.append(existing)

    if kept:
        print(f"\n{len(kept)} mevcut satır korundu, {len(fresh)} satır yazıldı/güncellendi.")
    return kept + fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", help="outputs/ altındaki model klasörleri")
    args = parser.parse_args(argv)

    root = (REPO_ROOT / args.root).resolve()
    outputs_root = root / "outputs"
    if not outputs_root.exists():
        raise SystemExit(f"Çıktı yok: {outputs_root}. Önce run_cli_reference.py.")

    gold = {task: {str(c["case_id"]): c for c in bench.load_gold(task)} for task in bench.TASKS}
    model_dirs = sorted(
        d
        for d in outputs_root.iterdir()
        if d.is_dir() and (not args.models or d.name in args.models)
    )

    rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        for output_path in sorted(model_dir.glob("*/*/output_r*.json")):
            case_id = output_path.parent.name
            task = output_path.parent.parent.name
            repeat = int(output_path.stem.split("_r")[-1])
            case = gold[task].get(case_id)
            if case is None:
                raise SystemExit(f"Altın kayıt yok: {task}/{case_id}")
            data = json.loads(output_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "result_id": f"{model_dir.name}|{task}|{case_id}|{repeat}",
                "model": model_dir.name,
                "provider": "anthropic-cli",
                "model_id": model_dir.name,
                "served_by": model_dir.name,
                "date": None,
                "task": task,
                "case_id": case_id,
                "repeat": repeat,
                # Harness içinden geçen çağrıda bu sayılar ölçülemez (bkz. modül notu).
                "mode": "manual-cli",
                "latency_s": None,
                "requests": None,
                "retries": None,
                "input_tokens": None,
                "output_tokens": None,
            }
            row.update(score_one(task, case, data.get("parsed"), data.get("parse_error")))
            rows.append(row)
            status = row["error"] or "ok"
            print(f"  {model_dir.name} {task}/{case_id} r{repeat}  {status}")

    if not rows:
        raise SystemExit("Puanlanacak çıktı bulunamadı.")

    results_path = root / "results.jsonl"
    merged = _merge_with_existing(results_path, rows)
    results_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8"
    )
    report_path = root / "report.md"
    report = bench.build_report(merged, bench.TASKS)
    report_path.write_text(
        "> Bu koşu `claude -p` üzerinden alındı; gecikme/token/retry kolonları "
        "ÖLÇÜLMEDİ (bkz. scripts/score_manual_outputs.py). Ücretsiz uç koşusuyla "
        "aynı tabloda toplanmamalıdır.\n\n" + report,
        encoding="utf-8",
    )
    ok = sum(1 for r in rows if r["error"] is None)
    print(f"\n{len(rows)} satır · {ok} geçti\n{results_path}\n{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
