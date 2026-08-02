"""Dökülmüş benchmark promptlarını `claude -p` ile koşar (referans koşusu).

Bu, `run_model_benchmark.py`'nin ÖLÇÜM hattı DEĞİLDİR ve onun yerine geçmez.
Ücretsiz uçlarla karşılaştırılabilir tek şey ÇIKTININ KALİTESİDİR: token, gecikme
ve retry sayıları Claude Code harness'ının içinden geçtiği için anlamsızdır ve
puanlayıcıya hiç taşınmaz (bkz. score_manual_outputs.py).

Ölçüm ortamının temizliği (hepsi ölçülerek doğrulandı, 2026-08-01):

  izole HOME       : `~/.claude/CLAUDE.md` + RTK.md + skills listesi normalde
                     boş bir dizinde bile modele system-reminder olarak giriyor.
                     Ölçüldü: 12.951 token bağlam. İzole HOME + --system-prompt
                     ile 208 token'a düşüyor; kalan tek enjeksiyon e-posta +
                     tarih satırı.
  --system-prompt  : Claude Code'un kendi sistem promptu tamamen benchmark'ın
                     sistem promptuyla değiştirilir.
  --tools ""       : araç yok. Aksi halde model dosya arar ve ölçtüğümüz şey
                     "tek atışta şemaya uyan çıktı" olmaktan çıkar.
  ayrı süreç       : her çağrı taze bağlam. Tek oturumda 2. tekrar 1. cevabı
                     görürdü ve `answer_consistency` şişerdi.

Kalan farklar (rapora yazılmalı): Agent SDK'nın sistem promptuna eklediği tek
cümlelik önek, e-posta/tarih system-reminder'ı ve şemanın araç tanımı yerine
metin olarak verilmesi.

Kullanım:
    python scripts/run_cli_reference.py --model claude-sonnet-5 --effort medium
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "runs" / "benchmark" / "referans-claude-cli"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def build_isolated_home(root: Path) -> Path:
    """Kullanıcının global CLAUDE.md/skill/hook'larını görmeyen bir HOME kurar.

    Yalnız oturum açma bilgisi kopyalanır; talimat taşıyan hiçbir dosya
    kopyalanmaz. Kopyalanırsa model benchmark'takinden fazla bilgi görür.
    """
    home = root / "isolated-home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    creds = Path.home() / ".claude" / ".credentials.json"
    if creds.exists():
        shutil.copy2(creds, home / ".claude" / ".credentials.json")
    src_config = Path.home() / ".claude.json"
    if src_config.exists():
        full = json.loads(src_config.read_text(encoding="utf-8"))
        keep = {
            key: full[key]
            for key in ("userID", "oauthAccount", "installMethod", "firstStartTime")
            if key in full
        }
        keep["hasCompletedOnboarding"] = True
        keep["projects"] = {}
        (home / ".claude.json").write_text(json.dumps(keep), encoding="utf-8")
    if not creds.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Oturum açma bilgisi bulunamadı (~/.claude/.credentials.json yok ve "
            "ANTHROPIC_API_KEY tanımsız)."
        )
    return home


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Model çıktısından JSON nesnesini çıkarır. Onarım YAPMAZ, yalnız ambalajı soyar.

    Kod bloğu/önsöz ayıklamak ölçüme müdahale değil: benchmark'ta bunun karşılığı
    araç çağrısı ambalajıdır. JSON'un İÇERİĞİNE dokunmak ise ölçtüğümüz şeyi
    bozardı, o yüzden ayrıştırma başarısızsa hata olduğu gibi kaydedilir.
    """
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
            return parsed, None
    return None, "çıktıdan JSON nesnesi ayrıştırılamadı"


def call_once(
    case_dir: Path, *, model: str, effort: str, home: Path, cwd: Path, timeout: float
) -> dict[str, Any]:
    system_prompt = (case_dir / "system.txt").read_text(encoding="utf-8")
    prompt = (case_dir / "PROMPT.md").read_text(encoding="utf-8")
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
        system_prompt,
        "--output-format",
        "json",
        prompt,
    ]
    env = {**os.environ, "HOME": str(home)}
    env.pop("ANTHROPIC_API_KEY", None)
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 — sabit komut, kabuk yok
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"cli_error": f"timeout ({timeout:.0f}s)", "wall_s": round(timeout, 3)}
    wall = round(time.monotonic() - started, 3)
    if completed.returncode != 0:
        return {
            "cli_error": f"exit {completed.returncode}: {completed.stderr.strip()[:400]}",
            "wall_s": wall,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"cli_error": f"CLI JSON değil: {completed.stdout[:400]}", "wall_s": wall}
    payload["wall_s"] = wall
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    root = (REPO_ROOT / args.root).resolve()
    prompts_root = root / "prompts"
    if not prompts_root.exists():
        raise SystemExit(f"Prompt dökümü yok: {prompts_root}. Önce dump_benchmark_prompts.py.")

    home = build_isolated_home(root)
    cwd = root / "cleanroom"
    cwd.mkdir(parents=True, exist_ok=True)
    if any(cwd.iterdir()):
        raise SystemExit(f"{cwd} boş olmalı: içindeki dosyalar modele bağlam olarak girer.")
    out_root = root / "outputs" / args.model
    print(f"Model: {args.model} · effort: {args.effort} · tekrar: {args.repeats}")

    case_dirs = sorted(
        d
        for d in prompts_root.glob("*/*")
        if d.is_dir() and (not args.tasks or d.parent.name in args.tasks)
    )
    calls = failures = 0
    for case_dir in case_dirs:
        task, case_id = case_dir.parent.name, case_dir.name
        target = out_root / task / case_id
        target.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, args.repeats + 1):
            raw_path = target / f"raw_r{repeat}.json"
            if raw_path.exists():  # resume: yapılmış çağrı tekrarlanmaz
                continue
            payload = call_once(
                case_dir,
                model=args.model,
                effort=args.effort,
                home=home,
                cwd=cwd,
                timeout=args.timeout,
            )
            calls += 1
            raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if payload.get("cli_error") or payload.get("is_error"):
                failures += 1
                detail = payload.get("cli_error") or str(payload.get("result"))[:200]
                print(f"  {task}/{case_id} r{repeat}  HATA  {detail}")
                continue
            parsed, error = extract_json(str(payload.get("result", "")))
            (target / f"output_r{repeat}.json").write_text(
                json.dumps(
                    {"parsed": parsed, "parse_error": error, "raw_text": payload.get("result")},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            failures += error is not None
            mark = "ok" if error is None else "AYRIŞTIRILAMADI"
            print(f"  {task}/{case_id} r{repeat}  {mark}  {payload.get('wall_s')}s")

    print(f"\n{calls} çağrı · {failures} sorunlu\nÇıktı: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
