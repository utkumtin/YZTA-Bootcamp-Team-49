"""`results.jsonl`'deki hata sınıflarını güncel `classify_error` ile yeniden türetir.

Neden gerekli: koşu sırasında `classify_error` düzeltilirse çalışan süreç eski
sürümü yüklü tutar (Python modülü zaten import edilmiş), yani o koşunun kalan
satırları eski sınıflandırmayla yazılır. `error_detail` ham sağlayıcı mesajını
sakladığı için sınıf sonradan doğru şekilde yeniden türetilebilir.

Yalnız METİN tabanlı kuralları uygular: tip tabanlı kontroller (ValidationError vb.)
zaten doğru sınıfı üretmişti ve buradan geçirilmiyor — bu script bir satırın sınıfını
ancak güncel kurallar FARKLI bir sonuç veriyorsa değiştirir. İdempotent: ikinci koşuda
hiçbir şey değişmez.

Kullanım:
    python scripts/repair_error_classes.py runs/benchmark/tur-1/results.jsonl
    python scripts/repair_error_classes.py <path> --apply   # yazar (öncesi .bak-repair)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_model_benchmark import (  # noqa: E402
    classify_error,
    elimination_signal,
    schema_verdict,
)


def rederive(row: dict) -> str | None:
    """Satırın hata sınıfını `error_detail` metninden yeniden türetir.

    Tip bilgisi kaybolmuş durumda: metin tabanlı kurallar `hata:{name}`'e düşerse
    ORİJİNAL sınıf korunur — tip tabanlı bir karar olabilir ve onu bozmamalıyız.
    """
    detail = row.get("error_detail")
    if not detail:
        return row.get("error")
    kind = classify_error(RuntimeError(str(detail)))
    if kind.startswith("hata:"):
        return row.get("error")
    return kind


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--apply", action="store_true", help="dosyayı yaz (yoksa yalnız rapor)")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = []
    for row in rows:
        if row.get("error") is None:
            continue
        new_kind = rederive(row)
        if new_kind == row.get("error"):
            continue
        changed.append(
            f"{row['model_id']} {row['task']}/{row['case_id']}#{row['repeat']}: "
            f"{row['error']} (schema_ok={row.get('schema_ok')}) -> "
            f"{new_kind} (schema_ok={schema_verdict(new_kind)})"
        )
        row["error"] = new_kind
        row["schema_ok"] = schema_verdict(new_kind)
        row["reclassified"] = True
        # Yanlış sınıfla yazılmış bir `circuit_broken` bayrağı, yeni sınıf artık
        # elemeye saymıyorsa geçersizdir: bırakılırsa rapor var olmayan bir elemeyi
        # gösterir. Bayrağın kalkması o görevin ATLANAN çağrılarını geri getirmez —
        # onlar `done`'da olmadığı için sonraki koşuda kendiliğinden denenir.
        still_counts = elimination_signal(new_kind, float(row.get("latency_s") or 0.0))
        if row.get("circuit_broken") and not still_counts:
            row.pop("circuit_broken")
            changed[-1] += " · circuit_broken bayrağı kaldırıldı"

    print(f"{len(rows)} satır · yeniden sınıflanan: {len(changed)}")
    for line in changed:
        print(f"  {line}")
    if not changed:
        return 0
    if not args.apply:
        print("\n(rapor modu — yazmak için --apply)")
        return 0

    backup = args.path.with_suffix(f"{args.path.suffix}.bak-repair")
    backup.write_text(args.path.read_text(encoding="utf-8"), encoding="utf-8")
    args.path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"\nyazıldı · yedek: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
