"""Reprodüksiyon paketi — tek-tık zip export.

Ürün tezinin ("denetim izi = metot bölümü") somut çıktısı: bir koşunun tüm
artefaktları tek bir zip'te toplanır ve paketin içindeki tek komutluk script,
paketlenmiş sonuçları yeniden üretip tolerans eşitliğiyle doğrular. Uyuşmazlık
sessizce geçilmez, script sıfırdan farklı bir çıkış koduyla biter.

Girdiler AÇIKÇA yol olarak verilir, tek bir run_id'den türetilmez: temizleme
run_id'si (uuid) ile multiverse run_id'si farklıdır, üstelik panel sayfası
varsayılan olarak `runs/latest` aynasını okur ve o yolda run_id hiç geçmez.
Yolu çözmek çağıran katmanın (Streamlit sayfası) işidir.

Eksik artefakt paketi engellemez ama gizlenmez de: `missing_artifacts()` listesi
hem MANIFEST.json'a hem de arayüz uyarısına aynı kaynaktan beslenir.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.variance import summarize
from ..config import SETTINGS
from ..contracts import EstimationResult
from ..spec import Specification
from .methods import render_methods_section

PACKAGE_FORMAT = "pareto-reproduction/1"

# Zip girdilerinin zaman damgası sabitlenir: aynı koşudan iki kez üretilen paket
# byte düzeyinde aynı olsun (paketin kendisi de reprodüklenebilir olmalı).
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

# Eksik artefaktların sabit kimlikleri; arayüz bunları etiketlerle eşler.
ARTIFACT_LABELS: dict[str, str] = {
    "specs": "spesifikasyon listesi",
    "panel": "analiz paneli",
    "frozen_menu": "dondurulmuş estimand ve menü",
    "decision_ledger": "karar defteri",
    "cleaning_script": "üretilen temizleme script'i",
    "raw_panel": "ham veri (temizleme öncesi)",
    "requirements": "requirements.txt",
    "figures": "figürler",
}


class ReproPackageError(RuntimeError):
    """Paket kurulamadı: zorunlu artefakt yok ya da okunamıyor."""


@dataclass(frozen=True)
class ReproInputs:
    """Pakete girecek artefaktların çözülmüş yolları.

    `results_path` dışındaki her alan opsiyoneldir; verilmeyen ya da diskte
    bulunmayan artefakt eksik sayılır ve manifest'e eksik olarak yazılır.
    `figures`: dosya adı -> kendi kendine yeten HTML gövdesi.
    """

    run_id: str
    results_path: Path
    specs_path: Path | None = None
    panel_path: Path | None = None
    frozen_menu_path: Path | None = None
    ledger_path: Path | None = None
    cleaning_script_path: Path | None = None
    raw_panel_path: Path | None = None
    figures: Mapping[str, str] = field(default_factory=dict)


def _present(path: Path | None) -> Path | None:
    return path if path is not None and path.exists() else None


def _has_specs(path: Path | None) -> bool:
    """Spesifikasyon dosyası var ve DOLU mu.

    Yalnız yola bakmak yetmez: boş bir `specs.json` pakete hiç girmez, o yüzden
    "eksik yok" demek paketin doğrulanabildiği yanılgısını üretir.
    """
    resolved = _present(path)
    if resolved is None:
        return False
    try:
        payload = _read_json(resolved)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and bool(payload)


def missing_artifacts(inputs: ReproInputs) -> list[str]:
    """Pakete giremeyen artefaktların kimlikleri (deterministik sırada)."""
    missing = [] if _has_specs(inputs.specs_path) else ["specs"]
    missing += [
        key
        for key, path in (
            ("panel", inputs.panel_path),
            ("frozen_menu", inputs.frozen_menu_path),
            ("decision_ledger", inputs.ledger_path),
            ("cleaning_script", inputs.cleaning_script_path),
            ("raw_panel", inputs.raw_panel_path),
        )
        if _present(path) is None
    ]
    if _present(_requirements_path()) is None:
        missing.append("requirements")
    if not inputs.figures:
        missing.append("figures")
    return missing


def _requirements_path() -> Path:
    return Path(__file__).resolve().parents[2] / "requirements.txt"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Diske yazılmış paneli okur. `panel.pkl`'i bu repo yazar, aynı güven sınırı."""
    frame = pickle.loads(path.read_bytes())  # noqa: S301
    if not isinstance(frame, pd.DataFrame):
        raise ReproPackageError(f"Panel dosyası DataFrame içermiyor: {path}")
    return frame


def _dtype_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


def _pareto_version() -> str:
    try:
        return version("pareto")
    except PackageNotFoundError:
        return "bilinmiyor"


def _cleaning_decisions(ledger_path: Path | None) -> list[dict[str, Any]]:
    """Karar defterini (JSON Lines) metot bölümünün okuyacağı sadeleştirilmiş hâle getirir."""
    if ledger_path is None:
        return []
    decisions: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        decisions.append(
            {
                "bulgu": entry.get("bulgu", ""),
                "transform_name": entry.get("transform_name", ""),
                "gerekce": entry.get("gerekce", ""),
                "resolution": entry.get("resolution"),
                "belirsizlik_bayragi": bool(entry.get("belirsizlik_bayragi", False)),
            }
        )
    return decisions


def _build_manifest(
    inputs: ReproInputs,
    *,
    results: list[EstimationResult],
    specs: list[Specification],
    frozen: dict[str, Any] | None,
    decisions: list[dict[str, Any]],
    contents: list[str],
) -> dict[str, Any]:
    """Paketin tek makine-okunur kaydı. METHODS.md bunun insan render'ıdır."""
    return {
        "package_format": PACKAGE_FORMAT,
        "pareto_version": _pareto_version(),
        "run_id": inputs.run_id,
        "estimand_hash": (frozen or {}).get("estimand_hash"),
        "menu_hash": (frozen or {}).get("menu_hash"),
        "estimand": (frozen or {}).get("estimand"),
        "spec_count": len(specs),
        "spec_hashes": {spec.spec_id: spec.content_hash() for spec in specs},
        "determinism": {
            "seed": SETTINGS.seed,
            "env": dict(SETTINGS.deterministic_env),
            "llm_temperature": SETTINGS.llm_temperature,
        },
        "variance_summary": dict(summarize(results)),
        "cleaning_decisions": decisions,
        "contents": sorted(contents),
        "missing": missing_artifacts(inputs),
    }


def build_reproduction_package(inputs: ReproInputs) -> bytes:
    """Koşunun artefaktlarını tek bir zip'e toplar ve zip'in byte'larını döndürür.

    `bytes` döndürmek iki tüketiciyi de karşılar: Streamlit `download_button`'ı
    ve testteki `zipfile.ZipFile(io.BytesIO(...))`.
    """
    results_path = _present(inputs.results_path)
    if results_path is None:
        raise ReproPackageError(
            f"Sonuç dosyası yok: {inputs.results_path}. Reprodüksiyon paketi sonuçsuz kurulamaz."
        )
    results = [EstimationResult(**item) for item in _read_json(results_path)]

    specs_path = _present(inputs.specs_path)
    specs = (
        [Specification(**item) for item in _read_json(specs_path)] if specs_path is not None else []
    )

    frozen_path = _present(inputs.frozen_menu_path)
    frozen = _read_json(frozen_path) if frozen_path is not None else None

    ledger_path = _present(inputs.ledger_path)
    decisions = _cleaning_decisions(ledger_path)

    files: dict[str, str | bytes] = {}

    files["results.json"] = json.dumps(
        [r.model_dump() for r in results], ensure_ascii=False, indent=2
    )
    if specs:
        files["specs.json"] = json.dumps(
            [s.model_dump() for s in specs], ensure_ascii=False, indent=2
        )

    dtypes: dict[str, dict[str, str]] = {}
    panel_path = _present(inputs.panel_path)
    if panel_path is not None:
        panel = _read_dataframe(panel_path)
        files["data/panel.csv"] = panel.to_csv(index=False)
        dtypes["panel"] = _dtype_map(panel)

    raw_path = _present(inputs.raw_panel_path)
    if raw_path is not None:
        raw = _read_dataframe(raw_path)
        files["data/raw.csv"] = raw.to_csv(index=False)
        dtypes["raw"] = _dtype_map(raw)

    if dtypes:
        # CSV tip taşımaz: FIPS gibi öndeki sıfırları korunan kolonlar okurken
        # sayıya döner ve temizleme karşılaştırması yanlış yere patlar. Tipler
        # ayrı bir kayıtla taşınır, run script okurken geri uygular.
        files["data/dtypes.json"] = json.dumps(dtypes, ensure_ascii=False, indent=2)

    cleaning_script_path = _present(inputs.cleaning_script_path)
    if cleaning_script_path is not None:
        # Script yeniden RENDER EDİLMEZ, diskteki hâli kopyalanır: L4 kapısının
        # doğruladığı artefakt ile pakete giren artefakt aynı olmak zorunda.
        files["cleaning/cleaning_steps.py"] = cleaning_script_path.read_text(encoding="utf-8")
    if ledger_path is not None:
        files["cleaning/decision_ledger.jsonl"] = ledger_path.read_text(encoding="utf-8")

    requirements_path = _present(_requirements_path())
    if requirements_path is not None:
        files["requirements.txt"] = requirements_path.read_text(encoding="utf-8")

    for name, html in inputs.figures.items():
        files[f"figures/{name}"] = html
    if inputs.figures:
        from plotly.offline import get_plotlyjs

        # Figürler `include_plotlyjs="directory"` ile üretilir: kütüphane tek kopya
        # olarak yanlarına yazılır, paket internetsiz açılır.
        files["figures/plotly.min.js"] = get_plotlyjs()

    manifest = _build_manifest(
        inputs,
        results=results,
        specs=specs,
        frozen=frozen,
        decisions=decisions,
        contents=list(files),
    )
    files["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)
    files["METHODS.md"] = render_methods_section(manifest)
    files["README.md"] = _render_readme(manifest)
    files["run_reproduction.py"] = RUN_SCRIPT

    return _zip_bytes(files)


def _zip_bytes(files: Mapping[str, str | bytes]) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            payload = files[name]
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.encode("utf-8") if isinstance(payload, str) else payload)
    return buffer.getvalue()


def _render_readme(manifest: dict[str, Any]) -> str:
    missing = manifest.get("missing", [])
    missing_block = (
        "\n".join(f"- {ARTIFACT_LABELS.get(key, key)}" for key in missing)
        if missing
        else "- yok, paket eksiksiz"
    )
    return f"""# Reprodüksiyon paketi — {manifest["run_id"]}

Bu paket bir Pareto koşusunun denetim izidir: temizleme kararları, üretilen
temizleme script'i, dondurulmuş estimand ve spesifikasyon menüsü, çalıştırılan
spesifikasyonlar, sonuçlar ve figürler.

## Tek komutla doğrulama

Paket `pareto` paketinin kurulu olduğu bir ortamda koşar (paket PyPI'da değildir,
repo kopyasında `pip install -e .` ile kurulur):

```
pip install -r requirements.txt
python run_reproduction.py
```

Script paketlenmiş paneli ve spesifikasyonları yeniden koşar, ürettiği katsayı,
standart hata ve gözlem sayılarını `results.json` ile tolerans içinde karşılaştırır.
Ham veri ve temizleme script'i pakete girmişse önce temizleme adımını da tekrarlar.
Uyuşmazlıkta script sıfırdan farklı bir çıkış koduyla biter.

## İçerik

| Yol | Ne |
|-----|-----|
| `MANIFEST.json` | Makine-okunur tek kayıt: hash'ler, determinizm pinleri, özet |
| `METHODS.md` | Manifest'in metot bölümü taslağı olarak render'ı (LLM kullanılmaz) |
| `run_reproduction.py` | Tek komutluk doğrulama script'i |
| `specs.json` | Çalıştırılan spesifikasyonlar |
| `results.json` | Referans sonuçlar |
| `data/panel.csv` | Analize giren panel |
| `data/raw.csv` | Temizleme öncesi ham veri |
| `data/dtypes.json` | CSV'lerin orijinal kolon tipleri (okurken geri uygulanır) |
| `cleaning/cleaning_steps.py` | Karar defterinden render edilmiş temizleme script'i |
| `cleaning/decision_ledger.jsonl` | Karar defteri (insan kararlarıyla damgalı) |
| `figures/` | Panel figürleri, kendi kendine yeten HTML |

## Bu pakette eksik olanlar

{missing_block}

## Determinizm

Koşu `seed={manifest["determinism"]["seed"]}` ve
`{manifest["determinism"]["env"]}` pinleriyle yapılır. LLM sıcaklığı
{manifest["determinism"]["llm_temperature"]}; ölçüm yapan hiçbir adımda LLM yoktur.
Paketin kendisi de deterministiktir: aynı koşudan iki kez üretilen zip byte
düzeyinde aynıdır.
"""


RUN_SCRIPT = '''"""Pareto reprodüksiyon doğrulaması — tek komut.

Paketlenmiş paneli ve spesifikasyonları yeniden koşar, sonucu `results.json` ile
tolerans içinde karşılaştırır. Uyuşmazlık sessizce geçilmez.

Kullanım: python run_reproduction.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RTOL = 1e-4
ATOL = 1e-6


def _fail(message: str) -> None:
    print(f"BAŞARISIZ: {message}")
    raise SystemExit(1)


def _import_pareto():
    try:
        from pareto.analysis.runner import run_specs
        from pareto.spec import Specification
    except ImportError as exc:
        _fail(
            f"`pareto` paketi içe aktarılamadı ({exc}). Paket PyPI'da değildir; "
            "repo kopyasında `pip install -e .` ile kurun."
        )
    return run_specs, Specification


def _read_table(name: str) -> pd.DataFrame:
    """Paketlenmiş CSV'yi orijinal tipleriyle okur.

    CSV tip taşımaz; tipler `data/dtypes.json` ile birlikte paketlenir. Aksi hâlde
    öndeki sıfırları korunan kolonlar sayıya döner ve karşılaştırma yanlış yere patlar.
    """
    path = HERE / "data" / f"{name}.csv"
    dtypes_path = HERE / "data" / "dtypes.json"
    if not dtypes_path.exists():
        return pd.read_csv(path)

    declared = json.loads(dtypes_path.read_text(encoding="utf-8")).get(name, {})
    date_cols = [col for col, dtype in declared.items() if str(dtype).startswith("datetime64")]
    plain = {col: dtype for col, dtype in declared.items() if col not in date_cols}
    return pd.read_csv(path, dtype=plain, parse_dates=date_cols or None)


def _verify_cleaning(panel: pd.DataFrame) -> None:
    script_path = HERE / "cleaning" / "cleaning_steps.py"
    raw_path = HERE / "data" / "raw.csv"
    if not script_path.exists() or not raw_path.exists():
        print("ATLANDI: temizleme adımı — pakette ham veri ya da temizleme script'i yok.")
        return

    spec = importlib.util.spec_from_file_location("pareto_repro_cleaning", script_path)
    if spec is None or spec.loader is None:
        _fail(f"Temizleme script'i yüklenemedi: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reproduced = module.clean(_read_table("raw")).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            reproduced, panel, check_dtype=False, rtol=RTOL, atol=ATOL
        )
    except AssertionError as exc:
        _fail(f"temizleme çıktısı paketlenmiş panel ile eşleşmiyor.\\n{exc}")
    print(f"TAMAM: temizleme adımı yeniden üretildi ({len(panel)} satır).")


def _verify_estimates() -> None:
    run_specs, Specification = _import_pareto()

    panel = _read_table("panel")
    specs = [
        Specification(**item)
        for item in json.loads((HERE / "specs.json").read_text(encoding="utf-8"))
    ]
    expected = {
        item["spec_id"]: item
        for item in json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    }

    _verify_cleaning(panel)

    mismatches: list[str] = []
    for result in run_specs(panel, specs):
        reference = expected.get(result.spec_id)
        if reference is None:
            mismatches.append(f"{result.spec_id}: pakette referans sonuç yok")
            continue
        if result.status != reference.get("status"):
            mismatches.append(
                f"{result.spec_id}: status {result.status} != {reference.get('status')}"
            )
            continue
        for field_name in ("coefficient", "std_error", "n_obs"):
            got = getattr(result, field_name)
            want = reference.get(field_name)
            if got is None or want is None:
                if got is not want:
                    mismatches.append(f"{result.spec_id}.{field_name}: {got} != {want}")
                continue
            if abs(float(got) - float(want)) > ATOL + RTOL * abs(float(want)):
                mismatches.append(f"{result.spec_id}.{field_name}: {got} != {want}")

    if mismatches:
        _fail(
            f"{len(mismatches)} spesifikasyon paketlenmiş sonuçla eşleşmedi:\\n  "
            + "\\n  ".join(mismatches)
        )
    print(f"TAMAM: {len(specs)} spesifikasyon paketlenmiş sonuçlarla tolerans içinde eşleşti.")


def main() -> None:
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"Pareto reprodüksiyon paketi · run_id={manifest['run_id']}")
    if manifest.get("missing"):
        print(f"UYARI: pakette eksik artefaktlar var: {manifest['missing']}")
    if not (HERE / "data" / "panel.csv").exists() or not (HERE / "specs.json").exists():
        _fail("panel ya da spesifikasyon listesi pakette yok; doğrulama koşulamaz.")
    _verify_estimates()
    print("Reprodüksiyon doğrulandı.")
    sys.exit(0)


if __name__ == "__main__":
    main()
'''
