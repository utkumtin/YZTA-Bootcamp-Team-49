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

Aynı gerekçeyle PROVENANS de doğrulanır: temizleme koşusu ile multiverse koşusu
ayrı run_id'ler taşır, dolayısıyla pakete birbirine ait olmayan bir karar defteri
ile sonuç çifti girebilir. `_provenance()` temizlemenin çıktısını multiverse'in
girdisiyle parmak iziyle karşılaştırır ve sonucu manifest'e yazar.
"""

from __future__ import annotations

import hashlib
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
from ..cleaning.codegen import REPRO_ATOL, REPRO_RTOL
from ..config import SETTINGS
from ..contracts import EstimationResult
from ..spec import Specification
from .methods import render_methods_section

# /3: manifest `tolerances` kaydını taşır (doğrulama script'i eşikleri oradan okur)
# ve `frozen_menu.json` pakete girer. /2 manifest `provenance` kaydını getirmişti.
# Eski paketler bu script'le doğrulanamaz, format alanı bunun için var.
PACKAGE_FORMAT = "pareto-reproduction/3"

# Tahmin karşılaştırmasının eşiği. Temizleme eşiğinden (REPRO_RTOL/ATOL) ayrıdır ve
# daha gevşektir: temizleme eşiği L4 kapısının pandas transformlarını ölçer, bu eşik
# ise BLAS/estimator gürültüsünü. İkisi de manifest'e yazılır, run script oradan okur;
# script'te ikinci bir kopya olsaydı paket kendi içinde çelişen iki karar verebilirdi.
ESTIMATE_RTOL = 1e-4
ESTIMATE_ATOL = 1e-6

# Manifest'in kendisi ve ondan türeyen dosyalar `contents` hesaplandıktan SONRA
# eklenir (METHODS ve README manifest'ten render edilir). Envanterin eksik kalmaması
# için adları burada sabit: "contents" paketin tam listesi olmalı.
_GENERATED_FILES = ("MANIFEST.json", "METHODS.md", "README.md", "run_reproduction.py")

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

    `cleaning_run_id` ve `cleaned_panel_path` pakete GİRMEZ; ikisi de provenans
    denetimi içindir: temizleme koşusunun kimliği manifest'e ayrı bir alan olarak
    yazılır, çıktısı (`reproduced.pkl`) ise multiverse'in girdisi olan panelle
    karşılaştırılır.
    """

    run_id: str
    results_path: Path
    specs_path: Path | None = None
    panel_path: Path | None = None
    frozen_menu_path: Path | None = None
    ledger_path: Path | None = None
    cleaning_script_path: Path | None = None
    raw_panel_path: Path | None = None
    cleaned_panel_path: Path | None = None
    cleaning_run_id: str | None = None
    figures: Mapping[str, str] = field(default_factory=dict)


def figure_html(name: str, figure: Any) -> str:
    """Figürü paket için kendi kendine yeten HTML'e çevirir.

    `include_plotlyjs="directory"` kütüphaneyi HTML'in yanındaki tek bir dosyadan
    okur: paket internetsiz açılır ama her figür için 4 MB tekrar taşınmaz.

    `div_id` figürün adına sabitlenir. Plotly aksi hâlde her çağrıda yeni bir uuid
    basar ve aynı koşudan üretilen iki paket byte düzeyinde farklı çıkar — paketin
    hash'lenip atıf verilebilmesi buna bağlı. Çağrı Streamlit sayfasında değil
    burada durur ki test edilebilsin; sayfa import edilemez, koşulur.
    """
    return figure.to_html(full_html=True, include_plotlyjs="directory", div_id=Path(name).stem)


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
    except ReproPackageError:
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


def package_key(inputs: ReproInputs) -> str:
    """Hazırlanmış paketin önbellek anahtarı: girdilerin TAMAMINI kapsar.

    Arayüz paketi oturumda önbelleğe alır. Anahtar yalnız sonuç yolunu ve eksik
    listesini kapsasaydı `_provenance()`in yakalamak için var olduğu senaryo tam da
    önbellekte kaybolurdu: A'yı temizleyip koşan ve paketi hazırlayan kullanıcı,
    sonra B'yi temizlediğinde sonuç yolu, run_id ve eksik listesi aynı kalır —
    anahtar değişmez, A'nın paketi servis edilir ve "EŞLEŞMİYOR" uyarısı hiç çıkmaz.
    Bu yüzden anahtar temizleme artefaktlarının yollarını da taşır.

    Sayfa import edilemediği için (Streamlit script'i, modül değil) anahtar burada
    durur: test edilebilen tek yer burası.
    """
    return json.dumps(
        {
            "run_id": inputs.run_id,
            "results": str(inputs.results_path),
            "specs": str(inputs.specs_path),
            "panel": str(inputs.panel_path),
            "frozen_menu": str(inputs.frozen_menu_path),
            "ledger": str(inputs.ledger_path),
            "cleaning_script": str(inputs.cleaning_script_path),
            "raw_panel": str(inputs.raw_panel_path),
            "cleaned_panel": str(inputs.cleaned_panel_path),
            "cleaning_run_id": inputs.cleaning_run_id,
            "missing": missing_artifacts(inputs),
            # Figürler adla değil İÇERİKLE anahtarlanır: panel yeniden çizildiğinde
            # (ör. eksen seçimi değişti) dosya adı aynı kalır ama gövde değişir.
            "figures": {
                name: hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]
                for name, html in sorted(inputs.figures.items())
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _requirements_path() -> Path:
    return Path(__file__).resolve().parents[2] / "requirements.txt"


def _read_text(path: Path) -> str:
    """Metin artefaktını okur; okunamayan artefakt paketin kendi hatasıdır.

    Arayüz yalnız `ReproPackageError` yakalar: bozuk bir artefaktın ham traceback
    olarak sızması "sessiz hiçlik"in tersi kadar kötüdür, kullanıcı ne olduğunu
    anlamaz. Bu yüzden her okuma noktası tek bir hata tipine çevrilir.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReproPackageError(f"Artefakt okunamadı: {path} ({exc})") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ReproPackageError(f"Artefakt geçerli JSON değil: {path} ({exc})") from exc


def _parse_models(payload: Any, model: type, path: Path) -> list[Any]:
    """JSON listesini tipli kayıtlara çevirir; şema kayması sessizce geçmez."""
    if not isinstance(payload, list):
        raise ReproPackageError(f"Artefakt liste içermiyor: {path}")
    try:
        return [model(**item) for item in payload]
    except (TypeError, ValueError) as exc:  # pydantic ValidationError ⊂ ValueError
        raise ReproPackageError(
            f"Artefakt {model.__name__} şemasına uymuyor: {path} ({exc})"
        ) from exc


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Diske yazılmış paneli okur. `panel.pkl`'i bu repo yazar, aynı güven sınırı."""
    try:
        frame = pickle.loads(path.read_bytes())  # noqa: S301
    except Exception as exc:  # noqa: BLE001  # pickle her tipte hata fırlatabilir
        raise ReproPackageError(f"Panel dosyası okunamadı: {path} ({exc})") from exc
    if not isinstance(frame, pd.DataFrame):
        raise ReproPackageError(f"Panel dosyası DataFrame içermiyor: {path}")
    return frame


def _dtype_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


def _frame_fingerprint(frame: pd.DataFrame) -> str | None:
    """Panelin içerik parmak izi: kolon adları + satır hash'leri.

    Manifest'e kayıt olarak yazılır. `hash_pandas_object` her dtype'ı (interval,
    egzotik extension array) hash'leyemez; hash'lenemeyen panel provenansı
    çökertmez, yalnız parmak izi kaydı boş kalır — karar `_frames_agree()`de
    zaten ayrıca veriliyor.
    """
    try:
        hasher = hashlib.sha256()
        hasher.update("\x1f".join(str(col) for col in frame.columns).encode("utf-8"))
        hasher.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    except TypeError:
        return None
    return hasher.hexdigest()[:16]


def _frames_agree(cleaned: pd.DataFrame, panel: pd.DataFrame) -> bool:
    """Temizleme çıktısı ile analiz paneli aynı veri mi.

    Karşılaştırma L4 kapısının TOLERANSIYLA yapılır, bit eşitliğiyle değil:
    `verify_reproduction` sandbox çıktısını `REPRO_RTOL/REPRO_ATOL` eşiğinde kabul
    eder, dolayısıyla float'a dokunan bir transform eklendiğinde kapıdan geçmiş
    doğru bir koşu son bit'te ayrışabilir. Burada bit eşitliği arasaydık o koşuya
    "denetim izin sahte" derdik; her pakette çıkan uyarı uyarı olmaktan çıkar.

    Şekil/kolon farkı ise tolerans meselesi değil, başka bir verinin işaretidir.
    """
    if list(cleaned.columns) != list(panel.columns) or len(cleaned) != len(panel):
        return False
    try:
        pd.testing.assert_frame_equal(
            cleaned.reset_index(drop=True),
            panel.reset_index(drop=True),
            check_dtype=False,
            rtol=REPRO_RTOL,
            atol=REPRO_ATOL,
        )
    except AssertionError:
        return False
    return True


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
    for number, line in enumerate(_read_text(ledger_path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReproPackageError(
                f"Karar defteri {number}. satırda bozuk: {ledger_path} ({exc})"
            ) from exc
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


def _provenance(inputs: ReproInputs, panel: pd.DataFrame | None) -> dict[str, Any]:
    """Temizleme koşusu ile multiverse koşusu gerçekten aynı veriyi mi taşıyor.

    İki koşu ayrı run_id'ler taşır ve arayüzde ayrı oturum anahtarlarından gelir:
    A verisini temizleyip koşan, sonra B verisini temizleyen bir kullanıcının
    paketi B'nin karar defterini A'nın sonuçlarıyla birleştirir. Eksik artefakt
    yoktur, dolayısıyla "eksiksiz" görünür. Bu yüzden temizlemenin çıktısı
    (`reproduced.pkl`) multiverse'in girdisi olan panelle parmak izinden
    karşılaştırılır; uyuşmazlık manifest'e ve arayüze yazılır.
    """
    record: dict[str, Any] = {
        "results_run_id": inputs.run_id,
        "cleaning_run_id": inputs.cleaning_run_id,
        "panel_fingerprint": _frame_fingerprint(panel) if panel is not None else None,
        "cleaned_panel_fingerprint": None,
        # None = denetlenemedi (temizleme sandbox'ı ya da panel yok); False = uyuşmuyor.
        "cleaning_matches_panel": None,
    }
    cleaned_path = _present(inputs.cleaned_panel_path)
    if cleaned_path is None or panel is None:
        return record
    cleaned = _read_dataframe(cleaned_path)
    record["cleaned_panel_fingerprint"] = _frame_fingerprint(cleaned)
    record["cleaning_matches_panel"] = _frames_agree(cleaned, panel)
    record["comparison"] = {"rtol": REPRO_RTOL, "atol": REPRO_ATOL}
    return record


def _build_manifest(
    inputs: ReproInputs,
    *,
    results: list[EstimationResult],
    specs: list[Specification],
    frozen: dict[str, Any] | None,
    decisions: list[dict[str, Any]],
    provenance: dict[str, Any],
    contents: list[str],
) -> dict[str, Any]:
    """Paketin tek makine-okunur kaydı. METHODS.md bunun insan render'ıdır."""
    return {
        "package_format": PACKAGE_FORMAT,
        "pareto_version": _pareto_version(),
        "run_id": inputs.run_id,
        "provenance": provenance,
        "estimand_hash": (frozen or {}).get("estimand_hash"),
        "menu_hash": (frozen or {}).get("menu_hash"),
        "estimand": (frozen or {}).get("estimand"),
        # Spec menüsünü üreten modelin kimliği. Dondurma kaydından okunur, canlı
        # olarak yeniden çözülmez: paket aylar sonra da üretilebiliyor ve o anda
        # etkin olan model menüyü üreten model değildir. Kayıt yoksa None kalır.
        "judge_model": (frozen or {}).get("judge_model"),
        "spec_count": len(specs),
        "spec_hashes": {spec.spec_id: spec.content_hash() for spec in specs},
        # Doğrulama script'i eşikleri BURADAN okur. Script'te ikinci bir kopya
        # tutulsaydı paket kendi içinde çelişebilirdi: manifest "eşleşmiyor" derken
        # script aynı farkı yutup "doğrulandı" basardı.
        "tolerances": {
            "cleaning": {"rtol": REPRO_RTOL, "atol": REPRO_ATOL},
            "estimates": {"rtol": ESTIMATE_RTOL, "atol": ESTIMATE_ATOL},
        },
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
    results = _parse_models(_read_json(results_path), EstimationResult, results_path)

    specs_path = _present(inputs.specs_path)
    specs: list[Specification] = (
        _parse_models(_read_json(specs_path), Specification, specs_path)
        if specs_path is not None
        else []
    )

    frozen_path = _present(inputs.frozen_menu_path)
    frozen = _read_json(frozen_path) if frozen_path is not None else None
    frozen_text = _read_text(frozen_path) if frozen_path is not None else None

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

    if frozen_text is not None:
        # Dosya diskteki hâliyle kopyalanır, manifest'ten yeniden serileştirilmez:
        # `estimand_hash` ve `menu_hash` bu dosyanın içeriği üzerinden hesaplanır,
        # dolayısıyla okuyucunun hash'leri yeniden hesaplayabilmesi için pakete
        # BYTE olarak girmesi gerekir. Yalnız manifest'e kopyalansaydı iki hash
        # doğrulanamayan birer iddia olurdu.
        files["frozen_menu.json"] = frozen_text

    dtypes: dict[str, dict[str, str]] = {}
    panel: pd.DataFrame | None = None
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
        files["cleaning/cleaning_steps.py"] = _read_text(cleaning_script_path)
    if ledger_path is not None:
        files["cleaning/decision_ledger.jsonl"] = _read_text(ledger_path)

    requirements_path = _present(_requirements_path())
    if requirements_path is not None:
        files["requirements.txt"] = _read_text(requirements_path)

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
        provenance=_provenance(inputs, panel),
        contents=[*files, *_GENERATED_FILES],
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
    provenance = manifest.get("provenance") or {}
    matches = provenance.get("cleaning_matches_panel")
    if matches is True:
        provenance_block = (
            "Temizleme koşusunun çıktısı analize giren panelle birebir eşleşiyor "
            f"(temizleme run_id: `{provenance.get('cleaning_run_id')}`, "
            f"panel parmak izi: `{provenance.get('panel_fingerprint')}`)."
        )
    elif matches is False:
        provenance_block = (
            "**UYARI: temizleme koşusunun çıktısı analize giren panelle EŞLEŞMİYOR** "
            f"(temizleme: `{provenance.get('cleaned_panel_fingerprint')}`, "
            f"panel: `{provenance.get('panel_fingerprint')}`). Bu paketteki karar "
            "defteri ve temizleme script'i, rapor edilen sonuçları üreten veriye ait "
            "olmayabilir; denetim izi olarak kullanmayın."
        )
    else:
        provenance_block = (
            "Temizleme koşusu ile analiz paneli arasındaki bağ DENETLENEMEDİ "
            "(temizleme sandbox çıktısı ya da panel pakete girmedi). Karar defterinin "
            "bu sonuçlara ait olduğu doğrulanmış değildir."
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

Script önce paketlenmiş spesifikasyon kümesini MANIFEST'teki içerik hash'leriyle
karşılaştırır (küme sonradan kırpılamaz), sonra paneli ve spesifikasyonları yeniden
koşar; ürettiği katsayı ve standart hataları `results.json` ile MANIFEST'te
bildirilen tolerans içinde, gözlem sayılarını ise TAM eşitlikle karşılaştırır
(n_obs bir ölçüm değil sayımdır). Ham veri ve temizleme script'i pakete girmişse önce
temizleme adımını da tekrarlar. Uyuşmazlıkta script sıfırdan farklı bir çıkış
koduyla biter.

> Bu paket çalıştırılabilir Python kodu içerir (`run_reproduction.py` ve
> `cleaning/cleaning_steps.py`); doğrulama komutu temizleme script'ini içe aktarıp
> çalıştırır. Paketi yalnız kaynağına güveniyorsanız koşun.

## Provenans

{provenance_block}

## İçerik

| Yol | Ne |
|-----|-----|
| `MANIFEST.json` | Makine-okunur tek kayıt: hash'ler, determinizm pinleri, özet |
| `METHODS.md` | Manifest'in metot bölümü taslağı olarak render'ı (LLM kullanılmaz) |
| `run_reproduction.py` | Tek komutluk doğrulama script'i |
| `frozen_menu.json` | Dondurulmuş estimand ve menü (manifest'teki hash'lerin kaynağı) |
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

Doğrulama sonuçlarla BAŞLAMAZ, spesifikasyon KÜMESİYLE başlar: MANIFEST'teki
içerik hash'leri paketlenmiş `specs.json` ile karşılaştırılır ve sonuç kümesi iki
yönde birden denetlenir. Aksi hâlde menüden birkaç spesifikasyon silmek doğrulamayı
küçültürdü ve script yine "tamam" derdi — ürünün "rapor edilen küme sonradan
kırpılamaz" iddiası tam da burada sınanır.

Kullanım: python run_reproduction.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import NoReturn

import pandas as pd

HERE = Path(__file__).resolve().parent


def _fail(message: str) -> NoReturn:
    print(f"BAŞARISIZ: {message}")
    raise SystemExit(1)


def _tolerance(manifest: dict, section: str) -> tuple[float, float]:
    """Karşılaştırma eşiğini MANIFEST'ten okur.

    Eşikler script'e GÖMÜLMEZ: paketi kuran katman temizleme çıktısını kendi
    eşiğiyle denetleyip sonucu manifest'e yazıyor. Script ikinci bir kopya
    taşısaydı aradaki farka düşen bir koşuda paket kendi içinde çelişirdi:
    manifest "EŞLEŞMİYOR" derken script aynı farkı yutup "doğrulandı" basardı.
    """
    declared = (manifest.get("tolerances") or {}).get(section)
    if not isinstance(declared, dict):
        _fail(
            f"MANIFEST.json `{section}` karşılaştırma eşiğini taşımıyor; "
            "doğrulama eşiksiz koşulamaz."
        )
    try:
        return float(declared["rtol"]), float(declared["atol"])
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"MANIFEST.json `{section}` eşiği okunamadı ({exc}).")


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


def _verify_cleaning(panel: pd.DataFrame, manifest: dict) -> None:
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

    rtol, atol = _tolerance(manifest, "cleaning")
    reproduced = module.clean(_read_table("raw")).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            reproduced, panel, check_dtype=False, rtol=rtol, atol=atol
        )
    except AssertionError as exc:
        _fail(f"temizleme çıktısı paketlenmiş panel ile eşleşmiyor.\\n{exc}")
    print(f"TAMAM: temizleme adımı yeniden üretildi ({len(panel)} satır).")


def _verify_spec_set(specs, expected_ids: set, manifest: dict) -> None:
    """Paketlenmiş küme MANIFEST'te dondurulan kümeyle aynı mı.

    Yalnız "koşulan her spec'in referansı var mı" diye bakmak yetmez: o denetim
    `specs.json`'dan spec silmeyi görmez, küçülen küme kendi içinde tutarlıdır.
    Kırpmayı yakalayan şey manifest'teki içerik hash'leriyle karşılaştırmadır.
    """
    declared = manifest.get("spec_hashes")
    if not isinstance(declared, dict) or not declared:
        _fail("MANIFEST.json spesifikasyon hash'i taşımıyor; küme doğrulanamaz.")

    packaged = {spec.spec_id: spec.content_hash() for spec in specs}
    if packaged != declared:
        removed = sorted(set(declared) - set(packaged))
        added = sorted(set(packaged) - set(declared))
        changed = sorted(
            spec_id
            for spec_id in set(packaged) & set(declared)
            if packaged[spec_id] != declared[spec_id]
        )
        _fail(
            "paketlenmiş spesifikasyon kümesi MANIFEST ile eşleşmiyor "
            f"(eksilen: {removed or 'yok'}, eklenen: {added or 'yok'}, "
            f"içeriği değişen: {changed or 'yok'})."
        )

    if expected_ids != set(declared):
        _fail(
            "`results.json` MANIFEST'teki spesifikasyon kümesini taşımıyor "
            f"(eksilen: {sorted(set(declared) - expected_ids) or 'yok'}, "
            f"fazladan: {sorted(expected_ids - set(declared)) or 'yok'})."
        )

    declared_count = manifest.get("spec_count")
    if isinstance(declared_count, int) and declared_count != len(packaged):
        _fail(f"MANIFEST {declared_count} spesifikasyon bildiriyor, pakette {len(packaged)} var.")


def _verify_estimates(manifest: dict) -> None:
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

    _verify_spec_set(specs, set(expected), manifest)
    _verify_cleaning(panel, manifest)

    rtol, atol = _tolerance(manifest, "estimates")
    mismatches: list[str] = []
    reproduced_ids: set = set()
    for result in run_specs(panel, specs):
        reproduced_ids.add(result.spec_id)
        reference = expected.get(result.spec_id)
        if reference is None:
            mismatches.append(f"{result.spec_id}: pakette referans sonuç yok")
            continue
        if result.status != reference.get("status"):
            mismatches.append(
                f"{result.spec_id}: status {result.status} != {reference.get('status')}"
            )
            continue
        # Gözlem sayısı TAM eşitlikle karşılaştırılır, toleransla değil: n_obs bir
        # ölçüm değil, sayımdır. Göreli eşikle karşılaştırılsaydı n=10.000'de bir
        # satırlık fark (tolerans tam 1.0) sessizce geçerdi — örneklemi değişmiş bir
        # koşuya "yeniden üretildi" demek, doğrulamanın kendisini boşa çıkarır.
        if result.n_obs != reference.get("n_obs"):
            mismatches.append(f"{result.spec_id}.n_obs: {result.n_obs} != {reference.get('n_obs')}")

        for field_name in ("coefficient", "std_error"):
            got = getattr(result, field_name)
            want = reference.get(field_name)
            if got is None or want is None:
                if got is not want:
                    mismatches.append(f"{result.spec_id}.{field_name}: {got} != {want}")
                continue
            if abs(float(got) - float(want)) > atol + rtol * abs(float(want)):
                mismatches.append(f"{result.spec_id}.{field_name}: {got} != {want}")

    for spec_id in sorted(set(expected) - reproduced_ids):
        mismatches.append(f"{spec_id}: pakette referans sonuç var ama yeniden üretilmedi")

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

    provenance = manifest.get("provenance") or {}
    if provenance.get("cleaning_matches_panel") is False:
        print(
            "UYARI: temizleme koşusunun çıktısı analize giren panelle eşleşmiyor; "
            "paketteki karar defteri bu sonuçlara ait olmayabilir."
        )
    elif provenance.get("cleaning_matches_panel") is None:
        print("UYARI: temizleme koşusu ile panel arasındaki bağ denetlenmedi.")

    if not (HERE / "data" / "panel.csv").exists() or not (HERE / "specs.json").exists():
        _fail("panel ya da spesifikasyon listesi pakette yok; doğrulama koşulamaz.")
    _verify_estimates(manifest)
    print("Reprodüksiyon doğrulandı.")
    sys.exit(0)


if __name__ == "__main__":
    main()
'''
