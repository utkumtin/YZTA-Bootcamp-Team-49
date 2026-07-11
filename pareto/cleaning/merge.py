"""Panel-merge — çok kaynağı unit×time panele birleştir.

Hero yakıtı (Medicaid): CDC mortalite (ilçe-yıl) + KFF eyalet genişleme tarihleri +
Census/BLS kontrolleri → FIPS-vs-isim uyumsuzluğu, tarih formatları, eksik kırsal
ilçeler. Çıktı `CleanPanel` (long df + manifest). Merge DETERMİNİSTİK ve
config.yaml-güdümlü; rol/anahtar eşlemesi dataset config'inden gelir, çekirdek
hiçbir dataset adı/kolonu hardcode etmez (veri-agnostik çekirdek).

Tasarım kararları:
- Tüm kaynaklar `dtype=str` okunur → FIPS leading-zero evrensel korunur. Sayısal
  kolonlar SONRA, yalnız panel bloğunda deklare edilen roller (outcome/covariate/
  weight) için fail-loud coerce edilir.
- NA-marker kuralı (Suppressed/Unreliable/… → NA) burada uygulanır; suppressed
  hücre yeniden kurulmaz (CDC WONDER DUA-uyumu).
- Spine = config'teki İLK kaynak (unit×time taşımak zorunda). Kalan kaynaklar
  anahtarına göre eklenir: unit×time varsa onunla, yoksa `merge.on` (eyalet-seviye
  broadcast). Sağ taraf anahtar başına tekil olmalı (m:1), değilse fail-loud.
  Rol çakışmasında spine önceliklidir (sağdaki kopya atılır + loglanır).
- `treatment:` bloğu varsa deterministik `treatment_cohort` + `never_treated`
  kolonları türetilir → CleanPanel Tier1 (DiD) olur. Committed-baseline göstergesi
  (treated_post) estimand/menü aşamasının işi, burada türetilmez.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..contracts import CleanPanel, PanelManifest

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- config & I/O


def load_dataset_config(dataset_dir: str | Path) -> dict[str, Any]:
    """`data/<dataset>/config.yaml` descriptor'ını yükler (dizin ya da dosya yolu verilebilir)."""
    path = Path(dataset_dir)
    if path.is_dir():
        path = path / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config bulunamadı: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "sources" not in config or "panel" not in config:
        raise ValueError(f"Geçersiz dataset config ({path}): 'sources' + 'panel' blokları zorunlu.")
    # YAML 1.1 footgun: çıplak `on:` anahtarı boolean True'ya parse edilir (Norway problemi).
    merge_cfg = config.get("merge")
    if isinstance(merge_cfg, dict) and True in merge_cfg:
        merge_cfg["on"] = merge_cfg.pop(True)
    return config


def _rows_until_marker(path: Path, marker: str) -> int | None:
    """Footer-marker'lı dosyada (CDC WONDER `"---"`) veri satırı sayısı; marker yoksa None."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip().strip('"') == marker:
                return max(i - 1, 0)  # başlık satırı veri sayılmaz
    return None


def _read_source_file(path: Path, fmt: str, read: dict[str, Any]) -> pd.DataFrame:
    """Tek dosyayı `read:` spesifikasyonuyla okur; her kolon str (leading-zero korunur)."""
    if fmt == "xlsx":
        return pd.read_excel(
            path,
            header=int(read.get("header_row", 0)),
            skipfooter=int(read.get("skipfooter", 0)),
            dtype=str,
        )
    sep = read.get("sep", "\t" if fmt == "tsv" else ",")
    nrows: int | None = None
    marker = read.get("skipfooter_marker")
    if marker:
        nrows = _rows_until_marker(path, str(marker))
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        nrows=nrows,
        keep_default_na=bool(read.get("keep_default_na", True)),
    )


def load_sources(config: dict[str, Any], base_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Config'teki her kaynağı ham DataFrame'e okur.

    `file:` glob olabilir (örn. yıl-başına LAUS dosyaları) → sıralı okunup alt alta
    eklenir. Dosya yoksa fail-loud.
    """
    base = Path(base_dir)
    out: dict[str, pd.DataFrame] = {}
    for name, src in config["sources"].items():
        pattern = str(src["file"])
        is_glob = any(ch in pattern for ch in "*?[")
        paths = sorted(base.glob(pattern)) if is_glob else [base / pattern]
        if not paths or any(not p.exists() for p in paths):
            raise FileNotFoundError(f"'{name}' kaynağı için dosya yok: {base / pattern}")
        fmt = str(src.get("format") or Path(pattern).suffix.lstrip("."))
        read = src.get("read") or {}
        frames = [_read_source_file(p, fmt, read) for p in paths]
        out[name] = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return out


# --------------------------------------------------------------------------- merge çekirdeği


def _select_roles(name: str, df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Kaynak kolonlarını rol adlarına çevirir; deklare edilen kolon yoksa fail-loud."""
    mapping = {src: role for role, src in columns.items()}
    missing = [src for src in mapping if src not in df.columns]
    if missing:
        raise ValueError(f"'{name}' kaynağında config'in deklare ettiği kolon(lar) yok: {missing}")
    return df[list(mapping)].rename(columns=mapping)


def _apply_derives(df: pd.DataFrame, derive: dict[str, Any]) -> pd.DataFrame:
    """`merge.derive` kurallarını uygular (hedef yoksa + girdiler varsa).

    `concat` girdileri türetimden sonra atılır: iki-parça FIPS anahtarı hedefte
    aynen içerildiğinden panelde kopya taşımanın anlamı yok.
    """
    for target, rule in derive.items():
        if target in df.columns:
            continue
        if "take_first" in rule:
            src = rule["from"]
            if src in df.columns:
                df[target] = df[src].str[: int(rule["take_first"])]
        elif "concat" in rule:
            parts = list(rule["concat"])
            if all(p in df.columns for p in parts):
                first, *rest = parts
                df[target] = df[first].str.cat([df[p] for p in rest])
                df = df.drop(columns=parts)
        else:
            raise ValueError(f"Bilinmeyen derive kuralı: {target} → {rule}")
    return df


def _prepare(
    name: str,
    df: pd.DataFrame,
    src_cfg: dict[str, Any],
    time: str,
    derive: dict[str, Any],
) -> pd.DataFrame:
    """Tek kaynağı merge'e hazırlar: rol-rename → NA-marker → derive → time sayısallaştır."""
    df = _select_roles(name, df, src_cfg["columns"])
    na_markers = (src_cfg.get("read") or {}).get("na_markers")
    if na_markers:
        df = df.replace(list(na_markers), pd.NA)  # suppressed → NA; yeniden kurulmaz (DUA)
    df = _apply_derives(df, derive)
    if time in df.columns:
        df[time] = pd.to_numeric(df[time], errors="raise")
    return df


def _join_keys(cols: Any, unit: str, time: str, on: str | None) -> list[str] | None:
    if unit in cols and time in cols:
        return [unit, time]
    if on and on in cols and time in cols:
        return [on, time]
    if on and on in cols:
        return [on]
    return None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _to_year(s: pd.Series) -> pd.Series:
    """Kohort kolonu → yıl (Int64). Değerler zaten yılsa aynen, değilse tarih parse edilir.

    Sayısal-yıl yolu ÖNCE denenir: `pd.to_datetime(2014)` epoch-nanosaniye sayar
    (sessiz 1970) — o tuzak buradan kapalı.
    """
    as_num = pd.to_numeric(s, errors="coerce")
    valid = as_num.dropna()
    all_numeric = len(valid) == int(s.notna().sum())
    if not valid.empty and all_numeric and valid.between(1800, 2200).all():
        return as_num.astype("Int64")
    return pd.to_datetime(s, errors="raise").dt.year.astype("Int64")


def merge_to_panel(sources: dict[str, pd.DataFrame], config: dict[str, Any]) -> CleanPanel:
    """Birden çok kaynağı config'teki rol/anahtar eşlemesine göre panele birleştirir."""
    src_cfg: dict[str, Any] = config["sources"]
    if set(sources) != set(src_cfg):
        diff = sorted(set(sources) ^ set(src_cfg))
        raise ValueError(f"sources ile config.sources uyuşmuyor: {diff}")

    panel_cfg = config["panel"]
    unit, time = panel_cfg["unit"], panel_cfg["time"]
    merge_cfg = config.get("merge") or {}
    derive = merge_cfg.get("derive") or {}
    on = merge_cfg.get("on")
    how = merge_cfg.get("how", "left")

    panel: pd.DataFrame | None = None
    spine_name = ""
    for name, cfg in src_cfg.items():  # yaml sırası deterministik; ilk kaynak = spine
        df = _prepare(name, sources[name], cfg, time, derive)
        if panel is None:
            spine_keys = [unit, time]
            if any(k not in df.columns for k in spine_keys):
                raise ValueError(f"Spine kaynağı '{name}' unit ({unit}) + time ({time}) taşımalı.")
            df = df.dropna(subset=spine_keys)
            df[time] = df[time].astype(int)
            n_dup = int(df.duplicated(spine_keys).sum())
            if n_dup:
                raise ValueError(f"Spine '{name}' {unit}×{time} başına tekil değil: {n_dup} kopya.")
            panel, spine_name = df, name
            continue

        keys = _join_keys(df.columns, unit, time, on)
        if keys is None:
            raise ValueError(
                f"'{name}' kaynağını panele bağlayacak anahtar yok (unit+time ya da '{on}')."
            )
        df = df.dropna(subset=keys)
        if time in keys:
            df[time] = df[time].astype(int)
        clash = [c for c in df.columns if c in panel.columns and c not in keys]
        if clash:
            logger.warning(
                "panel-merge: '%s' çakışan kolonları atıldı (spine '%s' önceliği): %s",
                name,
                spine_name,
                clash,
            )
            df = df.drop(columns=clash)
        try:
            panel = panel.merge(df, on=keys, how=how, validate="m:1")
        except pd.errors.MergeError as exc:
            raise ValueError(f"'{name}' kaynağı {keys} başına tekil değil (m:1 ihlali).") from exc

    assert panel is not None  # src_cfg boş olamaz (load_dataset_config valide etti)

    # Deklare edilen ölçüm rolleri sayısala coerce edilir (marker'lar zaten NA) — fail-loud.
    outcomes = _as_tuple(panel_cfg.get("outcome"))
    covariates = _as_tuple(panel_cfg.get("covariates"))
    weight = panel_cfg.get("weight")
    for col in (*outcomes, *covariates, *((weight,) if weight else ())):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="raise")

    # Tedavi türetmesi (config-güdümlü, deterministik) → Tier1.
    treatment_cohort_col: str | None = None
    never_treated_col: str | None = None
    treat_cfg = config.get("treatment") or {}
    cohort_from = treat_cfg.get("cohort_from")
    if cohort_from:
        if cohort_from not in panel.columns:
            raise ValueError(f"treatment.cohort_from kolonu panelde yok: {cohort_from}")
        panel["treatment_cohort"] = _to_year(panel[cohort_from])
        treatment_cohort_col = "treatment_cohort"
    never_when = treat_cfg.get("never_treated_when")
    if never_when:
        missing = [c for c in never_when if c not in panel.columns]
        if missing:
            raise ValueError(f"treatment.never_treated_when kolon(lar)ı panelde yok: {missing}")
        mask = pd.Series(True, index=panel.index)
        for col, val in never_when.items():
            mask &= panel[col] == val
        panel["never_treated"] = mask
        never_treated_col = "never_treated"

    manifest = PanelManifest(
        unit_col=unit,
        time_col=time,
        treatment_cohort_col=treatment_cohort_col,
        never_treated_col=never_treated_col,
        outcome_cols=outcomes,
        covariate_cols=covariates,
        weight_col=weight,
        provenance={name: str(cfg.get("file", "")) for name, cfg in src_cfg.items()},
    )
    clean = CleanPanel(df=panel.reset_index(drop=True), manifest=manifest)
    clean.validate_contract()
    return clean


def build_panel(dataset_dir: str | Path) -> CleanPanel:
    """Tek çağrılık kolaylık: config yükle → kaynakları oku → panele birleştir."""
    path = Path(dataset_dir)
    base = path.parent if path.is_file() else path  # config.yaml yolu da kabul edilir
    config = load_dataset_config(path)
    sources = load_sources(config, base)
    return merge_to_panel(sources, config)
