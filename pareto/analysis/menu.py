"""Spec Menüsü + Dondurma.

İki parça:
  1. `FrozenSpecMenu` (immutable + hash) — reprodüksiyon garantisinin KAYNAĞI.
     Onayda dondurulur; runtime bir daha LLM'e sormaz → tekrarlanabilirlik modelin
     kararlılığından değil, dondurmadan gelir.
  2. `expand_to_specs()` — aktif eksenlerin faktöriyel çarpımı → Specification listesi.
     Prototipteki `multiverse.py` kartezyen-çarpım mantığından migre; yeni eksenler
     (örneklem, never-treated) eklendi, sert tavan 500→24 (review sorun: config.py).

LLM menü ÜRETİMİ (context → savunulabilir seviyeler + baseline + gerekçe) Sprint-2;
aşağıda dikiş olarak işaretli. Aktif-eksen/aydınlık-yol etkileşimi de Sprint-2 UI.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging

from pydantic import BaseModel

from ..config import SETTINGS
from ..spec import SUPPORTED_ESTIMATORS, Estimator, Specification

logger = logging.getLogger(__name__)


class SpecMenu(BaseModel):
    """Savunulabilir seviyeler menüsü. Baseline = sessiz eksenlerin pinlendiği değer."""

    control_sets: list[list[str]]  # kontrol seti ekseni
    samples: list[str | None] = [None]  # örneklem ekseni (pandas query; None = tümü)
    pre_period_windows: list[int | None] = [None]
    clustering_levels: list[str]  # clustering ekseni
    include_never_treated: list[bool] = [True]  # never-treated dahil/hariç ekseni
    estimators: list[Estimator] = list(SUPPORTED_ESTIMATORS)
    active_axes: list[str] = []  # aydınlık yol: hangi eksenler faktöriyel açılır
    rationale: str = ""

    def freeze(self) -> FrozenSpecMenu:
        blob = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, default=list)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return FrozenSpecMenu(menu=self, menu_hash=digest)


class FrozenSpecMenu(BaseModel):
    """Onaylanmış, immutable menü + hash. Reprodüksiyon dondurmadan gelir."""

    model_config = {"frozen": True}
    menu: SpecMenu
    menu_hash: str


def expand_to_specs(
    frozen: FrozenSpecMenu,
    *,
    outcome: str,
    treatment: str,
    unit_col: str,
    time_col: str,
) -> list[Specification]:
    """Aktif eksenlerin faktöriyel çarpımını Specification listesine açar.

    Sessiz eksenler baseline'a (ilk seviye) pinli. Sert tavan 24:
    aşımda UYAR + LOGLA, sessiz kırpma YOK.
    """
    m = frozen.menu
    active = set(m.active_axes) if m.active_axes else _default_active(m)

    def levels(axis: str, all_levels: list):  # noqa: ANN001, ANN202
        return all_levels if axis in active else all_levels[:1]  # sessiz → baseline-pin

    combos = itertools.product(
        levels("control_set", m.control_sets),
        levels("sample", m.samples),
        levels("pre_period", m.pre_period_windows),
        levels("clustering", m.clustering_levels),
        levels("never_treated", m.include_never_treated),
        levels("estimator", m.estimators),
    )

    specs: list[Specification] = []
    for i, (controls, sample, window, cluster, never_t, est) in enumerate(combos):
        if est not in SUPPORTED_ESTIMATORS:
            raise ValueError(f"Desteklenmeyen estimator: {est} (staggered = fast-follow).")
        specs.append(
            Specification(
                spec_id=f"spec_{i:04d}",
                outcome=outcome,
                treatment=treatment,
                controls=tuple(controls),
                unit_fe=unit_col if est == "TWFE" else None,
                time_fe=time_col if est == "TWFE" else None,
                pre_period_window=window,
                cluster_by=cluster,
                estimator=est,
                sample_filter=sample,
                include_never_treated=never_t,
            )
        )

    cap = SETTINGS.max_specifications
    if len(specs) > cap:
        raise ValueError(
            f"{len(specs)} spesifikasyon üretildi, sert tavan {cap}. "
            "Aktif eksen sayısını azaltın veya seviye kırpın — sessiz kırpma yapılmaz."
        )
    logger.info("expand_to_specs: %d spec (aktif eksenler: %s)", len(specs), sorted(active))
    return specs


def _default_active(menu: SpecMenu) -> set[str]:
    """Birden çok seviyesi olan her ekseni default aktif say (acemi tek-tık)."""
    active = set()
    if len(menu.control_sets) > 1:
        active.add("control_set")
    if len(menu.samples) > 1:
        active.add("sample")
    if len(menu.pre_period_windows) > 1:
        active.add("pre_period")
    if len(menu.clustering_levels) > 1:
        active.add("clustering")
    if len(menu.include_never_treated) > 1:
        active.add("never_treated")
    if len(menu.estimators) > 1:
        active.add("estimator")
    return active


def generate_spec_menu(research_context, baseline_spec, available_columns):  # noqa: ANN001
    """LLM: context → savunulabilir seviyeler + baseline + gerekçe.

    SPRINT-2: yargı modeli (router JUDGE rolü) kapalı spec listesi değil, her eksende
    savunulabilir seviyeler önerir; kullanıcı 4 fiille (onayla/düzenle/çıkar/ekle)
    aydınlık yolu çizer; kullanıcı eklemeleri savunulabilirlik kapısından geçer.
    """
    raise NotImplementedError("LLM spec-menü üretimi Sprint-2 kapsamında (bkz docs/scrum).")
