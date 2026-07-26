"""Çekirdeğin veri-agnostikliği: `pareto/` hiçbir dataset adını/kolonunu bilmez.

SCOPE §11'in taşıyıcı iddiası, jürinin kendi verisini yükleyebilmesi. Bu ancak
dataset bilgisi `data/<set>/config.yaml` descriptor'larında kalırsa doğru olur;
çekirdeğe sızan tek bir kolon adı bile iddiayı sessizce yalanlar. Smoke matrisi
bunu davranışsal olarak gösterir, bu test statik olarak zorlar.

Tarama yalnız KODU görür: docstring ve yorumlar kapsam dışıdır, çünkü merge ve
event-study modülleri hero veri setinden meşru olarak söz eder; yasak olan, kodun
o veriye göre dallanmasıdır.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CORE_DIR = REPO_ROOT / "pareto"

# Panelin jenerik sözlüğü: bu kelimeler bir dataset'in kolon adı olarak da geçiyor
# ama çekirdekte tamamen başka bir anlamla duruyorlar. Her biri tek tek gerekçeli;
# listeye yeni bir şey eklemek, çekirdeğe bir dataset kavramı sızdırmakla eş anlamlıdır.
GENERIC_VOCABULARY = {
    "chain": "LLM failover zinciri (router/providers), Card-Krueger'ın restoran zinciri değil",
    "population": "estimand'in hedef popülasyon alanı + ayarlardaki tek varsayılan ağırlık kolonu",
    "state": "yerel değişken adı (analysis_state), eyalet kolonu değil",
    "status": "EstimationResult'ın koşu durumu alanı",
    "treated": "never-treated ekseninin sözlüğü (include_never_treated, _pareto_never_treated)",
    "weight": "panel manifest'indeki jenerik ağırlık rolü",
    "year": "zaman kolonu sezgisi ve kohort yıl dönüşümü (_to_year)",
}


def _dataset_tokens() -> set[str]:
    """Dataset descriptor'larından yasak token kümesi.

    Dataset adı, dizin adı, kaynak dosya adı, rol adları ve ham kolon adları:
    çekirdeğin bilmemesi gereken her şey config'te zaten yazılı, ayrıca elle
    listelenmez — liste config'lerle birlikte kendiliğinden büyür.
    """
    tokens: set[str] = set()
    for config_path in sorted(DATA_DIR.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tokens.add(config_path.parent.name)
        tokens.add(str(config.get("dataset") or config_path.parent.name))
        for source in config["sources"].values():
            tokens.add(Path(str(source["file"])).stem)
            for role, column in (source.get("columns") or {}).items():
                tokens.update({role, str(column)})
        panel = config["panel"]
        for value in (panel.get("outcome") or []) + (panel.get("covariates") or []):
            tokens.add(str(value))
        for key in ("unit", "time", "weight"):
            if panel.get(key):
                tokens.add(str(panel[key]))
    lowered = {token.lower() for token in tokens if token}
    return lowered - {"none", "null"} - set(GENERIC_VOCABULARY)


def _code_tokens(path: Path) -> set[str]:
    """Modülün kod token'ları: identifier'lar + docstring olmayan string literal'ler."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                docstrings.add(id(first.value))

    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                tokens.update(w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        elif isinstance(node, ast.Name):
            tokens.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            tokens.add(node.arg.lower())
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            tokens.add(node.name.lower())
        elif isinstance(node, ast.keyword) and node.arg:
            tokens.add(node.arg.lower())
    return tokens


def test_core_never_names_a_dataset_or_its_columns() -> None:
    """Çekirdek bir dataset adını/kolonunu anarsa jüri kendi verisiyle duvara toslar."""
    forbidden = _dataset_tokens()
    assert forbidden, "Dataset descriptor'larından token türetilemedi; tarama boşa koşuyor."

    leaks: dict[str, list[str]] = {}
    for path in sorted(CORE_DIR.rglob("*.py")):
        leaked = _code_tokens(path) & forbidden
        if leaked:
            leaks[str(path.relative_to(REPO_ROOT))] = sorted(leaked)
    assert not leaks, f"Çekirdeğe dataset bilgisi sızmış: {leaks}"


def test_generic_vocabulary_only_shelters_words_a_dataset_really_uses() -> None:
    """Gerekçesiz büyüyen allowlist testi boşaltır; her madde bir kolona karşılık gelmeli."""
    tokens: set[str] = set()
    for config_path in sorted(DATA_DIR.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for source in config["sources"].values():
            for role, column in (source.get("columns") or {}).items():
                tokens.update({role.lower(), str(column).lower()})

    stale = sorted(set(GENERIC_VOCABULARY) - tokens)
    assert not stale, f"Allowlist'te artık hiçbir dataset'in kullanmadığı token var: {stale}"
