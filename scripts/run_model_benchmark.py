"""Free model benchmark koşucusu — yalnız JUDGE görevleri.

`providers.py`'deki küratörlü model listeleri bugün yer tutucu (dosyadaki
`TODO(ekip)` notu). Bu script o listeyi kanıta bağlamak için var: adayları Pareto'nun
kendi dört JUDGE görevinde koşar, üretimdeki fail-loud doğrulayıcıları puanlayıcı
olarak kullanır ve üzerine literatüre dayalı altın seti uygular.

Ölçülen dört görev ve puanlayıcıları:

    temizleme  cleaning.generate_ledger        -> _validate_referenced_columns
    estimand   hypothesis.draft_tac_proposal   -> kolon listesi kısıtı + expected_sign
    spec_menu  menu.generate_spec_menu         -> menu.evaluate_menu_defensibility
    narrative  narrative.generate_narrative    -> _validate_axes + sayı uydurma

Üretim yolunu ölçer, taklit etmez: model enjeksiyonu `PARETO_JUDGE_PROVIDER` +
slotun `*_JUDGE_MODEL` değişkeni üzerinden yapılır, çağrılar gerçek
`generate_*` fonksiyonlarına gider. Böylece prompt, guardrail, şema zorlaması ve
retry davranışı kullanıcının göreceğiyle aynı olur.

Katalog sayfaları güvenilmez (Groq katalogunda hâlâ görünen Kimi K2 free tier'da
yok). Bu yüzden `--preflight` var: her model ID'ye küçük bir tipli çağrı atar,
cevap vermeyeni veya şema zorlayamayanı gerekçesiyle eler.

Koşu resume edilebilir: her sonuç anında `results.jsonl`'e eklenir, kotaya
çarpınca aynı komut kaldığı yerden devam eder.

Kullanım:
    python scripts/run_model_benchmark.py --dry-run
    PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --preflight
    PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --models gemini-3.6-flash

Bu script test süitine girmez (canlı sağlayıcı çağırır). Puanlayıcıları
`tests/test_model_benchmark.py` API yakmadan doğrular.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Doğrudan çalıştırma her çalışma dizininden kararlı kalsın: bu script paket
# giriş noktası olarak kurulmuyor (run_e2e_check.py ile aynı desen).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Doğrudan çalıştırma, Pareto importlarından önce repo kökünü sys.path'e ister.
from pydantic import TypeAdapter  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelResponse  # noqa: E402
from pydantic_ai.models import Model, ModelRequestParameters  # noqa: E402
from pydantic_ai.models.wrapper import WrapperModel  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402

import pareto.llm.cache as llm_cache  # noqa: E402
from pareto.analysis.hypothesis import (  # noqa: E402
    FrozenEstimand,
    SocraticDeclaration,
    TACProposal,
    draft_tac_proposal,
    freeze_estimand,
)
from pareto.analysis.menu import (  # noqa: E402
    ALL_AXES,
    SpecMenuProposal,
    evaluate_menu_defensibility,
    generate_spec_menu,
)
from pareto.cleaning.agent import TransformCall, generate_ledger  # noqa: E402
from pareto.cleaning.ledger import LedgerEntry  # noqa: E402
from pareto.cleaning.merge import build_panel, load_dataset_config  # noqa: E402
from pareto.config import PrivacyMode  # noqa: E402
from pareto.llm.narrative import VarianceNarrative, generate_narrative  # noqa: E402
from pareto.llm.providers import judge_slots_for  # noqa: E402
from pareto.profiling import profile_dataframe  # noqa: E402

BENCH_DIR = REPO_ROOT / "benchmarks"
GOLD_DIR = BENCH_DIR / "gold"
MODELS_PATH = BENCH_DIR / "models.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "benchmark"

TASKS = ("cleaning", "estimand", "spec_menu", "narrative")

_TRANSFORM_ADAPTER: TypeAdapter[Any] = TypeAdapter(TransformCall)


# --------------------------------------------------------------------------- #
# Ölçüm sarmalayıcısı
# --------------------------------------------------------------------------- #
@dataclass
class Meter:
    """Tek bir görev çağrısının ham sayaçları."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def retries(self) -> int:
        """Şema retry sayısı.

        `generate_*` fonksiyonları `AgentRunResult`'ı değil ayrıştırılmış çıktıyı
        döndürüyor, yani retry sayısı doğrudan okunamıyor. Ama pydantic-ai'nin
        şema retry'ı modele YENİ BİR İSTEK olarak gidiyor: ilk denemede şemayı
        tutturan model 1 istek yapar, her retry bir istek daha ekler.
        """
        return max(self.requests - 1, 0)


class MeteredModel(WrapperModel):
    """Sarmalanan modele giden her isteği verilen `Meter`'a işler."""

    def __init__(self, wrapped: Model | str, meter: Meter) -> None:
        # KnownModelName Literal'ı dışındaki "provider:model" stringleri de geçerli
        super().__init__(wrapped)  # type: ignore[arg-type]
        self._meter = meter

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._meter.requests += 1
        response = await super().request(messages, model_settings, model_request_parameters)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._meter.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self._meter.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        return response


@contextmanager
def metered(meter: Meter) -> Iterator[Meter]:
    """`wrap_with_cache` dikişine ölçüm sarmalayıcısını takar.

    Neden bu nokta: `router._resolve_model` modeli tam olarak burada sarmalıyor,
    yani zincir kurulumu / `extra_model_settings` / `thinking` / temperature
    hepsi üretimdeki gibi kalıyor — yalnız sarmalayıcı değişiyor. Cache zaten
    kapalı olmak zorunda (`require_cache_disabled`), o durumda `wrap_with_cache`
    modeli olduğu gibi döndürüyordu; yerine ölçen sarmalayıcı konuyor.

    `use_test_model` bu iş için YANLIŞ seam olurdu: o, zinciri tümden atlayıp
    `extra_model_settings`'i boşaltıyor (bkz. router._resolve_model), yani Gemini
    slotunun `thinking="medium"` ayarını sessizce düşürür ve ölçtüğümüz yol
    üretimdekinden farklı olurdu.
    """
    original = llm_cache.wrap_with_cache

    def _wrap(model: Model | str) -> Model | str:
        return MeteredModel(model, meter)

    llm_cache.wrap_with_cache = _wrap  # type: ignore[assignment]
    try:
        yield meter
    finally:
        llm_cache.wrap_with_cache = original  # type: ignore[assignment]


def require_cache_disabled() -> None:
    """Cache açıkken gecikme ve retry ölçülemez — fail-loud."""
    if llm_cache.cache_enabled():
        raise SystemExit(
            "PARETO_LLM_CACHE=0 gerekli. Cache açıkken ikinci koşu diskten döner; "
            "gecikme, retry ve token sayıları gerçeği göstermez.\n"
            "  PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py ..."
        )


# --------------------------------------------------------------------------- #
# Kota / throttle
# --------------------------------------------------------------------------- #
class QuotaExhausted(RuntimeError):
    """Modelin günlük isteği veya toplam kredisi bitti; koşu sonra devam eder."""


def _print_rpm_wait(pool: str, seconds: float) -> None:
    """RPM aralığını doldurmak için beklerken terminale basar.

    Sessiz kalırsa (ör. gemini-3.6-flash rpm=5 -> istekler arası 12sn) koşu
    donmuş gibi görünür; kullanıcı neyin beklendiğini görmeli.
    """
    print(f"  [rpm] {pool}: {seconds:.1f}sn bekleniyor (rpm sınırı)")


@dataclass
class Throttle:
    """RPM aralığı + istek/kredi tavanı, KOTA HAVUZU başına.

    Havuz kavramı şart: OpenRouter'da tüm `:free` modeller aynı 50 istek/gün
    havuzunu paylaşır, NVIDIA NIM'de 1.000 kredi hesabın tamamı içindir. Bunları
    model başına saymak kotayı olduğundan kat kat büyük gösterir ve koşu 429'a
    çarpar. Google ve Groq'ta limit gerçekten model başınadır; onlarda havuz
    anahtarı modelin kendisidir (`benchmarks/models.json: pool`).

    `now`/`sleep` enjekte edilebilir: testler sahte saatle kotayı tüketirken
    gerçekten beklemez.

    Tavan bir TAKVİM GÜNÜ değil, bir KOŞU için geçerlidir. Koşu kotaya çarpınca
    durur; ertesi gün aynı komut `results.jsonl`'den devam eder. Takvim gününü
    burada takip etmek, script gün ortasında başlatıldığında yanlış sonuç verirdi.
    """

    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    # Enjekte edilebilir: testler beklemeyi sessizce yakalar, üretim varsayılanı
    # terminale basar. `now`/`sleep` ile aynı gerekçe — gerçek I/O'yu testten ayır.
    on_wait: Callable[[str, float], None] = _print_rpm_wait
    _last_call: dict[str, float] = field(default_factory=dict)
    _count: dict[str, int] = field(default_factory=dict)

    def seed(self, pool: str, already_done: int) -> None:
        """Resume: önceki koşudan gelen çağrılar havuz kotasından düşülür."""
        self._count[pool] = self._count.get(pool, 0) + already_done

    def used(self, pool: str) -> int:
        return self._count.get(pool, 0)

    def acquire(self, pool: str, *, rpm: int | None, cap: int | None) -> None:
        used = self._count.get(pool, 0)
        if cap is not None and used >= cap:
            raise QuotaExhausted(f"{pool}: kota doldu ({used}/{cap})")
        if rpm:
            interval = 60.0 / float(rpm)
            last = self._last_call.get(pool)
            if last is not None:
                wait = interval - (self.now() - last)
                if wait > 0:
                    self.on_wait(pool, wait)
                    self.sleep(wait)
        self._last_call[pool] = self.now()
        self._count[pool] = used + 1


# --------------------------------------------------------------------------- #
# Aday matrisi + altın set
# --------------------------------------------------------------------------- #
def load_models(path: Path = MODELS_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["models"])


def load_gold(task: str, gold_dir: Path = GOLD_DIR) -> list[dict[str, Any]]:
    data = json.loads((gold_dir / f"{task}.json").read_text(encoding="utf-8"))
    return list(data["cases"])


def model_key(model: dict[str, Any]) -> str:
    return f"{model['provider']}/{model['id']}"


def quota_pool(model: dict[str, Any]) -> str:
    """Bu modelin kotayı paylaştığı havuz. Yoksa modelin kendisi tek başına havuzdur."""
    return str(model.get("pool") or model_key(model))


_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


def by_priority(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yüksek öncelikli modeller önce koşar (dosya sırası eşitlikte korunur).

    Paylaşımlı havuzda sıra sonucu belirler: NVIDIA'nın 1.000 kredisi biterse
    kuyruğun sonundaki modeller hiç koşmaz. O yüzden kredinin `deepseek-v4-pro`
    ve `inkling` gibi asıl adaylara gitmesi gerekir.
    """
    return sorted(models, key=lambda m: _PRIORITY_RANK.get(str(m.get("priority", "normal")), 1))


def quota_cap(model: dict[str, Any]) -> int | None:
    """Havuzun tavanı: günlük istek sayısı veya tek seferlik kredi bütçesi."""
    if model.get("budget") is not None:
        return int(model["budget"])
    if model.get("rpd") is not None:
        return int(model["rpd"])
    return None


# --------------------------------------------------------------------------- #
# Girdi hazırlama (dataset başına bir kez)
# --------------------------------------------------------------------------- #
@dataclass
class DatasetInputs:
    profile: dict[str, Any]
    columns: list[str]
    panel_cfg: dict[str, Any]


_DATASET_CACHE: dict[str, DatasetInputs] = {}


def dataset_inputs(dataset_dir: str) -> DatasetInputs:
    """Panel + profil + panel config; dataset başına bir kez kurulur.

    `run_e2e_check.py:_seam_load/_seam_profile` ile aynı yol: JUDGE üretimde de
    tam bu profili görüyor.
    """
    if dataset_dir not in _DATASET_CACHE:
        full = REPO_ROOT / dataset_dir
        config = load_dataset_config(full)
        panel = build_panel(full)
        _DATASET_CACHE[dataset_dir] = DatasetInputs(
            profile=profile_dataframe(panel.df),
            columns=list(panel.df.columns),
            panel_cfg=dict(config["panel"]),
        )
    return _DATASET_CACHE[dataset_dir]


def frozen_estimand_from_gold(case: dict[str, Any]) -> FrozenEstimand:
    """Altın estimand alanlarından deterministik FrozenEstimand kurar.

    spec_menu görevi estimand görevinin LLM çıktısına bağlı olmasın diye: her
    model aynı estimand'dan başlar, yoksa iki görevin hatası birbirine karışır.
    """
    proposal = TACProposal(**case["estimand"])
    return freeze_estimand(proposal, approved=True)


# --------------------------------------------------------------------------- #
# Puanlayıcılar — hepsi saf: çıktı + altın kayıt -> metrik sözlüğü
# --------------------------------------------------------------------------- #
def _referenced_columns(entry: LedgerEntry) -> tuple[str, ...]:
    """LedgerEntry'yi vetted çağrıya geri çevirip dokunduğu kolonları okur.

    Üretimdeki tipli şemayı yeniden kullanır (TransformCall discriminated union),
    böylece puanlayıcı transform parametre adlarını ikinci kez bilmek zorunda kalmaz.
    """
    call = _TRANSFORM_ADAPTER.validate_python(
        {"transform_name": entry.transform_name, **entry.params}
    )
    return tuple(call.referenced_columns())


def score_cleaning(entries: list[LedgerEntry], case: dict[str, Any]) -> dict[str, Any]:
    """Temizleme kararlarını altın MUST_FIX / FORBIDDEN / ACCEPTABLE'a göre puanlar."""
    proposed: list[tuple[str, str]] = []
    flag_by_pair: dict[tuple[str, str], bool] = {}
    for entry in entries:
        # Kolonsuz transform da bir öneridir ve precision'a girmeli:
        # `drop_duplicates(subset=None)` referenced_columns() olarak boş dizi
        # döndürür, kolon başına döngü onu tamamen görünmez bırakırdı.
        columns = _referenced_columns(entry) or ("",)
        for col in columns:
            pair = (entry.transform_name, col)
            proposed.append(pair)
            # Aynı çift birden çok kez önerilirse bayraklı olan kazanır: gatekeeper
            # açısından belirleyici olan, kararın insana gidip gitmediğidir.
            flag_by_pair[pair] = flag_by_pair.get(pair, False) or entry.belirsizlik_bayragi

    proposed_set = set(proposed)
    must = [(g["transform"], g["column"]) for g in case.get("must_fix", [])]
    forbidden = [(g["transform"], g["column"]) for g in case.get("forbidden", [])]
    acceptable = [(g["transform"], g["column"]) for g in case.get("acceptable", [])]

    found = [p for p in must if p in proposed_set]
    missed = [p for p in must if p not in proposed_set]
    hit_forbidden = [p for p in forbidden if p in proposed_set]
    # Precision'ın paydası TÜM öneriler, payı yalnız istenen + kabul edilebilir
    # olanlar. Yasak öneriler de precision'ı düşürür: veriyi bozan bir karar hem
    # ağır ihlaldir hem de isabetsizdir. (Önceki sürümde forbidden dışlanıyordu,
    # o yüzden yalnız yasak transform öneren model precision=1.0 alıyordu.)
    known = set(must) | set(acceptable)
    extraneous = [p for p in sorted(proposed_set) if p not in known]

    structural = set(case.get("structurally_missing", []))
    overconfident = [
        f"{t}:{c}"
        for (t, c), flagged in sorted(flag_by_pair.items())
        if c in structural and not flagged
    ]

    # Gereksiz çekingenlik: bilinen-doğru bir çifti (must_fix ya da acceptable,
    # yani gold zaten belirsizliksiz kabul ediyor) yine de insana sormak.
    # `structurally_missing`'in TAMAMLAYICISI değil — orası yalnız never-treated
    # sinyalini enumerate ediyor, "flag'in gerekçesiz olduğu her yer" değil. Gerçek
    # ölçüde belirsiz bir karara flag koymak gatekeeper'ın istediği davranıştır;
    # yanlış çapa (README:166-172'deki bad_control/config çelişkisiyle aynı hata
    # sınıfı) doğru davranan modeli cezalandırırdı. `c not in structural` şart:
    # medicaid'de implementation_date HEM must_fix HEM structurally_missing —
    # onu flag'lemek `overconfident_on_structural_missing`'i temiz tutan TAM
    # davranış; structural hariç tutulmazsa aynı eylem bir sayaçta ödüllenip
    # diğerinde cezalandırılır.
    unnecessary_flags = [
        f"{t}:{c}"
        for (t, c), flagged in sorted(flag_by_pair.items())
        if flagged and (t, c) in known and c not in structural
    ]

    n_prop = len(proposed_set)
    return {
        "n_decisions": len(entries),
        "proposed": sorted(f"{t}:{c}" for t, c in proposed_set),
        "must_fix_recall": (len(found) / len(must)) if must else None,
        "must_fix_missed": [f"{t}:{c}" for t, c in missed],
        "forbidden_hits": [f"{t}:{c}" for t, c in hit_forbidden],
        "precision": ((n_prop - len(extraneous)) / n_prop) if n_prop else None,
        "extraneous": [f"{t}:{c}" for t, c in extraneous],
        # Yapısal eksikliği olan kolona (never-treated sinyali) insana sormadan
        # dokunmak: gatekeeper'ın atlanması.
        "overconfident_on_structural_missing": overconfident,
        "unnecessary_flags": unnecessary_flags,
    }


def score_estimand(
    proposal: TACProposal, case: dict[str, Any], *, available_columns: list[str]
) -> dict[str, Any]:
    """Estimand eşlemesini, işaret korumasını ve clarification davranışını puanlar."""
    expect_clarify = bool(case["expect_needs_clarification"])
    declared_sign = case["declaration"]["expected_sign"]
    known = set(available_columns)

    hallucinated = [
        name for name in (proposal.treatment_coding, proposal.outcome) if name and name not in known
    ]

    accept_t = set(case.get("accept_treatment", []))
    accept_o = set(case.get("accept_outcome", []))
    if proposal.needs_clarification:
        # Clarification isteyen öneride kolon alanları anlamsızdır; eşleme
        # doğruluğu ölçülmez (None), yalnız clarification doğruluğu sayılır.
        treatment_ok: bool | None = None
        outcome_ok: bool | None = None
        hallucinated = []
    else:
        treatment_ok = proposal.treatment_coding in accept_t if accept_t else None
        outcome_ok = proposal.outcome in accept_o if accept_o else None

    return {
        "treatment_ok": treatment_ok,
        "outcome_ok": outcome_ok,
        "proposed_treatment": proposal.treatment_coding,
        "proposed_outcome": proposal.outcome,
        # Prompt açıkça yasaklıyor: "expected_sign değerini değiştirme".
        # İhlali her zaman hatadır, adversarial vakada bile.
        "sign_preserved": proposal.expected_sign == declared_sign,
        "needs_clarification": proposal.needs_clarification,
        # Abstention confusion matrix raporda burayı gold'u ikinci kez yüklemeden
        # okuyabilsin diye: puanlayıcılar "çıktı + altın kayıt -> metrik sözlüğü"
        # saf, ama gold'un beklentisini metrik sözlüğüne kopyalamak sözleşmeyi
        # bozmuyor (narrative zaten expected_driving_axes için aynısını yapıyor).
        "expect_needs_clarification": expect_clarify,
        "clarification_correct": proposal.needs_clarification == expect_clarify,
        "hallucinated_columns": hallucinated,
    }


def _control_columns(level: str) -> list[str]:
    if level.strip().lower() in ("none", "", "[]"):
        return []
    return [c.strip() for c in level.split("+") if c.strip()]


def score_spec_menu(
    proposal: SpecMenuProposal,
    case: dict[str, Any],
    *,
    available_columns: list[str],
    panel_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Menüyü deterministik kapıdan geçirir, sonra altın savunulabilirlik etiketlerini uygular."""
    estimand = case["estimand"]
    defensible, reasons, n_specs = evaluate_menu_defensibility(
        proposal,
        available_columns=available_columns,
        outcome=estimand["outcome"],
        treatment=estimand["treatment_coding"],
        unit_col=str(panel_cfg["unit"]),
        time_col=str(panel_cfg["time"]),
    )

    by_axis = {axis.axis_name: axis for axis in proposal.axes}
    missing_axes = [a for a in ALL_AXES if a not in by_axis]

    baseline_rules = case.get("baseline_must_be_in", {})
    baseline_violations = [
        f"{axis}={by_axis[axis].baseline_level}"
        for axis, allowed in baseline_rules.items()
        if axis in by_axis and by_axis[axis].baseline_level not in allowed
    ]

    indefensible_rules = case.get("indefensible_levels", {})
    indefensible_hits = []
    for axis, levels in indefensible_rules.items():
        axis_obj = by_axis.get(axis)
        if axis_obj is None:
            continue
        for level in [axis_obj.baseline_level, *axis_obj.candidate_levels]:
            if level in levels:
                indefensible_hits.append(f"{axis}={level}")

    bad_controls = case.get("bad_control_columns", {})
    bad_control_hits = []
    control_axis = by_axis.get("control_set")
    if control_axis is not None:
        for level in [control_axis.baseline_level, *control_axis.candidate_levels]:
            for col in _control_columns(level):
                if col in bad_controls:
                    bad_control_hits.append(f"{col} ({level})")

    return {
        "deterministic_gate_passed": defensible,
        "deterministic_reasons": reasons,
        "n_specs": n_specs,
        "missing_axes": missing_axes,
        # Cevap parmak izi: tekrarlar-arası tutarlılık ve order-check hem bu
        # alanı hem cleaning'in `proposed`'ını, estimand'ın treatment/outcome'ını
        # aynı mekanizmayla karşılaştırır (bkz. answer_fingerprint).
        "baseline_by_axis": {axis.axis_name: axis.baseline_level for axis in proposal.axes},
        "baseline_violations": baseline_violations,
        "indefensible_levels": sorted(set(indefensible_hits)),
        "bad_control_hits": sorted(set(bad_control_hits)),
        "needs_clarification": proposal.needs_clarification,
    }


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
# Türkçe teşhisi için sinyal kümesi: dile özgü harfler + sık işlev sözcükleri.
# Sezgisel olduğu bilinerek kullanılıyor; amaç "model İngilizce yazdı mı"yı
# yakalamak, dil bilimsel bir sınıflandırma yapmak değil.
_TR_CHARS = set("ğşıçöüĞŞİÇÖÜ")
_TR_WORDS = (
    "ve",
    "bir",
    "için",
    "ile",
    "olarak",
    "işaret",
    "eksen",
    "koşu",
    "değişim",
    "sonuç",
    "spesifikasyon",
    "yüzde",
    "dönüş",
)


def looks_turkish(text: str) -> bool:
    """Metin Türkçe mi (sezgisel): dile özgü harf VEYA en az iki işlev sözcüğü."""
    if any(ch in _TR_CHARS for ch in text):
        return True
    words = set(re.findall(r"\w+", text.lower()))
    return sum(1 for w in _TR_WORDS if w in words) >= 2


def _payload_numbers(payload: Any, into: set[float]) -> set[float]:
    """Girdideki her sayıyı toplar; oranların yüzde karşılığını da ekler."""
    if isinstance(payload, bool):
        return into
    if isinstance(payload, (int, float)):
        value = float(payload)
        into.add(value)
        into.add(abs(value))
        if 0 < abs(value) <= 1:  # 0.75 -> "%75" olarak yazılabilir
            into.add(round(abs(value) * 100, 4))
            into.add(float(round(abs(value) * 100)))
        return into
    if isinstance(payload, dict):
        for key, val in payload.items():
            _payload_numbers(key, into)
            _payload_numbers(val, into)
        return into
    if isinstance(payload, (list, tuple)):
        for item in payload:
            _payload_numbers(item, into)
        return into
    if isinstance(payload, str):
        for match in _NUMBER_RE.findall(payload):
            try:
                into.add(abs(float(match.replace(",", "."))))
            except ValueError:
                continue
    return into


def fabricated_numbers(text: str, allowed: set[float]) -> list[str]:
    """Metindeki, girdiden gelmeyen ve türetilebilir olmayan sayılar."""
    out: list[str] = []
    for token in _NUMBER_RE.findall(text):
        try:
            value = abs(float(token.replace(",", ".")))
        except ValueError:
            continue
        if not any(abs(value - a) < 1e-6 for a in allowed):
            out.append(token)
    return out


def score_narrative(narrative: VarianceNarrative, case: dict[str, Any]) -> dict[str, Any]:
    """Anlatıyı eksen atfı, halüsinasyon ve sayı uydurma açısından puanlar."""
    mentioned = [c.axis for c in narrative.eksen_yorumlari]
    expected = list(case["expected_driving_axes"])
    forbidden = set(case["forbidden_axes"])

    text = " ".join([narrative.ozet, *(c.yorum for c in narrative.eksen_yorumlari)])

    allowed = _payload_numbers({"s": case["summary"], "d": case["diagnosis"]}, set())
    for key in case.get("allowed_derived", {}):
        try:
            allowed.add(abs(float(key)))
        except ValueError:
            continue

    # Beklenen eksen ozet'te anılmalı: panelin okuyucuya söylediği tek cümle orası.
    driver_named = all(axis in narrative.ozet for axis in expected) if expected else None

    return {
        "axes_commented": mentioned,
        "expected_driving_axes": expected,
        "driver_named_in_ozet": driver_named,
        # Sinyal yokken eksen yorumu yazmak: prompt "skip axes with no signal" diyor.
        "commented_without_signal": (not expected) and bool(mentioned),
        "forbidden_axes_mentioned": sorted(set(mentioned) & forbidden),
        # Girdide olmayan sayı: sistem promptu sayı üretmeyi yasaklıyor ama
        # üretimde bunu zorlayan doğrulayıcı YOK. Burada ölçülüyor.
        "fabricated_numbers": fabricated_numbers(text, allowed),
        "is_turkish": looks_turkish(text),
    }


# --------------------------------------------------------------------------- #
# Cevap parmak izi — tekrarlar-arası tutarlılık + order-check için tek mekanizma
# --------------------------------------------------------------------------- #
def answer_fingerprint(task: str, scores: dict[str, Any]) -> tuple[Any, ...]:
    """Görevin 'cevabı'nı karşılaştırılabilir, deterministik bir tuple'a indirger.

    Aynı vaka + aynı model iki kez koşulduğunda (tekrar ya da kolon sırası
    değişince) bu iki sonucun AYNI cevaba mı vardığını ölçer — yalnız ikisinin
    de geçip geçmediğini değil. İki-üç doğru ama farklı cevap, üç kez aynı yanlış
    cevaptan daha güvenilir bir JUDGE adayı gibi görünebilir; parmak izi bunu
    ayırt eder.
    """
    if task == "cleaning":
        return tuple(scores.get("proposed") or ())
    if task == "estimand":
        return (
            scores.get("proposed_treatment"),
            scores.get("proposed_outcome"),
            scores.get("needs_clarification"),
        )
    if task == "spec_menu":
        return tuple(sorted((scores.get("baseline_by_axis") or {}).items()))
    if task == "narrative":
        return tuple(sorted(scores.get("axes_commented") or ()))
    raise ValueError(f"bilinmeyen görev: {task}")


# --------------------------------------------------------------------------- #
# Görev kayıt defteri
# --------------------------------------------------------------------------- #
def _run_cleaning(case: dict[str, Any]) -> dict[str, Any]:
    data = dataset_inputs(case["dataset_dir"])
    entries = generate_ledger(data.profile)
    return score_cleaning(entries, case)


def _run_estimand(case: dict[str, Any]) -> dict[str, Any]:
    data = dataset_inputs(case["dataset_dir"])
    proposal = draft_tac_proposal(
        research_story=case["research_story"],
        available_columns=data.columns,
        declaration=SocraticDeclaration(**case["declaration"]),
    )
    return score_estimand(proposal, case, available_columns=data.columns)


def _run_spec_menu(case: dict[str, Any]) -> dict[str, Any]:
    data = dataset_inputs(case["dataset_dir"])
    proposal = generate_spec_menu(
        frozen=frozen_estimand_from_gold(case),
        available_columns=data.columns,
    )
    return score_spec_menu(proposal, case, available_columns=data.columns, panel_cfg=data.panel_cfg)


def _run_narrative(case: dict[str, Any]) -> dict[str, Any]:
    narrative = generate_narrative(case["summary"], case["diagnosis"])
    return score_narrative(narrative, case)


TASK_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cleaning": _run_cleaning,
    "estimand": _run_estimand,
    "spec_menu": _run_spec_menu,
    "narrative": _run_narrative,
}


# --------------------------------------------------------------------------- #
# Sıra duyarlılığı (--order-check, isteğe bağlı) — yalnız kolon sırası
# GİRDİDE ters çevrilir; skorlayıcıya gerçek (ters çevrilmemiş) kolonlar gider,
# çünkü skorlama kümeye bakıyor (bkz. score_estimand/score_spec_menu), sıraya
# değil. Cleaning/narrative'de model girdisi bir kolon LİSTESİ değil (narrative
# hiç kolon görmüyor, cleaning profildeki sırayı zaten kendi üretiyor), o yüzden
# yalnız estimand + spec_menu'de anlamlı.
# --------------------------------------------------------------------------- #
ORDER_CHECK_VARIANT = "order-reversed"


def _run_estimand_reversed(case: dict[str, Any]) -> dict[str, Any]:
    data = dataset_inputs(case["dataset_dir"])
    proposal = draft_tac_proposal(
        research_story=case["research_story"],
        available_columns=list(reversed(data.columns)),
        declaration=SocraticDeclaration(**case["declaration"]),
    )
    return score_estimand(proposal, case, available_columns=data.columns)


def _run_spec_menu_reversed(case: dict[str, Any]) -> dict[str, Any]:
    data = dataset_inputs(case["dataset_dir"])
    proposal = generate_spec_menu(
        frozen=frozen_estimand_from_gold(case),
        available_columns=list(reversed(data.columns)),
    )
    return score_spec_menu(proposal, case, available_columns=data.columns, panel_cfg=data.panel_cfg)


ORDER_CHECK_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "estimand": _run_estimand_reversed,
    "spec_menu": _run_spec_menu_reversed,
}


# --------------------------------------------------------------------------- #
# Model enjeksiyonu
# --------------------------------------------------------------------------- #
@contextmanager
def pinned_model(model: dict[str, Any]) -> Iterator[None]:
    """JUDGE slotunu bu modele sabitler (env üzerinden, üretim çözüm yolundan).

    `chain_for` zinciri her çağrıda yeniden kurduğu için env'i ayarlamak yeter;
    router'a dokunulmaz. Slotun `model_env` adı koddan okunur, burada tekrarlanmaz.
    """
    provider = model["provider"]
    slots = judge_slots_for(PrivacyMode.PUBLIC)
    if provider not in slots:
        raise SystemExit(
            f"Bilinmeyen sağlayıcı {provider!r}. Kayıtlı PUBLIC judge sağlayıcıları: "
            f"{', '.join(slots)}"
        )
    slot = slots[provider]
    previous = {
        "PARETO_JUDGE_PROVIDER": os.environ.get("PARETO_JUDGE_PROVIDER"),
        slot.model_env: os.environ.get(slot.model_env),
    }
    os.environ["PARETO_JUDGE_PROVIDER"] = provider
    os.environ[slot.model_env] = str(model["id"])
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def classify_error(exc: BaseException) -> str:
    """Hatayı rapora yazılacak tek kelimeye indirger.

    Sıra önemli ve TİP, metinden önce gelir. pydantic-ai'nin şema retry mesajları
    sık sık "Unknown tool name ... not found" gibi ifadeler taşır; metne önce
    bakılırsa modelin kendi şema kusuru "model_yok" diye raporlanır ve aday
    listesinden yanlış nedenle çıkarılır. Tip bilgisi bu konuda metinden çok daha
    güçlü kanıt.
    """
    name = type(exc).__name__
    text = str(exc).lower()
    if isinstance(exc, OSError) and "tanımlı değil" in str(exc):
        return "anahtar_yok"
    if name in ("UnexpectedModelBehavior", "ValidationError", "ModelRetry"):
        return "sema_tutmadi"
    if "429" in text or "rate limit" in text or "quota" in text:
        return "kota"
    if "404" in text or "not found" in text or "does not exist" in text:
        return "model_yok"
    if "401" in text or "403" in text or "unauthorized" in text:
        return "yetki"
    if isinstance(exc, ValueError):
        return "dogrulayici_reddetti"
    return f"hata:{name}"


# Modele hiç ulaşamadığımız hatalar: şema hakkında BİLGİ vermezler.
# Bunlarda schema_ok=True demek "şemayı tutturdu" diye okunur ve yanlıştır.
_INFRA_ERRORS = frozenset({"anahtar_yok", "kota", "model_yok", "yetki"})


def schema_verdict(error_kind: str | None) -> bool | None:
    """Şema kararı: geçti / tutmadı / hakkında bilgi yok (None)."""
    if error_kind is None:
        return True
    if error_kind == "sema_tutmadi":
        return False
    return None if error_kind in _INFRA_ERRORS else True


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_one(model: dict[str, Any]) -> dict[str, Any]:
    """Tek modele küçük bir tipli çağrı: uç yaşıyor mu, şema zorluyor mu.

    En ucuz gerçek görevi (narrative, sabit küçük payload) kullanır; ayrı bir
    oyuncak şema uydurmak, ölçtüğümüz şeyle ölçtüğümüzü sandığımız şeyi ayırırdı.
    """
    case = load_gold("narrative")[0]
    meter = Meter()
    started = time.monotonic()
    try:
        with pinned_model(model), metered(meter):
            generate_narrative(case["summary"], case["diagnosis"])
        return {
            "model": model_key(model),
            "status": "ok",
            "latency_s": round(time.monotonic() - started, 3),
            "requests": meter.requests,
            "detail": f"{meter.requests} istek (retry: {meter.retries})",
        }
    except Exception as exc:  # noqa: BLE001 — sınıflandırıp rapora yazıyoruz
        return {
            "model": model_key(model),
            "status": "elendi",
            "latency_s": round(time.monotonic() - started, 3),
            "requests": 0,
            "detail": f"{classify_error(exc)}: {str(exc)[:180]}",
        }


# --------------------------------------------------------------------------- #
# Koşu
# --------------------------------------------------------------------------- #
def result_id(model: dict[str, Any], task: str, case_id: str, repeat: int | str) -> str:
    return f"{model_key(model)}|{task}|{case_id}|{repeat}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """`results.jsonl`'i tolere ederek okur: dosya yoksa boş, yarım satır atlanır.

    Koşu kill edilirse son satır yarım kalabilir. Bunun tüm resume'u ya da rapor
    üretimini çöpe atmaması gerekir — tek eksik satır, biriken sonuçlardan
    kıymetli değil.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  [uyarı] ayrıştırılamayan satır atlandı: {line[:80]}", file=sys.stderr)
    return rows


def read_done(path: Path) -> dict[str, int]:
    """Tamamlanmış çağrılar: sonuç kimliği -> 1."""
    return {row["result_id"]: 1 for row in read_rows(path) if row.get("result_id")}


def _call_and_score(
    model: dict[str, Any],
    task: str,
    case: dict[str, Any],
    *,
    rid: str,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Tek çağrının ortak iskeleti: pin -> ölç -> puanla -> hatayı sınıflandır.

    Normal tekrar döngüsü ve --order-check aynı iskeleti paylaşır; ayrılırlarsa
    order-check satırları farklı bir ölçüm yoluna gider ve ana satırlarla
    karşılaştırılamaz hale gelir.
    """
    meter = Meter()
    started = time.monotonic()
    row: dict[str, Any] = {
        "result_id": rid,
        "model": model_key(model),
        "provider": model["provider"],
        "model_id": model["id"],
        "task": task,
        "case_id": case["case_id"],
        **extra,
    }
    try:
        with pinned_model(model), metered(meter):
            scores = runner(case)
        row.update(schema_ok=True, validator_passed=True, error=None, scores=scores)
    except Exception as exc:  # noqa: BLE001 — sessiz atlama yok
        kind = classify_error(exc)
        row.update(
            schema_ok=schema_verdict(kind),
            validator_passed=False,
            error=kind,
            error_detail=str(exc)[:400],
            scores=None,
        )
    row.update(
        latency_s=round(time.monotonic() - started, 3),
        requests=meter.requests,
        retries=meter.retries,
        input_tokens=meter.input_tokens,
        output_tokens=meter.output_tokens,
    )
    return row


def run_matrix(
    models: list[dict[str, Any]],
    tasks: tuple[str, ...],
    repeats: int,
    out_dir: Path,
    throttle: Throttle,
    *,
    order_check: bool = False,
) -> list[dict[str, Any]]:
    """Matrisi koşar; her sonuç anında diske yazılır (resume edilebilirlik).

    `order_check=True` ise estimand + spec_menu vakalarına, normal tekrarlardan
    SONRA, kolon sırası ters çevrilmiş bir ekstra çağrı eklenir (bkz.
    ORDER_CHECK_RUNNERS). Varsayılan kapalı: kota zaten en dar havuzda (Gemini
    Flash 20/gün) tam matrisi zar zor karşılıyor, bunu ikiye katlamak opt-in olmalı.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = read_done(results_path)

    for model in models:
        already = sum(1 for rid in done if rid.startswith(f"{model_key(model)}|"))
        if already:
            throttle.seed(quota_pool(model), already)

    rows: list[dict[str, Any]] = []
    for model in models:
        key = model_key(model)
        pool = quota_pool(model)
        cap = quota_cap(model)
        rpm = model.get("rpm")
        exhausted = False
        for task in tasks:
            if exhausted:
                break
            for case in load_gold(task):
                if exhausted:
                    break
                for repeat in range(repeats):
                    rid = result_id(model, task, case["case_id"], repeat)
                    if rid in done:
                        continue
                    try:
                        throttle.acquire(pool, rpm=rpm, cap=cap)
                    except QuotaExhausted as exc:
                        print(f"  [kota] {exc} — bu havuzdaki kalan işler atlanıyor")
                        exhausted = True
                        break

                    row = _call_and_score(
                        model,
                        task,
                        case,
                        rid=rid,
                        runner=TASK_RUNNERS[task],
                        extra={"repeat": repeat},
                    )
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    flag = "ok" if row.get("error") is None else row["error"]
                    print(f"  {key} · {task}/{case['case_id']}#{repeat} → {flag}")

                if exhausted or not order_check or task not in ORDER_CHECK_RUNNERS:
                    continue
                rid = result_id(model, task, case["case_id"], ORDER_CHECK_VARIANT)
                if rid in done:
                    continue
                try:
                    throttle.acquire(pool, rpm=rpm, cap=cap)
                except QuotaExhausted as exc:
                    print(f"  [kota] {exc} — bu havuzdaki kalan işler atlanıyor")
                    exhausted = True
                    continue

                row = _call_and_score(
                    model,
                    task,
                    case,
                    rid=rid,
                    runner=ORDER_CHECK_RUNNERS[task],
                    extra={"repeat": ORDER_CHECK_VARIANT, "order_variant": "reversed"},
                )
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                flag = "ok" if row.get("error") is None else row["error"]
                print(f"  {key} · {task}/{case['case_id']}#order-reversed → {flag}")
    return rows


# --------------------------------------------------------------------------- #
# Rapor
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], pct: int) -> float | None:
    """En yakın-sıra (nearest-rank) yüzdelik — yalnız GÖRÜLMÜŞ bir değeri döndürür.

    `statistics.quantiles(..., n=100)` bu ölçekte (model başına ~onlarca çağrı)
    enterpolasyonla var olmayan bir hassasiyet uydurur; nearest-rank uydurmaz.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-(pct * len(ordered)) // 100))  # ceil(pct/100 * n)
    return ordered[min(rank, len(ordered)) - 1]


def _answer_consistency(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    """Aynı (görev, vaka) için tekrarlar aynı cevaba mı varıyor (bkz. answer_fingerprint).

    Yalnız BAŞARILI çağrılar sayılır (kota/hata 'tutarsızlık' değil, yokluktur) ve
    grup büyüklüğü >=2 olmalı — tek tekrarlı bir grup otomatik '%100 tutarlı'
    okunursa, kotaya en sık çarpan model (gemini-3.6-flash: 20/gün, bkz. README)
    en tutarlı görünür. Dönen ikinci değer (sayılabilir grup sayısı) bu yüzden
    orana eşlik etmeden raporlanmamalı.
    """
    groups: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    for row in rows:
        if row.get("error") is not None or not row.get("scores") or row.get("order_variant"):
            continue
        key = (row["task"], row["case_id"])
        fp = answer_fingerprint(row["task"], row["scores"])
        groups.setdefault(key, []).append(fp)

    countable = [fps for fps in groups.values() if len(fps) >= 2]
    if not countable:
        return None, 0
    agree = sum(1 for fps in countable if len(set(fps)) == 1)
    return agree / len(countable), len(countable)


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # order-check varyantları ayrı bir soruya cevap veriyor (sıra değişince cevap
    # değişiyor mu); ana eleme tablosunun ok_rate/latency/token istatistiklerine
    # karışırsa o istatistikleri gerçekte koşulmamış bir yük dağılımıyla kirletir.
    rows = [r for r in rows if not r.get("order_variant")]
    n = len(rows)
    ok = [r for r in rows if r.get("error") is None]
    # Gecikme yalnız BAŞARILI çağrılardan: hemen dönen bir 404, yavaş ama çalışan
    # bir modeli hızlı gösterirdi.
    lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    out_tokens = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]
    consistency, consistency_n = _answer_consistency(rows)
    return {
        "n": n,
        "ok": len(ok),
        "ok_rate": (len(ok) / n) if n else None,
        "retries": sum(int(r.get("retries") or 0) for r in rows),
        "latency_median": (statistics.median(lat) if lat else None),
        "latency_p95": _percentile(lat, 95),
        "tokens_out_median": (statistics.median(out_tokens) if out_tokens else None),
        "tokens_in_total": sum(int(r.get("input_tokens") or 0) for r in rows),
        "tokens_out_total": sum(int(r.get("output_tokens") or 0) for r in rows),
        "answer_consistency": consistency,
        "answer_consistency_n": consistency_n,
        "errors": sorted({str(r["error"]) for r in rows if r.get("error")}),
    }


def _cleaning_abstention_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(gatekeeper atlandı, gereksiz flag) — cleaning satırları üzerinden toplam.

    order-check varyantları hariç (bkz. _estimand_confusion) — bugün cleaning'de
    order-check yok ama sayaç sözleşmesi ikisinde de aynı olmalı.
    """
    missed_gate = 0
    over_caution = 0
    for row in rows:
        if row.get("task") != "cleaning" or row.get("order_variant"):
            continue
        scores = row.get("scores") or {}
        missed_gate += len(scores.get("overconfident_on_structural_missing") or ())
        over_caution += len(scores.get("unnecessary_flags") or ())
    return missed_gate, over_caution


def _estimand_confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Clarification confusion matrix — SAYIM, precision/recall değil (bkz. build_report notu).

    order-check varyantları hariç: dahil edilirse "tekrar=3'te 9/3 örnek" notu
    yanlış olur (--order-check ile 4. bir örnek farklı bir deneysel koşuldan gelir).
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
    for row in rows:
        if row.get("task") != "estimand" or row.get("order_variant"):
            continue
        scores = row.get("scores") or {}
        if "expect_needs_clarification" not in scores:
            continue
        actual = bool(scores.get("needs_clarification"))
        expect = bool(scores.get("expect_needs_clarification"))
        counts["n"] += 1
        counts["tp" if actual and expect else "fp" if actual else "fn" if expect else "tn"] += 1
    return counts


def _order_check_lines(rows: list[dict[str, Any]]) -> list[str]:
    """--order-check ile üretilmiş varyant satırlarını temel çağrıyla karşılaştırır.

    Hiç order-check koşulmadıysa boş liste döner (bölüm hiç basılmaz) — sessiz bir
    'karşılaştırılamadı' satırıyla var olmayan bir ölçüm varmış izlenimi vermez.
    """
    variants = [r for r in rows if r.get("order_variant")]
    if not variants:
        return []

    baseline_fp: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for row in rows:
        if row.get("order_variant") or row.get("error") is not None or not row.get("scores"):
            continue
        key = (row["model"], row["task"], row["case_id"])
        baseline_fp.setdefault(key, answer_fingerprint(row["task"], row["scores"]))

    lines = [
        "",
        "## Sıra duyarlılığı (order-check)",
        "",
        "Kolon listesi TERS çevrilip aynı vaka bir kez daha koşuldu. Cevap değiştiyse",
        "model girdi sırasına duyarlı — literatürde belgeli bir kırılganlık (order/position",
        "bias). Baz alınan cevap aynı vakanın normal-sıra tekrarlarından ilkidir.",
        "",
        "UYARI: aynı vakanın normal-sıra tekrarları zaten kendi aralarında tutarsızsa",
        "(yukarıdaki Cevap tutarlılığı sütununa bakın) bir HAYIR burada sıra duyarlılığını",
        "değil, modelin genel kararsızlığını gösterebilir — ikisini birlikte okuyun.",
        "",
        "| Model | Görev/Vaka | Sıra-kararlı mı |",
        "|---|---|---|",
    ]
    for row in sorted(variants, key=lambda r: (r["model"], r["task"], r["case_id"])):
        key = (row["model"], row["task"], row["case_id"])
        base = baseline_fp.get(key)
        if base is None or row.get("error") is not None or not row.get("scores"):
            verdict = "? karşılaştırılamadı (biri hata verdi)"
        else:
            verdict = (
                "evet"
                if answer_fingerprint(row["task"], row["scores"]) == base
                else "HAYIR — cevap değişti"
            )
        lines.append(f"| `{row['model']}` | {row['task']}/{row['case_id']} | {verdict} |")
    return lines


def build_report(rows: list[dict[str, Any]], tasks: tuple[str, ...]) -> str:
    """Model tablosu (tüm görevler havuzlu) + ağır ihlaller. Elenen/hatalı hiçbir şey gizlenmez.

    Görev kırılımı yok: bu tablo sıralama değil eleme aracı, tek bir ağır ihlal
    zaten yeterli diskalifiye nedeni (bkz. 'Nasıl okunmalı'). Görev başına kırılım
    isteniyorsa `results.jsonl`'i `task` alanına göre filtrelemek yeterli.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    lines = ["# JUDGE model benchmark raporu", ""]
    lines.append(f"Toplam çağrı: {len(rows)} · model: {len(by_model)} · görev: {', '.join(tasks)}")
    lines.append("")
    lines.append(
        "| Model | Çağrı | Şema+doğrulayıcı geçti | Retry | Medyan gecikme (s) | "
        "p95 gecikme (s) | Çıktı token (medyan) | Cevap tutarlılığı | Hatalar |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(by_model):
        a = _agg(by_model[name])
        rate = f"{a['ok']}/{a['n']}" if a["n"] else "-"
        med = f"{a['latency_median']:.2f}" if a["latency_median"] is not None else "-"
        p95 = f"{a['latency_p95']:.2f}" if a["latency_p95"] is not None else "-"
        tok = f"{a['tokens_out_median']:.0f}" if a["tokens_out_median"] is not None else "-"
        cons = (
            f"{a['answer_consistency']:.0%} (n={a['answer_consistency_n']})"
            if a["answer_consistency"] is not None
            else "- (yetersiz tekrar)"
        )
        lines.append(
            f"| `{name}` | {a['n']} | {rate} | {a['retries']} | {med} | {p95} | {tok} | {cons} | "
            f"{', '.join(a['errors']) or '-'} |"
        )
    lines += [
        "",
        "$ maliyet hesaplanmadı: aday matrisi 'bugün kalıcı ücretsiz' filtresiyle",
        "seçildi (README), `models.json`'da fiyat alanı yok. Çıktı token medyanı bir",
        "verimlilik proxy'si — gerçek $ maliyeti değil, uydurmamak için eklenmedi.",
    ]

    lines += ["", "## Ağır ihlaller", ""]
    severe: list[str] = []
    for row in rows:
        scores = row.get("scores") or {}
        for label, field_name in (
            ("YASAK transform", "forbidden_hits"),
            ("uydurma kolon", "hallucinated_columns"),
            ("uydurma eksen", "forbidden_axes_mentioned"),
            ("uydurma sayı", "fabricated_numbers"),
            ("savunulamaz seviye", "indefensible_levels"),
            ("bad control", "bad_control_hits"),
            ("gatekeeper atlandı", "overconfident_on_structural_missing"),
        ):
            hits = scores.get(field_name)
            if hits:
                severe.append(
                    f"- `{row['model']}` · {row['task']}/{row['case_id']} · "
                    f"{label}: {', '.join(map(str, hits))}"
                )
        if scores.get("sign_preserved") is False:
            severe.append(
                f"- `{row['model']}` · {row['task']}/{row['case_id']} · "
                "expected_sign DEĞİŞTİRİLDİ (prompt bunu açıkça yasaklıyor)"
            )
    lines += severe or ["- yok"]

    lines += [
        "",
        "## Abstention (gatekeeper) sayaçları",
        "",
        "Oran değil SAYIM: estimand'da gold 3 olumsuz/1 olumlu vaka (tekrar=3'te 9/3",
        "örnek) — precision/recall/F1 bu ölçekte kesinlik uydurur, sayım uydurmaz.",
        "spec_menu'da 'sormalı mıydı' gold etiketi yok, o yüzden tabloda değil.",
        "",
        "| Model | Cleaning: gatekeeper atlandı | Cleaning: gereksiz flag | "
        "Estimand TP/FP/FN/TN (n) |",
        "|---|---|---|---|",
    ]
    for name in sorted(by_model):
        missed_gate, over_caution = _cleaning_abstention_counts(by_model[name])
        conf = _estimand_confusion(by_model[name])
        lines.append(
            f"| `{name}` | {missed_gate} | {over_caution} | "
            f"{conf['tp']}/{conf['fp']}/{conf['fn']}/{conf['tn']} (n={conf['n']}) |"
        )
    lines += [
        "",
        "TP/FN: veri gerçekten eksikken doğru/yanlış davrandı (sor / sormadı).",
        "FP/TN: veri yeterliyken gereksiz sordu / doğru şekilde sormadı.",
    ]

    lines += _order_check_lines(rows)

    lines += [
        "",
        "## Nasıl okunmalı",
        "",
        "Bu tablo sıralama değil eleme aracıdır. Ağır ihlal listesinde görünen bir model,",
        "gecikmesi ve şema uyumu ne olursa olsun JUDGE slotuna aday değildir: uydurulmuş",
        "bir kolon adı ya da değiştirilmiş bir `expected_sign`, kullanıcının savunulabilir",
        "sonuç vaadini doğrudan çürütür.",
        "",
        "Sonuçların `providers.py`'deki `_JUDGE_*_OPTIONS` listelerine ve",
        "`ModelOption.performance_note` alanlarına aktarılması AYRI bir iştir.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def plan_lines(
    models: list[dict[str, Any]],
    tasks: tuple[str, ...],
    repeats: int,
    *,
    order_check: bool = False,
) -> list[str]:
    """--dry-run çıktısı: matris + KOTA HAVUZU başına yük ve gün tahmini.

    Gün tahmini havuz seviyesinde yapılır. Model başına yapmak, aynı kotayı
    paylaşan uçlarda (OpenRouter :free, NVIDIA kredileri) süreyi olduğundan
    kat kat kısa gösterirdi.

    `order_check`'i saymazsak tavan tahmini gerçek koşudan düşük çıkar ve koşu
    ortasında beklenmedik bir 429 yer — order-check her vakaya tekrar sayısından
    BAĞIMSIZ +1 çağrı ekliyor, yalnız estimand + spec_menu'de.
    """
    n_cases = sum(len(load_gold(t)) for t in tasks)
    order_extra = 0
    if order_check:
        order_extra = sum(len(load_gold(t)) for t in tasks if t in ORDER_CHECK_RUNNERS)
    per_model = n_cases * repeats + order_extra
    lines = [
        f"Görev: {', '.join(tasks)} · vaka: {n_cases} · tekrar: {repeats}"
        + (f" · order-check: +{order_extra}/model" if order_check else ""),
        f"Model başına çağrı: {per_model} · toplam: {per_model * len(models)}",
        "",
        f"{'Model':<48} {'sağlayıcı':<11} {'havuz':<18}",
    ]
    pools: dict[str, dict[str, Any]] = {}
    for model in models:
        pool = quota_pool(model)
        shown = "(kendi)" if pool == model_key(model) else pool
        lines.append(f"{model['id']:<48} {model['provider']:<11} {shown:<18}")
        bucket = pools.setdefault(pool, {"calls": 0, "cap": quota_cap(model), "n": 0})
        bucket["calls"] += per_model
        bucket["n"] += 1
        # Aynı havuzdaki uçlar farklı tavan bildirirse en dar olanı bağlayıcıdır;
        # ilk modelinkini almak süreyi olduğundan kısa gösterirdi.
        caps = [c for c in (bucket["cap"], quota_cap(model)) if c is not None]
        bucket["cap"] = min(caps) if caps else None

    lines += ["", f"{'Kota havuzu':<28} {'model':>6} {'çağrı':>7} {'tavan':>7} {'gün':>5}"]
    for pool, info in sorted(pools.items()):
        cap = info["cap"]
        days = "-" if not cap else str(-(-info["calls"] // cap))  # yukarı yuvarlama
        lines.append(f"{pool:<28} {info['n']:>6} {info['calls']:>7} {str(cap or '-'):>7} {days:>5}")
    slowest = max(
        (-(-i["calls"] // i["cap"]) for i in pools.values() if i["cap"]),
        default=0,
    )
    lines += [
        "",
        f"En yavaş havuz {slowest} koşu-günü sürer; toplam süreyi o belirler.",
        "Tavanlar benchmarks/models.json'dan okunur ve DEĞİŞİR — gerçeği",
        "`--preflight` ve sağlayıcının kendi konsolu söyler.",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="çağrı yapmadan matrisi bas")
    parser.add_argument("--preflight", action="store_true", help="yalnız uç/şema yoklaması")
    parser.add_argument("--models", nargs="*", help="yalnız bu model ID'leri")
    parser.add_argument("--providers", nargs="*", help="yalnız bu sağlayıcılar")
    parser.add_argument("--tasks", nargs="*", choices=TASKS, help="yalnız bu görevler")
    parser.add_argument("--repeats", type=int, default=3, help="vaka başına tekrar (varsayılan 3)")
    parser.add_argument("--out", type=Path, help="çıktı dizini (varsayılan runs/benchmark/<ts>)")
    parser.add_argument(
        "--order-check",
        action="store_true",
        help=(
            "estimand+spec_menu vakalarına kolon sırası ters çevrilmiş 1 ekstra "
            "çağrı ekle (sıra duyarlılığı raporu); çağrı sayısını bu görevlerde "
            "ikiye katlar, varsayılan kapalı"
        ),
    )
    args = parser.parse_args(argv)

    models = load_models()
    if args.providers:
        models = [m for m in models if m["provider"] in set(args.providers)]
    if args.models:
        models = [m for m in models if m["id"] in set(args.models)]
    if not models:
        print("Filtreye uyan model yok.", file=sys.stderr)
        return 2
    # Kuyruk sırası paylaşımlı havuzda sonucu belirler (bkz. by_priority).
    models = by_priority(models)
    tasks: tuple[str, ...] = tuple(args.tasks) if args.tasks else TASKS

    if args.dry_run:
        print("\n".join(plan_lines(models, tasks, args.repeats, order_check=args.order_check)))
        return 0

    require_cache_disabled()

    if args.preflight:
        print(f"Preflight: {len(models)} model\n")
        print(f"{'Model':<52} {'durum':<8} {'s':>6}  ayrıntı")
        failed = 0
        for model in models:
            row = preflight_one(model)
            failed += row["status"] != "ok"
            print(
                f"{row['model']:<52} {row['status']:<8} {row['latency_s']:>6.2f}  {row['detail']}"
            )
        print(f"\n{len(models) - failed} geçti, {failed} elendi.")
        return 1 if failed else 0

    out_dir = args.out or (DEFAULT_OUT_ROOT / time.strftime("%Y%m%d-%H%M%S"))
    print(f"Çıktı: {out_dir}")
    rows = run_matrix(
        models, tasks, args.repeats, out_dir, Throttle(), order_check=args.order_check
    )
    # Rapor tüm koşuyu kapsasın: bu oturumda atlanan (resume) satırlar da dahil.
    # Dosya hiç oluşmamış olabilir (ilk çağrıdan önce kota bittiyse) — read_rows
    # bunu boş liste olarak döndürür, koşu rapor üretirken çökmez.
    all_rows = read_rows(out_dir / "results.jsonl")
    if not all_rows:
        print("\nHiç sonuç üretilmedi (kota mı bitti?). Rapor yazılmadı.", file=sys.stderr)
        return 1
    report_path = out_dir / "report.md"
    report_path.write_text(build_report(all_rows, tasks), encoding="utf-8")
    print(f"\nYeni sonuç: {len(rows)} · toplam: {len(all_rows)}\nRapor: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
