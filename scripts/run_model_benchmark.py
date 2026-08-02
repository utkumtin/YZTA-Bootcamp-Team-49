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
from dataclasses import dataclass, field, replace
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
import pareto.llm.providers as llm_providers  # noqa: E402
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

# Vaka başına tekrar. 3 -> 2: teslim matrisi (16 vaka) gemini havuzunun birleşik
# 40/gün kotasına TEK GÜNDE sığmak zorunda; 16x3=48 sığmıyor, 16x2=32 sığıyor ve
# 8 çağrılık retry payı bırakıyor. Vaka kırpmak yerine tekrar düşürüldü çünkü
# dört veri setinin her biri diğerinin ölçemediği bir kusuru taşıyor
# (bkz. benchmarks/README.md: "Altın etiketler nereden geliyor").
# KAYIT: `_answer_consistency` n>=2 ile çalışmaya devam eder, ama n=2'de "tüm
# tekrarlar aynı" olmak n=3'ten kolaydır — yeni tutarlılık yüzdeleri eski
# 3-tekrarlı koşularla kıyaslanamaz.
DEFAULT_REPEATS = 2

# Kaç ARDIŞIK pahalı hatadan sonra bir (model, GÖREV) çifti elenir.
# Kapsam görev bazlı, model bazlı değil: gemma-4-31b-it `cleaning/medicaid`'i 3/3
# ~15sn'de geçip `cleaning/card_krueger`'da 504 aldı (2026-07-30) — modeli tümden
# elemek onun diğer 3 görevdeki performansını da silerdi, oysa "bu model şu görevde
# boğuluyor" tam olarak öğrenmek istediğimiz şey.
CIRCUIT_BREAK_ERRORS = 2

# Bir başarısız çağrı bu süreden uzun sürdüyse, hata sınıfı ne olursa olsun devre
# kesiciye sayılır. Devre kesicinin varlık sebebi süre ve (yenilenmeyen NVIDIA)
# kredi korumak; doğru ölçüt hata etiketi değil MALİYET. Canlı kanıt: 504
# DEADLINE_EXCEEDED @ 181sn "altyapı" sınıfında ama tek modelde 48 çağrı × 180sn
# = 2,4 saat eder. Hızlı 429/503 (~0,5sn) bu eşiğin altında kalır ve saymaz.
SLOW_FAILURE_S = 60.0

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

    def _wrap(model: Model | str, *, canned_mode: bool = False) -> Model | str:
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
    """RPM/TPM aralığını doldurmak için beklerken terminale basar.

    Sessiz kalırsa (ör. gemini-3.6-flash rpm=5 -> istekler arası 12sn) koşu
    donmuş gibi görünür; kullanıcı neyin beklendiğini görmeli.
    """
    print(f"  [hız] {pool}: {seconds:.1f}sn bekleniyor (rpm/tpm/cooldown sınırı)")


# Görev başına muhafazakâr token tahmini (girdi + çıktı), TPM throttle'ı için.
# ÖLÇÜLDÜ, uydurulmadı: runs/benchmark/*/results.jsonl'deki input_tokens +
# output_tokens medyanlarının üstüne yuvarlandı (2026-07-31, 429 satırları hariç).
#   cleaning  6.288 (gemini-3.6-flash) · spec_menu 1.380-4.817 · estimand 2.408
#   narrative 867-1.861
# Neden girdi+çıktı: Google TPM'i yalnız GİRDİ sayıyor (rate-limits dok.), ama
# ikisini birden saymak yanlış yönde hata yapar — fazladan beklemek 429'dan ucuz.
# Koşu sırasında gerçek ölçüm bu tahmini aşarsa Throttle onu kullanır (bkz. observe).
TASK_TOKEN_ESTIMATE: dict[str, int] = {
    "cleaning": 7000,
    "spec_menu": 5000,
    "estimand": 3000,
    "narrative": 2500,
}
DEFAULT_TOKEN_ESTIMATE = 7000

# Bir vakayı onaylarken tavanda bırakılacak fazladan çağrı payı (bkz. case_server).
# Şema retry'ı `acquire`'dan SONRA, `_call_and_score` içinde gidiyor: vakanın son
# hücresine gelindiğinde önceki hücrelerin retry'ları tavanı doldurmuş olmamalı.
# Koşul: (needed - 1) * (1 + r) < needed + margin, burada r = bir hücrenin
# ürettiği EKSTRA istek sayısı. runs/benchmark/tur-1'de görülen en büyük değer
# requests=4 (gpt-oss-20b/120b), yani r=3; needed=2 için margin > 2 çıkıyor.
# Teslim setinde retry hiç görülmedi (70 satırın hepsi requests=1), pay yine de
# ölçülen en kötü hale göre seçildi — bütçe şansa değil doğruya dayanmalı.
RETRY_MARGIN = 3


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

    Tavan bu SINIFTA bir koşu içindir; takvim günü bilgisi burada yok. Hangi eski
    çağrının bugünün kotasından sayılacağına `seed_from_prior()` karar verir —
    ayrım orada, çünkü havuzun günlük mü (`rpd`) ömür-boyu mu (`budget`) olduğunu
    bilmek `models.json` bilgisi gerektirir, saat değil.
    """

    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    # Enjekte edilebilir: testler beklemeyi sessizce yakalar, üretim varsayılanı
    # terminale basar. `now`/`sleep` ile aynı gerekçe — gerçek I/O'yu testten ayır.
    on_wait: Callable[[str, float], None] = _print_rpm_wait
    # Çağrılar arası TABAN bekleme, saniye. 0 = kapalı (rpm/tpm ne diyorsa o).
    # Açıkken aralık başlangıçtan değil CEVAPTAN sayılır (bkz. release): serbest
    # katmanda sağlayıcıyı yormamak için istenen davranış "cevabı bekle, sonra
    # bu kadar daha bekle". rpm/tpm aralığı yine tabanla birlikte geçerli, büyüğü
    # bağlar.
    cooldown: float = 0.0
    _last_call: dict[str, float] = field(default_factory=dict)
    _count: dict[str, int] = field(default_factory=dict)
    _observed: dict[str, int] = field(default_factory=dict)

    def seed(self, pool: str, already_done: int) -> None:
        """Resume: önceki koşudan gelen çağrılar havuz kotasından düşülür.

        Şema retry'ı da buradan işlenir: pydantic-ai'nin retry'ı modele YENİ BİR
        İSTEK gider ve sağlayıcının kotasından düşer, ama `acquire` hücre başına
        bir kez çağrılıyor. Fark işlenmezse tavan sessizce aşılır (bkz.
        run_matrix: meter.requests - 1).
        """
        self._count[pool] = self._count.get(pool, 0) + already_done

    def observe(self, task: str, tokens: int) -> None:
        """Bir görevde gerçekten harcanan token; sonraki çağrının tahminini besler.

        TASK_TOKEN_ESTIMATE ölçülmüş ama ESKİ bir koşudan geliyor; prompt veya
        model değişince gerçek maliyet büyüyebilir. Gördüğümüz en büyük değeri
        saklamak, throttle'ın tahmine değil ölçüme yaklaşmasını sağlar.
        """
        if tokens > self._observed.get(task, 0):
            self._observed[task] = tokens

    def estimate(self, task: str) -> int:
        """Bu görevin bir sonraki çağrısı için token tahmini (tahmin vs ölçüm: büyüğü)."""
        return max(
            TASK_TOKEN_ESTIMATE.get(task, DEFAULT_TOKEN_ESTIMATE),
            self._observed.get(task, 0),
        )

    def used(self, pool: str) -> int:
        return self._count.get(pool, 0)

    def remaining(self, pool: str, cap: int | None) -> int | None:
        """Havuzda kalan çağrı hakkı; tavan yoksa None (= sınırsız).

        Vaka bazlı sunucu seçimi bunu ister: bir vakanın TÜM tekrarları aynı uçtan
        gitmek zorunda (bkz. case_server), dolayısıyla çağrı çağrı `acquire`
        denemek yetmez — vakaya başlamadan önce yeterli hak olduğunu bilmek gerekir.
        """
        if cap is None:
            return None
        return max(cap - self._count.get(pool, 0), 0)

    def acquire(
        self,
        pool: str,
        *,
        rpm: int | None,
        cap: int | None,
        tpm: int | None = None,
        est_tokens: int = 0,
    ) -> None:
        used = self._count.get(pool, 0)
        if cap is not None and used >= cap:
            raise QuotaExhausted(f"{pool}: kota doldu ({used}/{cap})")
        interval = max(self.interval(rpm=rpm, tpm=tpm, est_tokens=est_tokens), self.cooldown)
        if interval > 0:
            last = self._last_call.get(pool)
            if last is not None:
                wait = interval - (self.now() - last)
                if wait > 0:
                    self.on_wait(pool, wait)
                    self.sleep(wait)
        self._last_call[pool] = self.now()
        self._count[pool] = used + 1

    def release(self, pool: str) -> None:
        """Çağrı bitti: cooldown açıksa damgayı CEVAP anına taşı.

        `acquire` damgayı isteği göndermeden önce basar, yani varsayılan aralık
        başlangıçtan başlangıcadır ve modelin cevap süresi aralığın İÇİNDEN sayılır.
        Cooldown modunda istenen bu değil: cevap ne kadar sürerse sürsün üstüne tam
        bir taban bekleme konmalı. Damgayı burada yenilemek aralığı bitişten
        başlangıca çevirir.

        Cooldown kapalıyken hiçbir şey yapmaz — mevcut rpm/tpm davranışı aynen kalır.
        """
        if self.cooldown:
            self._last_call[pool] = self.now()

    @staticmethod
    def interval(*, rpm: int | None, tpm: int | None = None, est_tokens: int = 0) -> float:
        """İki hız kısıtından bağlayıcı olanın gerektirdiği istekler-arası saniye.

        Bazı uçlarda bağlayıcı kısıt istek DEĞİL token: gemma-4-31b-it'in rpd'si
        14.400 ama tpm'i 16.000, yani ~6.3K'lık bir cleaning çağrısında dakikada
        ancak ~2 istek kaldırıyor. rpm=30'a bakan bir throttle 2sn aralıkla
        gider ve ilk dakikada 429 yer (canlı kanıt: 2026-07-30, 7 çağrı).
        """
        by_rpm = 60.0 / float(rpm) if rpm else 0.0
        by_tpm = 60.0 * float(est_tokens) / float(tpm) if (tpm and est_tokens) else 0.0
        return max(by_rpm, by_tpm)


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


def ship_matrix(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`ship: true` adaylar + onların kota uzantıları (dosya sırası korunur).

    Uzantı ELLE eklenmemeli: `fallback_only` bir model kendi satırlarını üretmez
    ama tavanı birincilin havuzuna eklenir (bkz. plan_lines). Listeden düşerse
    `--dry-run` gemini havuzunu 40 değil 20 sanar ve teslim matrisi olduğundan
    dar görünür — planı yanlış kuran tam olarak bu.
    """
    shipped = [m for m in models if m.get("ship")]
    extenders = {str(m.get("fallback_id")) for m in shipped if m.get("fallback_id")}
    keep = {m["id"] for m in shipped} | extenders
    return [m for m in models if m["id"] in keep]


def quota_cap(model: dict[str, Any]) -> int | None:
    """Havuzun tavanı: günlük istek sayısı veya tek seferlik kredi bütçesi."""
    if model.get("budget") is not None:
        return int(model["budget"])
    if model.get("rpd") is not None:
        return int(model["rpd"])
    return None


def local_date() -> str:
    """Kota gününün etiketi (yerel takvim tarihi).

    Tek yerde toplandı ki testler/simülasyonlar günü değiştirebilsin — gün geçişi
    bu koşucunun en kritik davranışı ve gerçek gece yarısını bekleyerek test edilemez.
    Sağlayıcıların sıfırlama saati garanti değil; gün granülasyonu yeterli yaklaşım.
    """
    return time.strftime("%Y-%m-%d")


def is_daily_pool(model: dict[str, Any]) -> bool:
    """Havuz her gün yenilenen bir hak mı (`rpd`), yoksa ömür-boyu kredi mi (`budget`)?

    `quota_cap()` ikisini de tek sayıya indiriyor, ama resume'da davranışları
    ZITTIR: dünkü NVIDIA kredisi gerçekten harcandı (bir daha gelmiyor), dünkü
    Gemini isteği ise bugünün 20'sinden düşmez. Bu ayrımı yapmamak koşuyu ilk
    günden sonra kalıcı olarak durduruyordu.
    """
    return model.get("budget") is None and model.get("rpd") is not None


def seed_from_prior(
    throttle: Throttle,
    prior_rows: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    today: str,
) -> None:
    """Önceki koşuların çağrılarını havuz sayaçlarına işler.

    Satır bazlı, model listesi bazlı DEĞİL: `fallback_id` ile gelen satırlar
    birincil modelin kimliğiyle yazılıyor, ama kotayı `served_by`'daki uç harcadı.
    Havuzu satırın kendi `served_by`'ından çözmek bunu doğru sayan tek yol.

    Günlük havuzlarda yalnız BUGÜNÜN satırları sayılır (`date` alanı); ömür-boyu
    kredi havuzlarında hepsi. `date`'i olmayan eski satırlar günlük havuzlarda
    yok sayılır — bu yönde hata yapmak, koşuyu kalıcı olarak durdurmaktansa en
    kötü halde sağlayıcıdan bir 429 yemeye yol açar (o da satır olarak raporlanır).

    Reddedilmiş çağrılar (429/503/404) kota TÜKETMEDİ ve tekrar denenecekler
    (bkz. is_retryable_error) — bunları saymak koşuyu gereksiz kısıtlar.
    """
    for row in prior_rows:
        if is_retryable_error(row.get("error")):
            continue
        served = lookup.get(str(row.get("served_by") or row.get("model_id") or ""))
        if served is None:
            continue
        if is_daily_pool(served) and row.get("date") != today:
            continue
        throttle.seed(quota_pool(served), 1)


def case_server(
    throttle: Throttle,
    model: dict[str, Any],
    *,
    needed: int,
    pinned_id: str | None,
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Bir vakanın TÜM kalan çağrılarını koşacak tek ucu seçer; yoksa None.

    Neden vaka bazlı: kota sınırı vaka ortasına düşerse aynı vakanın tekrarları
    farklı modellere dağılır ve `answer_consistency` model-içi tutarlılık değil
    iki-model-uyuşması ölçmeye başlar (aynı şey görev bazlı okuma için de geçerli).
    O yüzden vakaya, tüm tekrarlarını karşılayacak hak yoksa hiç başlanmaz.

    `pinned_id`: bu vakanın bir kısmı ÖNCEKİ koşuda zaten koşmuşsa, kalanı da aynı
    uçtan gitmek zorunda — vaka saflığı günler arasında da korunur. Pinli uçta yer
    yoksa vaka bu koşuda hiç koşmaz, ertesi güne kalır (fallback'e KAYDIRILMAZ).

    Seçim İKİ TURLU: önce `RETRY_MARGIN` payı olan bir uç aranır, hiçbiri yoksa
    pay şartı düşürülüp yalnız `needed` aranır. Tek turlu (paysız) seçim, tavanı
    `needed`'ın tam katı olan dar havuzlarda vakayı SIFIR payla onaylıyordu: tek
    bir şema retry'ı vaka ortasında tavanı doldurup `acquire`'ı QuotaExhausted'a
    düşürüyordu (ölçüldü: gemini-3.6-flash tavan 20, 10. vaka, hücre 19). Tek
    turlu yapıp payı zorunlu kılmak ise havuzun son artığını hiç kullanmazdı —
    ikinci tur o artığı kullanır, mid-case dolma riskini `run_matrix`'teki
    QuotaExhausted yakalaması karşılar.

    None dönmesi "bu havuz bitti" demektir; çağıran modelin kalanını atlar.
    """
    candidates = [model]
    fallback_id = model.get("fallback_id")
    fallback = lookup.get(str(fallback_id)) if fallback_id else None
    if fallback is not None:
        candidates.append(fallback)  # tek seviye: fallback'in fallback'i izlenmez
    if pinned_id:
        candidates = [c for c in candidates if c["id"] == pinned_id]

    for margin in (RETRY_MARGIN, 0):
        for candidate in candidates:
            left = throttle.remaining(quota_pool(candidate), quota_cap(candidate))
            if left is None or left >= needed + margin:
                if candidate is not model:
                    print(
                        f"  [kota] {model_key(model)} bu vakaya yetmiyor, "
                        f"{needed} çağrı {candidate['id']}'e veriliyor"
                    )
                return candidate
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

    # Kolon adı taşıyan alanlar `treatment` ve `outcome`. `treatment_coding`
    # kodlamanın serbest metin tarifi ("1 = genişleyen eyalet x post, 0 = diğer");
    # onu kolon listesine karşı sınamak her doğru öneriyi "uydurma kolon"
    # damgalar (ölçüldü: tur-1'de 70 puanlanabilir satırın 69'u).
    hallucinated = [
        name for name in (proposal.treatment, proposal.outcome) if name and name not in known
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
        treatment_ok = proposal.treatment in accept_t if accept_t else None
        outcome_ok = proposal.outcome in accept_o if accept_o else None

    return {
        "treatment_ok": treatment_ok,
        "outcome_ok": outcome_ok,
        "proposed_treatment": proposal.treatment,
        # Kodlama tarifi de kaydediliyor: `treatment` düzeltmesinden sonra eski
        # koşularla karışmasın diye ayrı anahtar, ve tarif alanı gerileme
        # yaparsa (kolon adı yazmaya başlarsa) ham satırdan görülebilsin.
        "proposed_treatment_coding": proposal.treatment_coding,
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
        identification_assumption=str(estimand.get("identification_assumption", "parallel_trends")),
        outcome=estimand["outcome"],
        # `treatment` = kolon adı. Bu satır önceden `treatment_coding` okuyordu ve
        # yalnız altın kayıt iki alanı ters doldurduğu için çalışıyordu; ikisi
        # birlikte düzeltildi (bkz. benchmarks/gold/spec_menu.json).
        treatment=estimand["treatment"],
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


@contextmanager
def judge_call_settings(
    model: dict[str, Any], *, thinking_on: bool, timeout: float
) -> Iterator[None]:
    """Bu modelin sağlayıcı slotuna geçici bir `timeout` uygular (+ varsa NVIDIA reasoning).

    Kod tabanında (`pareto/llm/*.py`) HİÇBİR LLM çağrısında timeout yok — hem NVIDIA
    hem Google free-tier uçları preflight'ı süresiz asabiliyor (ikisi de ESTAB
    bağlantıda `ep_poll`'da beklerken gözlendi, 2026-07-30). Bu fonksiyon,
    sağlayıcı ne olursa olsun, en azından bir üst sınır
    koyarak "sessizce sonsuza kadar asılı kal"ı "temiz, bilgilendirici hata"ya çevirir.

    NVIDIA'nın "thinking" modelleri (deepseek-v4-pro, nemotron-ultra, inkling — bkz.
    build.nvidia.com model sayfaları, 2026-07-30) ayrıca `chat_template_kwargs`/
    `reasoning_effort` alanı gönderilmezse kendi varsayılan reasoning derinliğiyle
    çalışır. `models.json:reasoning_control` yalnız bu 3 modelde dolu; diğer
    sağlayıcılarda/modellerde (glm-5.2, minimax-m3, Google, Groq, OpenRouter) no-op
    — yalnız `timeout` uygulanır.

    Neden `providers.JUDGE_NVIDIA_SLOT` gibi modül-seviyesi adı değil
    `_JUDGE_SLOTS_BY_PROVIDER`'ı değiştiriyoruz: o sözlük import anında donuyor
    (bkz. providers.py:243), yani slotu yeniden atamak `judge_slots_for()`'un
    okuduğu girdiyi değiştirmez.

    pydantic-ai'nin genel `ModelSettings.thinking` alanı NVIDIA için İŞE YARAMAZ:
    yalnız `openai_supports_reasoning` profili bilinen modellerde `reasoning_effort`e
    çevriliyor, NVIDIA NIM'deki topluluk model ID'leri bu profile girmiyor (bkz.
    pydantic_ai.models.openai._translate_thinking) — bu yüzden `extra_body`
    doğrudan gönderiliyor.
    """
    key = "on" if thinking_on else "off"
    extra_body = (model.get("reasoning_control") or {}).get(key)
    extra_model_settings: dict[str, Any] = {"timeout": timeout}
    if extra_body:
        extra_model_settings["extra_body"] = extra_body

    provider = model["provider"]
    original = llm_providers._JUDGE_SLOTS_BY_PROVIDER[provider]
    llm_providers._JUDGE_SLOTS_BY_PROVIDER[provider] = replace(
        original, extra_model_settings=extra_model_settings
    )
    try:
        yield
    finally:
        llm_providers._JUDGE_SLOTS_BY_PROVIDER[provider] = original


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
    # 5xx: sağlayıcı tarafı. `kota`'dan SONRA gelmeli — gövdesinde "quota" geçen bir
    # 503 kota olarak sınıflanmalı (kota mesajı daha bağlayıcı bilgi). Canlı örnek:
    # gemini-3.5-flash "This model is currently experiencing high demand" (2026-07-30).
    if any(s in text for s in ("500", "502", "503", "504", "overloaded", "high demand")):
        return "sunucu"
    # 413: istek modelin kapasitesini aşıyor. `sunucu`/`kota`dan AYRI bir sınıf çünkü
    # KALICI ve deterministik: payload küçülmediği sürece her denemede aynı sonucu
    # verir (canlı örnek: llama-3.1-8b-instant · cleaning, 2026-07-31). Bu yüzden
    # tekrar denenmez (`_INFRA_ERRORS`'ta değil) ama koşulsuz elemeye sayılır —
    # gecikme kapısına takılırsa 12 çağrının hepsi aynı 413'ü yer.
    if any(s in text for s in ("413", "request too large", "too large for model")):
        return "kapasite"
    # Ayrıştırılamayan structured output = ŞEMA kusuru, taşıyıcısı ne olursa olsun.
    # Groq bunu HTTP 400 + "Parsing failed. The model generated output that could not
    # be parsed" ile döndürüyor (canlı örnek: gpt-oss-20b, 2026-07-31), pydantic-ai
    # ise retry tükenmesiyle → `sema_tutmadi`. Aynı kusur, iki farklı yol. Ayırmazsak
    # `hata:*` sınıfına düşüyor ve İKİ yanlış sonuç doğuruyor: schema_ok=True
    # (başarısızlık BAŞARI sayılır) ve elemeye koşulsuz sayılma (şema hatası transport
    # gibi davranır). Altyapı kontrollerinden SONRA duruyor ki gerçek 429/503
    # sinyalleri öncelik korusun.
    # `output_parse_failed` Groq'un makine-okunur `code` alanı — İngilizce mesaj
    # metninden daha sağlam imza, o yüzden listede ilk sırada.
    if any(
        s in text
        for s in (
            "output_parse_failed",
            "parsing failed",
            "could not be parsed",
            "failed_generation",
        )
    ):
        return "sema_tutmadi"
    if isinstance(exc, ValueError):
        return "dogrulayici_reddetti"
    return f"hata:{name}"


# Modele hiç ulaşamadığımız hatalar: şema hakkında BİLGİ vermezler.
# Bunlarda schema_ok=True demek "şemayı tutturdu" diye okunur ve yanlıştır.
# Aynı sebeple bunlar modelin kusuru DEĞİL: ne devre kesiciyi tetiklerler ne de
# matris hücresini "ölçüldü" sayarlar (bkz. is_retryable_error).
_INFRA_ERRORS = frozenset({"anahtar_yok", "kota", "model_yok", "yetki", "sunucu"})

# Modelin KENDİ kusuru olan hatalar: bunlar tam olarak ölçmek istediğimiz şey.
_MEASURED_ERRORS = frozenset({"sema_tutmadi", "dogrulayici_reddetti"})


def is_retryable_error(kind: str | None) -> bool:
    """Bu hata matris hücresini "ölçüldü" saymamalı mı (sonraki koşuda tekrar denensin)?

    Geçici bir 429/503 hücreyi kalıcı olarak zehirlerse o model o vakayı BİR DAHA
    hiç koşmaz ve rapor eksik n'i sessizce final okuma gibi gösterir. Modelin kendi
    kusurları (şema/doğrulayıcı) ise tekrar DENENMEMELİ — ikinci çekilişte geçmesi
    ölçümü yumuşatır.
    """
    return kind is not None and kind in _INFRA_ERRORS


def counts_toward_elimination(kind: str | None) -> bool:
    """Bu hata sınıfı KOŞULSUZ elemeye sayılır mı? (transport/`hata:*`)

    Yalnız "uca hiç ulaşamıyoruz" sınıfı. Burada gecikme kapısı YOK ve olmamalı:
    `nvidia:hesap`'ta hızlı başarısızlık da çağrı başına bir kredi yakıyor ve kredi
    yenilenmiyor — 2 çağrıda durmak 48'de durmaktan çok farklı.
    Gecikme kapısı yalnız altyapı sınıfına ait, çünkü orada 429 gerçekten bedava.

    - Altyapı hataları: sağlayıcı kaynaklı, model masum, hızlıysa maliyeti ~0,5sn.
      Canlı kanıt: gemma-4-31b-it 3 başarılı çağrıdan sonra 429 yedi, dakikalar
      sonra aynı uç HTTP 200 döndü (2026-07-30) — elenmesi yanlıştı.
    - Şema/doğrulayıcı hataları: benchmark'ın ÖLÇTÜĞÜ şey. Eleyerek susturursak
      benchmark kendi sorusunu cevaplayamaz.
    """
    return kind is not None and kind not in _INFRA_ERRORS and kind not in _MEASURED_ERRORS


def elimination_signal(kind: str | None, latency_s: float) -> bool:
    """Bu çağrı devre kesici sayacını artırmalı mı?

    İki yol: koşulsuz transport sınıfı, VEYA yavaş bir altyapı hatası (504 @ 181sn
    gibi — etiketi "altyapı" ama tek modelde 48 × 180sn = 2,4 saat eder).

    Ölçüm hataları hiçbir gecikmede saymaz. Canlı kanıt (2026-07-31):
    gpt-oss-120b `cleaning`'de 9/9 `sema_tutmadi` verdi, bazıları 72-90sn sürdü —
    ama yavaşlığın SEBEBİ ölçülen kusurun kendisi (pydantic-ai şema retry'ı,
    `requests=2..4`), uç asılması değil. Bunu elemeye saymak "şema hataları
    ölçümdür" kuralını arka kapıdan iptal ediyordu.
    """
    if counts_toward_elimination(kind):
        return True
    return kind in _INFRA_ERRORS and latency_s >= SLOW_FAILURE_S


def resets_streak(kind: str | None) -> bool:
    """Bu çağrı ardışıklık sayacını sıfırlamalı mı?

    Ölçüt: uç GERÇEKTEN bir model cevabı üretti mi. Başarı ve ölçüm hatası
    (şema/doğrulayıcı reddi) ikisi de üretti — uç canlı, sayaç sıfırlanır.
    Hızlı altyapı hatası ne sayar ne sıfırlar: iki timeout arasına düşen bir 429
    seriyi bozmamalı.

    Sıfırlamayı atlamak "ardışık" kelimesini yalan yapıyordu: 72sn'lik bir hata,
    araya 3 ölçüm hatası girdikten sonra gelen 90sn'lik hatayla "2 ardışık"
    sayılıp gpt-oss-120b'nin cleaning görevini haksız yere kesti (2026-07-31).
    """
    return kind is None or kind in _MEASURED_ERRORS


def schema_verdict(error_kind: str | None) -> bool | None:
    """Şema kararı: geçti / tutmadı / hakkında bilgi yok (None).

    Varsayılan `None` — yani "kanıt yok". Beyaz liste yerine kara liste kullanmak
    (bilinmeyen her sınıfı True saymak) birincil metriği sessizce şişiriyordu:
    413 ve `hata:ConnectError` gibi MODELE HİÇ ULAŞMAMIŞ çağrılar "şema geçti"
    diye sayılıyordu. Yeni sınıf eklendiğinde de güvenli tarafta kalır.
    """
    if error_kind is None:
        return True
    if error_kind == "sema_tutmadi":
        return False
    if error_kind == "dogrulayici_reddetti":
        # Şema TUTTU; reddedilen şey içerik (ör. kombinatoryal bütçe, uydurma kolon).
        return True
    return None


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
        settings = judge_call_settings(model, thinking_on=False, timeout=60)
        with pinned_model(model), settings, metered(meter):
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


def read_many(paths: list[Path]) -> list[dict[str, Any]]:
    """Birden çok koşuyu tek rapor için okur; hücreler koşular arasında KARIŞMAZ.

    `result_id` koşu tarihini içermiyor (`model|task|case_id|repeat`), yani
    `latest_per_cell` iki farklı koşunun aynı hücresini tek satıra indirirdi:
    3-tekrarlı eski koşu ile 2-tekrarlı yeni koşu sessizce birleşir ve rapor
    EKSİK değil YANLIŞ sayı üretir. Kaynak dosya yolunu `result_id`'ye önek
    yapmak dedupe'u koşunun içinde tutar; `source` alanı da hangi satırın
    nereden geldiğini raporda göstermeye yarar.

    YALNIZ RAPOR YOLU. Çıktısı `run_matrix`'e VERİLMEZ: önekli `result_id`,
    resume'un "bu hücre yapıldı" kontrolüyle (bkz. run_matrix: done) eşleşmez,
    yani tüm matris yapılmamış görünür ve kota baştan harcanır.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        src = str(path)
        for row in read_rows(path):
            row["source"] = src
            if row.get("result_id") is not None:
                row["result_id"] = f"{src}::{row['result_id']}"
            rows.append(row)
    return rows


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


def latest_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aynı matris hücresinin birden fazla satırı varsa SON'unu tutar (sıra korunur).

    Geçici altyapı hatası alan hücre sonraki koşuda tekrar denenir ve aynı
    `result_id` ile ikinci bir satır yazılır (bkz. is_retryable_error). İkisini de
    saymak `n`'i şişirir ve başarı oranını olduğundan düşük gösterir: 429 yemiş bir
    çağrı, modelin bir başarısızlığı gibi okunur. Ham satırlar `results.jsonl`'de
    kalır — burada yalnız rapor okuması düzeltilir.
    """
    keep: dict[str, int] = {}
    for index, row in enumerate(rows):
        rid = row.get("result_id")
        if rid is None:
            continue
        keep[str(rid)] = index
    kept = set(keep.values())
    return [row for i, row in enumerate(rows) if i in kept or row.get("result_id") is None]


def _call_and_score(
    model: dict[str, Any],
    task: str,
    case: dict[str, Any],
    *,
    rid: str,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    extra: dict[str, Any],
    served: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tek çağrının ortak iskeleti: pin -> ölç -> puanla -> hatayı sınıflandır.

    Normal tekrar döngüsü ve --order-check aynı iskeleti paylaşır; ayrılırlarsa
    order-check satırları farklı bir ölçüm yoluna gider ve ana satırlarla
    karşılaştırılamaz hale gelir.

    `model` satırın KİMLİĞİ, `served` çağrıyı fiilen karşılayan uç (varsayılan:
    ikisi aynı). İkisi `fallback_id` eşlemesinde ayrışır — pin/reasoning/kota
    `served`'a, rapor başlığı `model`'e gider. `served_by` alanı bu ayrışmayı
    results.jsonl'de görünür tutar; olmazsa rapor sessizce iki ucu karıştırırdı.
    """
    served = served or model
    meter = Meter()
    started = time.monotonic()
    row: dict[str, Any] = {
        "result_id": rid,
        "model": model_key(model),
        "provider": model["provider"],
        "model_id": model["id"],
        "served_by": served["id"],
        # Günlük kotanın hangi güne yazıldığı (bkz. seed_from_prior).
        "date": local_date(),
        "task": task,
        "case_id": case["case_id"],
        **extra,
    }
    try:
        settings = judge_call_settings(served, thinking_on=True, timeout=180)
        with pinned_model(served), settings, metered(meter):
            scores = runner(case)
        row.update(schema_ok=True, validator_passed=True, error=None, scores=scores)
    except Exception as exc:  # noqa: BLE001 — sessiz atlama yok
        kind = classify_error(exc)
        row.update(
            schema_ok=schema_verdict(kind),
            validator_passed=False,
            error=kind,
            # 1200: 400'de Google'ın 429 gövdesi tam kota metriğinin adından ÖNCE
            # kesiliyordu ("Quota exceeded for metric: ...generat"), yani hangi limitin
            # aşıldığını (istek/dakika mı token/dakika mı) söyleyen tek bilgi kayboluyordu.
            error_detail=str(exc)[:1200],
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

    Kota birimi VAKA'dır, çağrı değil: bir vakanın tüm kalan çağrıları (tekrarlar
    + varsa order-check varyantı) tek uçtan gider ya da vakaya hiç başlanmaz
    (bkz. case_server). Bu yüzden havuzda tekrar sayısından küçük bir artık
    kalabilir — o artık bilinçli olarak kullanılmaz.

    `CIRCUIT_BREAK_ERRORS` ardışık hata gören model elenir ve kalan çağrıları
    atlanır. Gerekçe: NVIDIA'nın ~1.000 kredisi tek seferlik, ölü bir uç 48 çağrı
    boyunca kredi ve saat yakabilir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    prior = read_rows(results_path)
    # Geçici altyapı hatası alan hücre "yapıldı" SAYILMAZ: yoksa bir 429 o vakayı
    # kalıcı olarak siler ve rapor eksik n'i final okuma gibi gösterir.
    done = {
        row["result_id"]: 1
        for row in prior
        if row.get("result_id") and not is_retryable_error(row.get("error"))
    }
    # `fallback_id` hedefi --models/--providers filtresiyle `models`'ten düşmüş
    # olabilir; katalog tam dosyadan gelir, çağıranın dictleri (testler dahil) üstün.
    lookup: dict[str, dict[str, Any]] = {m["id"]: m for m in load_models()}
    lookup.update({m["id"]: m for m in models})
    # Yarım kalmış vakanın kalanı da AYNI uçtan gitmeli (vaka saflığı günler arası).
    served_of: dict[tuple[str, str, str], str] = {}
    for row in prior:
        ident = (str(row.get("model")), str(row.get("task")), str(row.get("case_id")))
        served = str(row.get("served_by") or row.get("model_id") or "")
        if served and all(ident):
            served_of.setdefault(ident, served)

    seed_from_prior(throttle, prior, lookup, local_date())

    rows: list[dict[str, Any]] = []
    for model in models:
        if model.get("fallback_only"):
            print(f"  [atla] {model_key(model)}: yalnız kota uzantısı, kendi başına ölçülmez")
            continue
        key = model_key(model)
        stop = False  # havuz/kota bitti -> bu modelin TÜM görevleri
        for task in tasks:
            if stop:
                break
            # Devre kesici sayacı her görevde sıfırdan başlar (bkz. CIRCUIT_BREAK_ERRORS).
            consecutive_errors = 0
            task_broken = False
            for case in load_gold(task):
                if stop or task_broken:
                    break
                case_id = case["case_id"]
                plan: list[tuple[int | str, Callable[..., Any], dict[str, Any]]] = [
                    (r, TASK_RUNNERS[task], {"repeat": r}) for r in range(repeats)
                ]
                if order_check and task in ORDER_CHECK_RUNNERS:
                    plan.append(
                        (
                            ORDER_CHECK_VARIANT,
                            ORDER_CHECK_RUNNERS[task],
                            {"repeat": ORDER_CHECK_VARIANT, "order_variant": "reversed"},
                        )
                    )
                todo = [p for p in plan if result_id(model, task, case_id, p[0]) not in done]
                if not todo:
                    continue

                server = case_server(
                    throttle,
                    model,
                    needed=len(todo),
                    pinned_id=served_of.get((key, task, case_id)),
                    lookup=lookup,
                )
                if server is None:
                    print(
                        f"  [kota] {key}: {task}/{case_id} için {len(todo)} çağrılık hak "
                        "kalmadı — bu modelin kalanı sonraki koşuya bırakılıyor"
                    )
                    stop = True
                    break

                for variant, runner, extra in todo:
                    rid = result_id(model, task, case_id, variant)
                    # `case_server` `needed` kadar hak doğruladı, ama şema retry'ı
                    # `acquire`'dan SONRA gidiyor: önceki hücrelerin retry'ları
                    # tavanı vaka ortasında doldurabilir. Yani burası invariant
                    # ihlali DEĞİL, ulaşılabilir bir durum — RETRY_MARGIN payı
                    # nadirleştirir, tüketmez (dar havuzda payın kendisi de biter).
                    # Yutmamak tüm koşuyu öldürürdü: teslim koşusunda gemini'nin
                    # 19. hücresindeki tek bir retry gemma'yı ve inkling'i hiç
                    # koşturmadan traceback'e düşürüyordu. Sağlayıcının 429'uyla
                    # (aşağıdaki `kind == "kota"`) aynı davranış: modeli durdur,
                    # diğer modeller sürsün, kalanı sonraki koşuya kalsın.
                    try:
                        throttle.acquire(
                            quota_pool(server),
                            rpm=server.get("rpm"),
                            cap=quota_cap(server),
                            tpm=server.get("tpm"),
                            est_tokens=throttle.estimate(task),
                        )
                    except QuotaExhausted as exc:
                        print(
                            f"  [kota] {key}: {exc} — vaka ortasında doldu (retry payı "
                            "yetmedi); bu modelin kalanı sonraki koşuya bırakılıyor"
                        )
                        stop = True
                        break
                    try:
                        row = _call_and_score(
                            model, task, case, rid=rid, runner=runner, extra=extra, served=server
                        )
                    finally:
                        # `finally`: çağrı istisnayla düşerse damga acquire anında
                        # kalır ve sonraki çağrı kısalmış bir aralık öder — tam da
                        # işler kötü giderken sağlayıcıyı hızlandırmak olurdu.
                        throttle.release(quota_pool(server))
                    # Şema retry'ı sağlayıcıya ayrı bir istek olarak gitti ve kotadan
                    # düştü; `acquire` yalnız birini saydı. Farkı işlemezsek tavan
                    # sessizce aşılır (bkz. Throttle.seed).
                    throttle.seed(quota_pool(server), max(int(row.get("requests") or 1) - 1, 0))
                    throttle.observe(
                        task, int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
                    )
                    kind = row.get("error")
                    if elimination_signal(kind, float(row.get("latency_s") or 0.0)):
                        consecutive_errors += 1
                        if consecutive_errors >= CIRCUIT_BREAK_ERRORS:
                            row["circuit_broken"] = True
                    elif resets_streak(kind):
                        consecutive_errors = 0
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    if not is_retryable_error(kind):
                        done[rid] = 1
                    served_of.setdefault((key, task, case_id), server["id"])

                    flag = "ok" if kind is None else kind
                    via = "" if server is model else f" (via {server['id']})"
                    label = "order-reversed" if variant == ORDER_CHECK_VARIANT else f"#{variant}"
                    print(f"  {key}{via} · {task}/{case_id}{label} → {flag}")
                    if row.get("circuit_broken"):
                        print(
                            f"  [devre kesici] {key} · {task}: {CIRCUIT_BREAK_ERRORS} ardışık "
                            "pahalı hata — bu GÖREVİN kalanı atlanıyor, diğer görevler sürüyor"
                        )
                        task_broken = True
                        break
                    if kind == "kota":
                        # Sağlayıcı "havuz bitti" diyor; Throttle ne sanıyorsa sansın,
                        # iterasyona devam etmek yalnızca hızlı 429 dizisi üretir.
                        # ELEME DEĞİL: hücre retry edilebilir kalır, model masum.
                        print(
                            f"  [kota] {key}: sağlayıcı 429 döndü — bu modelin kalanı "
                            "sonraki koşuya bırakılıyor"
                        )
                        stop = True
                        break
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


def _measured_total(rows: list[dict[str, Any]], field: str) -> int | None:
    """Toplam; hiçbir satırda ölçüm yoksa `None`.

    `sum(... or 0)` ÖLÇÜLMEDİ ile SIFIR'ı aynı hücreye düşürüyordu: CLI referans
    koşusunda retry/token bilerek `None` yazılıyor (harness içinden ölçülemez),
    tablo ise "Retry: 0" / "0 token" basıyor ve okuyan "retry olmadı" anlıyor.
    Rapor önsözündeki uyarı tabloya taşınmıyor, o yüzden hücrenin kendisi
    ölçümsüzlüğü söylemeli: tablo `None`'ı diğer ölçülmemiş sütunlarla aynı
    biçimde `-` basıyor.
    """
    measured = [r[field] for r in rows if r.get(field) is not None]
    return sum(int(v) for v in measured) if measured else None


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
        "retries": _measured_total(rows, "retries"),
        "latency_median": (statistics.median(lat) if lat else None),
        "latency_p95": _percentile(lat, 95),
        "tokens_out_median": (statistics.median(out_tokens) if out_tokens else None),
        "tokens_in_total": _measured_total(rows, "input_tokens"),
        "tokens_out_total": _measured_total(rows, "output_tokens"),
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


def _served_by_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Bir modelin satırları birden fazla uçtan geldiyse bunu AÇIKÇA yazar.

    `fallback_id` eşlemesi (bkz. models.json) satırları birincil modelin başlığı
    altında topluyor. Bunu raporda söylemezsek tablo, tek bir ucun 48 çağrısı gibi
    okunur — oysa skorlar iki uca dağılmış olabilir. Eşleme yoksa bölüm basılmaz.
    """
    split: dict[str, dict[str, int]] = {}
    for row in rows:
        served = str(row.get("served_by") or row.get("model_id") or "")
        counts = split.setdefault(str(row.get("model")), {})
        counts[served] = counts.get(served, 0) + 1
    mixed = {name: c for name, c in split.items() if len(c) > 1}
    if not mixed:
        return []
    lines = [
        "",
        "## Çağrıyı karşılayan uçlar (kota uzantısı)",
        "",
        "Aşağıdaki başlıklar TEK model değil, kotayı paylaşan eşlenmiş uçlardır",
        "(models.json: fallback_id). Skorlar bu uçların BİRLEŞİMİDİR.",
        "",
        "| Rapor başlığı | Uç | Çağrı |",
        "|---|---|---|",
    ]
    for name in sorted(mixed):
        for served, count in sorted(mixed[name].items()):
            lines.append(f"| `{name}` | `{served}` | {count} |")
    return lines


# Görev kırılımının basıldığı en fazla model sayısı. Üstünde tablo okunmaz hale
# gelir ve zaten oradaki soru "hangisi elenir", "hangisi hangi dikişte iyi" değil.
TASK_BREAKDOWN_MAX_MODELS = 4


def _per_task_lines(rows: list[dict[str, Any]], tasks: tuple[str, ...]) -> list[str]:
    """Görev başına geçme oranı — yalnız aday sayısı azaldığında.

    Havuzlu ana tablo bir ELEME aracı ve öyle kalıyor (bkz. build_report): tek bir
    ağır ihlal zaten diskalifiye eder, görev kırılımı orada gürültü olurdu. Ama
    teslim setinde soru değişiyor: "hangi model hangi dikişte iyi" — cleaning'de
    üstün olup narrative'de düşen bir modeli havuzlu oran gizler. O yüzden kırılım
    ana tablonun YERİNE değil YANINA geliyor ve yalnız <= TASK_BREAKDOWN_MAX_MODELS
    öznede basılıyor.
    """
    by_model = sorted({str(r["model"]) for r in rows})
    if len(by_model) > TASK_BREAKDOWN_MAX_MODELS:
        return []
    lines = [
        "",
        "## Görev kırılımı (geçti/çağrı · r=şema retry)",
        "",
        "Retry burada AYRI yazılıyor çünkü kotayı o yiyor: her şema retry'ı",
        "sağlayıcıya ekstra bir istek gider. Koşu erken durduysa hangi görevin",
        "payı tükettiğini havuzlu tablo değil bu sütun söyler.",
        "",
        "| Model | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for name in by_model:
        cells = []
        for task in tasks:
            task_rows = [r for r in rows if r["model"] == name and r["task"] == task]
            task_rows = [r for r in task_rows if not r.get("order_variant")]
            if not task_rows:
                cells.append("-")
                continue
            ok = sum(1 for r in task_rows if r.get("error") is None)
            retries = sum(int(r.get("retries") or 0) for r in task_rows)
            cell = f"{ok}/{len(task_rows)}"
            cells.append(f"{cell} · r={retries}" if retries else cell)
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def _source_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Rapor birden çok koşudan derlendiyse hangi koşudan kaç satır geldiğini yazar.

    Koşular arasında tekrar sayısı ve prompt sürümü değişmiş olabilir (bkz.
    read_many); bunu söylemeyen birleşik rapor, farklı koşulların ortalamasını
    tek bir ölçüm gibi gösterir.
    """
    counts: dict[str, int] = {}
    for row in rows:
        src = row.get("source")
        if src:
            counts[str(src)] = counts.get(str(src), 0) + 1
    if len(counts) < 2:
        return []
    lines = [
        "",
        "## Kaynak koşular",
        "",
        "Bu rapor birden çok koşudan derlendi. Koşular arasında tekrar sayısı,",
        "prompt sürümü ve tarih FARKLI olabilir — satırlar tek bir koşuymuş gibi",
        "okunmamalı.",
        "",
        "| Koşu | Satır |",
        "|---|---|",
    ]
    for src in sorted(counts):
        lines.append(f"| `{src}` | {counts[src]} |")
    return lines


def _circuit_break_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Devre kesiciyle elenen modelleri listeler.

    Elenen modelin tablodaki `n`'i küçük kalır; bunu ayrıca söylemezsek okuyucu
    'kota bitti' ile 'model ölü' arasını ayırt edemez — ikisi bambaşka kararlar.
    """
    broken = [r for r in rows if r.get("circuit_broken")]
    if not broken:
        return []
    lines = [
        "",
        f"## Devre kesici: {CIRCUIT_BREAK_ERRORS} ardışık pahalı hata sonrası elenenler",
        "",
        "Eleme (model, GÖREV) bazlı: yalnız o görevin kalan çağrıları yapılmadı, aynı",
        f"modelin diğer görevleri koştu. Pahalı = hata + ≥{SLOW_FAILURE_S:.0f}sn (süre ve",
        "yenilenmeyen kredi korumak için). Tablodaki düşük çağrı sayısı kota bitmesinden",
        "değil elemeden geliyor.",
        "",
    ]
    for row in broken:
        lines.append(
            f"- `{row['model']}` (uç: `{row.get('served_by', '?')}`) · "
            f"son hata: {row.get('error')} · {row['task']}/{row['case_id']}"
        )
    return lines


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
    rows = latest_per_cell(rows)
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
        # Ölçülmemiş retry `-`; `0` yalnız GERÇEKTEN ölçülüp sıfır çıktığında.
        ret = f"{a['retries']}" if a["retries"] is not None else "-"
        med = f"{a['latency_median']:.2f}" if a["latency_median"] is not None else "-"
        p95 = f"{a['latency_p95']:.2f}" if a["latency_p95"] is not None else "-"
        tok = f"{a['tokens_out_median']:.0f}" if a["tokens_out_median"] is not None else "-"
        cons = (
            f"{a['answer_consistency']:.0%} (n={a['answer_consistency_n']})"
            if a["answer_consistency"] is not None
            else "- (yetersiz tekrar)"
        )
        lines.append(
            f"| `{name}` | {a['n']} | {rate} | {ret} | {med} | {p95} | {tok} | {cons} | "
            f"{', '.join(a['errors']) or '-'} |"
        )
    lines += [
        "",
        "$ maliyet hesaplanmadı: aday matrisi 'bugün kalıcı ücretsiz' filtresiyle",
        "seçildi (README), `models.json`'da fiyat alanı yok. Çıktı token medyanı bir",
        "verimlilik proxy'si — gerçek $ maliyeti değil, uydurmamak için eklenmedi.",
    ]
    lines += _per_task_lines(rows, tasks)
    lines += _source_lines(rows)
    lines += _served_by_lines(rows)
    lines += _circuit_break_lines(rows)

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
        "**Cevap tutarlılığı tekrar sayısına duyarlıdır.** Ölçüt 'bir vakanın TÜM",
        "tekrarları aynı cevaba vardı mı'; n=2'de bunu tutturmak n=3'ten kolaydır.",
        "Farklı `--repeats` ile koşulmuş raporların tutarlılık yüzdeleri",
        "birbiriyle kıyaslanamaz.",
        "",
        "**spec_menu, 2026-07-31 prompt düzeltmesinden öncesi/sonrası kıyaslanamaz.**",
        "`generate_spec_menu` promptu o tarihe kadar HARD_CAP'ten (spesifikasyon",
        "bütçesi) hiç söz etmiyordu; menüler kartezyen çarpımda tavanı aşıp",
        "reddediliyordu. Düzeltme öncesi ölçülen spec_menu sayıları bayattır.",
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

    `fallback_only` modeller ÖZNE sayılmaz (kendi çağrıları yok) ama tavanları
    eşlendikleri birincil modelin havuzuna eklenir — yoksa dry-run gün tahmini
    kotayı olduğundan dar gösterir ve planı yanlış kurarsınız.
    """
    n_cases = sum(len(load_gold(t)) for t in tasks)
    order_extra = 0
    if order_check:
        order_extra = sum(len(load_gold(t)) for t in tasks if t in ORDER_CHECK_RUNNERS)
    per_model = n_cases * repeats + order_extra
    by_id = {m["id"]: m for m in models}
    subjects = [m for m in models if not m.get("fallback_only")]
    lines = [
        f"Görev: {', '.join(tasks)} · vaka: {n_cases} · tekrar: {repeats}"
        + (f" · order-check: +{order_extra}/model" if order_check else ""),
        f"Model başına çağrı: {per_model} · toplam: {per_model * len(subjects)}",
        "",
        f"{'Model':<48} {'sağlayıcı':<11} {'havuz':<18}",
    ]
    pools: dict[str, dict[str, Any]] = {}
    for model in models:
        pool = quota_pool(model)
        shown = "(kendi)" if pool == model_key(model) else pool
        if model.get("fallback_only"):
            shown = "→ kota uzantısı"
        lines.append(f"{model['id']:<48} {model['provider']:<11} {shown:<18}")
        if model.get("fallback_only"):
            continue  # kendi çağrısı yok; tavanı aşağıda birincilin havuzuna eklenir
        cap = quota_cap(model)
        extender = by_id.get(str(model.get("fallback_id") or ""))
        if extender is not None and extender.get("fallback_only"):
            extra_cap = quota_cap(extender)
            if cap is not None and extra_cap is not None:
                cap += extra_cap
            pool = f"{pool} + {extender['id']}"
        bucket = pools.setdefault(pool, {"calls": 0, "cap": cap, "n": 0, "seconds": 0.0})
        bucket["calls"] += per_model
        bucket["n"] += 1
        # Süre tahmini görev başına ayrı hesaplanır: TPM bağlayıcıysa aralık
        # görevin token maliyetiyle değişir (cleaning ~7K, narrative ~2.5K).
        # Havuz tavanına bakan gün sayısı bunu göremez — gemma'nın rpd'si 14.400,
        # yani "1 gün" der ve TPM throttle'ının olup olmadığını ayırt edemez.
        for task in tasks:
            per_task_calls = len(load_gold(task)) * repeats
            if order_check and task in ORDER_CHECK_RUNNERS:
                per_task_calls += len(load_gold(task))
            bucket["seconds"] += per_task_calls * Throttle.interval(
                rpm=model.get("rpm"),
                tpm=model.get("tpm"),
                est_tokens=TASK_TOKEN_ESTIMATE.get(task, DEFAULT_TOKEN_ESTIMATE),
            )
        # Aynı havuzdaki uçlar farklı tavan bildirirse en dar olanı bağlayıcıdır;
        # ilk modelinkini almak süreyi olduğundan kısa gösterirdi.
        caps = [c for c in (bucket["cap"], cap) if c is not None]
        bucket["cap"] = min(caps) if caps else None

    lines += [
        "",
        f"{'Kota havuzu':<44} {'model':>6} {'çağrı':>7} {'tavan':>7} {'gün':>5} {'dk':>6}",
    ]
    total_minutes = 0.0
    for pool, info in sorted(pools.items()):
        cap = info["cap"]
        days = "-" if not cap else str(-(-info["calls"] // cap))  # yukarı yuvarlama
        minutes = info["seconds"] / 60.0
        total_minutes += minutes
        lines.append(
            f"{pool:<44} {info['n']:>6} {info['calls']:>7} {str(cap or '-'):>7} "
            f"{days:>5} {minutes:>6.0f}"
        )
    slowest = max(
        (-(-i["calls"] // i["cap"]) for i in pools.values() if i["cap"]),
        default=0,
    )
    lines += [
        "",
        f"En yavaş havuz {slowest} koşu-günü sürer; kaç GÜNE yayılacağını o belirler.",
        f"Tek günlük süre ~{total_minutes:.0f} dk: koşucu sıralı, havuzlar toplanır.",
        "Dakika sütunu istek ARALIĞINDAN gelir (rpm ve varsa tpm'den bağlayıcı olanı,",
        "bkz. Throttle.interval); model cevap süresi buna EKLENİR, dahil değildir.",
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
    parser.add_argument(
        "--ship",
        action="store_true",
        help=(
            "yalnız uygulamada sunulacak adaylar (models.json: ship) + kota "
            "uzantıları — teslim koşusunun tek komutu"
        ),
    )
    parser.add_argument("--tasks", nargs="*", choices=TASKS, help="yalnız bu görevler")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"vaka başına tekrar (varsayılan {DEFAULT_REPEATS})",
    )
    parser.add_argument("--out", type=Path, help="çıktı dizini (varsayılan runs/benchmark/<ts>)")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help=(
            "cevap geldikten SONRA bir sonraki çağrıya kadar beklenecek taban süre "
            "(sn); rpm/tpm aralığıyla birlikte büyüğü bağlar. 0 = kapalı"
        ),
    )
    parser.add_argument(
        "--report-from",
        nargs="+",
        type=Path,
        metavar="RESULTS.JSONL",
        help=(
            "çağrı YAPMADAN, verilen koşuların sonuçlarından birleşik rapor üret "
            "(teslim funnel'ı). --out ile birlikte kullanılır"
        ),
    )
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
    if args.ship:
        models = ship_matrix(models)
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

    if args.report_from:
        # Çağrı yok: yalnız diskteki satırlardan rapor. `require_cache_disabled`
        # burada aranmaz — cache ayarı sonuç ÜRETİRKEN önemli, okurken değil.
        rows = read_many(list(args.report_from))
        if not rows:
            print("Verilen dosyalarda satır yok.", file=sys.stderr)
            return 1
        out_dir = args.out or (DEFAULT_OUT_ROOT / "birlesik")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.md"
        report_path.write_text(build_report(rows, tasks), encoding="utf-8")
        print(f"{len(rows)} satır · {len(args.report_from)} koşu\nRapor: {report_path}")
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
        models,
        tasks,
        args.repeats,
        out_dir,
        Throttle(cooldown=args.cooldown),
        order_check=args.order_check,
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
