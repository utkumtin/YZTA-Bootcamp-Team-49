# JUDGE model benchmark

`pareto/llm/providers.py`'deki küratörlü model listeleri yer tutucu — dosyanın kendi notu:

> `TODO(ekip): aşağıdaki Groq/OpenRouter listeleri yer tutucudur — ekibin kendi
> testlerinden geçirdiği gerçek model ID'leri + performance_note/*_cost_note ile
> değiştirilmeli`

Bu dizin o listeyi kanıta bağlamak için var. Genel leaderboard'lar Pareto'nun işini
ölçmüyor; ölçen tek şey Pareto'nun kendi dört JUDGE görevi.

**Durum: keşif turu koşuldu (11 model, `runs/benchmark/tur-1`), teslim turu bekliyor.**
Uygulamada sunulacak üç aday `possible-models.md`'de sabit: `gemini-3.6-flash`,
`gemma-4-31b-it`, `thinkingmachines/inkling`. Onları ölçen tek komut `--ship`
(aşağıda). Sonucu `providers.py`'ye taşımak ayrı bir iş.

---

## Ne ölçülüyor

| Görev | Üretim giriş noktası | Deterministik doğrulayıcı | Altın set eki |
|---|---|---|---|
| Temizleme | `cleaning/agent.py:generate_ledger` | `_validate_referenced_columns` | MUST_FIX / FORBIDDEN / ACCEPTABLE |
| Estimand | `analysis/hypothesis.py:draft_tac_proposal` | kolon listesi kısıtı | doğru eşleme + `expected_sign` koruması |
| Spec menüsü | `analysis/menu.py:generate_spec_menu` | `evaluate_menu_defensibility` | savunulabilir baseline, bad control |
| Anlatı | `llm/narrative.py:generate_narrative` | `_validate_axes` | doğru eksen + sayı uydurma + Türkçe |

Puanlamanın yarısı zaten üretimde: dört görevin de arkasında halüsinasyonu fail-loud
yakalayan bir doğrulayıcı var. Altın set, doğrulayıcıların göremediği kısmı ekliyor —
"kolon uydurmadı" ile "doğru kolonu seçti" arasındaki fark.

Her çağrıda ayrıca: `schema_ok`, `retries`, `latency_s`, `input/output_tokens`, hata sınıfı.

Bunun üstüne dört metrik daha, hepsi mevcut çağrılardan türetiliyor (ekstra API
maliyeti yok): p95 gecikme, çıktı token medyanı (verimlilik proxy'si — `$`
DEĞİL, adaylar ücretsiz ve `models.json`'da fiyat alanı yok), tekrarlar-arası
**cevap tutarlılığı** (aynı vaka 3 kez koşulunca aynı cevaba mı varılıyor —
"3/3 doğru" ile "2/3 doğru, hep farklı" pass-rate'te aynı görünürdü, tutarlılık
ayırır) ve **abstention sayaçları** (gatekeeper atlandı / gereksiz flag —
cleaning'te; TP/FP/FN/TN — estimand'ın clarification kararında). Beşinci
metrik, `--order-check`, ekstra çağrı gerektirdiği için opsiyonel (aşağıda).

---

## Kullanım

```bash
# 1. Matrisi ve kota takvimini gör (çağrı yapmaz, anahtar istemez)
python scripts/run_model_benchmark.py --dry-run --ship

# 2. Hangi uçlar gerçekten yaşıyor ve şema zorluyor (model başına 1 çağrı)
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --preflight --ship

# 3. Tek model dumanı
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py \
    --models gemini-3.6-flash --tasks narrative --repeats 1

# 4. TESLİM KOŞUSU: uygulamada sunulacak adaylar, tek çalıştırma (~16 dk + gecikme)
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --ship --out runs/benchmark/teslim

# 4b. Tam keşif matrisi (12 özne, 576 çağrı, çok günlü; resume edilebilir)
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --repeats 3 --out runs/benchmark/tur-1

# 6. Birden çok koşuyu tek rapora derle (çağrı yapmaz) — teslim funnel'ı
python scripts/run_model_benchmark.py \
    --report-from runs/benchmark/teslim/results.jsonl runs/benchmark/tur-1/results.jsonl \
    --out runs/benchmark/birlesik

# 5. + sıra duyarlılığı: estimand/spec_menu vakalarına kolon sırası TERS
#    çevrilmiş 1 ekstra çağrı ekler (bu iki görevde çağrı sayısı 2 katına çıkar,
#    o yüzden ayrı bayrak; --dry-run bunu havuz tavanına dahil eder).
#    --ship ile birlikte gemini payını sıfırlar: ayrı bir güne koy.
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --order-check
```

`PARETO_LLM_CACHE=0` zorunlu; unutulursa koşucu durur. Cache açıkken ikinci koşu
diskten döner ve gecikme/retry sayıları gerçeği göstermez.

Çıktı: `results.jsonl` (her satır bir çağrı) + `report.md` (model tablosu — görev
kırılımı yok, bu bir sıralama değil eleme aracı; ağır ihlaller; abstention
sayaçları; `--order-check` koşulduysa sıra duyarlılığı bölümü). `runs/` git'e
girmez.

### Anahtarlar

`.env` içine (bkz. `.env.example`): `GEMINI_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`. Model enjeksiyonu `PARETO_JUDGE_PROVIDER` +
slotun `*_JUDGE_MODEL` değişkeniyle yapılır — koşucu bunları çağrı başına kendisi
ayarlar, elle set etmeye gerek yok.

---

## Kota takvimi

Koşucu kotayı **havuz** başına takip eder. Aynı hesap limitini paylaşan uçlar tek
havuzdur; model başına saymak kotayı kat kat büyük gösterir ve koşu 429 yer.

**Bağlayıcı kısıt her zaman istek sayısı değil.** Üç ayrı eksen var ve hangisinin
bağladığı uca göre değişir:

| Model (teslim seti) | Uç | RPM | TPM | RPD | Bağlayıcı |
|---|---|---|---|---|---|
| `gemini-3.6-flash` | Google AI Studio | 5 | 250K | **20** | RPD |
| `gemini-3.5-flash` | Google AI Studio | 5 | 250K | **20** | RPD |
| `gemma-4-31b-it` | Google AI Studio | 30 | **16K** | 14.400 | **TPM** |
| `thinkingmachines/inkling` | NVIDIA NIM | ~40 | - | ~1.000 kredi (ömür boyu) | kredi |

AI Studio panelinden doğrulandı, 2026-07-31. Diğer havuzlar: Groq model başına
1.000 istek/gün **ama 200.000 token/gün** (canlı 429 gövdesi; gpt-oss-20b kotasını
43 çağrıda bitirdi), `openrouter:free` 50/gün ortak havuz.

`gemma-4-31b-it`'in geniş RPD'si yanıltıcı: 16K TPM'de ~7K'lık bir cleaning çağrısı
dakikada ancak ~2 istek bırakır. 2026-07-30'da rpm'e bakan throttle 7 çağrıda 429
yedi. `Throttle.interval` artık ikisinden bağlayıcı olanı uyguluyor ve `--dry-run`
havuz başına **dakika** tahmini basıyor — gün sütunu bunu göremez (gemma'da her
hâlükârda 1 çıkar), yani throttle'ın çalıştığını gösteren tek sütun dakikadır.

Google limitleri **proje başına, anahtar başına değil**. `gemini-3.5-flash`'ın kota
uzantısı olarak eşlenmesi bu yüzden işe yarıyor: ayrı *model*, ayrı 20'lik RPD.
İkinci bir API anahtarı aynı işi yapmaz. RPD Pasifik saatiyle gece yarısı sıfırlanır.

**Retry payı — 40'lık havuz yalnız planlamada var.** `--dry-run` iki ucun tavanını
toplayıp 40 gösterir, ama koşuda iki ayrı havuz vardır ve her birinin tavanı 20'dir;
32 çağrı ancak devir gerçekten olursa sığar. Şema retry'ı sağlayıcıya ayrı bir istek
gider ve `acquire`'dan SONRA olur, yani bir vaka tam `remaining == needed` ile
onaylanırsa tek retry tavanı vaka ORTASINDA doldurur. İki dikiş birlikte kapatıyor:

- `case_server` bir vakayı onaylarken `RETRY_MARGIN` (3) çağrılık pay arar; hiçbir
  uçta pay kalmamışsa payı düşürüp havuzun artığını yine de kullanır.
- `run_matrix` `acquire`'ın `QuotaExhausted`'ını yakalar: o modeli durdurur, diğer
  modeller koşmaya devam eder. Yakalanmadığı sürümde gemini'de tek bir retry gemma
  ve inkling'i hiç koşturmadan tüm koşuyu traceback'e düşürüyordu.

Pratik sonuç: teslim koşusu gemini'yi **16/20 + 16/20** kullanır, her uçta 4 çağrı
retry ve 429 tekrarı için artar.

**Teslim matrisi:** 3 özne × 4 görev × 4 vaka × 2 tekrar = **96 çağrı**, tek gün,
tek çalıştırma (`--ship`). Gemini'nin 32 çağrısı iki uca **16 + 16** dağılır ve her
uçta 4 çağrılık retry payı kalır (aşağıda: retry payı). Tam keşif matrisi
(`--repeats 3`) **12 özne × 48 = 576 çağrı** ve çok günlüdür; `--dry-run` her
ikisinin dağılımını da basar.

`--order-check` teslim koşusunu 96 → **120 çağrıya** çıkarır; gemini'nin iki ucunun
birleşik tavanı da tam 40'tır, yani pay diye bir şey kalmaz. Ölçüldü (sahte saatle
simülasyon): retry olmasa bile koşu **118/120** ile biter, eksik 2 satır ertesi güne
kalır. Sıra duyarlılığını ölçeceksen ayrı bir güne koy.

Koşu sırası `models.json`'daki `priority` alanına göre: `high` → `normal` → `low`.
Paylaşımlı havuzda sıra sonucu belirler — NVIDIA kredisi biterse kuyruğun sonundaki
modeller hiç koşmaz, o yüzden kredi `deepseek-v4-pro`/`inkling` gibi asıl adaylara gider.

Kotalar **2026-07-27'de** doğrulandı ve değişir. Gerçeği `--preflight` ve sağlayıcının
kendi konsolu söyler.

---

## Aday listesi nasıl seçildi

Filtre: (a) bugün kalıcı ücretsiz, (b) `tools` veya native structured output desteği —
**zorunlu**, yoksa pydantic-ai `output_type` çalışmaz, (c) bağlam ≥ 128K, (d) aşağıdaki
eksenlerde yayınlanmış skor. Gerekçeler `models.json`'daki `evidence` alanında.

| Pareto'nun ihtiyacı | Bakılan dış benchmark |
|---|---|
| Şemaya ilk denemede uyan tipli çıktı | BFCL v4 · Tau2/Tau3 · MCP Atlas |
| "Yalnız vetted registry'den seç" kısıtına uyma | IFBench / IFEval |
| Olmayan kolon/eksen uydurmama | AA-Omniscience · SimpleQA Verified |
| Ekonometrik yargı | MMLU-Pro · GPQA Diamond |
| Tablo/veri-analizi muhakemesi | DABStep · BLADE |
| Türkçe anlatı | MMMLU · Global-MMLU-Lite · MMLU-ProX |

Dış skorlar yalnız listeyi daraltmak için; kararı bu benchmark verecek.

**Katalog sayfalarına güvenilmedi.** Groq'un katalog sayfasında hâlâ görünen
`moonshotai/kimi-k2-instruct` ve `llama-4-scout/maverick` deprecate edilmiş ve free
tier tablosunda yok — listeye alınmadı. NVIDIA NIM model ID'leri
`https://integrate.api.nvidia.com/v1/models` (auth'suz açık) üzerinden tek tek
doğrulandı. `--preflight` bunu koşu anında tekrar eder.

**Kontrol grubu şart:** `gemini-3.6-flash` (`JUDGE_SLOT` defaultu) ve
`thinkingmachines/inkling` (AA-Omniscience/IFBench referansı) matriste. Diğer her model
bunlara göre okunur; yoksa "daha iyi" demenin ölçüsü olmaz.

### Ücretsiz modeller PRIVATE slotu dolduramaz

`:free` OpenRouter uçları ZDR taşımıyor; NVIDIA NIM'in ücretsiz ucu da taşımıyor. Bu
yüzden `JUDGE_NVIDIA_SLOT` yalnız PUBLIC katmanda kayıtlı (`no_train=False`) ve bu
benchmark'ın çıktısı yalnız **PUBLIC** judge listelerini besleyecek.
`JUDGE_*_PRIVATE_SLOT` bu işin kapsamı dışında.

---

## Altın etiketler nereden geliyor

Dört referans veri seti: `data/card_krueger`, `data/castle`, `data/divorce`,
`data/medicaid`. Hepsi kanonik ekonometri veri setleri; "savunulabilir seçim" kaynak
makalelerde belgeli. Rol tanımları her veri setinin `config.yaml`'ında.

| Veri seti | Kaynak | Rolü |
|---|---|---|
| card_krueger | Card & Krueger (1994), NJ-PA asgari ücret | 2×2 DiD |
| castle | Cheng & Hoekstra (2013), castle doctrine (Cunningham, *Mixtape*) | kademeli yasalaşma |
| divorce | Stevenson & Wolfers (2006); Goodman-Bacon (2021) örnek verisi | estimator-flip'in kanonik örneği |
| medicaid | ACA Medicaid genişlemesi (CDC WONDER · KFF · SAHIE · SAIPE · BLS LAUS) | tek gerçek temizleme vitrini |

**Temizleme etiketleri ölçüldü, uydurulmadı.** Her `dtype`/eksiklik değeri 2026-07-27'de
gerçek panellerin profilinden okundu. Tekrar üretmek için:

```bash
python -c "import sys;sys.path.insert(0,'.');\
from pareto.cleaning.merge import build_panel;from pareto.profiling import profile_dataframe;\
print(profile_dataframe(build_panel('data/castle').df)['columns'])"
```

`tests/test_model_benchmark.py` bu bağı canlı tutuyor: altın setteki her kolonun panelde,
her transform'un `REGISTRY`'de olduğunu doğruluyor ve `coerce_numeric` MUST_FIX'lerinin
gerçekten metin dtype'lı kolonlara işaret ettiğini kontrol ediyor. Pipeline ileride
dtype'ı düzeltirse test düşer ve altın set güncellenir — yoksa benchmark var olmayan bir
kusuru aramaya devam ederdi.

**Savunulabilirlik etiketleri literatürden:**

- `clustering='none'` panel veride savunulamaz: hata terimi birim içinde seri
  korelasyonludur, kümelemesiz standart hatalar DiD'de ciddi biçimde küçük çıkar
  (Bertrand, Duflo & Mullainathan 2004). Kümeleme tedavinin atandığı seviyede olmalı —
  Medicaid'de eyalet, ilçe değil (genişleme kararı eyaletindir).
- `bad_control_columns`: tedavi sonrası belirlenen değişkeni kontrol olarak koymak yolu
  tıkar ve katsayıyı yanlılaştırır (Angrist & Pischke, *bad controls*). En güçlü örnek
  medicaid'de `pct_uninsured` — genişlemenin birincil sonucu, mortalite regresyonunda
  kontrol etmek tam da ölçülmek istenen mekanizmayı kapatır. Ayrıca her vakada sonucun
  kendisi ve (medicaid'de) sonucun pay/paydası.

  **Bu liste veri setinin kendi `config.yaml: panel.covariates` listesiyle çakışamaz** —
  `test_spec_menu_gold_bad_controls_do_not_contradict_config` bunu zorluyor. İlk
  taslakta castle'da `prisoner`/`police`, divorce'ta `homicide_rate`/`afdc_cases` bad
  control diye etiketlenmişti; oysa ikisi de o veri setlerinin covariate listesinde ve
  kaynak makaleler bunları kontrol olarak kullanıyor. Repo'nun kendi artefaktıyla çelişen
  bir etiket, benchmark'ı doğru davranan modeli cezalandırır hale getirirdi.

**card_krueger vakası bilerek zor:** kullanıcının beyanı `expected_sign="negative"`
(standart ders kitabı önermesi) ama literatürdeki ünlü bulgu bunun tersi. Model
literatürü "düzeltmeye" kalkarsa `expected_sign`'ı çevirir ve yakalanır. Estimand'ın işi
çıpalamayı öldürmek: kullanıcının teorik beyanı sonuç görülmeden dondurulur, model onu
düzeltmez.

---

## Koşudan çıkan üretim bulgusu: spesifikasyon bütçesi

Keşif turunda güçlü modellerin yarısı spec_menu'de aynı duvara çarptı — inkling
6/12, llama-3.3-70b 6/12, nemotron-nano 6/12, gpt-oss-120b 8/12 — ve hata hep aynıydı:
*"1920 spesifikasyon üretildi, sert tavan 24"*.

Sebep model değil prompt: `generate_spec_menu` `HARD_CAP`'ten (24, `config.py:
max_specifications`) hiç söz etmiyordu. `_axis_levels` her eksende
`[baseline_level, *candidate_levels]` döndürüyor, `expand_to_specs` bunların
**kartezyen çarpımını** alıyor. 7 eksenin hepsine ikişer aday seviye = 128
spesifikasyon. Üretimde de aynı yol: kullanıcı JUDGE'ın menüsünü olduğu gibi
onaylarsa eğri yerine hata görür.

2026-07-31'de prompt'a bir **spesifikasyon bütçesi** bloğu eklendi (menu.py:
`_SPEC_BUDGET_RULE`): çarpımın nasıl hesaplandığı, örnek aritmetik, bütçeyi
harcama politikası (kullanmadığın ekseni `candidate_levels: []` ile pinle) ve
cevaptan önce çarpımı hesaplama adımı. Sayı ayardan f-string ile geliyor.

**Şema validator'ı bilerek eklenmedi.** Çarpımı kontrol eden bir pydantic
validator pydantic-ai retry'ını tetikler ve model geri bildirimle kendini
düzeltir; o zaman bu dikiş "açıkça bildirilmiş sert bir kısıta İLK denemede uydu
mu" ölçmeyi bırakır, "geri bildirimle düzelebiliyor mu" ölçmeye başlar. Birincisi
JUDGE seçiminde aradığımız talimat-uyumu ekseninin ta kendisi (aday listesindeki
IFBench/IFEval gerekçesi). Kısıt prompt'ta söylenir, `expand_to_specs` fail-loud kalır.

**Sonuç:** düzeltme öncesi spec_menu sayıları BAYAT. `tur-1`'deki 11 modelin
spec_menu oranları yeni koşuyla kıyaslanamaz; rapor bunu kendi içinde de yazıyor.

## Bilinerek yapılmayanlar

- **Sayı uydurma doğrulayıcısı üretimde yok.** Anlatı sistem promptu "you never compute,
  estimate, or invent numbers" diyor ama bunu zorlayan bir doğrulayıcı yok
  (`_validate_axes` yalnız eksen adlarına bakar). Benchmark ölçüyor; üretime eklemek
  ayrı bir karar.
- **MECHANICAL slot ölçülmüyor** — bu tur yalnız JUDGE.
- **Tekrar sayısı 3'ten 2'ye indi** (`DEFAULT_REPEATS`). 16 vaka × 3 = 48 çağrı
  gemini havuzunun birleşik 40/gün kotasına sığmıyordu, 16 × 2 = 32 sığıyor.
  Vaka kırpmak yerine tekrar düşürüldü: dört veri setinin her biri diğerinin
  ölçemediği bir kusuru taşıyor. **Cevap tutarlılığı bundan etkilenir** — ölçüt
  "bir vakanın TÜM tekrarları aynı cevaba vardı mı" ve n=2'de bunu tutturmak
  n=3'ten kolaydır. Farklı `--repeats` ile koşulmuş raporların tutarlılık
  yüzdeleri kıyaslanamaz; rapor bunu yazıyor.
- **`gemini-3.6-flash` skorları saf değil.** `gemini-3.5-flash` kota uzantısı
  olarak eşli (models.json: `fallback_id`), yani satırların bir kısmı 3.5'ten
  gelmiş olabilir; rapor "Çağrıyı karşılayan uçlar" tablosunda dağılımı basıyor.
  Bu bilinçli bir takas: 20/gün ile 40/gün arasındaki fark, teslim matrisinin tek
  güne sığıp sığmaması demek. Teslim metni "3.6/3.5-flash ailesi" demeli, saf
  3.6 iddiası kurmamalı. İkisinin gerçekten denk olduğu ÖLÇÜLMEDİ, ekip kararı.
- **Türkçe teşhisi sezgisel:** dile özgü harf veya en az iki işlev sözcüğü. Amaç "model
  İngilizce yazdı mı"yı yakalamak, dil bilimsel sınıflandırma yapmak değil.
- **Cerebras · Cohere · Mistral · Cloudflare** sağlayıcı dalları eklenmedi. pydantic-ai
  2.5.0 `cerebras`/`cohere`/`mistral` provider'larını hazır getiriyor, ileride eklemek
  birkaç satır. Not: Mistral'in ücretsiz katmanı veri eğitimine opt-in zorunlu kılıyor.

## Koşarken görülmesi gereken

`never_treated` kolonu `castle` ve `divorce` panellerinde tümüyle `False` geliyor,
oysa iki `config.yaml` da never-treated birimlerden söz ediyor (castle'da 29 eyalet).
`card_krueger` ve `medicaid`'de iki değerli. Bu yüzden `never_treated` ekseni altın
sette **etiketsiz** bırakıldı — "doğru" seviye veriden okunamıyor. Panel kurulumundaki
bu tutarsızlık bu işin kapsamı dışında; ayrıca ele alınmalı.
