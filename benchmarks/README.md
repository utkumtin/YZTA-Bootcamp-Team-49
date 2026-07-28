# JUDGE model benchmark

`pareto/llm/providers.py`'deki küratörlü model listeleri yer tutucu — dosyanın kendi notu:

> `TODO(ekip): aşağıdaki Groq/OpenRouter listeleri yer tutucudur — ekibin kendi
> testlerinden geçirdiği gerçek model ID'leri + performance_note/*_cost_note ile
> değiştirilmeli`

Bu dizin o listeyi kanıta bağlamak için var. Genel leaderboard'lar Pareto'nun işini
ölçmüyor; ölçen tek şey Pareto'nun kendi dört JUDGE görevi.

**Durum: benchmark kurulu, henüz koşulmadı.** Aday listesi ve puanlayıcılar hazır,
testler geçiyor. Tam matrisi koşmak ve sonucu `providers.py`'ye taşımak ayrı bir iş.

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

---

## Kullanım

```bash
# 1. Matrisi ve kota takvimini gör (çağrı yapmaz, anahtar istemez)
python scripts/run_model_benchmark.py --dry-run

# 2. Hangi uçlar gerçekten yaşıyor ve şema zorluyor (model başına 1 çağrı)
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --preflight

# 3. Tek model dumanı
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py \
    --models gemini-3.6-flash --tasks narrative --repeats 1

# 4. Tam matris (resume edilebilir; aynı komut kaldığı yerden devam eder)
PARETO_LLM_CACHE=0 python scripts/run_model_benchmark.py --out runs/benchmark/tur-1
```

`PARETO_LLM_CACHE=0` zorunlu; unutulursa koşucu durur. Cache açıkken ikinci koşu
diskten döner ve gecikme/retry sayıları gerçeği göstermez.

Çıktı: `results.jsonl` (her satır bir çağrı) + `report.md` (model × görev tablosu +
ağır ihlaller). `runs/` git'e girmez.

### Anahtarlar

`.env` içine (bkz. `.env.example`): `GEMINI_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`. Model enjeksiyonu `PARETO_JUDGE_PROVIDER` +
slotun `*_JUDGE_MODEL` değişkeniyle yapılır — koşucu bunları çağrı başına kendisi
ayarlar, elle set etmeye gerek yok.

---

## Kota takvimi

Koşucu kotayı **havuz** başına takip eder. Aynı hesap limitini paylaşan uçlar tek
havuzdur; model başına saymak kotayı kat kat büyük gösterir ve koşu 429 yer.

| Havuz | Tavan | Not |
|---|---|---|
| Google, model başına | `gemma-4-*` 14.400/gün · `gemini-3.*-flash` **20/gün** · flash-lite 500/gün | Flash'ın 20/gün'ü matristeki en dar kota |
| Groq, model başına | 1.000/gün (llama-3.1-8b: 14.400) | |
| `nvidia:hesap` | **1.000 kredi, tek seferlik** | Günlük yenilenmiyor. Telefon doğrulaması gerekiyor. |
| `openrouter:free` | 50/gün (hesapta $10 kredi varsa 1.000) | Tüm `:free` modeller **ortak havuz** |

17 model × 4 görev × 4 vaka × 3 tekrar = 816 çağrı. `--dry-run` havuz başına dağılımı
ve gün tahminini basar. Şu anki matriste en yavaş havuz Gemini Flash (3 koşu-günü);
NVIDIA tek seferlik bütçenin ~%29'unu kullanır.

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

## Bilinerek yapılmayanlar

- **Sayı uydurma doğrulayıcısı üretimde yok.** Anlatı sistem promptu "you never compute,
  estimate, or invent numbers" diyor ama bunu zorlayan bir doğrulayıcı yok
  (`_validate_axes` yalnız eksen adlarına bakar). Benchmark ölçüyor; üretime eklemek
  ayrı bir karar.
- **MECHANICAL slot ölçülmüyor** — bu tur yalnız JUDGE.
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
