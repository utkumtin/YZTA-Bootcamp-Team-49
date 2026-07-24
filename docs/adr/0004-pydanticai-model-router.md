# 4. LLM katmanı: PydanticAI + model router

- Durum: accepted
- Tarih: 2026-07-05
- İlgili: review.md sorun #3

## Bağlam
Prototip (`prototype/pareto/llm_client.py`) ham Anthropic SDK'ya sabitlenmişti (tek sağlayıcı,
paid) + regex ile JSON ayıklama. Rubrikteki "model seçimi" (20p) ve "maliyet" (10p) bu
soyutlamadan besleniyor; test stratejisi de model-agnostik test gerektiriyor.

## Karar
`llm/router.py` PydanticAI Agent kurar (tipli `output_type` → şema-zorlaması + retry, regex
gitti). `llm/providers.py` rol → sağlayıcı zinciri: yargı PİNLİ Gemini 3.5 Flash (thinking ON),
mekanikte failover (Flash-Lite → Groq → OpenRouter). Test: `use_test_model` ile PydanticAI
`TestModel`/`FunctionModel` (API yakmaz). Private modda yalnız no-train uçlar (fail-loud).

## Sonuç
Model-agnostik, test edilebilir, privacy-aware. Reprodüksiyon dondurmadan gelir, model
stabilitesinden değil. Paid-frontier escalation provize-kapalı.

## Notlar (2026-07-09 güncellemesi)
- Google provider prefix'i `google` olarak standardize edildi (legacy `google-gla` kullanılmıyor).
- BYOK akışı yalnız Google ile sınırlı değil; Groq/OpenRouter anahtarları da aynı oturum panelinden
  alınır ve router zincirine yansıtılır.

## Notlar (2026-07-22 güncellemesi)
- Model ID'leri artık kodda sabit değil: her zincir slotu `.env`/`st.secrets`'tan çözülür
  (`GEMINI_JUDGE_MODEL`, `GEMINI_JUDGE_PRIVATE_MODEL`, `GEMINI_MECHANICAL_MODEL`,
  `GROQ_MECHANICAL_MODEL`, `OPENROUTER_MECHANICAL_MODEL`). Yeni model çıktığında kod değişmez.
- "Pin" anlamı korunuyor: yargı zinciri **tek üyeli** kalır (failover yok). Değişen tek şey,
  o tek üyenin model ID'sinin artık konfigürasyon olması.
- Env yalnız `model_id`'yi kontrol eder. `provider` / `api_key_env` / `no_train` kodda pinli
  kalır — private moddaki no-train emniyeti (`providers.py`) ancak böyle anlamlı olur.
- Zincirler import anında değil **çağrı anında** kurulur: `.env` yükleme sırası ve oturum-içi
  UI seçimi ancak böyle yansır.
- Uygulama-içi model seçimi (BYOK paneli) yalnız yargı slotunu ve yalnız **public** modu etkiler;
  seçim `os.environ`'a yazılmaz, oturumda kalır. Private uçlar deploy sahibinin kontrolünde.

## Notlar (2026-07-24 güncellemesi)
- JUDGE artık her (sağlayıcı × privacy) kombinasyonu için ayrı bir slot: Gemini/Groq/OpenRouter
  × public/private = 6 slot (önceki: yalnız Gemini × 2). Kullanıcı UI'dan önce sağlayıcı, sonra o
  sağlayıcının küratörlü model listesinden bir model seçer (`streamlit_ui.py: _render_model_choice`).
- "Pinli tek üye" değişmezi korunuyor: `chain_for(JUDGE, ...)` hâlâ tam olarak 1 `ProviderModel`
  döndürür — failover'a girmiyor. Değişen, *hangi* slotun pinli olduğunun artık seçilebilir olması.
- **Private mod artık JUDGE için de UI'dan seçilebilir** — önceki "private = yalnız deploy sahibi"
  kısıtı JUDGE'a özel olarak gevşetildi. Kullanıcı yalnız küratörlü 3 sağlayıcılık private slot
  kümesinden seçer; `provider`/`api_key_env`/`no_train` yine kodda pinli, seçim bunları asla
  değiştiremez. **MECHANICAL rolü bu kapsamın dışında** — davranışı, UI'da hiç gösterilmemesi dahil,
  değişmedi.
- ZDR/no-train garantisi sağlayıcıya göre asimetrik sağlanıyor: Groq **hesap-seviyesinde** ZDR
  sunuyor, console.groq.com/settings/data-controls adresinden açılır; kod bunu doğrulayamaz, ops
  tarafından `GROQ_API_KEY`'in bağlı hesapta fiilen açık olduğu teyit edilmeli. UI da private modda
  Groq seçildiğinde kullanıcıyı bu ayarı doğrulaması gerektiği konusunda uyarır ve sorumluluğu
  kullanıcıya bırakır (`streamlit_ui.py: _render_model_choice`). OpenRouter ise **istek-bazlı**:
  `router.py: _model_from_provider` artık OpenRouter için genel `"provider:model"` string yerine
  açık bir `OpenRouterModel`/`OpenRouterProvider` kurar, private OpenRouter judge slotu
  `extra_model_settings={"openrouter_provider": {"zdr": True}}` deklare eder; bu,
  `_resolve_model`/`build_agent` üzerinden `Agent`'ın `model_settings`'ine taşınır.
- Model başına performans/maliyet notu alanları hazırlandı (`ModelOption.performance_note`,
  `input_cost_note`, `output_cost_note`) — ekip kendi test sonuçlarıyla dolduracak. Otomatik
  maliyet hesaplama/analiz-başı fatura tahmini altyapısı bilinçli olarak bu turun kapsamı dışında.
- Groq/OpenRouter judge slotlarındaki model ID'leri şu an yer tutucu — merge öncesi ekibin gerçek
  küratörlü listesiyle değiştirilmeli (bkz. `providers.py` içindeki `TODO(ekip)` notu).

## Notlar (2026-07-24 güncellemesi #2) — thinking parametresi (planned-issues.md madde 2a/4)

- **Tespit edilen boşluk:** `ProviderModel.thinking`/`ModelSlot.thinking` alanı hiçbir yerde
  tüketilmiyordu — `router.py:_model_from_provider` bu bayrağı hiç okumuyordu. Bu ADR'nin ve
  `dev-docs/SCOPE.md:21,114`'ün "yargı modeli thinking ON" iddiası kodda karşılıksızdı.
- **Araştırma** (kurulu `pydantic-ai==2.5.0` kaynağı + `ai.pydantic.dev/thinking`): pydantic-ai'de
  cross-provider birleşik bir `ModelSettings.thinking` alanı var (`settings.py:291`) — Google, Groq
  ve OpenRouter'ın hepsi destekliyor. Her sağlayıcının model sınıfı bunu kendi native parametresine
  çeviriyor (Google: `google_thinking_config`/`include_thoughts`/`thinking_level`/`thinking_budget`;
  Groq: `reasoning_format`/`reasoning_effort`; OpenRouter: `reasoning`). Model profili thinking
  desteklemiyorsa ayar sessizce elenir (`models/__init__.py:prepare_request`) — reasoning yapmayan
  modellere (ör. Groq mekanik) göndermek zararsız, no-op.
- **Kritik bulgu — neden salt bool yetmiyordu:** Gemini'de salt `thinking=True` yalnız
  `include_thoughts=True` üretiyor (`models/google.py:_translate_thinking`), `thinking_budget`/
  `thinking_level` set ETMİYOR. Flash-tier bir model varsayılan olarak reasoning yapmıyorsa, bool
  `True` reasoning'i açmaz — yalnız "varsa" izini response'a dahil eder. Orijinal `thinking: bool`
  tasarımı "thinking ON" iddiasını garanti etmiyordu.
- **Karar:** alan `bool`'dan küratörlü bir effort seviyesine (`ThinkingChoice = "off"|"low"|"medium"|
  "high"`) genişletildi — pydantic-ai'nin tam skalası (`minimal`..`xhigh`) değil, mevcut "serbest
  metin yok, küratörlü liste" felsefesiyle tutarlı bir alt küme. `ModelSlot.default_thinking` +
  `thinking_options` (boşsa UI'da gösterilmez) `options`/`_session_choice` deseninin birebir aynısı.
  `router.py:_resolve_model`, tek üyeli zincirde `pm.thinking != "off"` olduğunda `extra_model_settings
  ["thinking"]`'e yazıyor — `"off"` hiç key eklemiyor, model kendi varsayılanını kullanıyor.
- **Kullanıcıya açıldı:** mekanizma sağlayıcıdan bağımsız çalıştığı için, JUDGE'ın **6 slotunun**
  hepsinde (Gemini/Groq/OpenRouter × public/private) BYOK panelinden ("Thinking" selectbox,
  `streamlit_ui.py:_render_model_choice`) kullanıcı thinking seviyesini seçebiliyor. Kimlik alanları
  (`provider`/`api_key_env`/`no_train`) yine pinli, yalnız bir davranış tercihi eklendi.
  Varsayılanlar: Gemini judge (public+private) `"medium"` (iddiayı fiilen karşılamak için); Groq/
  OpenRouter judge `"off"` (önceki davranış korunuyor — bu sağlayıcılar için hiçbir zaman "thinking
  ON" iddiası yoktu). MECHANICAL dokunulmadı — UI'da hâlâ gösterilmiyor.
- **Canlı doğrulama:** `tests/test_router_smoke.py::test_gemini_canli_thinking_ile_yapili_cikti_uretir`
  gerçek pinli modelle (`gemini-3.5-flash`) denendi; bu oturumda Google tarafında geçici bir `503
  UNAVAILABLE` ("high demand") ile karşılaşıldı — pydantic-ai'yi tamamen atlayan çıplak bir
  `google-genai` çağrısıyla da aynı hata doğrulandı, yani bizim koddan kaynaklanmıyor. Mekanizmanın
  kendisi (thinking + `output_type` birlikte, GH pydantic/pydantic-ai#793/#2293 riskine karşı) aynı
  hesap/anahtarla erişilebilen başka güncel modellerde (`gemini-3-flash-preview`,
  `gemini-3.1-flash-lite`) ayrıca doğrulandı: `ThinkingPart` üretiliyor ve yapılı çıktı doğru
  parse ediliyor. Test suit'te kalıcı — CI'da anahtar yoksa zaten skip olur, pinli model tekrar
  erişilebilir olduğunda gerçek bir regresyon sinyali verir.
- Madde 4'ün bullet 2'si (no_train doğrulanabilirliği) ve bullet 3'ü (kimlik alanlarının .env'e
  açılmaması) zaten yukarıdaki 2026-07-24 notunda cevaplanmıştı; bu güncelleme yalnız bullet 1'i
  (thinking parametre eşlemesi) kapatıyor.
- **Cache anahtarı kontrolü:** `thinking` artık kullanıcı tarafından oturum içinde değiştirilebilen
  bir eksen olduğu için, `CachedModel`'in (`cache.py:105-125`) anahtarının bunu içerip içermediği
  ayrıca doğrulandı — içermeseydi, kullanıcı thinking seviyesini değiştirip aynı prompt'u tekrar
  çalıştırdığında eski seviyenin cache'lenmiş yanıtı sessizce geri dönerdi (savunulabilirlik
  tezine doğrudan aykırı). `CachedModel` ham `model_settings` sözlüğünü (pydantic-ai'nin
  `prepare_request()`'i `thinking`'i ayıklamadan ÖNCE) hash'lediği için anahtar zaten `thinking`'i
  içeriyor — hem koda bakarak hem `FunctionModel` ile canlı bir denemeyle doğrulandı
  (`tests/test_router_smoke.py::test_cache_farkli_thinking_ayri_girdi_olur`).
