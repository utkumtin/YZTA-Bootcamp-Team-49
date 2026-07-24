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
