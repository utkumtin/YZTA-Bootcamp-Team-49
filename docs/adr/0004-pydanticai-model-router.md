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
