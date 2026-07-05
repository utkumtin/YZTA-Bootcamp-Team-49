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
