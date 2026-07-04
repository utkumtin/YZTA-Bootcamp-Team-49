# Pareto

Nicel araştırmacıların veri temizleme ve nedensel çıkarım (causal inference) analizlerini
yaparken yaşadıkları zaman kaybına ve "sonuca güven" problemine çözüm üreten platform.

## Felsefe

- **Kara kutu yok**: Her adımın kodu üretilir, karar defterine (decision ledger) yazılır.
- **Varyans bir özelliktir, gürültü değil**: Araştırmacı serbestlik derecelerinin (researcher
  degrees of freedom) dağılımı şeffaf şekilde gösterilir.
- **Tek motor, fork yok**: Estimator'lar `statsmodels` / `linearmodels` / `pyfixest`
  kütüphanelerinden alınır. Biz sadece orkestrasyon ve varyans muhakemesi katmanını yazarız.
- **Human-in-the-loop**: Yüksek güvenli işlemler otomatik, gerçek belirsizlikler kullanıcıya
  sorulur.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-..."   # LLM agent'ları için gerekli
```

## Klasör Yapısı

```
pareto/
  config.py                    # Genel ayarlar
  llm_client.py                # Anthropic API sarmalayıcısı (JSON-mode destekli)
  p1_cleaning/
    ingestion.py               # Veri yükleme + profiling
    decision_ledger.py         # LLM tabanlı temizlik kararları -> JSON
    gatekeeper.py               # Belirsizlik bayrağı -> kullanıcı onayı
    code_executor.py            # Kararlardan Python kodu üretir, denetim izi bırakır, çalıştırır
    pipeline.py                  # P1'i uçtan uca bağlayan orkestratör
  p2_p3_analysis/
    spec_generator.py           # LLM: kontrol seti / pencere / clustering önerileri
    multiverse.py                # Kartezyen çarpım -> N adet spesifikasyon
    runner.py                    # Spesifikasyonları paralel çalıştırır (OLS / TWFE)
  p4_dashboard/
    app.py                       # Streamlit "Varyans Paneli"
main.py                          # CLI orkestrasyon girişi
```

## Çalıştırma

```bash
# 1) Temizleme + analiz motorunu CLI üzerinden koş
python main.py --input veri.csv --outcome y --treatment d --unit firm_id --time year

# 2) Sonuç ortaya çıktıktan sonra dashboard'u aç
streamlit run pareto/p4_dashboard/app.py -- --results runs/latest_results.json
```

## Durum

Bu, dört promptta tarif edilen mimarinin **çalışan bir iskeletidir**:
staggered adoption (Callaway-Sant'Anna) estimator'ı bilinçli olarak dışarıda bırakıldı
(stretch goal). Temel OLS / TWFE çokluevreni uçtan uca çalışır.
