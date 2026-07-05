# 3. Multiverse runner: subprocess, ProcessPool değil

- Durum: accepted
- Tarih: 2026-07-05
- İlgili: review.md sorun #2

## Bağlam
Prototip `ProcessPoolExecutor` kullanıyordu. Streamlit'te `session_state` child process'e
geçmez → KeyError; `asyncio.run()` de Tornado loop'ta kırık. CLI'da çalışır, Streamlit'e
taşındığı gün patlar.

## Karar
`analysis/runner.py`: estimator-agnostik sıralı çekirdek (`run_specs`) + standalone worker
(`python -m pareto.analysis.runner --job ...`) sonuç+ilerlemeyi diske yazar; Streamlit yalnız
okur (`launch_multiverse` + `RunHandle`). Determinizm env (seed/PYTHONHASHSEED/OMP) subprocess'e
enjekte edilir. Demo küçük → sıralı yeter; gerekirse subprocess *içinde* joblib-threading.

## Sonuç
Streamlit-safe. Standalone-script + sandbox paterniyle hizalı.
