# S2-13 Uçtan Uca Entegrasyon Doğrulaması

## Provenance

Şu komutla üretildi:

```bash
.venv/bin/python scripts/run_e2e_check.py --out docs/verification/s2-13-e2e-check.md
```

- Veri seti: medicaid
- Run: s2-13-e2e
- Kaynak commit: dff0fcc
- Python: 3.11.15
- Platform: Linux
- JUDGE modu: PydanticAI test modeli (sahte tipli çıktı, canlı sağlayıcı çağrısı yok)

## Genel Sonuç

**GEÇTİ**: yedi dikiş de koştu ve beklenen çıktıyı üretti

## Definition of Success Kontrol Listesi

| Madde | Durum | Not |
| --- | --- | --- |
| Yükle: config güdümlü panel merge | GEÇTİ | 34154 satır, 18 kolon, tier tier1_panel_did |
| Profille: deterministik kolon profili | GEÇTİ | 18 kolon profillendi |
| Temizle: JUDGE karar defteri, gatekeeper, codegen | GEÇTİ | 2 karar (1 tanesi gatekeeper'dan geçti), L4 reprodüksiyon doğrulandı |
| Estimand: TAC önerisi ve dondurma | GEÇTİ | estimand donduruldu (hash 13094356f88602fb) |
| Menü: JUDGE spec menüsü, dondurma, faktöriyel açılım | GEÇTİ | menü donduruldu (hash 91f68b4b6594f2a5), 8 spesifikasyon açıldı |
| Multiverse: subprocess runner ve disk çıktıları | GEÇTİ | 8 sonuç yazıldı (0 başarısız spec) |
| Varyans paneli: özet, eksen atfı, narrative, efektif N, pre-trend | GEÇTİ | bant robust, 1 eksen yorumu, 8 pre-trend katsayısı |
| Her LLM adımı test modeliyle koşulabilir | GEÇTİ | dört JUDGE adımı da test modeliyle koştu |
| CI kalite kapısı | DIŞ KONTROL | bu harness çalıştırmaz, CI doğrular: ruff format --check ., ruff check ., mypy pareto, pytest -m "not live" |

## Dikişler

### Yükle: config güdümlü panel merge

- Durum: **GEÇTİ**
- Sonuç: 34154 satır, 18 kolon, tier tier1_panel_did
- Ölçümler:
  - n_rows: 34154
  - n_cols: 18
  - tier: "tier1_panel_did"
  - n_sources: 5
  - unit_col: "county_fips"
  - time_col: "year"

### Profille: deterministik kolon profili

- Durum: **GEÇTİ**
- Sonuç: 18 kolon profillendi
- Ölçümler:
  - n_rows: 34154
  - n_cols: 18
  - duplicate_row_count: 0
  - n_potential_join_keys: 18

### Temizle: JUDGE karar defteri, gatekeeper, codegen

- Durum: **GEÇTİ**
- Sonuç: 2 karar (1 tanesi gatekeeper'dan geçti), L4 reprodüksiyon doğrulandı
- Ölçümler:
  - n_decisions: 2
  - n_gated: 1
  - n_applied: 2
  - audit_script: "runs/audit_trail/s2-13-e2e_cleaning_steps.py"
  - repro_dir: "runs/audit_trail/s2-13-e2e_repro"
  - ledger: "runs/audit_trail/s2-13-e2e_decision_ledger.jsonl"
  - n_rows_after_clean: 34154

### Estimand: TAC önerisi ve dondurma

- Durum: **GEÇTİ**
- Sonuç: estimand donduruldu (hash 13094356f88602fb)
- Ölçümler:
  - committed_cohort: 2014
  - n_rows: 23606
  - n_treated_post_rows: 7104
  - freeze_hash: "13094356f88602fb"
  - outcome: "pct_uninsured"
  - treatment_coding: "treated_post"

### Menü: JUDGE spec menüsü, dondurma, faktöriyel açılım

- Durum: **GEÇTİ**
- Sonuç: menü donduruldu (hash 91f68b4b6594f2a5), 8 spesifikasyon açıldı
- Ölçümler:
  - menu_hash: "91f68b4b6594f2a5"
  - n_specs: 8
  - hard_cap: 24
  - n_control_sets: 2
  - n_estimator_levels: 2
  - n_weighting_levels: 2
  - estimators: ["OLS", "TWFE"]
  - mapping_warnings: ["Hash 13094356f88602fb: parallel_trends varsayımı OLS ile eşleştirildi. Bu durum hata değildir ancak önerilen yöntem değildir."]

### Multiverse: subprocess runner ve disk çıktıları

- Durum: **GEÇTİ**
- Sonuç: 8 sonuç yazıldı (0 başarısız spec)
- Ölçümler:
  - run_dir: "runs/s2-13-e2e"
  - n_results: 8
  - n_failed_specs: 0
  - failed_specs: []
  - progress: {"done": 8, "total": 8}
  - mirror: "runs/latest/results.json"

### Varyans paneli: özet, eksen atfı, narrative, efektif N, pre-trend

- Durum: **GEÇTİ**
- Sonuç: bant robust, 1 eksen yorumu, 8 pre-trend katsayısı
- Ölçümler:
  - summary: {"n_total": 8, "n_ok": 8, "n_failed": 0, "sign_agreement": 1.0, "significance_rate": 1.0, "point_min": -10.968781, "point_max": -1.884269, "modal_sign": -1, "band": "robust"}
  - n_used_in_diagnosis: 8
  - anova_partial_r2: {"control_set": 0.400736, "sample": null, "pre_period": null, "clustering": null, "never_treated": null, "estimator": 0.978261, "weighting": 0.362169}
  - diagnosis_warnings: ["sample: tek seviyeli eksen; partial-R² hesaplanmadı.", "pre_period: tek seviyeli eksen; partial-R² hesaplanmadı.", "clustering: tek seviyeli eksen; partial-R² hesaplanmadı.", "never_treated: tek seviyeli eksen; partial-R² hesaplanmadı."]
  - narrative_axes: ["estimator"]
  - effective_n: {"spec_0000": 23606, "spec_0001": 23606, "spec_0002": 23606, "spec_0003": 23606, "spec_0004": 23518, "spec_0005": 23518, "spec_0006": 23518, "spec_0007": 23518}
  - event_study_n_obs: 23606
  - event_study_points: 8

## Kapsam Sınırları

- JUDGE adımları test modeliyle koşar, canlı sağlayıcı davranışı ölçülmez
- gatekeeper kararları programatik onaylanır, arayüz etkileşimi kapsam dışı
- Streamlit sayfaları render edilmez, doğrulanan şey sayfaların çağırdığı zincir
- ham CDC dosyası repoda tutulmadığından bu koşu yerel veri gerektirir
