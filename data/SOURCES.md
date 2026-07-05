# Veri Kaynakları (DCAS — Data & Code Availability Statement)

LICENSE **kodu** kapsar; veri kendi şartlarında. Kullanıcı **kendi ürettiği çıktının**
(temizleme kodu/metot) sahibidir. Üç kaynak da redistribute edilebilir (aşağıdaki atıflarla).

## Dizin düzeni (SCOPE §12)
```
data/
  SOURCES.md                 # bu dosya
  fetch_data.py              # veri-erişim yardımcısı (DCAS)
  medicaid/
    config.yaml              # kolon rolleri / merge / treatment-kohort (veri-agnostik core yükler)
    raw/
      cdc_wonder_mortality_1999_2020.tsv   # büyük ham → git-ignore'lu (manuel export)
      kff_expansion_dates.csv               # küçük, redistribute → repo'da
```
Türev/temiz extract'ler (canned demo yakıtı) repo'da tutulur; büyük ham girdiler değil.

## Kaynaklar

### 1. CDC WONDER — Multiple Cause of Death, 1999-2020 (mortalite)
- **Rol:** ilçe-yıl mortalite (temizleme yakıtı + sonuç değişkeni).
- **Lisans:** ABD kamu malı (public domain).
- **DUA-uyumu:** hücre-suppression KORUNUR (Deaths < 10 → suppressed). Suppressed hücreler
  NA/efektif-N olarak ele alınır, **yeniden kurulmaz**; yeniden-kimliklendirme yapılmaz.
- **Edinim:** CDC WONDER interaktif sorgu aracıdır (programatik indirme yok). Sorgu:
  ilçe × yıl gruplaması, tüm ilçeler, 1999-2020. Export edilen `.tsv` `raw/`'a konur.
  Dosya sonundaki `---` sonrası metadata footer'ı beklenen davranıştır (loader atlar).
- **Atıf:** Centers for Disease Control and Prevention, National Center for Health
  Statistics. Multiple Cause of Death 1999-2020, CDC WONDER. http://wonder.cdc.gov/mcd.html

### 2. KFF — Eyalet Medicaid genişleme tarihleri (tedavi)
- **Rol:** tedavi kohortu (genişleme yılı) + never-treated maskesi.
- **Lisans:** gerçek/kamuya açık; atıfla redistribute (küçük tablo repo'da).
- **Atıf:** KFF (Kaiser Family Foundation), "Status of State Medicaid Expansion Decisions."

### 3. Census / ACS — kontroller (Sprint-2)
- **Rol:** kontrol değişkenleri (gelir, işsizlik, demografi).
- **Lisans:** ABD kamu malı; atıfla.
- **Edinim:** `fetch_data.py` (Census API / ACS 5-yıl); henüz eklenmedi (Sprint-2).
