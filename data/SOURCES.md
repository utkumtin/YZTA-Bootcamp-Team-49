# Veri Kaynakları (DCAS — Data & Code Availability Statement)

LICENSE **kodu** kapsar; veri kendi şartlarında. Kullanıcı **kendi ürettiği çıktının**
(temizleme kodu/metot) sahibidir. Tüm kaynaklar aşağıdaki atıflarla redistribute edilebilir.

Edinim iki yolla: `python data/fetch_data.py --fetch` (programatik kaynaklar) ve
manuel export (CDC WONDER, BLS LAUS — `--check` eksikleri ve yönergeleri listeler).

## Dizin düzeni (SCOPE §12)
```
data/
  SOURCES.md                 # bu dosya
  fetch_data.py              # veri-erişim yardımcısı (DCAS): --check / --fetch
  medicaid/                  # HERO: ACA Medicaid genişlemesi (uçtan uca demo)
    config.yaml
    raw/
      cdc_wonder_mortality_2009_2019.tsv    # büyük ham → git-ignore'lu (manuel export)
      kff_expansion_dates.csv               # küçük, redistribute → repo'da
      sahie_uninsured_2009_2019.csv         # türev extract (fetch) → repo'da
      saipe_income_poverty_2009_2019.csv    # türev extract (fetch) → repo'da
      laus/laucnty{09..19}.xlsx             # manuel export (BLS) → repo'da (küçük)
  divorce/                   # kanonik flip seti (Sprint-1 de-risk, primer)
    config.yaml
    raw/{bacon_example.dta†, divorce.csv}
  castle/                    # kanonik flip seti (ikincil/smoke)
    config.yaml
    raw/{castle.dta†, castle.csv}
  card_krueger/              # minik 2×2 fixture (CI smoke matrisi)
    config.yaml
    raw/{njmin.zip, codebook, read.me, card_krueger.csv}
```
† `.dta` git-ignore'lu; CSV export'lar (türev/temiz extract) repo'da tutulur.

## Hero: Medicaid genişlemesi

### 1. CDC WONDER — Multiple Cause of Death (mortalite)
- **Rol:** ilçe-yıl mortalite (temizleme yakıtı + "kırılgan" sonuç değişkeni).
- **Kapsam:** 2009-2019. (1999-2008 export hatası nedeniyle dışarıda; 2020 COVID
  nedeniyle **bilerek** dışarıda.)
- **Lisans:** ABD kamu malı (public domain).
- **DUA-uyumu:** hücre-suppression KORUNUR (Deaths < 10 → suppressed). Suppressed hücreler
  NA/efektif-N olarak ele alınır, **yeniden kurulmaz**; yeniden-kimliklendirme yapılmaz.
- **Edinim:** MANUEL — CDC WONDER interaktif sorgu aracıdır (programatik indirme yok).
  Sorgu: ilçe × yıl gruplaması, tüm ilçeler. Export edilen `.tsv` `raw/`'a konur.
  Dosya sonundaki `---` sonrası metadata footer'ı beklenen davranıştır (loader atlar).
- **Atıf:** Centers for Disease Control and Prevention, National Center for Health
  Statistics. Multiple Cause of Death, CDC WONDER (export: 2009-2019). https://wonder.cdc.gov/mcd.html

### 2. KFF — Eyalet Medicaid genişleme tarihleri (tedavi)
- **Rol:** tedavi kohortu (genişleme yılı) + never-treated maskesi.
- **Lisans:** gerçek/kamuya açık; atıfla redistribute (küçük tablo repo'da).
- **Atıf:** KFF (Kaiser Family Foundation), "Status of State Medicaid Expansion Decisions."

### 3. Census SAHIE — Small Area Health Insurance Estimates (kapsam sonucu)
- **Rol:** ilçe-yıl sigortasız % (18-64) — ikinci sonuç değişkeni, hero demonun
  "robust" yüzü (kapsam etkisi spec'ler arası sağlam).
- **Lisans:** ABD kamu malı; atıfla.
- **Edinim:** `fetch_data.py --fetch` — key'siz statik sunucu
  `www2.census.gov/programs-surveys/sahie/datasets/time-series/estimates-acs/sahie-{yıl}-csv.zip`
  (Census API artık zorunlu key istiyor; o yüzden statik dosyalar). Yıllık zip indirilir,
  ilçe × (18-64, tüm gelir/ırk/cinsiyet toplamı) satırlarına filtrelenip tek CSV'ye
  birleştirilir; zip'ler saklanmaz. Türev extract repo'da.
- **Atıf:** U.S. Census Bureau, Small Area Health Insurance Estimates (SAHIE) Program.

### 4. Census SAIPE — Small Area Income and Poverty Estimates (kontroller)
- **Rol:** kontrol değişkenleri — medyan hane geliri + yoksulluk oranı (ilçe-yıl).
- **Lisans:** ABD kamu malı; atıfla.
- **Edinim:** `fetch_data.py --fetch` — `www2.census.gov/programs-surveys/saipe/datasets/`
  altındaki `est{yy}all.txt` dosyaları parse edilir ('.' = suppressed → NA), yalnız ilçe
  satırları tutulur, FIPS'ler zero-pad'lenir. Türev extract repo'da.
- **Atıf:** U.S. Census Bureau, Small Area Income and Poverty Estimates (SAIPE) Program.

### 5. BLS LAUS — ilçe yıllık işsizlik (kontrol)
- **Rol:** kontrol değişkeni (işsizlik oranı, ilçe-yıl).
- **Lisans:** ABD kamu malı; atıfla.
- **Edinim:** MANUEL — BLS, script indirmelerini 403 ile engelliyor. Tarayıcıdan
  `https://www.bls.gov/lau/laucnty{yy}.xlsx` (yy = 09..19) indirilip
  `medicaid/raw/laus/` altına konur. `fetch_data.py --check` eksikleri listeler.
  Dosyalar KASITLI ham tutulur (başlık/footer satırları, birleşik ilçe+eyalet
  kolonu) — temizleme agent'ının Excel-ingest vitrini; okuma ipuçları config.yaml'da.
- **Atıf:** U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics.

## Kanonik flip setleri (Sprint-1 de-risk spike; SCOPE §4)

### 6. `divorce` — Stevenson-Wolfers no-fault divorce (primer flip seti)
- **Rol:** estimator-flip'in en temiz belgeli örneği (Goodman-Bacon 2021'in çalışma
  verisi); veri-tesisatından bağımsız spike. Eyalet-yıl 1964-1996, kademeli yasalaşma.
- **Lisans:** yayımlanmış replikasyon verisi; atıfla redistribute.
- **Edinim:** `fetch_data.py --fetch` — http://pped.org/bacon_example.dta → CSV export.
- **Atıf:** Stevenson & Wolfers (2006), "Bargaining in the Shadow of the Law: Divorce
  Laws and Family Distress," QJE. Veri: Goodman-Bacon (2021), "Difference-in-differences
  with variation in treatment timing," Journal of Econometrics (bacondecomp örnek verisi).

### 7. `castle` — Cheng-Hoekstra castle doctrine (ikincil flip/smoke seti)
- **Rol:** ikincil flip/smoke; kademeli yasalaşma (2005-2009), zengin kontrol seti.
- **Lisans:** yayımlanmış replikasyon verisi; atıfla redistribute.
- **Edinim:** `fetch_data.py --fetch` — github.com/scunning1975/mixtape (castle.dta) → CSV.
- **Atıf:** Cheng & Hoekstra (2013), "Does Strengthening Self-Defense Law Deter Crime or
  Escalate Violence?" JHR. Veri: Cunningham, "Causal Inference: The Mixtape."

## Fixture & breadth

### 8. Card-Krueger NJ-PA (minik 2×2 fixture, CI smoke)
- **Rol:** hızlı test + genelleme garantisi (SCOPE §11 smoke matrisi). 410 restoran ×
  2 dalga, GENİŞ format (reshape temizleme egzersizi).
- **Lisans:** yazarın sitesinden kamuya açık; atıfla.
- **Edinim:** `fetch_data.py --fetch` — davidcard.berkeley.edu/data_sets/njmin.zip;
  `public.dat` codebook'taki 46 kolonla CSV'ye çevrilir ('.' → NA).
- **Atıf:** Card & Krueger (1994), "Minimum Wages and Employment: A Case Study of the
  Fast-Food Industry in New Jersey and Pennsylvania," AER.

### 9. Silberzahn kırmızı kart (opsiyonel upside — EDİNİLMEDİ)
- **Rol:** yalnız saf-OLS-multiverse breadth upside'ı inerse (DiD-dışı, Silberzahn
  many-analysts verisi). Şimdilik yalnız pointer.
- **Edinim:** OSF projesi — https://osf.io/gvm2z/ (`CrowdstormingDataJuly1st.csv`).
- **Atıf:** Silberzahn et al. (2018), "Many Analysts, One Data Set," AMPPS.
