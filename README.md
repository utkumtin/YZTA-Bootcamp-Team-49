# Team 49 · Pareto

> **Pareto**: a defensible-inference engine for empirical social-science researchers.
> *Not one confident answer, but a distribution over defensible choices, and where & why it is fragile.*

---

## Team Members

| Member | Role | Social |
|--------|------|--------|
| [Utku Metin](https://github.com/utkumtin) | Product Owner | [LinkedIn](https://www.linkedin.com/in/utkumtn/) |
| [Ozan Çelik](https://github.com/Ozan7146) | Scrum Master | [LinkedIn](https://www.linkedin.com/in/ozan-%C3%A7elik-7b8062221/) |
| [Betül Bostan](https://github.com/betul-bostan) | Developer | [LinkedIn](https://www.linkedin.com/in/bet%C3%BCl-bostan-2105942b2/) |
| [Utku Uzunhüseyin](https://github.com/utkuzuunhuseyin) | Developer | [LinkedIn](https://www.linkedin.com/in/utku-uzunh%C3%BCseyin/) |

---

<details open>
<summary><h2>Product Description</h2></summary>

**Pareto** is a defensible-inference engine for empirical social-science researchers. Instead
of a single confident number, it returns a **distribution over defensible choices**, and tells
you *where* and *why* the result is fragile.

Two moves, one promise (**a defensible result**):

1. **Black-box-free, human-in-the-loop cleaning.** Every fix is a logged decision (a *decision
   ledger*) plus generated, reproducible code: your audit trail and methods section, for free.
2. **Context-aware multiverse analysis.** Pareto runs controls × sample × estimator
   specifications in parallel and surfaces the spread in a **variance panel** that says not just
   *"fragile"* but *why*: "8 of 10 runs are positive; the 2 that flip do so when the estimator
   goes TWFE → Callaway-Sant'Anna."

<details>
<summary><h4>Türkçe açıklama</h4></summary>

**Pareto**, ampirik sosyal-bilim araştırmacıları için bir *savunulabilir çıkarım* motorudur.
Tek bir kesin sayı yerine, **savunulabilir seçimler üzerinde bir dağılım** verir; sonucun
*nerede* ve *neden* kırılgan olduğunu söyler.

İki hamle, tek söz (**savunulabilir sonuç**):

1. **Kara-kutu olmayan, human-in-the-loop temizleme.** Her düzeltme loglanmış bir karardır
   (*karar defteri*) + üretilen tekrarlanabilir kod: denetim iziniz ve metot bölümünüz hazır gelir.
2. **Bağlam-farkında çokluevren analizi.** Kontrol × örneklem × estimator spesifikasyonları
   paralel koşulur; dağılım **varyans panelinde** yalnız *"kırılgan"* değil *nedenini* söyleyerek
   gösterilir: "10 koşunun 8'i pozitif; dönen 2'si estimator TWFE → Callaway-Sant'Anna olunca döner."

</details>
</details>

---

<details>
<summary><h2>Product Features</h2></summary>

- **Ingest & profiling**: CSV/Excel (committed), `.dta` (cheap). Deterministic column profiling
  sends only a *summary payload* to the LLM; raw rows never leave the machine.
- **Human-in-the-loop cleaning agent**: auto-applies high-confidence fixes, asks only genuine
  ambiguities, logs a **decision ledger**, and emits reproducible code from a **closed, vetted
  transform library** (no arbitrary code generation). Panel-merges multiple sources into a
  unit×time panel.
- **Estimand-first hypothesis**: Socratic + TAC elicitation produces a **frozen, hashed
  Estimand** (H0/H1, treatment, identification assumption) *before* any result is seen; kills anchoring.
- **Defensible spec-menu**: a judge model proposes defensible levels per axis + a baseline; the
  researcher "draws the lit path" with 4 verbs (approve / edit / remove / add). The menu is
  **frozen + hashed** for reproducibility.
- **Multiverse runner**: a deterministic subprocess engine runs the factorial cross-product of
  active axes (hard cap **24 specs**) via `pyfixest`, with live disk progress.
- **Variance panel**: specification curve + summary (% positive, sign changes) + deterministic
  matched-pair / ANOVA **axis attribution** + a **3-band robust/fragile label** + pre-trend
  event-study + an LLM narrative that *explains* (never measures).
- **Cross-cutting**: a model router (cheap model for mechanical work, a pinned strong model for
  judgment), persistent project memory, a one-click reproduction bundle, and **privacy-by-design**
  (public/demo vs private no-train modes).

<details>
<summary><h4>Türkçe açıklama</h4></summary>

- **Ingest & profilleme**: CSV/Excel (committed), `.dta` (ucuz). Deterministik kolon profillemesi
  LLM'e yalnız *özet payload* gönderir; ham satırlar makineden çıkmaz.
- **Human-in-the-loop temizleme ajanı**: yüksek-güvenli düzeltmeleri otomatik uygular, yalnız
  gerçek belirsizlikleri sorar, bir **karar defteri** tutar ve **kapalı/vetted transform
  kütüphanesinden** tekrarlanabilir kod üretir (keyfi kod üretimi YOK). Çok kaynağı unit×time
  panele birleştirir.
- **Estimand-first hipotez**: Socratic + TAC ile, herhangi bir sonuç görülmeden **donmuş, hash'li
  Estimand** (H0/H1, tedavi, tanımlama varsayımı) üretilir; anchoring'i öldürür.
- **Savunulabilir spec-menü**: yargı modeli her eksende savunulabilir seviyeler + baseline önerir;
  araştırmacı 4 fiille (onayla / düzenle / çıkar / ekle) "aydınlık yolu çizer". Menü reprodüksiyon
  için **dondurulur + hash'lenir**.
- **Multiverse runner**: deterministik subprocess motoru aktif eksenlerin faktöriyel çarpımını
  (sert tavan **24 spec**) `pyfixest` ile koşar, diske canlı ilerleme yazar.
- **Varyans paneli**: spec-curve + özet (% pozitif, işaret değişimi) + deterministik matched-pair /
  ANOVA **eksen atfı** + **3-bant robust/fragile etiketi** + pre-trend event-study + *açıklayan*
  (asla ölçmeyen) bir LLM narrative.
- **Kesişen**: model router (mekanik iş ucuz modele, yargı pinli güçlü modele), kalıcı proje
  hafızası, tek-tık reprodüksiyon paketi ve **tasarımdan-gizlilik** (public/demo vs private no-train).

</details>
</details>

---

<details>
<summary><h2>Target Audience</h2></summary>

- **Beachhead:** PhD students and early-career empirical social-science researchers at the
  **econometrics / causal-inference / difference-in-differences** intersection: those who feel the
  data-drudgery pain and retraction fear most, and are least locked into legacy workflows.
- **Broader field:** economics, political science, public-health policy, sociology, empirical finance.

<details>
<summary><h4>Türkçe açıklama</h4></summary>

- **Beachhead:** **ekonometri / nedensel çıkarım / fark-farkın-farkı (DiD)** kesişimindeki doktora
  ve erken-kariyer ampirik sosyal-bilim araştırmacıları: veri angaryasını ve geri-çekilme (retraction)
  korkusunu en çok yaşayan, eski iş akışlarına en az bağlı grup.
- **Geniş alan:** ekonomi, siyaset bilimi, halk sağlığı politikası, sosyoloji, ampirik finans.

</details>
</details>

---

## Product Backlog

We run the project from a **GitHub Projects Kanban** board (Backlog · Ready · In Progress · In
Review · Done). Each card is a user story with story points, an owner, and a task/acceptance checklist.

- **Board:** [GitHub Projects Kanban](https://github.com/users/utkumtin/projects/3)
- **Screenshot:** ![Kanban board (Sprint 2)](docs/img/kanban-board-0507.png)

---

# Sprint 1

<details>
<summary><h2>Product Status</h2></summary>

Sprint 1 was design/foundation-heavy, so by design there is no empirical run yet. The product
status is a running, end-to-end **Streamlit walking skeleton** on top of the typed core: the
landing page (mode + BYOK) and the three flow pages. LLM-driven steps land in Sprint 2, and each
page says so explicitly instead of faking output (fail-loud):

| Landing: mode + BYOK + flow overview | 1 · Cleaning: upload + profiling entry |
|---|---|
| ![Landing page](docs/img/landing.png) | ![Cleaning page](docs/img/cleaning.png) |

| 2 · Analysis: estimand-first seam (Sprint 2) | 3 · Variance panel: reads runner output |
|---|---|
| ![Analysis page](docs/img/analysis.png) | ![Variance panel page](docs/img/variance-panel.png) |

<details>
<summary><h4>Türkçe açıklama</h4></summary>

Sprint 1 tasarım/temel ağırlıklıydı; tasarım gereği henüz ampirik koşu yok. Ürün durumu, tipli
çekirdeğin üzerinde uçtan uca çalışan **Streamlit yürüyen iskeleti**: açılış sayfası (mod + BYOK)
ve üç akış sayfası. LLM'li adımlar Sprint 2'de gelir; her sayfa bunu sahte çıktı göstermek yerine
açıkça söyler (fail-loud). Görseller: açılış · 1·Temizleme (yükleme + profilleme girişi) ·
2·Analiz (estimand-first dikişi) · 3·Varyans Paneli (runner çıktısını okur).

</details>
</details>

<details>
<summary><h2>Project Management / Board</h2></summary>

GitHub Projects Kanban with per-column WIP limits (In Progress 4 · In Review 3) and a
numeric `Estimate` field per card (Fibonacci story points):

![GitHub Projects board](docs/img/kanban-board-0507.png)

</details>

- **Sprint Notes:**
  * Locked the idea & thesis **"a defensible result"**: the variance *is* the product.
  * Wrote the scope document (SCOPE): committed / fast-follow / upside; **hero dataset = ACA
    Medicaid expansion**.
  * Accepted **5 architecture decisions (ADR 0001–0005):** single engine (`Specification` atom) ·
    closed/vetted codegen (the LLM never writes arbitrary code) · subprocess runner (not
    ProcessPool) · PydanticAI + model router · single estimator library (**pyfixest**).
  * Migrated the prototype into a typed committed core, resolving **the 4 critical review
    conflicts** (codegen RCE `exec` → vetted-transform render · ProcessPool → subprocess · raw
    SDK + regex-JSON → PydanticAI · missing contracts/axes; hard cap 500 → 24).
  * Landed the skeleton + typed core + CI (uv · ruff · mypy · pre-commit + gitleaks · GitHub Actions).
- **Expected point completion within Sprint:** `34` Points
- **Point Completion Logic:** We estimate stories in **Fibonacci story points** (1·2·3·5·8·13),
  sizing *relative effort / complexity / uncertainty* rather than hours; stories above 13 are split.
  We do **not** pre-fix a grand total: each sprint's commitment is set by team availability and
  velocity is tracked sprint over sprint. **Sprint 1** (design / foundation) committed and completed
  **~34 points**; **Sprint 2** commits **84** (heaviest: committed core, end-to-end); **Sprint 3**
  (polish + ship) is estimated at its start.
- **Daily Scrum:** No daily stand-ups this sprint; the team synced in **3 Slack Huddle calls**
  (20 Jun · 21 Jun · 4 Jul), with async coordination in a shared WhatsApp group. A fixed daily
  cadence + channel (with archived notes) is a Sprint 2 action item (see Retrospective).
  <details>
  <summary>Huddle screenshots</summary>

  ![Slack Huddle, 20 Jun](docs/img/huddle-2006.png)
  ![Slack Huddle, 21 Jun](docs/img/huddle-2106.png)
  ![Slack Huddle, 4 Jul](docs/img/huddle-0407.png)

  </details>
- **Product Backlog URL:** [GitHub Projects Kanban](https://github.com/users/utkumtin/projects/3)
- **Sprint Review:**
  * The first sprint was spent on ideation, scoping, and locking the architecture; no empirical
    run yet, by design.
  * Product thesis and target audience were fixed; the committed / fast-follow / upside boundary
    was drawn and the ACA Medicaid hero chosen.
  * 5 ADRs were accepted and the prototype was migrated into a clean, typed core with green CI.
  * Decided that the empirical estimator-flip spike (`divorce` / `castle` + Medicaid core axis)
    moves to **Sprint 2** as an early go/no-go.
- **Sprint Review Participants:** Utku Metin, Ozan Çelik, Betül Bostan, Utku Uzunhüseyin
- **Sprint Retrospective:**
  * Estimate **story points at the start** of the sprint (the scheme was set up only at this
    sprint's end) → Sprint 2 begins with a fully-pointed board.
  * Fix the **daily-scrum cadence + channel** and archive screenshots regularly.
  * Pull the **empirical flip spike** earlier (go/no-go up front).
  * Adopt **availability-based planning** (U is unavailable in week 1 → cleaning backend pulled forward).
  * Write a **`TestModel` test for every new LLM step** from the start (intent over behaviour).
  * **Assign the Scrum Master** role explicitly (done: Ozan Çelik).

<details>
<summary><h4>Türkçe açıklama</h4></summary>

- **Sprint Notları:**
  * Fikir & tez kilitlendi, **"savunulabilir sonuç"**: varyansın kendisi üründür.
  * Kapsam dokümanı (SCOPE) yazıldı: committed / fast-follow / upside; **hero veri = ACA Medicaid genişlemesi**.
  * **5 mimari karar (ADR 0001–0005) kabul edildi:** tek motor (`Specification` atomu) · kapalı/vetted
    codegen (LLM keyfi kod yazmaz) · subprocess runner (ProcessPool değil) · PydanticAI + model
    router · tek estimator kütüphanesi (**pyfixest**).
  * Prototip tipli committed core'a migre edildi; review'in **4 kritik çatışması** giderildi (codegen
    RCE `exec` → vetted-transform render · ProcessPool → subprocess · ham SDK + regex-JSON →
    PydanticAI · eksik kontrat/eksen; sert tavan 500 → 24).
  * İskelet + tipli çekirdek + CI (uv · ruff · mypy · pre-commit + gitleaks · GitHub Actions) indirildi.
- **Sprint İçinde Tamamlanması Beklenen Puan:** `34` Puan
- **Puan Tamamlama Mantığı:** Story'ler **Fibonacci story point** (1·2·3·5·8·13) ile puanlanır;
  saat değil *göreli efor / karmaşıklık / belirsizlik* ölçülür; 13 üstü story bölünür. **Sabit bir
  proje-toplamı belirlenmez:** her sprint taahhüdü ekip müsaitliğine göre kurulur, velocity sprint
  sprint izlenir. **Sprint 1** (tasarım / temel) **~34 puan** taahhüt edip tamamladı; **Sprint 2**
  **84** taahhüt eder (en ağır: committed çekirdek, uçtan uca); **Sprint 3** (cila + ship) sprint
  başında tahmin edilir.
- **Daily Scrum:** Bu sprint günlük stand-up yapılmadı; **3 Slack Huddle görüşmesi** (20.06 ·
  21.06 · 04.07) ve WhatsApp grubunda asenkron koordinasyon ile senkronize olundu (ekran
  görüntüleri yukarıdaki İngilizce bölümde). Sabit günlük kadans + kanal (arşivlenen notlarla)
  Sprint 2 aksiyon maddesidir (bkz Retrospektif).
- **Sprint Gözden Geçirilmesi (Review):**
  * İlk sprint fikir, kapsam ve mimari kilidiyle geçti; tasarım gereği henüz ampirik koşu yok.
  * Ürün tezi ve hedef kitle sabitlendi; committed / fast-follow / upside sınırı çizildi, Medicaid hero seçildi.
  * 5 ADR kabul edildi, prototip temiz tipli çekirdeğe migre edildi, CI yeşil.
  * Ampirik estimator-flip spike'ının (`divorce` / `castle` + Medicaid çekirdek ekseni) erken go/no-go
    olarak **Sprint 2'ye** taşınmasına karar verildi.
- **Sprint Gözden Geçirme Katılımcıları:** Utku Metin, Ozan Çelik, Betül Bostan, Utku Uzunhüseyin
- **Sprint Retrospektifi:**
  * Story-point tahmini sprint **başında** yapılmalı (şema bu sprint sonunda kuruldu) → Sprint 2 baştan puanlı board ile başlar.
  * Daily-scrum kadansı + kanalı netleştir, screenshot'ları düzenli arşivle.
  * Ampirik **flip spike** öne alınmalı (go/no-go baştan).
  * **Müsaitlik-bazlı planlama** benimsendi (U Hafta-1 yok → temizleme backend'i öne çekildi).
  * Her yeni LLM adımı için baştan **`TestModel` testi** yazılmalı (davranış değil niyet).
  * **Scrum Master** rolü net atanmalı (atandı: Ozan Çelik).

</details>

---

## Technical Details

**Single engine.** The atomic unit is a **Specification** (`pareto/spec.py`):
`{outcome, regressors, fixed_effects, clustering, sample, estimator}`. OLS, TWFE-DiD and staggered
estimators are different points in this space; the estimator is just one axis → no "two forked
systems" risk.

```
app/                     # Streamlit UI (main + pages: 1_cleaning, 2_analysis, 3_variance_panel)
pareto/
  spec.py                # Specification (Pydantic atom)
  profiling.py           # deterministik kolon profilleme (LLM'e özet payload)
  contracts.py           # CleanPanel kontratı + fail-loud validate_contract() + EstimationResult
  config.py              # ayarlar: model-router rolleri, privacy modu, sert tavan 24
  cleaning/              # agent, ledger, merge, codegen, transforms (kapalı/vetted kütüphane)
  analysis/              # hypothesis (estimand), menu (dondurma+faktöriyel), runner, estimators, variance
  llm/                   # router (PydanticAI), providers (model zincirleri), guardrails (spotlighting)
  memory/store.py        # proje-store / hafıza (disk)
data/  notebooks/  docs/{adr,scrum}  tests/
```

**Stack:** Python · Streamlit (Community Cloud, canned-default + BYOK) · pandas · **pyfixest**
(OLS/TWFE, ADR 0005) · **PydanticAI** + model router (judge pinned = Gemini Flash, thinking on;
mechanical → Gemini Flash-Lite / Groq failover; `TestModel` for API-free tests) · subprocess
multiverse runner (seed + `PYTHONHASHSEED` pinned) · uv · ruff · mypy · pre-commit + gitleaks ·
GitHub Actions. Independent validation via R (`did` / `differences`) in `notebooks/` only.

## Setup

```bash
uv sync --extra dev          # veya: pip install -e ".[dev]"
cp .env.example .env         # BYOK anahtarlarını doldur (opsiyonel; demo canned)
streamlit run app/main.py
```

## License

Apache-2.0.
