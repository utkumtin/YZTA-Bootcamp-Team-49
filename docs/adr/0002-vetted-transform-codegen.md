# 2. Codegen: kapalı/vetted transform kütüphanesi (LLM keyfi kod üretmez)

- Durum: accepted
- Tarih: 2026-07-05
- İlgili: review.md sorun #1 (kritik)

## Bağlam
Prototip (`prototype/pareto/p1_cleaning/code_executor.py`) LLM'e keyfi pandas kodu
ürettirip `exec` ile aynı süreçte çalıştırıyordu; `__builtins__` global namespace'e
veriliyordu → üretilen kod `__import__` ile FS/network'e çıkabilir (RCE). SCOPE L3 bunu
açıkça yasaklıyor.

## Karar
Kod, kapalı/vetted transform sözlüğünden (`pareto/cleaning/transforms.py`) SABİT şablonla
RENDER edilir. LLM yalnız transform_name + tipli params seçer (Pydantic-kapalı sözlük,
`ledger.LedgerEntry` validator ile zorlanır). `exec` yok; uygulama saf `apply` fonksiyonlarıyla.

## Sonuç
Blast-radius sınırlı (L3). Kapsam SCOPE §2 katman 2 ile sınırlı (type-coercion/leading-zero/
tarih/NA/rename/duplicate). Subprocess sandbox + eşitlik-assert (L4) reprodüksiyon paketiyle
Sprint-2/3. Genişletme = registry'e yeni vetted transform eklemek.
