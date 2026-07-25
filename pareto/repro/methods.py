"""Metot bölümü taslağı — deterministik şablon, LLM YOK.

Ürünün iddiası "denetim izi = metot bölümü". Bu modül o iddiayı somutlaştırır:
paketin MANIFEST.json'unu okur ve makalenin metot bölümüne doğrudan taşınabilir
bir Markdown taslağı render eder. Ölçen ya da yorumlayan hiçbir adımda LLM yoktur;
şablon aynı manifest için her zaman aynı metni üretir.
"""

from __future__ import annotations

from typing import Any

_BAND_TEXT: dict[str, str] = {
    "robust": "dayanıklı",
    "mixed": "karışık",
    "fragile": "kırılgan",
}


def _percent(value: Any) -> str:
    return f"%{float(value) * 100:.0f}" if isinstance(value, int | float) else "hesaplanamadı"


def _estimand_section(manifest: dict[str, Any]) -> list[str]:
    estimand = manifest.get("estimand")
    if not isinstance(estimand, dict):
        return [
            "Dondurulmuş estimand kaydı bu pakette yok; aşağıdaki spesifikasyonlar "
            "estimand beyanı olmadan raporlanmaktadır.",
        ]
    lines = [
        "Analiz, spesifikasyonlar üretilmeden ÖNCE dondurulan bir estimand üzerine kuruludur "
        f"(estimand hash: `{manifest.get('estimand_hash')}`).",
        "",
        f"- Estimand tipi: {estimand.get('estimand_type', 'belirtilmemiş')}",
        f"- Çıktı değişkeni (outcome): `{estimand.get('outcome', 'belirtilmemiş')}`"
        + (f", birim: {estimand['outcome_unit']}" if estimand.get("outcome_unit") else ""),
        f"- Müdahale kodlaması (treatment): `{estimand.get('treatment_coding', 'belirtilmemiş')}`",
    ]
    for field_name, label in (
        ("population", "Hedef popülasyon"),
        ("time_scope", "Zaman kapsamı"),
        ("identification_assumption", "Tanımlayıcı varsayım"),
        ("expected_sign", "Beklenen etki yönü"),
        ("h0", "Sıfır hipotezi"),
        ("h1", "Alternatif hipotez"),
    ):
        value = estimand.get(field_name)
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _provenance_note(manifest: dict[str, Any]) -> list[str]:
    """Karar defterinin bu sonuçlara ait olduğu doğrulandı mı.

    Temizleme ve multiverse ayrı koşulardır; taslak, doğrulanmamış bir defteri
    doğrulanmış gibi sunarsa metot bölümü olduğu iddiasını kaybeder.
    """
    provenance = manifest.get("provenance") or {}
    matches = provenance.get("cleaning_matches_panel")
    if matches is True:
        return []
    if matches is False:
        return [
            "> **UYARI:** Aşağıdaki karar defteri, analize giren panelle eşleşmeyen "
            "bir temizleme koşusundan gelmektedir "
            f"(temizleme run_id: `{provenance.get('cleaning_run_id')}`). Bu bölüm "
            "doğrulanana kadar yayımlanmamalıdır.",
            "",
        ]
    return [
        "> **Not:** Karar defterinin bu sonuçları üreten veriye ait olduğu "
        "doğrulanamamıştır (temizleme sandbox çıktısı pakete girmedi).",
        "",
    ]


def _cleaning_section(manifest: dict[str, Any]) -> list[str]:
    decisions = manifest.get("cleaning_decisions") or []
    if not decisions:
        return [
            "Bu pakette karar defteri bulunmuyor; veri temizleme adımları "
            "burada raporlanamamaktadır.",
        ]

    note = _provenance_note(manifest)
    flagged = sum(1 for d in decisions if d.get("belirsizlik_bayragi"))
    rejected = sum(1 for d in decisions if d.get("resolution") == "rejected")
    lines = [
        *note,
        f"Veri temizleme {len(decisions)} karardan oluşur. Bunların {flagged} tanesi "
        "belirsizlik bayrağı taşıdığı için insan onayına sunulmuş, "
        f"{rejected} tanesi reddedilmiştir. Her karar kapalı bir transform sözlüğünden "
        "seçilir; uygulanan kod bu kararlardan deterministik olarak render edilmiştir "
        "(`cleaning/cleaning_steps.py`).",
        "",
        "| # | Bulgu | Transform | Gerekçe | Karar |",
        "|---|-------|-----------|---------|-------|",
    ]
    for index, decision in enumerate(decisions, start=1):
        resolution = decision.get("resolution") or "otomatik onay"
        lines.append(
            f"| {index} | {decision.get('bulgu', '')} | `{decision.get('transform_name', '')}` "
            f"| {decision.get('gerekce', '')} | {resolution} |"
        )
    return lines


def _menu_section(manifest: dict[str, Any]) -> list[str]:
    spec_count = manifest.get("spec_count", 0)
    if not spec_count:
        return ["Bu pakette spesifikasyon listesi bulunmuyor."]
    return [
        f"Dondurulan spesifikasyon menüsü (menü hash: `{manifest.get('menu_hash')}`) "
        f"{spec_count} savunulabilir spesifikasyona genişletilmiştir. Menü, sonuçlar "
        "görülmeden dondurulur; her spesifikasyonun içerik hash'i manifest'te kayıtlıdır, "
        "yani rapor edilen küme sonradan kırpılamaz.",
    ]


def _results_section(manifest: dict[str, Any]) -> list[str]:
    summary = manifest.get("variance_summary") or {}
    band = _BAND_TEXT.get(str(summary.get("band")), "belirlenemedi")
    lines = [
        f"Menüdeki {summary.get('n_total', 0)} spesifikasyonun "
        f"{summary.get('n_ok', 0)} tanesi başarıyla tahmin edilmiştir "
        f"({summary.get('n_failed', 0)} tahmin başarısız olmuştur ve gizlenmemiştir).",
        "",
        f"- İşaret uyumu: {_percent(summary.get('sign_agreement'))}",
        f"- Anlamlılık oranı: {_percent(summary.get('significance_rate'))}",
    ]
    point_min = summary.get("point_min")
    point_max = summary.get("point_max")
    if isinstance(point_min, int | float) and isinstance(point_max, int | float):
        lines.append(f"- Nokta tahmini aralığı: {point_min:.4f} ile {point_max:.4f} arası")
    lines += [
        "",
        f"Bu dağılım deterministik üç-bant kuralına göre **{band}** olarak "
        "etiketlenmiştir. Etiket betimleyicidir, formal bir birleşik test değildir.",
    ]
    return lines


def render_methods_section(manifest: dict[str, Any]) -> str:
    """Paket manifest'inden metot bölümü taslağını render eder."""
    determinism = manifest.get("determinism") or {}
    sections: list[str] = [
        f"# Metot bölümü taslağı — {manifest.get('run_id', 'bilinmeyen koşu')}",
        "",
        "Bu taslak, koşunun denetim izinden deterministik bir şablonla üretilmiştir; "
        "hiçbir cümlesi dil modeli tarafından yazılmamıştır. Metni olduğu gibi "
        "kullanmadan önce alan bilgisiyle gözden geçirin.",
        "",
        "## 1. Veri ve temizleme",
        "",
        *_cleaning_section(manifest),
        "",
        "## 2. Estimand",
        "",
        *_estimand_section(manifest),
        "",
        "## 3. Spesifikasyon menüsü",
        "",
        *_menu_section(manifest),
        "",
        "## 4. Sonuçlar",
        "",
        *_results_section(manifest),
        "",
        "## 5. Reprodüksiyon",
        "",
        f"Tüm koşu `seed={determinism.get('seed')}` ve `{determinism.get('env')}` "
        "determinizm pinleriyle yapılmıştır. Paketin kökündeki `run_reproduction.py` "
        "script'i paneli ve spesifikasyonları yeniden koşar ve sonuçları tolerans "
        "içinde doğrular; uyuşmazlıkta sıfırdan farklı bir çıkış koduyla biter.",
    ]
    missing = manifest.get("missing") or []
    if missing:
        sections += [
            "",
            "## 6. Bu taslakta eksik kalanlar",
            "",
            "Aşağıdaki artefaktlar pakete giremediği için ilgili bölümler eksiktir: "
            + ", ".join(str(item) for item in missing)
            + ".",
        ]
    return "\n".join(sections) + "\n"
