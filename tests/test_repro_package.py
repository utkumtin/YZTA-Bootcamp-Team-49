"""S3-04 reprodüksiyon paketi testleri.

Paketin sözü iki tanedir ve ikisi de burada sınanır: (1) denetim izinin hiçbir
parçası sessizce düşmez, eksik varsa manifest'e yazılır; (2) paketteki tek
komutluk script paketlenmiş sonuçları gerçekten yeniden üretir, uyuşmazlıkta
sıfırdan farklı çıkış koduyla biter.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pareto.repro import (
    ReproInputs,
    ReproPackageError,
    build_reproduction_package,
    figure_html,
    missing_artifacts,
    package_key,
    render_methods_section,
)
from pareto.spec import Specification

pyfixest = pytest.importorskip("pyfixest")  # estimator dep yoksa atla, CI'da koşar

from pareto.analysis.runner import run_specs  # noqa: E402
from pareto.cleaning.codegen import REPRO_ATOL, REPRO_RTOL, render_audit_script  # noqa: E402
from pareto.cleaning.ledger import LedgerEntry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _panel(effect: float = 0.8, seed: int = 0, units: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(units):
        treated = unit % 2
        u_fe = rng.normal()
        for year in range(6):
            post = 1 if year >= 3 else 0
            d = treated * post
            y = effect * d + u_fe + 0.1 * year + rng.normal(0, 0.3)
            rows.append({"y": y, "d": d, "unit": unit, "year": year})
    return pd.DataFrame(rows)


def _specs() -> list[Specification]:
    return [
        Specification(spec_id="s1", outcome="y", treatment="d", cluster_by="unit"),
        Specification(
            spec_id="s2", outcome="y", treatment="d", controls=("year",), cluster_by=None
        ),
    ]


def _figure_html(name: str) -> str:
    """Figürü panelin kullandığı yardımcıyla üretir.

    Testte elle sabit bir HTML string'i vermek determinizm testini boşa çıkarır:
    Plotly varsayılan olarak her çağrıda yeni bir uuid div id'si basar, yani gerçek
    yolda paket byte'ları değişir ama sabit string'li test bunu göremez. Aynı
    nedenle HTML burada elle üretilmez, panelin çağırdığı `figure_html` çağrılır.
    """
    import plotly.graph_objects as go

    return figure_html(name, go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 9])]))


def _other_figure():
    """Aynı adla çizilmiş BAŞKA bir figür (panel yeniden çizdiğinde olan şey)."""
    import plotly.graph_objects as go

    return go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[9, 4, 1])])


def _make_run(tmp_path: Path, *, with_cleaning: bool = True, units: int = 40) -> ReproInputs:
    """Gerçek bir koşunun diskteki artefakt düzenini kurar."""
    panel = _panel(units=units)
    specs = _specs()
    results = run_specs(panel, specs)

    run_dir = tmp_path / "runs" / "demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "panel.pkl").write_bytes(pickle.dumps(panel))
    (run_dir / "specs.json").write_text(
        json.dumps([s.model_dump() for s in specs], ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps([r.model_dump() for r in results], ensure_ascii=False), encoding="utf-8"
    )

    store_dir = tmp_path / "store" / "demo-run"
    store_dir.mkdir(parents=True)
    (store_dir / "frozen_menu.json").write_text(
        json.dumps(
            {
                "estimand_hash": "abc123deadbeef01",
                "menu_hash": "feed0123cafe4567",
                "spec_count": len(specs),
                "estimand": {
                    "estimand_type": "ATT",
                    "outcome": "y",
                    "outcome_unit": "puan",
                    "treatment": "d",
                    "treatment_coding": "d",
                    "population": "tedavi edilen birimler",
                    "time_scope": "1-6 dönem",
                    "expected_sign": "positive",
                    "identification_assumption": "parallel_trends",
                    "h0": "etki yok",
                    "h1": "etki pozitif",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ledger_path: Path | None = None
    script_path: Path | None = None
    raw_path: Path | None = None
    cleaned_path: Path | None = None
    cleaning_run_id: str | None = None
    if with_cleaning:
        audit_dir = tmp_path / "audit_trail"
        audit_dir.mkdir(parents=True)
        entry = LedgerEntry(
            bulgu="Panelde tekrar eden satırlar var.",
            transform_name="drop_duplicates",
            params={"subset": None},
            gerekce="Tekrar eden satırlar tahmini yanlı hale getirir.",
            belirsizlik_bayragi=False,
            resolution="approved",
        )
        script_path = audit_dir / "clean_cleaning_steps.py"
        script_path.write_text(render_audit_script([entry]), encoding="utf-8")
        ledger_path = audit_dir / "clean_decision_ledger.jsonl"
        ledger_path.write_text(
            json.dumps(entry.model_dump(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cleaning_run_id = "clean"
        raw_path = audit_dir / "clean_repro" / "raw.pkl"
        raw_path.parent.mkdir(parents=True)
        # Ham veri = panel + tekrar eden satırlar; temizleme script'i tam olarak
        # paneli geri vermeli.
        raw = pd.concat([panel, panel.head(5)], ignore_index=True)
        raw_path.write_bytes(pickle.dumps(raw))
        # L4 sandbox'ının çıktısı: provenans denetimi bunu analiz paneliyle
        # karşılaştırır, yani gerçek bir koşuda ikisi aynı olmalı.
        cleaned_path = raw_path.with_name("reproduced.pkl")
        cleaned_path.write_bytes(pickle.dumps(panel))

    return ReproInputs(
        run_id="demo-run",
        results_path=run_dir / "results.json",
        specs_path=run_dir / "specs.json",
        panel_path=run_dir / "panel.pkl",
        frozen_menu_path=store_dir / "frozen_menu.json",
        ledger_path=ledger_path,
        cleaning_script_path=script_path,
        raw_panel_path=raw_path,
        cleaned_panel_path=cleaned_path,
        cleaning_run_id=cleaning_run_id,
        figures={"specification_curve.html": _figure_html("specification_curve.html")},
    )


def _names(payload: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return set(archive.namelist())


def _read(payload: bytes, name: str) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(name).decode("utf-8")


def _run_package(extract_dir: Path) -> subprocess.CompletedProcess:
    """Açılmış paketin doğrulama script'ini koşar.

    `pareto` PyPI'da değil: sandbox'ta da repo kökü PYTHONPATH'e eklenir
    (verify_reproduction ile aynı desen).
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": "0"}
    return subprocess.run(
        [sys.executable, "run_reproduction.py"],
        cwd=extract_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_package_carries_every_audit_trail_artifact(tmp_path):
    # NEDEN: ürün tezi "denetim izi = metot bölümü"; parçalardan biri pakete
    # girmezse paket tezi taşımıyor demektir.
    payload = build_reproduction_package(_make_run(tmp_path))
    assert _names(payload) >= {
        "README.md",
        "MANIFEST.json",
        "METHODS.md",
        "requirements.txt",
        "run_reproduction.py",
        "specs.json",
        "frozen_menu.json",
        "results.json",
        "data/panel.csv",
        "data/raw.csv",
        "cleaning/cleaning_steps.py",
        "cleaning/decision_ledger.jsonl",
        "figures/specification_curve.html",
        "figures/plotly.min.js",
    }

    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert manifest["missing"] == []
    assert manifest["estimand_hash"] == "abc123deadbeef01"
    assert manifest["menu_hash"] == "feed0123cafe4567"
    # Donmuş menünün her spec'i içerik hash'iyle kayıtlı → rapor edilen küme
    # sonradan kırpılamaz.
    assert set(manifest["spec_hashes"]) == {"s1", "s2"}
    assert manifest["determinism"]["seed"] == 20260704


@pytest.mark.parametrize("with_cleaning", [True, False])
@pytest.mark.parametrize("empty_specs", [False, True])
def test_manifest_contents_lists_every_packaged_file(tmp_path, with_cleaning, empty_specs):
    # NEDEN: MANIFEST paketin "tek makine-okunur kaydı"dır. Envanteri kendini ve
    # ondan türeyen dosyaları saymazsa, kaydı okuyup paketi denetleyen bir araç
    # eksik bir listeyi tam sanır — eksikliği gizleyen bir kayıt, kayıt değildir.
    #
    # Eksik artefaktlı kurulumlarla birlikte koşulur: `specs.json`, `frozen_menu.json`
    # ve temizleme dosyaları KOŞULLU eklenir, yani envanterin kırılgan olduğu yer
    # eksiksiz paket değil, eksik pakettir.
    inputs = _make_run(tmp_path, with_cleaning=with_cleaning)
    if empty_specs:
        assert inputs.specs_path is not None
        inputs.specs_path.write_text("[]", encoding="utf-8")

    payload = build_reproduction_package(inputs)
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert manifest["contents"] == sorted(_names(payload))


def test_frozen_menu_is_packaged_so_its_hashes_can_be_recomputed(tmp_path):
    # NEDEN: manifest `estimand_hash` ve `menu_hash` BEYAN eder. Hash'lerin
    # hesaplandığı dosya pakete girmezse okuyucu bu iki değeri hiçbir şeye karşı
    # doğrulayamaz; "menü sonuçlar görülmeden donduruldu" iddiası denetlenemez
    # bir cümleye düşer.
    inputs = _make_run(tmp_path)
    payload = build_reproduction_package(inputs)
    assert inputs.frozen_menu_path is not None
    assert _read(payload, "frozen_menu.json") == inputs.frozen_menu_path.read_text(encoding="utf-8")

    packaged = json.loads(_read(payload, "frozen_menu.json"))
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert packaged["estimand_hash"] == manifest["estimand_hash"]
    assert packaged["menu_hash"] == manifest["menu_hash"]


def test_package_key_changes_when_the_cleaning_run_changes(tmp_path):
    # NEDEN: arayüz paketi bu anahtarla önbelleğe alır. Anahtar temizleme
    # artefaktlarını kapsamazsa provenans uyarısı TAM DA gerektiği senaryoda
    # yutulur: A'yı temizleyip koşan ve paketi hazırlayan kullanıcı sonra B'yi
    # temizlediğinde sonuç yolu, run_id ve eksik listesi aynı kalır — anahtar
    # değişmezse A'nın paketi servis edilir ve "EŞLEŞMİYOR" hiç görünmez.
    first = _make_run(tmp_path)

    assert first.raw_panel_path is not None
    second_dir = tmp_path / "audit_trail_b" / "temiz-b_repro"
    second_dir.mkdir(parents=True)
    (second_dir / "raw.pkl").write_bytes(first.raw_panel_path.read_bytes())
    (second_dir / "reproduced.pkl").write_bytes(pickle.dumps(_panel(effect=0.2, seed=7)))
    second = replace(
        first,
        raw_panel_path=second_dir / "raw.pkl",
        cleaned_panel_path=second_dir / "reproduced.pkl",
        cleaning_run_id="temiz-b",
    )

    # Hiçbir artefakt EKSİK değil ve sonuç yolu aynı: anahtarı ayıran tek şey
    # temizleme koşusunun kimliği ve yolları olmalı.
    assert missing_artifacts(first) == missing_artifacts(second)
    assert first.results_path == second.results_path
    assert package_key(first) != package_key(second)


def test_package_key_changes_when_a_figure_is_redrawn(tmp_path):
    # NEDEN: figürler dosya ADIYLA anahtarlansaydı, aynı adla yeniden çizilen bir
    # figür (eksen seçimi değişti) önbellekteki eski gövdeyle indirilirdi; paketteki
    # figür raporlanan panelden başka bir şey gösterirdi.
    inputs = _make_run(tmp_path)
    name = next(iter(inputs.figures))
    redrawn = replace(inputs, figures={name: figure_html(name, _other_figure())})

    assert set(inputs.figures) == set(redrawn.figures)
    assert package_key(inputs) != package_key(redrawn)


def test_package_key_is_stable_for_unchanged_inputs(tmp_path):
    # NEDEN: anahtar her rerun'da değişseydi önbellek hiç tutmaz ve paket her
    # etkileşimde yeniden kurulurdu; uyarı da her seferinde sıfırlanırdı.
    inputs = _make_run(tmp_path)
    assert package_key(inputs) == package_key(replace(inputs))


def test_cleaning_script_is_copied_not_rerendered(tmp_path):
    # NEDEN: L4 kapısı diskteki script'i doğrular. Paket zamanında yeniden render
    # edilen script farklı bir artefakttır (zaman damgası değişir) ve doğrulanmamıştır.
    inputs = _make_run(tmp_path)
    payload = build_reproduction_package(inputs)
    assert inputs.cleaning_script_path is not None
    assert _read(payload, "cleaning/cleaning_steps.py") == inputs.cleaning_script_path.read_text(
        encoding="utf-8"
    )


def test_missing_artifacts_are_reported_not_hidden(tmp_path):
    # NEDEN: fail-loud. Karar defteri olmayan bir paket sessizce "eksiksiz" görünürse
    # kullanıcı olmayan bir denetim izine güvenir.
    inputs = _make_run(tmp_path, with_cleaning=False)
    assert set(missing_artifacts(inputs)) == {"decision_ledger", "cleaning_script", "raw_panel"}

    payload = build_reproduction_package(inputs)
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert set(manifest["missing"]) == {"decision_ledger", "cleaning_script", "raw_panel"}
    assert "karar defteri" in _read(payload, "README.md")
    assert "eksik" in _read(payload, "METHODS.md").lower()


def test_empty_spec_list_counts_as_missing(tmp_path):
    # NEDEN: boş bir specs.json pakete hiç girmez ve run script koşamaz. Yolun var
    # olması "eksik yok" demek için yetmez, yoksa paket doğrulanabilir görünür.
    inputs = _make_run(tmp_path)
    assert inputs.specs_path is not None
    inputs.specs_path.write_text("[]", encoding="utf-8")

    assert "specs" in missing_artifacts(inputs)
    payload = build_reproduction_package(inputs)
    assert "specs.json" not in _names(payload)
    assert "specs" in json.loads(_read(payload, "MANIFEST.json"))["missing"]


def test_column_dtypes_survive_the_csv_round_trip(tmp_path):
    # NEDEN: CSV tip taşımaz. FIPS gibi öndeki sıfırları korunan kolonlar okurken
    # sayıya dönerse temizleme karşılaştırması gerçek olmayan bir hatayla patlar.
    inputs = _make_run(tmp_path)
    assert inputs.panel_path is not None
    panel = pickle.loads(inputs.panel_path.read_bytes())
    panel["fips"] = ["05001"] * len(panel)
    inputs.panel_path.write_bytes(pickle.dumps(panel))

    payload = build_reproduction_package(inputs)
    declared = json.loads(_read(payload, "data/dtypes.json"))
    assert declared["panel"]["fips"] in {"object", "str"}

    restored = pd.read_csv(io.StringIO(_read(payload, "data/panel.csv")), dtype=declared["panel"])
    assert restored["fips"].iloc[0] == "05001"


def test_package_without_results_fails_loud(tmp_path):
    # NEDEN: sonuçsuz bir "reprodüksiyon paketi" yanlış güven üretir; sessiz boş
    # paket yerine patlamalı.
    with pytest.raises(ReproPackageError, match="Sonuç dosyası yok"):
        build_reproduction_package(
            ReproInputs(run_id="yok", results_path=tmp_path / "results.json")
        )


def test_package_bytes_are_deterministic(tmp_path):
    # NEDEN: reprodüksiyon paketinin kendisi de reprodüklenebilir olmalı; aynı
    # koşudan iki farklı zip, hash'lenerek atıf verilmesini imkansız kılar.
    #
    # Figürler İKİNCİ kez ayrıca render edilir: aynı HTML string'ini iki kez
    # paketlemek determinizmi inşa yoluyla garantiler ve testi boşa çıkarır. Gerçek
    # yolda panel her rerun'da figürü yeniden çizer, kırılgan yer tam orasıdır.
    inputs = _make_run(tmp_path)
    recaptured = replace(
        inputs,
        figures={name: _figure_html(name) for name in inputs.figures},
    )
    assert build_reproduction_package(inputs) == build_reproduction_package(recaptured)


def test_methods_draft_renders_decisions_without_llm(tmp_path):
    # NEDEN: metot bölümü taslağı deterministik şablondan gelir; karar defterindeki
    # her satır taslakta görünür olmalı, yoksa "denetim izi = metot bölümü" boş bir iddia.
    payload = build_reproduction_package(_make_run(tmp_path))
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    methods = _read(payload, "METHODS.md")

    assert methods == render_methods_section(manifest)
    assert "drop_duplicates" in methods
    assert "Tekrar eden satırlar tahmini yanlı hale getirir." in methods
    assert "abc123deadbeef01" in methods
    assert "parallel_trends" in methods


def test_run_script_reproduces_packaged_results(tmp_path):
    # NEDEN: paketin tek gerçek sözü bu. Zip açılır, tek komut koşar ve paketlenmiş
    # katsayılar tolerans içinde geri gelir; gelmezse çıkış kodu sıfır olmaz.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    proc = _run_package(extract_dir)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TAMAM: temizleme adımı yeniden üretildi" in proc.stdout
    assert "Reprodüksiyon doğrulandı." in proc.stdout


def test_run_script_fails_when_the_cleaning_script_is_tampered(tmp_path):
    # NEDEN: temizleme adımı doğrulamanın kendi kapısı. Pakete giren script, ham
    # veriden paketlenmiş paneli üretmeyi bırakırsa denetim izi sahtedir. Sonuç
    # dosyasına DOKUNULMAZ: tahminler yine tutar, dolayısıyla bu farkı yakalayan
    # tek şey `_verify_cleaning` — o kapı sessizse test de sessiz kalır.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "tampered_cleaning"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    script_file = extract_dir / "cleaning" / "cleaning_steps.py"
    original = script_file.read_text(encoding="utf-8")
    # Tekrar eden satırların atılması, karar defterinin bildirdiği tek adımdı.
    tampered = original.replace("df = df.drop_duplicates", "df = df  # drop_duplicates")
    # Bulunamayan desen sessiz bir no-op bırakır ve test kendi kendini boşa çıkarır.
    assert tampered != original, "temizleme adımı script'te bulunamadı"
    script_file.write_text(tampered, encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "BAŞARISIZ" in proc.stdout
    assert "temizleme çıktısı" in proc.stdout


def test_run_script_fails_when_spec_set_is_trimmed(tmp_path):
    # NEDEN: ürünün iddiası "rapor edilen küme sonradan kırpılamaz". Sonuçları
    # bozmadan menüden spec silmek, kalan kümeyi kendi içinde tutarlı bıraktığı
    # için katsayı karşılaştırmasına yakalanmaz; kümeyi MANIFEST hash'lerine karşı
    # denetlemezsek script kırpılmış bir pakete "doğrulandı" der.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "trimmed"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    specs_file = extract_dir / "specs.json"
    kept = json.loads(specs_file.read_text(encoding="utf-8"))[:1]
    specs_file.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "BAŞARISIZ" in proc.stdout
    assert "s2" in proc.stdout


def test_run_script_fails_when_a_spec_definition_is_edited(tmp_path):
    # NEDEN: kırpmanın ikizi. Bir spec'in içeriği (ör. kontrol değişkenleri)
    # sonradan değiştirilirse koşu yine tutarlı görünür; içerik hash'i MANIFEST'te
    # dondurulduğu için bu da yakalanmalı.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "edited"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    specs_file = extract_dir / "specs.json"
    edited = json.loads(specs_file.read_text(encoding="utf-8"))
    edited[0]["cluster_by"] = None
    specs_file.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "içeriği değişen" in proc.stdout


def test_provenance_mismatch_is_recorded_everywhere(tmp_path):
    # NEDEN: temizleme ve multiverse ayrı koşulardır. A'yı temizleyip koşan, sonra
    # B'yi temizleyen kullanıcının paketi B'nin karar defterini A'nın sonuçlarıyla
    # birleştirir; hiçbir artefakt EKSİK olmadığı için paket "eksiksiz" görünür.
    # Uyarı manifest'e, README'ye ve METHODS'a birden yazılmazsa kullanıcı olmayan
    # bir denetim izine güvenir.
    inputs = _make_run(tmp_path)
    assert inputs.cleaned_panel_path is not None
    # Başka bir koşunun temizleme çıktısı: aynı şema, farklı veri.
    inputs.cleaned_panel_path.write_bytes(pickle.dumps(_panel(effect=0.2, seed=7)))

    payload = build_reproduction_package(inputs)
    manifest = json.loads(_read(payload, "MANIFEST.json"))

    assert manifest["missing"] == []  # eksik yok; yanlış olan provenans
    assert manifest["provenance"]["cleaning_matches_panel"] is False
    assert manifest["provenance"]["cleaning_run_id"] == "clean"
    assert (
        manifest["provenance"]["cleaned_panel_fingerprint"]
        != manifest["provenance"]["panel_fingerprint"]
    )
    assert "EŞLEŞMİYOR" in _read(payload, "README.md")
    assert "UYARI" in _read(payload, "METHODS.md")


def test_provenance_confirms_matching_cleaning_run(tmp_path):
    # NEDEN: uyarı her pakette çıkarsa uyarı olmaktan çıkar. Doğru eşleşen koşuda
    # provenans olumlu kaydedilmeli ve taslakta uyarı görünmemeli.
    payload = build_reproduction_package(_make_run(tmp_path))
    manifest = json.loads(_read(payload, "MANIFEST.json"))

    assert manifest["provenance"]["cleaning_matches_panel"] is True
    assert "UYARI" not in _read(payload, "METHODS.md")


def test_provenance_tolerates_what_the_l4_gate_tolerated(tmp_path):
    # NEDEN: L4 kapısı sandbox çıktısını TOLERANSLA kabul eder (REPRO_RTOL/ATOL).
    # Provenans denetimi bit eşitliği arasaydı, float'a dokunan bir transform
    # eklendiğinde kapıdan geçmiş DOĞRU bir koşu "denetim izi sahte" damgası yerdi.
    # Her pakette çıkan bir uyarı, uyarı olmaktan çıkar.
    inputs = _make_run(tmp_path)
    assert inputs.cleaned_panel_path is not None
    cleaned = pickle.loads(inputs.cleaned_panel_path.read_bytes())
    # Toleransın içinde, ama bit düzeyinde farklı.
    cleaned["y"] = cleaned["y"] + 1e-12
    inputs.cleaned_panel_path.write_bytes(pickle.dumps(cleaned))

    manifest = json.loads(_read(build_reproduction_package(inputs), "MANIFEST.json"))
    assert manifest["provenance"]["cleaning_matches_panel"] is True
    # Parmak izleri farklı olabilir; kararı veren tolerans karşılaştırmasıdır.
    assert manifest["provenance"]["comparison"] == {"rtol": REPRO_RTOL, "atol": REPRO_ATOL}


def test_provenance_flags_structurally_different_cleaning_output(tmp_path):
    # NEDEN: tolerans gevşekliği her farkı yutmamalı. Kolon kümesi ya da satır
    # sayısı farklıysa bu tolerans meselesi değil, başka bir veridir.
    inputs = _make_run(tmp_path)
    assert inputs.cleaned_panel_path is not None
    cleaned = pickle.loads(inputs.cleaned_panel_path.read_bytes())
    inputs.cleaned_panel_path.write_bytes(pickle.dumps(cleaned.head(len(cleaned) - 1)))

    manifest = json.loads(_read(build_reproduction_package(inputs), "MANIFEST.json"))
    assert manifest["provenance"]["cleaning_matches_panel"] is False


def test_provenance_is_unverified_without_cleaning_sandbox(tmp_path):
    # NEDEN: "denetlenmedi" ile "denetlendi ve tamam" aynı şey değildir; ikisini
    # tek bayrakta toplamak doğrulanmamış bir defteri doğrulanmış gösterir.
    inputs = _make_run(tmp_path, with_cleaning=False)
    manifest = json.loads(_read(build_reproduction_package(inputs), "MANIFEST.json"))
    assert manifest["provenance"]["cleaning_matches_panel"] is None
    assert manifest["provenance"]["cleaning_run_id"] is None


def test_corrupt_ledger_raises_package_error_not_raw_traceback(tmp_path):
    # NEDEN: arayüz yalnız ReproPackageError yakalar. Bozuk bir artefakt ham
    # JSONDecodeError olarak sızarsa kullanıcı Streamlit traceback'i görür ve
    # neyin bozuk olduğunu anlamaz.
    inputs = _make_run(tmp_path)
    assert inputs.ledger_path is not None
    inputs.ledger_path.write_text("{bu json değil\n", encoding="utf-8")

    with pytest.raises(ReproPackageError, match="Karar defteri"):
        build_reproduction_package(inputs)


def test_schema_drifted_results_raise_package_error(tmp_path):
    # NEDEN: aynı gerekçe, pydantic tarafı. Eski şemalı bir results.json
    # ValidationError ile patlarsa hata mesajı kullanıcıya ulaşmaz.
    inputs = _make_run(tmp_path)
    inputs.results_path.write_text(
        json.dumps([{"spec_id": "s1"}], ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ReproPackageError, match="şemasına uymuyor"):
        build_reproduction_package(inputs)


def test_methods_draft_degrades_loudly_without_frozen_menu(tmp_path):
    # NEDEN: donmuş menü bulunamadığında (ör. run_id yanlış çözülürse) taslak
    # estimand bölümünü kaybeder. Sessizce kısalmak yerine eksikliği söylemeli.
    inputs = _make_run(tmp_path)
    assert inputs.frozen_menu_path is not None
    inputs.frozen_menu_path.unlink()

    methods = _read(build_reproduction_package(inputs), "METHODS.md")
    assert "Dondurulmuş estimand kaydı bu pakette yok" in methods
    assert "frozen_menu" in methods


def test_run_script_fails_when_packaged_results_are_tampered(tmp_path):
    # NEDEN: doğrulama gerçekten doğruluyor mu? Sonuç dosyası bozulduğunda script
    # sessizce "tamam" derse kapı yok demektir.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "tampered"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    results_file = extract_dir / "results.json"
    tampered = json.loads(results_file.read_text(encoding="utf-8"))
    tampered[0]["coefficient"] = float(tampered[0]["coefficient"]) + 1.0
    results_file.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0
    assert "BAŞARISIZ" in proc.stdout


def test_run_script_fails_on_a_single_row_n_obs_difference(tmp_path):
    # NEDEN: n_obs bir ölçüm değil, SAYIMDIR. Göreli toleransla karşılaştırılırsa
    # eşik örneklemle birlikte büyür (n=10.000'de tolerans tam 1.0) ve bir satırlık
    # fark sessizce geçer. Örneklemi değişmiş bir koşuya "yeniden üretildi" demek,
    # doğrulamanın kendisini boşa çıkarır — bu yüzden tam eşitlik aranmalı.
    #
    # Panel BİLEREK büyük: 240 satırda göreli eşik zaten 0.024'tür ve bir satırlık
    # farkı yakalar, yani küçük panelde bu test toleranslı kodda da geçerdi. Hata
    # ancak eşik 1.0'ı aştığında (n_obs > 10.000) ortaya çıkar.
    payload = build_reproduction_package(_make_run(tmp_path, units=1700))
    extract_dir = tmp_path / "n_obs"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    results_file = extract_dir / "results.json"
    tampered = json.loads(results_file.read_text(encoding="utf-8"))
    # Katsayı ve standart hataya DOKUNULMAZ: tek fark bir satırlık örneklem.
    tampered[0]["n_obs"] = int(tampered[0]["n_obs"]) - 1
    results_file.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "n_obs" in proc.stdout


def test_run_script_reads_its_tolerances_from_the_manifest(tmp_path):
    # NEDEN: eşik script'e gömülü olsaydı paket kendi içinde çelişebilirdi —
    # manifest temizleme çıktısı için "EŞLEŞMİYOR" derken, daha gevşek bir eşik
    # taşıyan script aynı farkı yutup "doğrulandı" basardı. Tek kaynak MANIFEST.
    payload = build_reproduction_package(_make_run(tmp_path))
    manifest = json.loads(_read(payload, "MANIFEST.json"))
    assert manifest["tolerances"]["cleaning"] == {"rtol": REPRO_RTOL, "atol": REPRO_ATOL}

    extract_dir = tmp_path / "no_tolerance"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    manifest_file = extract_dir / "MANIFEST.json"
    stripped = json.loads(manifest_file.read_text(encoding="utf-8"))
    del stripped["tolerances"]
    manifest_file.write_text(json.dumps(stripped, ensure_ascii=False), encoding="utf-8")

    # Eşiksiz kalan script sessizce bir varsayılana düşmemeli.
    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "eşi" in proc.stdout


def test_run_script_fails_readably_on_a_malformed_tolerance(tmp_path):
    # NEDEN: eksik eşiğin ikizi. Sayı olmayan bir eşik ham TypeError/ValueError
    # traceback'i olarak sızarsa kullanıcı neyin bozuk olduğunu göremez; modülün
    # her yerinde olduğu gibi burada da hata OKUNUR bir mesaja çevrilmeli.
    payload = build_reproduction_package(_make_run(tmp_path))
    extract_dir = tmp_path / "bad_tolerance"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extract_dir)

    manifest_file = extract_dir / "MANIFEST.json"
    broken = json.loads(manifest_file.read_text(encoding="utf-8"))
    broken["tolerances"]["cleaning"]["rtol"] = "bu sayı değil"
    manifest_file.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")

    proc = _run_package(extract_dir)
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}"
    assert "BAŞARISIZ" in proc.stdout
    assert "Traceback" not in proc.stderr
