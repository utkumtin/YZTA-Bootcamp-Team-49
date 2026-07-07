"""Veri-erişim yardımcısı (DCAS).

İki tür kaynak var:
- MANUEL: programatik indirme mümkün değil; dosyanın varlığı doğrulanır, eksikse
  nasıl edinileceği söylenir. (CDC WONDER interaktif sorgu aracı; BLS LAUS
  script'le indirmeyi 403 ile engelliyor — tarayıcı gerekir.)
- FETCH: `--fetch` ile indirilir/türetilir (Census API, replikasyon .dta'ları,
  Card-Krueger arşivi). İdempotent: mevcut dosya yeniden indirilmez.

Kullanım:
  python data/fetch_data.py --check   # beklenen dosyaları doğrula (default)
  python data/fetch_data.py --fetch   # fetch'lenebilir kaynakları indir/türet
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

USER_AGENT = "pareto-bootcamp-research"

# --- MANUEL kaynaklar: yalnız varlık kontrolü -------------------------------

PANEL_YEARS = range(2009, 2020)  # CDC WONDER export penceresi (2020 bilerek dışarıda: COVID)

EXPECTED_MANUAL = {
    "medicaid/raw/cdc_wonder_mortality_2009_2019.tsv": (
        "CDC WONDER Multiple Cause of Death (ilçe × yıl) — MANUEL export. "
        "http://wonder.cdc.gov/mcd.html · export'u bu yola koy (SOURCES.md)."
    ),
    "medicaid/raw/kff_expansion_dates.csv": (
        "KFF eyalet genişleme tarihleri — repo ile gelir (küçük, redistribute)."
    ),
    **{
        f"medicaid/raw/laus/laucnty{y % 100:02d}.xlsx": (
            "BLS LAUS ilçe yıllık işsizlik — MANUEL (BLS script'i 403'ler). Tarayıcıdan: "
            f"https://www.bls.gov/lau/laucnty{y % 100:02d}.xlsx"
        )
        for y in PANEL_YEARS
    },
}

# --- FETCH kaynakları --------------------------------------------------------
# Census API artık zorunlu key istiyor; onun yerine key'siz statik dosya sunucusu
# (www2.census.gov) kullanılır — tam programatik, secret gerektirmez.

SAHIE_URL = (
    "https://www2.census.gov/programs-surveys/sahie/datasets/time-series/"
    "estimates-acs/sahie-{year}-csv.zip"
)
SAIPE_URL = (
    "https://www2.census.gov/programs-surveys/saipe/datasets/{year}/"
    "{year}-state-and-county/est{yy}all.txt"
)

# Card-Krueger public.dat kolonları (arşivdeki `codebook` sırasıyla; 46 alan)
NJMIN_COLUMNS = [
    "SHEET",
    "CHAIN",
    "CO_OWNED",
    "STATE",
    "SOUTHJ",
    "CENTRALJ",
    "NORTHJ",
    "PA1",
    "PA2",
    "SHORE",
    "NCALLS",
    "EMPFT",
    "EMPPT",
    "NMGRS",
    "WAGE_ST",
    "INCTIME",
    "FIRSTINC",
    "BONUS",
    "PCTAFF",
    "MEALS",
    "OPEN",
    "HRSOPEN",
    "PSODA",
    "PFRY",
    "PENTREE",
    "NREGS",
    "NREGS11",
    "TYPE2",
    "STATUS2",
    "DATE2",
    "NCALLS2",
    "EMPFT2",
    "EMPPT2",
    "NMGRS2",
    "WAGE_ST2",
    "INCTIME2",
    "FIRSTIN2",
    "SPECIAL2",
    "MEALS2",
    "OPEN2R",
    "HRSOPEN2",
    "PSODA2",
    "PFRY2",
    "PENTREE2",
    "NREGS2",
    "NREGS112",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — sabit https hostlar
        return resp.read()


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[SKIP] {dest.relative_to(HERE)} zaten var")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(url))
    print(f"[OK  ] {dest.relative_to(HERE)} indirildi")


def fetch_sahie() -> None:
    """SAHIE — ilçe-yıl sigortasız % (18-64, tüm gelirler). Kapsam sonucu (robust yüz).

    Yıllık ~10MB zip indirilir, ilçe × (18-64, tüm gelir/ırk/cinsiyet) satırlarına
    filtrelenir, tek CSV'ye birleştirilir. Zip'ler diske yazılmaz.
    """
    import pandas as pd

    dest = HERE / "medicaid/raw/sahie_uninsured_2009_2019.csv"
    if dest.exists():
        print(f"[SKIP] {dest.relative_to(HERE)} zaten var")
        return
    frames = []
    for year in PANEL_YEARS:
        with zipfile.ZipFile(io.BytesIO(_get(SAHIE_URL.format(year=year)))) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".csv"))
            lines = zf.read(member).decode("latin-1").splitlines()
        # CSV başlığı ~80 satırlık not bloğundan sonra gelir
        hdr = next(i for i, ln in enumerate(lines) if ln.lower().startswith("year,version"))
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), dtype=str, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.apply(lambda col: col.str.strip())
        mask = (
            (df["geocat"] == "50")  # ilçe düzeyi
            & (df["agecat"] == "1")  # 18-64
            & (df["racecat"] == "0")
            & (df["sexcat"] == "0")
            & (df["iprcat"] == "0")  # tüm gelirler
        )
        keep = [
            "year",
            "statefips",
            "countyfips",
            "pctui",
            "pctui_moe",
            "state_name",
            "county_name",
        ]
        frames.append(df.loc[mask, keep])
        print(f"       sahie {year}: {int(mask.sum())} ilçe")
    pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
    print(f"[OK  ] {dest.relative_to(HERE)} yazıldı")


def fetch_saipe() -> None:
    """SAIPE — ilçe-yıl medyan hane geliri + yoksulluk oranı. Kontrol değişkenleri.

    est{yy}all.txt: sayısal alanlar boşluk-ayrımlı, ardından yer adı. İlk 21 sayısal
    token sabit: [5]=yoksulluk %, [20]=medyan hane geliri. '.' = suppressed/NA.
    Yalnız ilçe satırları tutulur (countyfips != 0); FIPS'ler zero-pad'lenir.
    """
    import pandas as pd

    dest = HERE / "medicaid/raw/saipe_income_poverty_2009_2019.csv"
    if dest.exists():
        print(f"[SKIP] {dest.relative_to(HERE)} zaten var")
        return
    rows = []
    for year in PANEL_YEARS:
        url = SAIPE_URL.format(year=year, yy=f"{year % 100:02d}")
        n = 0
        for line in _get(url).decode("latin-1").splitlines():
            toks = line.split()
            num: list[str] = []
            for t in toks:
                if any(ch.isalpha() for ch in t):
                    break
                num.append(t)
            if len(num) < 21 or num[1] == "0":  # kısa satır ya da eyalet/ABD toplamı
                continue
            # satır sonundaki dosya damgasını at ("... est09ALL.txt 22OCT2010")
            name = re.sub(
                r"\s+est\d+all\.txt.*$", "", " ".join(toks[len(num) :]), flags=re.IGNORECASE
            )
            rows.append(
                {
                    "year": year,
                    "statefips": num[0].zfill(2),
                    "countyfips": num[1].zfill(3),
                    "poverty_pct_all_ages": num[5],
                    "median_hh_income": num[20],
                    "name": name,
                }
            )
            n += 1
        print(f"       saipe {year}: {n} ilçe")
    pd.DataFrame(rows).to_csv(dest, index=False, na_rep="")
    print(f"[OK  ] {dest.relative_to(HERE)} yazıldı")


def _dta_to_csv(dta: Path, csv: Path) -> None:
    import pandas as pd

    if csv.exists():
        print(f"[SKIP] {csv.relative_to(HERE)} zaten var")
        return
    pd.read_stata(dta).to_csv(csv, index=False)
    print(f"[OK  ] {csv.relative_to(HERE)} yazıldı")


def fetch_divorce() -> None:
    """Stevenson-Wolfers no-fault divorce (Goodman-Bacon örneği) — kanonik flip seti."""
    dta = HERE / "divorce/raw/bacon_example.dta"
    _download("http://pped.org/bacon_example.dta", dta)
    _dta_to_csv(dta, HERE / "divorce/raw/divorce.csv")


def fetch_castle() -> None:
    """Cheng-Hoekstra castle doctrine — ikincil flip/smoke seti (Mixtape reposu)."""
    dta = HERE / "castle/raw/castle.dta"
    _download("https://raw.githubusercontent.com/scunning1975/mixtape/master/castle.dta", dta)
    _dta_to_csv(dta, HERE / "castle/raw/castle.csv")


def fetch_card_krueger() -> None:
    """Card-Krueger NJ-PA asgari ücret — minik 2×2 fixture (CI smoke)."""
    import pandas as pd

    zip_path = HERE / "card_krueger/raw/njmin.zip"
    csv = HERE / "card_krueger/raw/card_krueger.csv"
    _download("https://davidcard.berkeley.edu/data_sets/njmin.zip", zip_path)
    if csv.exists():
        print(f"[SKIP] {csv.relative_to(HERE)} zaten var")
        return
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("public.dat")
        # codebook/read.me kolon tanımları ve atıf için CSV'nin yanında dursun
        for member in ("codebook", "read.me"):
            target = zip_path.parent / member
            if not target.exists():
                target.write_bytes(zf.read(member))
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", names=NJMIN_COLUMNS, na_values=".")
    if len(df.columns) != len(NJMIN_COLUMNS) or len(df) != 410:  # codebook: 410 gözlem
        raise SystemExit(f"public.dat beklenen şekilde değil: {df.shape}")
    df.to_csv(csv, index=False)
    print(f"[OK  ] {csv.relative_to(HERE)} yazıldı ({len(df)} gözlem)")


FETCHERS = [fetch_sahie, fetch_saipe, fetch_divorce, fetch_castle, fetch_card_krueger]

EXPECTED_FETCHABLE = [
    "medicaid/raw/sahie_uninsured_2009_2019.csv",
    "medicaid/raw/saipe_income_poverty_2009_2019.csv",
    "divorce/raw/divorce.csv",
    "castle/raw/castle.csv",
    "card_krueger/raw/card_krueger.csv",
]


def fetch() -> None:
    for fn in FETCHERS:
        fn()


def check() -> int:
    missing = 0
    for rel, how in EXPECTED_MANUAL.items():
        path = HERE / rel
        status = "OK " if path.exists() else "EKSİK"
        if not path.exists():
            missing += 1
        print(f"[{status}] {rel}")
        if not path.exists():
            print(f"        → {how}")
    for rel in EXPECTED_FETCHABLE:
        path = HERE / rel
        status = "OK " if path.exists() else "EKSİK"
        if not path.exists():
            missing += 1
        print(f"[{status}] {rel}")
        if not path.exists():
            print("        → python data/fetch_data.py --fetch")
    if missing:
        print(f"\n{missing} dosya eksik. Yukarıdaki yönergelerle edinin.")
    else:
        print("\nTüm beklenen dosyalar mevcut.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto veri-erişim yardımcısı")
    parser.add_argument("--check", action="store_true", help="Beklenen dosyaları doğrula")
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch'lenebilir kaynakları indir/türet"
    )
    args = parser.parse_args()
    if args.fetch:
        fetch()
    raise SystemExit(check())


if __name__ == "__main__":
    main()
