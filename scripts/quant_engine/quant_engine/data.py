"""Download and cache official data published by the European Central Bank.

Sources (free of charge; the ECB permits reuse provided the source is
acknowledged, see https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html):

* Euro foreign exchange reference rates, full history since 4 January 1999:
  https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip
  Rates are quoted as units of foreign currency per 1 euro.
* Euro area yield curve, AAA-rated central government bonds, Svensson model
  parameters (ECB Data Portal, dataset YC, series
  YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0,BETA1,BETA2,BETA3,TAU1,TAU2}).

Downloads are cached under ``data/raw/ecb/`` in the repository (ignored by
Git). Set the environment variable ``QUANT_ENGINE_DATA_DIR`` to use another
cache folder. Command-line usage from the repository root:

    python -m quant_engine.data --fx --yield-curve --start 2004-09-06 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ECB_FX_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ECB_YC_URL = ("https://data-api.ecb.europa.eu/service/data/YC/"
              "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0+BETA1+BETA2+BETA3+TAU1+TAU2"
              "?format=csvdata&startPeriod={start}&endPeriod={end}")
USER_AGENT = "quant-engine/0.1 (research scripts; https://github.com/paolo993788/Quantitative-finance)"
SVENSSON_COLUMNS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def repository_root() -> Path:
    """Locate the repository root (the folder containing scripts/quant_engine)."""
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "scripts" / "quant_engine").is_dir():
            return base
    return Path(__file__).resolve().parents[3]


def cache_dir() -> Path:
    env = os.environ.get("QUANT_ENGINE_DATA_DIR")
    path = Path(env) if env else repository_root() / "data" / "raw" / "ecb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, destination: Path, timeout: float = 60.0) -> Path:
    """Download `url` to `destination` (written atomically)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(destination)
    return destination


def parse_ecb_fx_csv(text: str) -> pd.DataFrame:
    """Parse eurofxref-hist.csv: one row per business day, 'N/A' for missing."""
    frame = pd.read_csv(io.StringIO(text), na_values=["N/A"], index_col="Date", parse_dates=["Date"])
    frame = frame.loc[:, [c for c in frame.columns if c and not c.startswith("Unnamed")]]
    return frame.sort_index().astype(float)


def load_ecb_fx_rates(currencies=("USD", "GBP", "JPY", "CHF"), start=None, end=None, refresh=False) -> pd.DataFrame:
    """Euro reference rates (foreign currency per EUR) for the chosen currencies."""
    path = cache_dir() / "eurofxref-hist.zip"
    if refresh or not path.exists():
        download(ECB_FX_URL, path)
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        frame = parse_ecb_fx_csv(archive.read(name).decode("utf-8"))
    missing = [c for c in currencies if c not in frame.columns]
    if missing:
        raise KeyError(f"currencies not available in the ECB file: {missing}")
    frame = frame.loc[start:end, list(currencies)].dropna(how="all")
    frame.attrs["source"] = f"European Central Bank, euro foreign exchange reference rates ({ECB_FX_URL})"
    return frame


def parse_ecb_yield_curve_csv(text: str) -> pd.DataFrame:
    """Parse the ECB Data Portal `csvdata` output into a date x parameter table."""
    raw = pd.read_csv(io.StringIO(text))
    if "KEY" in raw.columns:
        parameter = raw["KEY"].str.split(".").str[-1]
    else:
        parameter = raw["DATA_TYPE_FM"]
    table = (raw.assign(parameter=parameter, date=pd.to_datetime(raw["TIME_PERIOD"]))
                .pivot_table(index="date", columns="parameter", values="OBS_VALUE", aggfunc="last"))
    missing = [c for c in SVENSSON_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"unexpected ECB response, missing parameters: {missing}")
    return table[SVENSSON_COLUMNS].sort_index().astype(float)


def load_ecb_svensson_parameters(start: str, end: str, refresh=False) -> pd.DataFrame:
    """Daily Svensson parameters of the ECB AAA euro area yield curve."""
    path = cache_dir() / f"yc_svensson_{start}_{end}.csv"
    if refresh or not path.exists():
        download(ECB_YC_URL.format(start=start, end=end), path)
    table = parse_ecb_yield_curve_csv(path.read_text(encoding="utf-8"))
    table.attrs["source"] = "European Central Bank, euro area yield curves (AAA), Svensson parameters"
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download official ECB data into the local cache.")
    parser.add_argument("--fx", action="store_true", help="euro foreign exchange reference rates")
    parser.add_argument("--yield-curve", action="store_true", help="AAA yield curve Svensson parameters")
    parser.add_argument("--start", default="2004-09-06")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args(argv)
    if args.fx:
        rates = load_ecb_fx_rates(refresh=True)
        print(f"FX reference rates: {rates.index.min().date()} to {rates.index.max().date()}, cached in {cache_dir()}")
    if args.yield_curve:
        curve = load_ecb_svensson_parameters(args.start, args.end, refresh=True)
        print(f"Svensson parameters: {len(curve)} days, cached in {cache_dir()}")
    if not (args.fx or args.yield_curve):
        parser.print_help()


if __name__ == "__main__":
    main()
