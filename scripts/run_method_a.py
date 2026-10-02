"""Estimate beta with empirical Method A and the analytical model for every station.

Writes to outputs/tables/:
    method_a_<network>.csv          one row per station (beta in %/hPa)
    method_a_<network>_monthly.csv  every accepted monthly regression
    method_a_<network>_pooled.csv.gz daily (dx, dln) pairs used in the regression figures

Usage:
    python scripts/run_method_a.py [--networks EU US NM]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config, io, method_a  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def save(results, network):
    config.TABLES.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.ok]
    summary = pd.DataFrame([r.summary() for r in ok])
    paper = {"EU": config.EU_STATIONS, "US": config.US_STATIONS}.get(network)
    summary["in_paper"] = True if paper is None else summary.station_id.isin(paper)
    monthly = pd.concat([r.monthly.assign(station_id=r.station_id, name=r.name) for r in ok])
    pooled = pd.concat([r.pooled.assign(station_id=r.station_id) for r in ok])
    tag = network.lower()
    summary.to_csv(config.TABLES / f"method_a_{tag}.csv", index=False)
    monthly.to_csv(config.TABLES / f"method_a_{tag}_monthly.csv", index=False)
    pooled.to_csv(config.TABLES / f"method_a_{tag}_pooled.csv.gz", index_label="date")
    print(f"{network}: {len(ok)}/{len(results)} stations with a valid beta")


def run_eu(jung):
    sites = io.eu_sites()
    results = []
    for s in sites.itertuples():
        if s.station_id in config.EU_SKIP:
            continue
        try:
            df = io.load_eu_station(s.station_id)
        except FileNotFoundError:          # e.g. SCC003: no processed file
            print(f"  EU {s.station_id:<11} {s.name:<16} not loaded (missing file)")
            continue
        r = method_a.crns_station(df, s.station_id, s.name, "EU", s.lat, s.lon, s.altitude,
                                  s.gv, jung, config.EU_SETTINGS)
        print(f"  EU {s.station_id:<11} {s.name:<16} months={len(r.monthly):3d}", flush=True)
        results.append(r)
    save(results, "EU")


def run_us(jung):
    sites = io.us_sites()
    results = []
    for s in sites.itertuples():
        if s.station_id in config.US_SKIP:
            continue
        num = s.station_id.zfill(3)
        try:
            df = io.load_us_station(num)
        except (OSError, KeyError, ValueError) as e:   # notebook: try/except "Error with"
            print(f"  US {s.station_id:<4} {s.name:<24} not loaded ({type(e).__name__})")
            continue
        r = method_a.crns_station(df, num, s.name, "US", s.lat, s.lon, s.altitude, s.gv,
                                  jung, config.US_SETTINGS)
        r.station_id = s.station_id
        print(f"  US {s.station_id:<4} {s.name:<24} months={len(r.monthly):3d}", flush=True)
        results.append(r)
    save(results, "US")


def run_nm():
    ref = io.load_nm_reference_median()
    state = {}   # carried across stations, see config.NMSettings.legacy_stale_regression
    results = []
    for s in io.nm_sites().itertuples():
        if s.station_id in config.NM_BLACKLIST:
            continue
        counts = io.load_nmdb_decadal("uncorrected", s.station_id)
        pressure = io.load_nmdb_decadal("pressure_mbar", s.station_id)
        if counts is None or pressure is None:
            print(f"  NM {s.station_id:<6} no data")
            continue
        r = method_a.nm_station(counts[s.station_id], pressure[s.station_id], ref,
                                s.station_id, s.name, s.lat, s.lon, s.altitude, s.gv,
                                state=state)
        print(f"  NM {s.station_id:<6} {s.name:<22} months={len(r.monthly):3d}", flush=True)
        results.append(r)
    save(results, "NM")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--networks", nargs="+", default=["EU", "US", "NM"],
                    choices=["EU", "US", "NM"])
    args = ap.parse_args()
    t0 = time.time()
    jung = io.load_jung() if {"EU", "US"} & set(args.networks) else None
    if "EU" in args.networks:
        run_eu(jung)
    if "US" in args.networks:
        run_us(jung)
    if "NM" in args.networks:
        run_nm()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
