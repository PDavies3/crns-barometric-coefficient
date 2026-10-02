"""Readers for the copied raw data (data/raw, gzip-compressed).

Ported from ``CRNS_lib.py`` (COSMOS loaders) and ``calc_beta1.py`` (NMDB loaders).
"""
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from . import config
from .corrections import remove_outliers_rolling


def _open(path):
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    return path


def read_csv(path, **kw):
    return pd.read_csv(_open(path), low_memory=False, **kw)


# ----------------------------------------------------------------- metadata ---------
def eu_sites():
    """COSMOS-Europe metadata: one row per station ID with GV, lat, lon, altitude, name."""
    add = read_csv(config.EU_DIR / "Additional_information.csv", encoding="latin1").iloc[:-3]
    gen = read_csv(config.EU_DIR / "General_ information.csv", encoding="latin1").iloc[:-3]
    gen = gen.rename(columns=lambda c: c.split(" (")[0].strip())
    out = gen.rename(columns={"File name": "station_id", "Station": "name", "Latitude": "lat",
                              "Longitude": "lon", "Altitude": "altitude",
                              "Detector Typ": "probe_type"})
    gv = add.set_index("Station ID")["Cutoff rigidity(GV)"]
    out["gv"] = out["station_id"].map(gv).astype(float)
    return out[["station_id", "name", "lat", "lon", "altitude", "gv", "probe_type"]]


def us_sites():
    """COSMOS-US metadata (site characteristics sheet + coordinates)."""
    s = pd.read_excel(config.US_DIR / "COSMOS_Site_Characteristics_WASCAL.xlsx",
                      skiprows=2, sheet_name="Tabelle1")
    ll = read_csv(config.US_DIR / "lonlat.csv")
    s = s.merge(ll, on="Site Name", how="left")
    out = pd.DataFrame({
        "station_id": s["ID"].astype(str),
        "name": s["Site Name"],
        "lat": s["lat"],
        "lon": s["lon"],
        "altitude": s["Elevation (m)"].astype(float),
        "gv": s["Cutoff Rigidity (GV)"].astype(float),
        "probe_type": s["Probe Type"],
    })
    return out


def nm_sites():
    """Neutron monitors of Väisänen et al. (2021), table S02, in file order."""
    df = read_csv(config.NMDB_DIR / "2020ja028941-sup-0006-table si-s02.csv",
                  index_col=0, sep=";", encoding="latin1")
    df = df.dropna(subset=["NMDB"])
    return pd.DataFrame({
        "station_id": df["NMDB"].values,
        "name": df.index.str.strip(),
        "lat": df["Latitude"].values,
        "lon": df["Longitude"].values,
        "altitude": df["Altitude"].values,
        "gv": df["Cut-off Rigidity"].values,
    })


# ----------------------------------------------------------------- CRNS --------------
def _hour(s):
    """Truncate to the hour, dropping any time zone but keeping the wall-clock time
    (the original code did this with strftime('%Y-%m-%d %H'))."""
    s = pd.to_datetime(s, format="mixed")
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("h")


def _parse_eu_time(t):
    """Parse COSMOS-Europe timestamps.

    Files use either ``dd.mm.yyyy HH:MM`` (day first) or ISO 8601 with a +00:00 offset.
    ISO strings must not be parsed with ``dayfirst`` (pandas would swap day and month
    for days <= 12); the original code relied on pandas 1.x's ISO fast path for this.
    """
    parsed = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    iso = t.str.match(r"^\d{4}-\d{2}-\d{2}")
    if iso.any():
        parsed[iso] = pd.to_datetime(t[iso], format="ISO8601", utc=True).dt.tz_localize(None)
    if (~iso).any():
        parsed[~iso] = pd.to_datetime(t[~iso].str.replace(".", "-", regex=False),
                                      format="%d-%m-%Y %H:%M")
    return parsed


def load_eu_station(station_id):
    """Hourly MOD, Pressure, TEMP, RH, SOILM for a COSMOS-Europe station."""
    raw = read_csv(config.EU_DIR / "raw_crns_and_meteorological_data" / f"{station_id}.csv")
    t = raw["DateTime_utc"].astype(str)
    if station_id in ("ALC001", "ALC002"):
        # These files mix text timestamps with spreadsheet day numbers.
        is_text = t.str.contains(":")
        parsed = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
        parsed[is_text] = _parse_eu_time(t[is_text])
        parsed[~is_text] = pd.to_datetime(t[~is_text].astype(float), unit="D",
                                           origin="1900-01-01").dt.round("s")
    else:
        parsed = _parse_eu_time(t)
    raw.index = _hour(parsed).rename("Date")

    sm = read_csv(config.EU_DIR / "processed_crns_data_and_diagnostics" / f"{station_id}.csv",
                  usecols=["DateTime_utc", "SoilMoisture_volumetric_MovAvg24h"])
    sm.index = _hour(sm["DateTime_utc"].str[:-9])
    sm = sm["SoilMoisture_volumetric_MovAvg24h"].rename("SOILM")

    cols = {}
    for key, pat in [("MOD", "NeutronCount_Epithermal_Cum1"), ("Pressure", "AirPressure"),
                     ("TEMP", "AirTemperature"), ("RH", "AirHumidity")]:
        cols[key] = raw.columns[raw.columns.str.contains(pat)][0]
    df = raw[[cols[k] for k in ("MOD", "Pressure", "TEMP", "RH")]]
    df.columns = ["MOD", "Pressure", "TEMP", "RH"]
    df = df.replace({"noData": np.nan, "NoData": np.nan}).astype(float)
    return df.join(sm)


def _load_mat(path):
    with gzip.open(_open(path), "rb") as f:
        return sio.loadmat(io.BytesIO(f.read()), squeeze_me=True, struct_as_record=False)


def _matlab_time(t):
    return pd.to_datetime(np.asarray(t, dtype=float) - 719529, unit="D").round("s")


def load_us_station(num):
    """Hourly MOD, Pressure, RH, TEMP, SOILM for COSMOS-US station ``num`` ('006', '016', ...)."""
    m = _load_mat(config.US_DIR / f"COSMOS_{num}.mat")
    l1, l3 = m["Level1"], m["Level3"]
    sm = pd.DataFrame({"SM12H": l3.SM12H, "SOILM": l3.SOILM},
                      index=_matlab_time(l3.TIME).rename("Date")).astype(float) / 100
    df = pd.DataFrame({"Pressure": l1.PRESS, "RH": l1.RH, "TEMP": l1.TEM, "MOD": l1.MOD},
                      index=_matlab_time(l1.TIME).rename("Date")).astype(float)
    df = df[df.MOD > 10]
    out = pd.concat([sm, df], axis=1)
    if num in ("099", "031"):
        out = out.iloc[1:]
    elif num == "061":
        out.loc[out["MOD"] < 4000, "MOD"] = np.nan
    if config.APPLY_US_PERIOD_REMOVAL:
        out = remove_us_periods(int(num), out)
    return out


# Manual data-quality exclusions for COSMOS-US (from CRNS_lib.remove_some_time_stamps).
_US_KEEP_BEFORE = {20: "2013-06-01", 49: "2014-08-01", 51: "2013-07-01", 89: "2017-01-01",
                   92: "2016-09-01", 1: "2012-05-01", 10: "2016-11-01", 84: "2015-12-01",
                   40: "2013-02", 14: "2019-01-01", 33: "2018-01-01", 42: "2018-01-01",
                   68: "2013-05-01", 110: "2016-06-01", 53: "2015-01-01"}
_US_KEEP_AFTER = {8: "2010-05-01", 11: "2010-10-01", 19: "2012-06-01", 24: "2017-01-01",
                  46: "2014-01-01", 105: "2015-01-01"}
_US_IGNORE = {9: [("2010-07-30", "2010-08-20")], 60: [("2013-02-01", "2013-06-01")],
              16: [("2015-01-01", "2016-06-30")], 20: [("2013-08-01", "2013-12-30")],
              25: [("2012-07-15", "2013-05-01"), ("2011-07-15", "2011-09-15")]}
_US_EXTRA = {19: (">", "2013-08-01"), 73: ("<", "2016-05-15"), 80: (">", "2011-06-15"),
             82: ("<", "2016-12-01")}


def remove_us_periods(sid, df):
    if sid in _US_KEEP_BEFORE:
        df = df[df.index < _US_KEEP_BEFORE[sid]]
    if sid in _US_KEEP_AFTER:
        df = df[df.index > _US_KEEP_AFTER[sid]]
    for start, end in _US_IGNORE.get(sid, []):
        df = df[(df.index < start) | (df.index > end)]
    if sid in _US_EXTRA:
        op, date = _US_EXTRA[sid]
        df = df[df.index > date] if op == ">" else df[df.index < date]
    return df


def load_jung():
    """Hourly Jungfraujoch NM count rate (column RCORR_E) used for the CRNS incoming correction."""
    df = read_csv(config.JUNG_FILE, sep=",", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")[["RCORR_E"]]


# ----------------------------------------------------------------- NMDB --------------
def _read_nmdb_file(path, code):
    with gzip.open(path, "rt") as f:
        lines = f.readlines()
    header = next((i for i, line in enumerate(lines) if "start_date_time" in line), None)
    if header is None:          # NMDB "Sorry, no data available" placeholder
        return None
    df = pd.read_csv(io.StringIO("".join(lines)), index_col=0, header=header,
                     names=["Datetime", code], sep=";", na_values="   null")
    df.index = pd.to_datetime(df.index)
    df[code] = df[code].astype(float)
    return df


def load_nmdb_decadal(kind, code, dtlist=("revori", "1h")):
    """Hourly NMDB series (``kind`` = 'uncorrected' or 'pressure_mbar').

    Revised-original data take precedence; gaps are filled with the 1-h table.  Each
    file is outlier-filtered first (``remove_outliers``, unc_fact=2), as in calc_beta1.
    """
    folder = config.NMDB_DIR / "full_series_decades" / kind
    parts = {}
    for dt in dtlist:
        frames = []
        for fn in sorted(folder.glob(f"{code}_{dt}_*.txt.gz")):
            d = _read_nmdb_file(fn, code)
            if d is None:
                continue
            d[code] = remove_outliers_rolling(d[code], unc_fact=2)
            frames.append(d)
        parts[dt] = pd.concat(frames) if frames else None

    out = None
    for dt in dtlist:
        if parts[dt] is None:
            continue
        p = parts[dt][~parts[dt].index.duplicated()]
        p = p[p.index == p.index.floor("h")]          # original joins onto an hourly grid
        if out is None:
            out = p
        else:
            idx = out.index.union(p.index)
            out = out.reindex(idx).fillna(p.reindex(idx))
    return out


def load_nm_reference_median():
    """Median of all efficiency-corrected NMs (reference for the McJannet correction)."""
    df = read_csv(config.NMDB_DIR / "from_2014" / "corr_for_efficiency" / "median.txt",
                  index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
