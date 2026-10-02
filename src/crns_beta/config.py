"""Paths, station lists and Method A settings.

Every number here is taken from the code that produced Davies et al. (2026),
Sensors 26, 925 (``5th_Activity.ipynb`` for CRNS, ``NM_method_A.ipynb`` for the
neutron monitors).  Where that code differs from what the paper's text says, the
code value is used (so the published tables are reproduced) and the difference is
noted next to the setting.
"""
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
OUTPUTS = REPO / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

EU_DIR = RAW / "cosmos_europe"
US_DIR = RAW / "cosmos_us"
NMDB_DIR = RAW / "nmdb"
JUNG_FILE = RAW / "jung" / "JUNG_Data_UTC.TXT.gz"
CUTOFF_FILE = RAW / "cutoff_rigidity" / "Cut-off_Rigidity_2020.nc"

# Gravity used to turn pressure (hPa) into atmospheric depth x = 10 P / g (g cm-2), Eq. (3).
G_DEPTH = 9.8


# --- Station sets shown in the paper (Table A2, Figures 3, A3, A4) -------------------
# COSMOS-Europe: the 50 stations of Figure A4 / Table A2.
EU_STATIONS = [
    "AAC001", "ACC001", "ALC002", "BBC001", "CAC001", "CUC001", "DEC001", "ELC001",
    "ESC001", "EUC001", "FEC001", "FIC001", "FSC001", "GBC001", "GLC001", "GRC001",
    "GSC001", "HAC001", "HDC001", "HHC001", "HOC001", "JEC001", "LUC001", "MEBCK001",
    "PEC001", "RIC001", "ROC001", "ROC002", "RUBCDKR001", "RUBCK002", "RUBCK003",
    "RUBCK004", "RUC004", "RUC005", "RUC006", "RUC007", "RUC008", "SAC001", "SCC001",
    "SCC004", "SEC001", "SHC001", "SHC002", "SHC003", "VOC001", "WAC001", "WEC001",
    "WUC001", "WUC002", "ZUC001",
]
# Stations the notebook never processes (no usable data).
EU_SKIP = ["BUC001", "LEC001", "CBC001"]

# COSMOS-US: the 21 stations of Figure A3 / Table A2 (IDs of COSMOS_<ID>.mat).
US_STATIONS = [
    "26", "46", "49", "30", "31", "16", "6", "38", "28", "29", "42", "87", "45", "18",
    "41", "88", "69", "95", "97", "43", "99",
]
US_SKIP = ["106", "107", "96", "801", "55"]

# Neutron monitors: all NMDB codes listed in Väisänen et al. (2021) table S02 are tried.
NM_BLACKLIST = ["MCRL"]          # mobile laboratory, no fixed site
NM_DROP_AFTER = ["NVBK"]         # processed, then dropped by the notebook
NM_QUANTILE_FILTER = ["PTFM", "NANM", "IRK3"]   # 1st-99th percentile filter on pooled data


@dataclass(frozen=True)
class MethodASettings:
    """Acceptance rules for one monthly regression of Eq. (4).

    ``beta_min``/``beta_max`` bound -slope (per g cm-2).  The paper's text quotes
    0.004-0.02 (Section 2.2) / 0.4-1 (Figure 1); the values below are those
    actually used to produce Tables A1/A2.
    """
    beta_min: float
    beta_max: float
    r_max: float = -0.9            # r <= -0.9  (paper: r^2 >= 0.9)
    min_days: int = 15             # paper text: 18 days
    max_daily_change: float = 0.10  # drop days whose count differs >10% from the next day
    ln_ratio_clip: float | None = None   # |ln(N_j/N_j+1)| limit applied before regression
    min_pooled_points: int = 0     # station kept only if pooled points > this ...
    min_valid_months: int = 0      # ... and at least this many valid months


EU_SETTINGS = MethodASettings(beta_min=0.0056, beta_max=0.0087, min_valid_months=4)
US_SETTINGS = MethodASettings(beta_min=0.0066, beta_max=0.0080, ln_ratio_clip=0.2,
                              min_pooled_points=50, min_valid_months=3)


@dataclass(frozen=True)
class NMSettings:
    beta_min: float = 0.005
    beta_max: float = 0.010
    r_max: float = -0.9
    min_days: int = 18
    pooled_beta_min: float = 0.005   # pooled slope must lie in (0.005, 0.0085)
    pooled_beta_max: float = 0.0085
    max_dx: float = 40.0             # |x_j - x_j+1| < 40 g cm-2
    max_dln: float = 0.7             # |ln(N_j/N_j+1)| < 0.7
    # Reproduce two behaviours of NM_method_A.ipynb that change the published numbers:
    #  - a month with < min_days days re-uses r/slope of the last regression run;
    #  - pressure is converted to g cm-2 before being passed to the McJannet & Desilets
    #    correction and to the analytical model, both of which expect hPa.
    legacy_stale_regression: bool = True
    legacy_depth_as_pressure: bool = True


NM_SETTINGS = NMSettings()

# CRNS_lib.remove_some_time_stamps holds manual exclusion periods for some COSMOS-US
# stations, but the notebooks pass the station ID as a string while the tables are keyed
# by integers, so no period was ever removed.  Off by default to match the published values.
APPLY_US_PERIOD_REMOVAL = False

# pandas < 3 padded NaNs before pct_change (fill_method="pad").  The notebooks ran on
# that behaviour, which keeps the day before each data gap in the analysis.
LEGACY_PCT_CHANGE_PAD = True


@dataclass(frozen=True)
class MapExtent:
    extent: list = field(default_factory=list)
    lat: tuple = ()
    lon: tuple = ()


US_MAP = MapExtent([-125, -65, 17, 60], (20, 55), (-140, -50))
EU_MAP = MapExtent([-13, 20, 37, 60.5], (40, 65), (-20, 30))
GLOBAL_MAP = MapExtent([-180, 180, -90, 85], (-90, 90), (-180, 180))
