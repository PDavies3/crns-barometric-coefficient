# SP_CRNS_Beta_correction

Site-specific barometric coefficient (β) for cosmic-ray neutron sensors (CRNS) and
neutron monitors (NM), using **empirical Method A** and the **analytical model of
Desilets & Zreda (2003)**. This repository reproduces the Method A and analytical
results of

> Davies, P., Baatz, R., Schattan, P., Quansah, E., Amekudzi, L. K., Bogena, H. R. (2026).
> On the Variability of the Barometric Effect and Its Relation to Cosmic-Ray Neutron
> Sensing. *Sensors* 26, 925. https://doi.org/10.3390/s26030925

## Method A in one paragraph

Hourly neutron counts are cleaned and corrected for air humidity (Rosolem et al., 2013)
and for incoming intensity (Jungfraujoch NM for CRNS; McJannet & Desilets, 2023 for
NM). They are then averaged to days. Days whose count differs by more than 10 % from the
next day are dropped. Pressure is converted to atmospheric depth, x = 10 P / g
(Eq. 3). For each calendar month, the day-to-day changes are regressed (Eq. 4):

    ln(N_j / N_j+1) = −β (x_j − x_j+1)

A month is accepted when r ≤ −0.9 and β lies inside the acceptance bounds (see
[Settings](#settings)). The station β is the **mean of the accepted monthly slopes**.
For each accepted month, the analytical β_p (Eq. 6) and β_eff (Eq. 7) are
evaluated at that month's mean pressure.

## Usage

```bash
pip install -e .[test]   # or use the anaconda base env (all dependencies present)
make run                 # β for every station -> outputs/tables/   (~10 min)
make plots               # figures             -> outputs/figures/
make test                # unit tests
```

`make` (or `make all`) runs `run` and then `plots`. Each script can also be run on its
own, e.g. `python scripts/run_method_a.py --networks EU US`.

## Layout

```
src/crns_beta/
  config.py        paths, station lists, acceptance rules
  analytical.py    Desilets & Zreda (2003) β_p and β_eff
  corrections.py   humidity, incoming-intensity, reference-pressure, outlier filters
  io.py            readers for COSMOS-Europe, COSMOS-US (.mat), NMDB, Jungfraujoch
  method_a.py      Method A for CRNS and NM stations
scripts/
  run_method_a.py                    β for all stations -> outputs/tables/
  plot_maps.py
  plot_beta_distance_gv.py
  plot_solstice.py
  plot_petzenkirchen_sensitivity.py
  plot_regressions.py
data/raw/          all input data, gzip-compressed (not tracked in git)
tests/             unit tests (analytical model vs. original code, synthetic Method A)
outputs/           results (not tracked in git)
```

## Outputs

`run_method_a.py` writes three files per network (`eu`, `us`, `nm`) to `outputs/tables/`:

- `method_a_<net>.csv`: one row per station with location, altitude, cut-off rigidity
  (`gv`), number of valid months and days, `beta_em_a` (mean, SD, median, per season),
  `beta_p`, `beta_eff`, mean absolute humidity and soil moisture, `in_paper`.
- `method_a_<net>_monthly.csv`: every accepted monthly regression (slope, r, standard
  error, number of days, mean pressure, analytical β_p and β_eff).
- `method_a_<net>_pooled.csv.gz`: the daily (x_j − x_j+1, ln N_j/N_j+1) pairs of the
  accepted months.

β values are in %/hPa. All processed stations are included (59 EU, 42 US, 46 NM).
`in_paper` marks the 50 EU and 21 US stations listed in the paper's appendix.

## Data

Everything the calculations read is in `data/raw/`, copied from
`/media/pdavies/T7 Shield/PhD_work/Manuscript/Roland_Manuscript_2`.

- `cosmos_europe/`: raw and processed station CSVs, station metadata
  (from `COSMOS_Europe_Data/`).
- `cosmos_us/`: `COSMOS_<ID>.mat` files, site sheet, coordinates
  (from `US_barometric_coeff/data/` and `lonlat.csv`).
- `jung/`: Jungfraujoch NM (`RCORR_E`) for the CRNS incoming correction
  (from `US_barometric_coeff/Current_analysis/`).
- `nmdb/`: NMDB hourly counts and pressure (1951–2030 decades), reference median,
  station table (from `Paul_Schattan/data/nmdb/` and `Paul_Schattan/*.csv`).
- `petzenkirchen/`: COSMOS-US 087 hourly record for the Petzenkirchen sensitivity
  analysis (from `Sensitivity_analysis/PEC001.csv`).
- `cutoff_rigidity/`: cut-off rigidity grid for the map contours
  (from `US_barometric_coeff/Online/`).

The files are gzip-compressed (about 550 MB instead of 2.6 GB) and are read directly in
that form. Stations whose files are missing or unreadable are skipped (EU: SCC003; US:
070, 086, 100), as are NMDB "Sorry, no data available" placeholder files.

## Settings

Behaviour of the original notebooks, kept on purpose. Each item can be changed in
`config.py`.

- β acceptance bounds (per g cm⁻²): EU 0.0056–0.0087, US 0.0066–0.0080, NM 0.005–0.010.
- Minimum days per month: CRNS 15, NM 18.
- Minimum valid months: EU 4; US 3 (and more than 50 days).
- US: ln-ratio clipped to ±0.2.
- NM analytical value: **median** of the monthly values (CRNS: mean).
- `APPLY_US_PERIOD_REMOVAL`: off. The notebooks pass the station ID as a string, so the
  manual exclusion periods in `remove_some_time_stamps` were never applied.
- `LEGACY_PCT_CHANGE_PAD`: NaNs are padded before `pct_change` (pandas < 3 behaviour).
- `NMSettings.legacy_stale_regression`: a month with fewer than 18 days reuses r/slope
  from the previous regression (including the previous station's pooled fit).
- `NMSettings.legacy_depth_as_pressure`: NM pressure is converted to g cm⁻² before the
  McJannet correction and the analytical model, which both expect hPa.

Neutron monitors with a cut-off rigidity of 0 GV are dropped, because the analytical
model is undefined there (the original code failed and skipped them).

The reported "%hPa" values are slopes per g cm⁻² × 100, as in the paper. Per hPa they
would be 10/9.8 ≈ 1.02 × larger.
