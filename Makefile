PY ?= python3

.PHONY: all run plots test

all: run plots

run:       ## Method A + analytical beta for every station -> outputs/tables
	$(PY) scripts/run_method_a.py

plots:     ## Figures 3, 4, A1-A5 -> outputs/figures
	$(PY) scripts/plot_regressions.py
	$(PY) scripts/plot_maps.py
	$(PY) scripts/plot_beta_distance_gv.py
	$(PY) scripts/plot_solstice.py
	$(PY) scripts/plot_petzenkirchen_sensitivity.py

test:
	$(PY) -m pytest -q
