# Stage 2 – Dataset exploration & validation (owner: Prithvish)

## What we implemented and why
Stage 1's audit was a throwaway C# script. Stage 2 rebuilds it as **reusable, tested Python**, because every later stage imports these loaders:

| Module | Purpose |
|---|---|
| `src/pathsnap/config.py` | Data and result paths. `PATHSNAP_DATA_DIR` overrides the data root (used on Laptop 1 for evaluator data) |
| `src/pathsnap/geo.py` | Haversine, **per-trip UTM projection** (`Projector`), bearing, angle difference |
| `src/pathsnap/data/loaders.py` | `load_trip()` returns a `Trip` (GPS, nodes, arcs, projected coords). `load_reference_route()` is **separate and refuses `evaluator_test`** |
| `src/pathsnap/data/network.py` | Network statistics: duplicates, reverse twins, weak/strong components, arc lengths |
| `src/pathsnap/data/audit.py` | Full audit, writing the CSV/JSON/log files below |
| `src/pathsnap/viz/eda_plots.py` | 7 report figures |
| `src/pathsnap/viz/folium_maps.py` | Reusable layered Folium map. The API reuses it later. It refuses to draw the reference route for evaluator trips |
| `src/pathsnap/viz/eda_maps.py` | Generates 5 showcase maps |
| `tests/test_data_stage2.py` | 6 tests: distances, projection, loaders, leakage guard |

**Input:** `data/train/*`, `data/student_test/*`, laid out as `data/<partition>/<id>/<id>.{track,nodes,arcs,route}`.
**Output:** `results/eda/` (listed below).

## Important code decisions (be ready to defend these)
- **Nodes and arcs are never dropped.** The row index *is* the ID, so dropping a row would silently re-number everything after it. Bad rows are kept and flagged instead.
- **The `.route` file is not read by `load_trip`.** Prediction code physically cannot see ground truth unless it calls `load_reference_route`, and that call raises `PermissionError` for evaluator trips.
- **The UTM zone is chosen from each trip's centroid.** The test `test_projection_matches_haversine` shows projected distances agree with great-circle distances to within 0.2 %.
- **Anomaly thresholds only flag at this stage; they never delete:** 60 s gap, 50 m/s jump, 50 m from route. `data_quality_log.csv` records each decision.

## Commands (Windows PowerShell, from `C:\ML_Hackathon`)

One-time setup:
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
pip install -e .
```

Run:
```bash
pytest -q
```
```bash
python -m pathsnap.data.audit
```
```bash
python -m pathsnap.viz.eda_plots
```
```bash
python -m pathsnap.viz.eda_maps
```

## How to verify (results from our run on 18-11-2026)
| Check | Expected |
|---|---|
| `pytest -q` | `6 passed` |
| `audit_summary.json` → train | 80 trips, 208,417 obs, 2,978.4 km, 96.5 % 1 s steps, 22 trips with a gap > 60 s, 1,524 jumps, 13 route breaks |
| `audit_summary.json` → student_test | 10 trips, 19,160 obs, 297.0 km, 98.62 % 1 s steps, 1 trip with a gap > 60 s, 8 jumps, 1 route break |
| Audit runtime | about 11 s |
| Cross-check against the Stage 1 C# audit | Identical on every count column for all 90 trips. GPS-to-route distances differ by at most 0.08 m, because UTM differs slightly from the C# equirectangular approximation |
| `results/eda/figures/` | 7 PNGs |
| `results/eda/maps/` | 5 HTML maps. Open them in a browser and toggle the layers in the top-right panel |

## New finding in Stage 2 (Stage 1 doc corrected)
Trips 54 and 33 are **not** routes that end early. Their far-from-route fixes (16,849 and 3,445) each sit inside a patch of about 1 km, which fits a long stop or a car park with GPS drift. The drift also inflates their path length. Decision: label these fixes *unmatched* and flag them as stationary in Stage 3. Look at `07_sample_trajectories.png` (middle panel) and `maps/train_00000033.html`.

## Common errors
| Symptom | Fix |
|---|---|
| `Python was not found` | Install Python 3.11 (`winget install Python.Python.3.11`) and reopen the terminal, or call `.venv\Scripts\python.exe` directly |
| `ModuleNotFoundError: pathsnap` | Run `pip install -e .` inside the activated venv |
| `FileNotFoundError ... data\train` | Put the dataset at `data/train` and `data/student_test`, or set `$env:PATHSNAP_DATA_DIR` |
| `pip` fails building `pyproj`/`shapely` | Upgrade pip. These packages have Windows wheels for 3.11, so avoid Python 3.13 |
| Map shows grey tiles | The map needs internet access for OSM tiles. The data layers still render without it |
| `PermissionError: Evaluator ground truth…` | Working as intended. Only the server-side evaluator may pass `allow_evaluator=True` |

## Report material produced (section 2.2)
Missing-value analysis (`02`), GPS trajectory plots (`07`), road-network visualisations (`04`, `07`), Folium maps (`maps/`). All of these use the **train + student_test** partitions; the figure legends say which.
