# PathSnap – Distributed Map-Matching & Route Reconstruction
UE24CS352A Machine Learning hackathon, PES University.

PathSnap matches noisy GPS fixes to directed road arcs and reconstructs a continuous route:

```
.track/.nodes/.arcs ─► per-trip UTM projection ─► STRtree candidates (50 m, top-8)
   ─► 14 location-free features ─► classifier P(candidate is correct)
   ─► HMM + Viterbi (road connectivity) ─► route ─► FastAPI (Laptop 1) ─► Leaflet UI (Laptop 2)
```

**Status:** Phase-1 submission. Every component is implemented and runs end to end on the real data. Results are first-pass and will be refined. Every number below is produced by the scripts listed, from files in `results/`.

## Where each deliverable lives
| Deliverable | Code | Outputs |
|---|---|---|
| Data processing / preprocessing | `src/pathsnap/data/` (loaders, audit, network stats), `src/pathsnap/geo.py` | `results/eda/*.csv`, `docs/STAGE1_DATASET_AUDIT.md` |
| Candidate generation | `src/pathsnap/candidates/generate.py` | `results/features/candidate_quality.csv` (candidate miss rate) |
| Label derivation | `src/pathsnap/labels/align.py` (monotonic DP alignment of fixes to the reference route) | – |
| Feature engineering | `src/pathsnap/features/build.py` | `results/features/*.parquet` (regenerated, not committed) |
| Model training (6 models) | `src/pathsnap/models/registry.py`, `train_all.py` | `results/models/comparison.csv`, `cv_folds.csv`, `configs.json` |
| Model evaluation | `src/pathsnap/eval/metrics.py`, `train_all.py` | `results/models/*.png` (CV/student-test F1, Top-1, log loss, confusion matrices, MLP loss curve) |
| Bayesian optimisation | `src/pathsnap/models/optimise.py` (Optuna TPE) | `results/optuna/` (trials, search space, best params, tuned vs untuned) |
| HMM + Viterbi | `src/pathsnap/hmm/viterbi.py`, `hmm/compare.py` | `results/hmm/` (decoder grid, classifier vs HMM, map) |
| Shared inference pipeline | `src/pathsnap/pipeline.py` | `models/final/` (model, feature list, HMM params) |
| API | `backend/app.py` (FastAPI) | `logs/api.log` at runtime |
| UI | `frontend/` (HTML + Leaflet) | – |
| Deployment | `docs/DEPLOY.md`, `requirements.txt` | – |
| Visualisations | `src/pathsnap/viz/` (Folium + matplotlib) | `results/eda/figures/`, `results/eda/maps/`, `results/hmm/*.html` |

## Setup (Python 3.11)
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
On Linux/macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.
Data is **not** in git. Copy it to `data/train/` and `data/student_test/` (see `data/README.md`).

## Reproduce everything (in order)
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
```bash
python -m pathsnap.features.build
```
```bash
python -m pathsnap.models.train_all
```
```bash
python -m pathsnap.models.optimise
```
```bash
python -m pathsnap.hmm.compare
```

## Run the system
See [docs/DEPLOY.md](docs/DEPLOY.md). In short:
- **Laptop 1:** `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
- **Laptop 2:** set the API IP in `frontend/config.js`, then `python -m http.server 5500 --directory frontend`

## Leakage safeguards
- Evaluator data is never used for training, tuning or selection.
- `load_trip` never reads `.route`.
- `load_reference_route` refuses `evaluator_test` unless the server-side evaluator passes `allow_evaluator=True`.
- Reference routes are never sent to the UI.
- GroupKFold by trip is used for all cross-validation.
- The HMM is tuned on held-out train trips only.

## Team and ownership
| Member | Workstream |
|---|---|
| Prithvish | data pipeline, EDA, candidates, features |
| Krishna | models, evaluation, Bayesian optimisation |
| Pulkit | label alignment, HMM / Viterbi |
| Priyanka | metrics, API, UI, deployment |

Docs: [roadmap](docs/PATHSNAP_ROADMAP.md) · [audit](docs/STAGE1_DATASET_AUDIT.md) · [stage 2](docs/STAGE2_GUIDE.md) · [git workflow](docs/GIT_WORKFLOW.md)

AI assistance (Claude) was used for scaffolding and is disclosed here as the event policy requires.
