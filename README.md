# PathSnap – Distributed Map-Matching & Route Reconstruction
UE24CS352A Machine Learning hackathon, PES University.

PathSnap matches noisy GPS fixes to directed road arcs and reconstructs a continuous route:

```
.track/.nodes/.arcs ─► per-trip UTM projection ─► STRtree candidates (50 m, top-8)
   ─► 14 location-free features ─► classifier P(candidate is correct)
   ─► HMM + Viterbi (road connectivity) ─► route ─► FastAPI (Laptop 1) ─► Leaflet UI (Laptop 2)
```

**Status:** Phase-1 submission. Every component is implemented and runs end to end on the real data. Results are first-pass and will be refined. Every number below is produced by the scripts listed, from files in `results/`.

## Phase-1 results (measured, reproducible from `results/`)

**Candidate generation** (50 m radius, top 8): candidate miss rate **0.72 %** on train and **0.27 %** on student-test. Of the GPS fixes, 185,226 / 208,417 (train) and 19,133 / 19,160 (student-test) are labelled; the rest are more than 50 m from the reference route.

**Model comparison:** GroupKFold(5) by trip on the 80 train trips; student-test (ST) is all 10 trips. Source: `results/models/comparison.csv`.

| Model | Family | CV macro-F1 | CV Top-1 | ST Top-1 | ST macro-F1 | ST log loss | ST continuity (classifier only) | Inference ms / 1k rows |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | Linear | 0.882 | 0.833 | 0.840 | 0.888 | 0.173 | 0.734 | 0.1 |
| KNN (k=25, distance-weighted) | Instance-based | 0.895 | 0.832 | 0.849 | 0.905 | 0.176 | 0.740 | 385 |
| Random forest | Bagging | 0.905 | 0.844 | 0.861 | 0.915 | 0.118 | 0.761 | 2.5 |
| HistGradientBoosting | Boosting | 0.906 | 0.847 | 0.868 | 0.916 | 0.114 | 0.767 | 2.5 |
| **LightGBM (selected)** | Boosting | **0.907** | **0.849** | **0.871** | **0.917** | **0.112** | **0.773** | 1.9 |
| MLP (64-32) | Neural net | 0.902 | 0.842 | 0.857 | 0.909 | 0.125 | 0.767 | 0.6 |

**Bayesian optimisation** (Optuna TPE, 20 trials, 433 s, objective = GroupKFold(3) Top-1 on train): the best trial reached CV Top-1 **0.8477**, versus **0.8483** for the initial configuration, so tuning gave **no improvement**. The final model was chosen on CV only, so the initial configuration was kept. Source: `results/optuna/`.

**HMM / Viterbi vs classifier only** (decoder tuned on 12 held-out train trips; mean over the 10 student-test trips). Source: `results/hmm/`.

| | Classifier only | + HMM/Viterbi |
|---|---|---|
| Pointwise accuracy | 0.883 | **0.894** |
| Route continuity | 0.773 | **0.910** |
| Invalid transitions per trip | 127.8 | **46.0** |
| Route arc F1 | 0.856 | **0.882** |
| Decoder runtime per trip | – | 308 ms |

The HMM improved 9 of 10 trips. On trip 77 it did not help (0.729 → 0.717).

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
