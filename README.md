# PathSnap – Distributed Map-Matching & Route Reconstruction
UE24CS352A Machine Learning hackathon, PES University.

PathSnap matches noisy GPS fixes to directed road arcs and reconstructs a continuous route. It uses a per-candidate ML classifier followed by HMM/Viterbi decoding, served by FastAPI (Laptop 1) to a web UI (Laptop 2).

## Status
| Stage | Owner | Status |
|---|---|---|
| 1 Docs + dataset inspection | all | Completed ([audit](docs/STAGE1_DATASET_AUDIT.md), [roadmap](docs/PATHSNAP_ROADMAP.md)) |
| 2 Exploration & validation | Prithvish | Testing ([guide](docs/STAGE2_GUIDE.md)) |
| 3 Preprocessing, labels, features | Prithvish / Pulkit | Not Started |
| 4 Candidate generation | Prithvish | Not Started |
| 5–7 Models, selection, Bayesian opt. | Krishna | Not Started |
| 8 HMM / Viterbi | Pulkit | Not Started |
| 9–11 API, UI, deployment | Priyanka | Not Started |
| 12 Report | all | Not Started |

## Setup (Windows, Python 3.11)
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

Data is **not** in git. Copy the dataset to `data/train/` and `data/student_test/`, one folder per trip.

## Run
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

## Layout
`src/pathsnap/` holds the library code (data, geo, viz; later candidates, features, models, hmm, eval). `results/` holds generated outputs. `docs/` holds documentation. `tests/` holds the tests. See the roadmap §7 for the full planned structure.
