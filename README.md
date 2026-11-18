<h1 align="center">PathSnap</h1>
<p align="center"><b>Distributed Map-Matching &amp; Route Reconstruction</b><br>
UE24CS352A Machine Learning Hackathon · PES University</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688">
  <img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-orange">
  <img alt="HMM" src="https://img.shields.io/badge/Decoder-HMM%20%2B%20Viterbi-6250d6">
  <img alt="Tests" src="https://img.shields.io/badge/tests-10%20passing-brightgreen">
</p>

PathSnap takes **noisy GPS traces**, matches every GPS fix to the **directed road segment** the vehicle was actually on, and rebuilds a **continuous, drivable route**. It combines a machine-learning classifier with a Hidden Markov Model decoded by Viterbi, and it runs as a **FastAPI server (Laptop 1)** that serves a **web map UI (Laptop 2)**.

---

## Table of contents
1. [How it works](#1-how-it-works)
2. [Results](#2-results)
3. [Project structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Add the dataset](#6-add-the-dataset)
7. [Quick start: run the app on one laptop](#7-quick-start--run-the-app-on-one-laptop)
8. [Two-laptop deployment](#8-two-laptop-deployment-evaluation-setup)
9. [Reproduce every experiment](#9-reproduce-every-experiment)
10. [API reference](#10-api-reference)
11. [Leakage safeguards](#11-leakage-safeguards)
12. [Troubleshooting](#12-troubleshooting)
13. [Team](#13-team)

---

## 1. How it works

```
 .track / .nodes / .arcs  (one road network per trip)
            │
            ▼
 ① Preprocess   per-trip UTM projection (metres), speed, heading, gap flags
            │
            ▼
 ② Candidates   R-tree (STRtree) → up to 8 road arcs within 50 m of each GPS fix
            │
            ▼
 ③ Features     14 location-free features per (fix, candidate)
            │
            ▼
 ④ Classifier   LightGBM → P(candidate is the true arc)
            │
            ▼
 ⑤ HMM+Viterbi  picks the most likely *connected* sequence of arcs
            │
            ▼
 ⑥ FastAPI      runs ①–⑤ live per request, scores it server-side ──► ⑦ Web UI (Leaflet map)
```

| Stage | Main file |
|---|---|
| Loading and preprocessing | `src/pathsnap/data/loaders.py`, `src/pathsnap/geo.py` |
| Candidate generation | `src/pathsnap/candidates/generate.py` |
| Labels (training only) | `src/pathsnap/labels/align.py` |
| Features | `src/pathsnap/features/build.py` |
| Six models + comparison | `src/pathsnap/models/registry.py`, `train_all.py` |
| Bayesian optimisation | `src/pathsnap/models/optimise.py` |
| HMM / Viterbi | `src/pathsnap/hmm/viterbi.py`, `compare.py` |
| One shared inference pipeline | `src/pathsnap/pipeline.py` |
| API | `backend/app.py` |
| UI | `frontend/` |

---

## 2. Results
All numbers come from files in `results/` and can be regenerated with [Section 9](#9-reproduce-every-experiment).

**Candidate generation** (50 m, top-8): candidate miss rate is **0.72 %** on train and **0.27 %** on student-test.

**Model comparison** uses GroupKFold(5) grouped by trip on the 80 train trips. ST = the 10 student-test trips. Source: `results/models/comparison.csv`.

| Model | Family | CV macro-F1 | CV Top-1 | ST Top-1 | ST macro-F1 | ST log loss | Inference (ms / 1k rows) |
|---|---|---|---|---|---|---|---|
| Logistic regression | Linear | 0.882 | 0.833 | 0.840 | 0.888 | 0.173 | 0.1 |
| KNN (k = 25) | Instance-based | 0.895 | 0.832 | 0.849 | 0.905 | 0.176 | 385 |
| Random forest | Bagging | 0.905 | 0.844 | 0.861 | 0.915 | 0.118 | 2.5 |
| HistGradientBoosting | Boosting | 0.906 | 0.847 | 0.868 | 0.916 | 0.114 | 2.5 |
| **LightGBM (selected)** | Boosting | **0.907** | **0.849** | **0.871** | **0.917** | **0.112** | 1.9 |
| MLP (64-32) | Neural network | 0.902 | 0.842 | 0.857 | 0.909 | 0.125 | 0.6 |

**Bayesian optimisation** used Optuna TPE: 20 trials, 433 s. The best trial reached CV Top-1 0.8477, versus 0.8483 for the initial configuration, so **tuning gave no gain**. The initial configuration was kept, chosen on CV only (`results/optuna/`).

**HMM / Viterbi vs classifier only** (mean over the 10 student-test trips; `results/hmm/`). The transition model uses **road-network distance** (Dijkstra over the directed arc graph, capped at the GPS step + 300 m). Parameters were tuned on 12 held-out train trips:

| Metric | Classifier only | + HMM / Viterbi |
|---|---|---|
| Pointwise accuracy | 0.883 | **0.928** |
| Route continuity | 0.773 | **0.922** |
| Invalid transitions per trip | 127.8 | **32.0** |
| Route arc F1 | 0.856 | **0.918** |
| Decoder time per trip | – | 464 ms |

The HMM improved all 10 trips. The earlier straight-line-distance version reached 0.894 accuracy and made trip 77 slightly worse; with road distance, trip 77 goes from 0.729 to 0.789.

---

## 3. Project structure

```
pathsnap/
├── backend/app.py              FastAPI server (Laptop 1)
├── frontend/                   Web UI (Laptop 2): index.html, app.js, config.js
├── src/pathsnap/               The Python package (imported as `pathsnap`)
│   ├── config.py               Paths and constants
│   ├── geo.py                  Distances, UTM projection, bearings
│   ├── data/                   Loaders, dataset audit, network statistics
│   ├── candidates/             Spatial index + candidate generation
│   ├── labels/                 GPS-to-reference-route alignment (training labels)
│   ├── features/               Feature table builder
│   ├── models/                 6 models, training/comparison, Optuna
│   ├── hmm/                    Viterbi decoder + classifier-vs-HMM comparison
│   ├── eval/                   Shared metric definitions
│   ├── viz/                    Folium maps + matplotlib figures
│   └── pipeline.py             predict_trip(): used by experiments AND the API
├── models/final/               Final model, feature list, HMM parameters
├── results/                    Every CSV / PNG / HTML the report uses
├── tests/                      pytest suite
├── docs/                       Audit, roadmap, deployment guide, speaker guide
├── tools/                      Helper scripts (speaker-guide PDF, Stage-1 audit)
├── data/                       Dataset goes here (NOT committed, see Section 6)
├── requirements.txt
└── pyproject.toml
```

> **Why `pathsnap/src/pathsnap`?** The outer folder is the repository. The inner `src/pathsnap` is the Python package, so code can run `from pathsnap... import ...`. This is the standard "src layout".

---

## 4. Prerequisites

| Requirement | Version | Check with |
|---|---|---|
| Python | **3.11** (3.10 also works; avoid 3.13) | `python --version` |
| Git | any recent | `git --version` |
| Disk space | ~1.5 GB (dependencies + data + generated tables) | |
| RAM | 8 GB minimum, 16 GB recommended | |
| Internet | for `pip install`, and for map tiles in the UI | |

Install Python on Windows if needed:
```bash
winget install Python.Python.3.11
```

---

## 5. Installation

### 5.1 Clone the repository
```bash
git clone https://github.com/GitGud-User/PathSnap_Team_51.git
```
```bash
cd PathSnap_Team_51
```

### 5.2 Create and activate a virtual environment

**Windows (PowerShell / CMD)**
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```

When it's active, your prompt starts with `(.venv)`.

> If PowerShell says *"running scripts is disabled"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use CMD instead.

### 5.3 Install dependencies and the project package
```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```
```bash
pip install -e .
```

`pip install -e .` registers `src/pathsnap` as an importable package, so every `python -m pathsnap...` command works from the project root.

### 5.4 Verify the install
```bash
pytest -q
```
Expected output: **`10 passed`**. Two of the tests need the dataset from Section 6.

---

## 6. Add the dataset

The dataset is **not in git**. Copy the supplied folders so that the layout looks exactly like this:

```
data/
├── train/
│   ├── 00000001/
│   │   ├── 00000001.track
│   │   ├── 00000001.nodes
│   │   ├── 00000001.arcs
│   │   └── 00000001.route
│   └── ...                      (80 trip folders)
├── student_test/                (10 trip folders, same layout)
└── evaluator_test/              (empty – filled on Laptop 1 on evaluation day)
```

From the supplied `Student folder.zip`, copy its `train` and `student_test` folders into `data/`.

If you keep the data somewhere else, point the project at it before running anything:

| Shell | Command |
|---|---|
| PowerShell | `$env:PATHSNAP_DATA_DIR = "D:\pathsnap_data"` |
| CMD | `set PATHSNAP_DATA_DIR=D:\pathsnap_data` |
| Linux / macOS | `export PATHSNAP_DATA_DIR=/path/to/data` |

---

## 7. Quick start – run the app on one laptop

The trained model is already committed in `models/final/`, so **no training is needed** to run the app.

**Terminal 1: start the API**
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Wait for this line:
```
... | INFO | model + HMM params loaded; serving partition 'student_test' (10 trips)
```

**Terminal 2: serve the UI**
```bash
python -m http.server 5500 --directory frontend
```

**Browser:** open **http://127.0.0.1:5500**. The status pill should turn green ("API live"). Click any trip, and the full pipeline runs on the server and draws the route.

Useful checks:
- http://127.0.0.1:8000/api/health returns `{"status": "ok", ...}`.
- http://127.0.0.1:8000/docs is FastAPI's interactive API page.

> The API serves `evaluator_test` automatically when that folder contains trips. Otherwise it serves `student_test` for rehearsal. To force one, set `PATHSNAP_EVAL_PARTITION` before starting uvicorn: in PowerShell `$env:PATHSNAP_EVAL_PARTITION = "student_test"`, in CMD `set PATHSNAP_EVAL_PARTITION=student_test`.

---

## 8. Two-laptop deployment (evaluation setup)

| Laptop | Role | What the panel sees |
|---|---|---|
| **Laptop 1** | FastAPI + model + HMM + server-side scoring | the **uvicorn console**, logging every request with stage timings (also saved to `logs/api.log`) |
| **Laptop 2** | Web UI only | trip buttons, the map, metrics |

Both laptops must be on the **same network**. If the venue Wi-Fi blocks device-to-device traffic, use a **phone hotspot**.

### Laptop 1 (API)
1. Complete Sections 5 and 6. On evaluation day, put the evaluator trips in `data/evaluator_test/`.
2. Find this laptop's IP address:
   ```bash
   ipconfig
   ```
   Note the **IPv4 Address** of the Wi-Fi adapter, e.g. `192.168.43.10`.
3. Allow port 8000 through the Windows firewall. Run this once in an **Administrator** terminal:
   ```bash
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```
4. Start the API listening on the network (keep this window visible):
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```

### Laptop 2 (UI)
1. Clone the repository. Only the `frontend/` folder and Python are needed; the dependencies aren't.
2. Edit **`frontend/config.js`** and set Laptop 1's IP:
   ```js
   window.PATHSNAP_API = "http://192.168.43.10:8000";
   ```
3. Test the connection by opening `http://192.168.43.10:8000/api/health` in a browser. It must show JSON.
4. Serve the UI:
   ```bash
   python -m http.server 5500 --directory frontend
   ```
5. Open **http://127.0.0.1:5500**. The panel picks any trip, and Laptop 1's console shows the request being processed.

**Different networks, or Wi-Fi that blocks devices from reaching each other?** Use a Cloudflare quick tunnel instead. On Laptop 1, run `cloudflared tunnel --url http://127.0.0.1:8000` next to the API, then put the printed `https://….trycloudflare.com` URL in Laptop 2's `frontend/config.js`. The API and its logs stay on Laptop 1.

More detail, including both options and how to view the Folium maps: [docs/DEPLOY.md](docs/DEPLOY.md).

---

## 9. Reproduce every experiment

Run these from the project root with the virtual environment active, **in this order**. Each step reads the outputs of the previous one. Times were measured on an 8-core / 16 GB laptop.

| # | Command | What it does | Main outputs | Approx. time |
|---|---|---|---|---|
| 1 | `python -m pathsnap.data.audit` | Validates all 90 trips (files, gaps, jumps, network, routes) | `results/eda/trip_audit.csv`, `audit_summary.json`, `data_quality_log.csv` | ~15 s |
| 2 | `python -m pathsnap.data.extra_checks` | Extra dataset checks | `results/eda/extra_checks.csv` | ~1 min |
| 3 | `python -m pathsnap.viz.eda_plots` | 7 EDA figures | `results/eda/figures/*.png` | ~30 s |
| 4 | `python -m pathsnap.viz.eda_maps` | 5 interactive Folium maps | `results/eda/maps/*.html` | ~30 s |
| 5 | `python -m pathsnap.features.build` | Candidates + labels + features for train and student-test | `results/features/*.parquet`, `candidate_quality.csv` | ~2 min |
| 6 | `python -m pathsnap.models.train_all` | Trains and compares the 6 models (GroupKFold 5) | `results/models/comparison.csv`, `cv_folds.csv`, plots | ~5–8 min |
| 7 | `python -m pathsnap.models.optimise --trials 20 --timeout 420` | Optuna TPE on LightGBM; saves the final model (chosen on CV) | `results/optuna/*`, `models/final/*` | ~8 min |
| 8 | `python -m pathsnap.hmm.compare` | Tunes the HMM on held-out train trips; classifier vs HMM on student-test | `results/hmm/*`, `models/final/hmm_params.json` | ~2 min |
| 9 | `python tools/make_speaker_guide.py` | Rebuilds the panel speaker guide from the results | `docs/PathSnap_Speaker_Guide.pdf` | ~5 s |

Run the tests again at any time:
```bash
pytest -q
```

> Step 5 writes large `.parquet` tables that are git-ignored; regenerate them with step 5. Everything else in `results/` is committed so the report can be checked without rerunning.

---

## 10. API reference

Base URL: `http://<laptop-1-ip>:8000`. Interactive docs are at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Status, active partition, model type, HMM parameters |
| `GET` | `/api/trips` | Trip IDs found on disk for the active partition |
| `POST` | `/api/predict/{trip_id}` | Runs the full pipeline on one trip and returns the routes and metrics |

Example:
```bash
curl -X POST http://127.0.0.1:8000/api/predict/00000056
```

The response contains `gps` (fixes), `hmm_route` and `classifier_route` (arc line segments), `metrics` (Top-1 accuracy, route continuity, invalid transitions, candidate miss rate, route arc F1, all computed server-side), and `timings_ms`.
**It never contains the reference route.**

Errors: `404` for an unknown trip ID, `500` if the pipeline fails (with a message, also logged).

---

## 11. Leakage safeguards
- Evaluator data is never used for training, tuning, feature selection or model choice.
- `load_trip()` **never reads** `.route` files. `load_reference_route()` raises `PermissionError` for `evaluator_test` unless the server-side scorer passes `allow_evaluator=True`.
- The reference route is used only inside the API's scoring function and is **never sent to the UI**.
- All cross-validation is **GroupKFold by trip**, so a trip's fixes never appear in both training and validation.
- HMM parameters are tuned on held-out **train** trips. Student-test is used for reporting only.
- Nothing is hard-coded or cached: every `/api/predict` call runs the pipeline live.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found / opens Microsoft Store | Install Python 3.11 (Section 4), reopen the terminal |
| `ModuleNotFoundError: No module named 'pathsnap'` | Activate `.venv` and run `pip install -e .` |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use CMD |
| `FileNotFoundError ... data\train` | The dataset isn't in `data/` (Section 6) or `PATHSNAP_DATA_DIR` is wrong |
| `FileNotFoundError models/final/model.joblib` | `git pull`, or run steps 5–8 of Section 9 |
| UI: "Cannot reach the API" / red pill | API not running; wrong IP in `frontend/config.js`; firewall rule missing; laptops on different networks |
| UI loads but the map is blank | No internet (map tiles come from OpenStreetMap). Routes still load |
| Folium map shows "403 Access blocked" tiles | The HTML was opened by double-clicking. Serve it with `python -m http.server 5600 --directory results` and open it via `http://127.0.0.1:5600/...` |
| `Address already in use` on port 8000 or 5500 | Another server is running; close it or use another port (`--port 8001`) |
| A long trip takes a few seconds | Expected for trips with ~19k fixes; the spinner shows progress and uvicorn logs each stage |
| `pip` fails building `shapely` / `pyproj` | Use Python 3.11 (prebuilt wheels exist); upgrade pip first |

---

## 13. Team

| Member | Ownership |
|---|---|
| **Prithvish** | Data pipeline, EDA, candidate generation, feature engineering |
| **Krishna** | Model training and comparison, evaluation, Bayesian optimisation |
| **Pulkit** | Label alignment, HMM and Viterbi decoding |
| **Priyanka** | Shared metrics, FastAPI backend, web UI, deployment |

Further documentation: [roadmap](docs/PATHSNAP_ROADMAP.md) · [dataset audit](docs/STAGE1_DATASET_AUDIT.md) · [Stage 2 guide](docs/STAGE2_GUIDE.md) · [deployment](docs/DEPLOY.md) · [Git workflow](docs/GIT_WORKFLOW.md) · [speaker guide (PDF)](docs/PathSnap_Speaker_Guide.pdf)

*AI assistance was used for parts of the scaffolding and is disclosed here, as the event's AI-use policy requires. Every experiment was run by the team, and every reported number is reproducible from this repository.*
