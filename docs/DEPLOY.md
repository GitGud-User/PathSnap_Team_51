# Two-laptop deployment

| Laptop | Runs | Shows the panel |
|---|---|---|
| **1 – API** | `uvicorn backend.app:app` (candidates → features → LightGBM → HMM/Viterbi → server-side scoring) | the uvicorn / `logs/api.log` console, where every request is logged with stage timings |
| **2 – UI** | static `frontend/` (Leaflet map) served with `python -m http.server` | trip buttons, map layers, metrics |

Both laptops must be on the **same network**. Use a phone hotspot if the venue Wi-Fi blocks laptop-to-laptop traffic (client isolation).

## Laptop 1 (API)
```bash
git clone https://github.com/GitGud-User/pathsnap.git
```
```bash
cd pathsnap
```
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
Copy the dataset into `data/` (see `data/README.md`). On evaluation day, put the 10 evaluator trips in `data/evaluator_test/<id>/`. The API serves `evaluator_test` automatically when that folder has trips; otherwise it serves `student_test` for rehearsals.

Allow the port through the firewall (run in an **Administrator** PowerShell):
```bash
netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
```
Find Laptop 1's IP: run `ipconfig` and note the IPv4 address of the Wi-Fi adapter, e.g. `192.168.43.10`.

Start the API (leave this window visible for the panel):
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Check it locally: open `http://127.0.0.1:8000/api/health` and `http://127.0.0.1:8000/docs`.

## Laptop 2 (UI)
Only the `frontend/` folder is needed. A clone of the repo is simplest.
1. Edit `frontend/config.js` and set `window.PATHSNAP_API = "http://<Laptop-1-IP>:8000";`.
2. Serve it:
```bash
python -m http.server 5500 --directory frontend
```
3. Open `http://127.0.0.1:5500` in a browser.

Connectivity test from Laptop 2: open `http://<Laptop-1-IP>:8000/api/health` in a browser. It should return JSON.

## Troubleshooting
| Symptom | Fix |
|---|---|
| UI says "Cannot reach API" | wrong IP in `config.js`, the firewall rule is missing, the laptops are on different networks, or venue Wi-Fi isolation (switch to a hotspot) |
| `FileNotFoundError models/final/model.joblib` | run `python -m pathsnap.models.optimise` first, or pull the committed `models/final/` |
| 404 trip not in partition | the trip folder is missing from `data/<partition>/` |
| Slow on a 19k-fix trip | expected (several seconds). The UI shows a spinner, and uvicorn logs each stage |
