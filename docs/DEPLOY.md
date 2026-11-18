# Two-laptop deployment

| Laptop | Runs | Shows the panel |
|---|---|---|
| **1 – API** | `uvicorn backend.app:app` (candidates → features → LightGBM → HMM/Viterbi → server-side scoring) | the uvicorn / `logs/api.log` console, where every request is logged with stage timings |
| **2 – UI** | static `frontend/` (Leaflet map) served with `python -m http.server` | trip buttons, map layers, metrics |

There are two ways to connect the laptops. Both keep the same roles: the API and its logs on Laptop 1, the UI on Laptop 2.

| Option | Use when | Needs |
|---|---|---|
| **A. Same Wi-Fi / hotspot** | Both laptops can reach each other on the local network | Same network, firewall rule, Laptop 1's local IP |
| **B. Cloudflare Tunnel** (what we demo with) | Venue Wi-Fi blocks laptop-to-laptop traffic, or the laptops are on different networks | Internet on both laptops, `cloudflared` on Laptop 1 |

---

## Laptop 1 setup (both options)
```bash
git clone https://github.com/GitGud-User/PathSnap_Team_51.git
```
```bash
cd PathSnap_Team_51
```
```bash
python -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python.exe -m pip install -e .
```
(On Linux/macOS, use `python3 -m venv .venv`, then `.venv/bin/python -m pip ...`. Calling the venv's Python directly avoids PowerShell's "running scripts is disabled" error with `activate`.)

Copy the dataset into `data/` (see `data/README.md`). On evaluation day, put the 10 evaluator trips in `data/evaluator_test/<id>/`. The API serves `evaluator_test` automatically when that folder has trips; otherwise it serves `student_test` for rehearsals.

---

## Option A – same network

**Laptop 1**
1. Find the IP address. On Windows run `ipconfig` and take the Wi-Fi adapter's IPv4 address. On Linux run `hostname -I` and take the first address. Example: `192.168.43.10`.
2. Open port 8000. Run this once on Windows in an **Administrator** terminal:
   ```bash
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```
   On Linux, only needed if `sudo ufw status` says active: `sudo ufw allow 8000/tcp`.
3. Start the API on the network. Leave this window visible for the panel:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```

**Laptop 2**
1. Check you can reach it: open `http://192.168.43.10:8000/api/health`. It must return JSON.
2. In `frontend/config.js` set `window.PATHSNAP_API = "http://192.168.43.10:8000";`.
3. Start the UI and open `http://127.0.0.1:5500`:
   ```bash
   python -m http.server 5500 --directory frontend
   ```

---

## Option B – Cloudflare Tunnel

A *quick tunnel* gives the API on Laptop 1 a temporary public HTTPS address, `https://<random>.trycloudflare.com`, so Laptop 2 can reach it from any network. All computation and all logs stay on Laptop 1. Cloudflare only relays the requests.

**Laptop 1**
1. Install cloudflared, once:
   ```bash
   winget install --id Cloudflare.cloudflared -e
   ```
   Close and reopen the terminal afterwards. On Linux, install it from Cloudflare's downloads page.
2. **Terminal 1: start the API.** Local-only is enough, because the tunnel connects locally:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```
3. **Terminal 2: start the tunnel.**
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
   Near the top of its output, copy the URL from the box *"Your quick Tunnel has been created! Visit it at …"*. A later `ERR Failed to initialize DNS local resolver` line is harmless; the line that matters is `Registered tunnel connection`.

**Laptop 2**
1. Check it: open `https://<random>.trycloudflare.com/api/health`. It must return `"status": "ok"`. The first request can take 10–30 s.
2. In `frontend/config.js` set the tunnel URL, with no trailing slash:
   ```js
   window.PATHSNAP_API = "https://<random>.trycloudflare.com";
   ```
3. Start the UI and open `http://127.0.0.1:5500`:
   ```bash
   python -m http.server 5500 --directory frontend
   ```

**Know before demoing**
- Keep the cloudflared terminal open the whole time. Every restart gives a **new URL**, so `config.js` must be updated each time.
- While the tunnel runs, the API is reachable by anyone with the URL. It only returns predictions and metrics; reference routes never leave Laptop 1. Stop it with `Ctrl + C` afterwards.
- It adds roughly 100–500 ms per request. The UI's "round trip" timing shows this.
- Opening the bare tunnel URL returns a JSON list of endpoints. Use `/docs` for the interactive API page.
- Don't commit `config.js` with a tunnel URL or IP in it. Change it back to `http://127.0.0.1:8000`.

---

## Viewing the Folium maps (report screenshots)
OpenStreetMap blocks map tiles for pages opened by double-clicking an HTML file (`file:///...`), showing "403 Access blocked". Serve the `results` folder instead:
```bash
.venv\Scripts\python.exe -m http.server 5600 --directory results
```
Then open:
- `http://127.0.0.1:5600/eda/maps/train_00000001.html` (EDA map, report section 2.4)
- `http://127.0.0.1:5600/hmm/map_student_test_00000000.html` (classifier vs HMM, report section 6)

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| UI shows the red "Offline" pill | API not running; wrong URL/IP in `config.js`; for Option A, firewall rule missing or different networks; for Option B, cloudflared stopped or URL changed |
| `{"detail":"Not Found"}` | You opened a path that doesn't exist. Use `/`, `/docs`, `/api/health` or `/api/trips` |
| Tunnel URL gives `502 Bad Gateway` | cloudflared is running but uvicorn isn't. Start the API on port 8000 |
| `Address already in use` / `WinError 10048` | Another server uses the port. Close it, or use `--port 8001` and update `config.js` |
| `FileNotFoundError models/final/model.joblib` | Pull the committed `models/final/`, or run the training steps in the README |
| 404 trip not in partition | The trip folder is missing from `data/<partition>/` |
| Map tiles show "403 Access blocked" | Open the page through `http://`, not by double-clicking (see above) |
| Slow on a 19k-fix trip | Expected (several seconds). The UI shows a spinner and uvicorn logs each stage |
