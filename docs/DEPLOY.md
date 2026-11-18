# Two-laptop deployment

| Laptop | Runs | Shows the panel |
|---|---|---|
| **1 – API** | `uvicorn backend.app:app` (candidates → features → LightGBM → HMM/Viterbi → server-side scoring, **and the `frontend/` UI at `/ui/`**) | the uvicorn / `logs/api.log` console, where every request is logged with stage timings |
| **2 – UI** | a browser. Either the page served by Laptop 1 (`/ui/`, or the tunnel link), or a static `frontend/` copy served with `python -m http.server` and pointed at Laptop 1 in `config.js` | trip buttons, map layers, metrics |

There are two ways to connect the laptops. Both keep the models, data and logs on Laptop 1.

| Option | Use when | Needs |
|---|---|---|
| **A. Same Wi-Fi / hotspot** | Both laptops can reach each other on the local network | Same network, firewall rule, Laptop 1's local IP |
| **B. Cloudflare Tunnel** (what we demo with) | Venue Wi-Fi blocks laptop-to-laptop traffic, or the laptops are on different networks | Internet on Laptop 1, `cloudflared` on Laptop 1, the link shared with Laptop 2 |

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
1. Open `http://192.168.43.10:8000/ui/` in a browser. That is Laptop 1 serving the UI and the API on one origin, so nothing else is needed. `/api/health` must return `"status": "ok"`.
2. Optional, only if Laptop 2 should serve its own copy of the UI: in `frontend/config.js` set `window.PATHSNAP_API = "http://192.168.43.10:8000";`, then run `python -m http.server 5500 --directory frontend` and open `http://127.0.0.1:5500`.

---

## Option B – Cloudflare Tunnel

A *quick tunnel* gives the app a temporary public HTTPS address, `https://<random>.trycloudflare.com`. Because the API also serves the UI (`/ui/`), that **single link** is the whole demo: anyone who opens it gets the map, and the map calls the API on the same origin. All computation, data and logs stay on Laptop 1; Cloudflare only relays the requests.

**Laptop 1**
1. Install cloudflared, once:
   ```bash
   brew install cloudflared                            # macOS
   winget install --id Cloudflare.cloudflared -e       # Windows
   sudo apt install cloudflared                        # Debian/Ubuntu (or see Cloudflare's downloads page)
   ```
   On Windows, close and reopen the terminal afterwards.
2. **Terminal 1: start the app.** Local-only is enough, because the tunnel connects locally:
   ```bash
   .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000           # macOS / Linux
   .venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000   # Windows
   ```
3. **Terminal 2: start the tunnel** — either the helper script, which checks that the API is up first, or cloudflared directly:
   ```bash
   ./tools/tunnel.sh
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
   Near the top of its output, copy the URL from the box *"Your quick Tunnel has been created! Visit it at …"*. A later `ERR Failed to initialize DNS local resolver` line is harmless; the line that matters is `Registered tunnel connection`.

**Any device with the link**
1. Open `https://<random>.trycloudflare.com`. It redirects to `/ui/` and the status pill turns green once `/api/health` answers. The first request can take 10–30 s.
2. `https://<random>.trycloudflare.com/api/health`, `/api` and `/docs` work too.

No `frontend/config.js` edit is needed: `window.PATHSNAP_API = ""` means "same origin". Only the separate-static-server setup (Option A, or `python -m http.server 5500`) needs the absolute URL.

**Know before demoing**
- Keep the cloudflared terminal open the whole time. Every restart gives a **new URL**, so re-share the link after a restart.
- While the tunnel runs, the app is reachable by anyone with the URL. It only returns predictions and metrics; reference routes never leave Laptop 1. Stop it with `Ctrl + C` afterwards.
- It adds roughly 100–500 ms per request. The UI's "round trip" timing shows this.
- Leave `config.js` at `""` in git. A tunnel URL or LAN IP belongs only in a local edit.

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
| UI shows the red "Offline" pill | API not running, or the page is calling the wrong origin. A page served by the API (`/ui/`, LAN IP, tunnel link) needs `config.js` at `""`; only a separately served `frontend/` copy needs Laptop 1's absolute URL. For Option A also check the firewall rule and that both laptops are on the same network |
| Tunnel link does not load at all (no page, no pill) | cloudflared isn't running, or it was restarted — every run prints a **new** URL. Restart it and re-share that link; no `config.js` change is needed |
| `{"detail":"Not Found"}` | You opened a path that doesn't exist. Use `/ui/` for the map, `/docs`, `/api`, `/api/health` or `/api/trips` |
| Tunnel URL gives `502 Bad Gateway` | cloudflared is running but uvicorn isn't. Start the API on port 8000 |
| `Address already in use` / `WinError 10048` | Another server uses the port. Close it, or use `--port 8001` and update `config.js` only if the UI is served separately |
| `FileNotFoundError models/final/model.joblib` | Pull the committed `models/final/`, or run the training steps in the README |
| 404 trip not in partition | The trip folder is missing from `data/<partition>/` |
| Map tiles show "403 Access blocked" | Open the page through `http://`, not by double-clicking (see above) |
| Slow on a 19k-fix trip | Expected (several seconds). The UI shows a spinner and uvicorn logs each stage |
