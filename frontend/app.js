// PathSnap evaluator UI (Laptop 2). Everything shown comes from the live API response.
const API = window.PATHSNAP_API;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (id) => document.getElementById(id);

const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
const layers = { gps: L.layerGroup(), clf: L.layerGroup(), hmm: L.layerGroup() };
L.control.layers(null, {
  "Raw GPS": layers.gps, "Classifier-only route": layers.clf, "HMM/Viterbi route": layers.hmm,
}, { collapsed: false }).addTo(map);
Object.values(layers).forEach((l) => l.addTo(map));

function status(msg, isErr = false, busy = false) {
  $("status").className = "status" + (isErr ? " err" : "");
  $("status").innerHTML = (busy ? '<span class="spinner"></span>' : "") + msg;
}

const LABELS = {
  top1_accuracy_hmm: ["Top-1 accuracy (HMM)", "pct"],
  top1_accuracy_classifier: ["Top-1 accuracy (classifier only)", "pct"],
  route_continuity_hmm: ["Route continuity (HMM)", "pct"],
  route_continuity_classifier: ["Route continuity (classifier)", "pct"],
  invalid_transitions_hmm: ["Invalid transitions (HMM)", "int"],
  invalid_transitions_classifier: ["Invalid transitions (classifier)", "int"],
  candidate_miss_rate: ["Candidate miss rate", "pct"],
  route_f1_hmm: ["Route arc F1 (HMM)", "pct"],
  route_precision_hmm: ["Route arc precision (HMM)", "pct"],
  route_recall_hmm: ["Route arc recall (HMM)", "pct"],
  labelled_fixes: ["Labelled GPS fixes", "int"],
  unmatched_fixes_hmm: ["Unmatched fixes (HMM)", "int"],
};
const fmt = (v, k) => k === "pct" ? (100 * v).toFixed(2) + " %" : Number(v).toLocaleString();

function renderMetrics(m) {
  $("metrics").innerHTML = Object.entries(LABELS).filter(([k]) => k in m)
    .map(([k, [label, kind]]) => `<tr><td>${label}</td><td>${fmt(m[k], kind)}</td></tr>`).join("");
}
function renderTimings(t) {
  $("timings").innerHTML = Object.entries(t)
    .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${Math.round(v).toLocaleString()}</td></tr>`).join("");
}

async function runTrip(id, btn) {
  document.querySelectorAll(".trips button").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  status(`Running full pipeline for trip ${id} on the server…`, false, true);
  Object.values(layers).forEach((l) => l.clearLayers());
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/api/predict/${id}`, { method: "POST" });
    if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail || r.statusText}`);
    const d = await r.json();
    d.gps.forEach(([la, lo]) => L.circleMarker([la, lo], { radius: 2, color: css("--gps"), weight: 0, fillOpacity: .7 }).addTo(layers.gps));
    L.polyline(d.classifier_route, { color: css("--clf"), weight: 4, opacity: .8 }).addTo(layers.clf);
    const hmm = L.polyline(d.hmm_route, { color: css("--hmm"), weight: 5, opacity: .9 }).addTo(layers.hmm);
    const b = L.latLngBounds(d.gps);
    map.fitBounds(b.isValid() ? b : hmm.getBounds(), { padding: [20, 20] });
    renderMetrics(d.metrics);
    renderTimings({ ...d.timings_ms, round_trip_ms: performance.now() - t0 });
    status(`Trip ${id}: ${d.n_fixes.toLocaleString()} GPS fixes processed.`);
  } catch (e) {
    status(`Request failed – ${e.message}. Is the API running at ${API}?`, true);
  }
}

async function init() {
  try {
    const r = await fetch(`${API}/api/trips`);
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    $("partition").textContent = `(${d.partition}, ${d.trips.length})`;
    d.trips.forEach((id) => {
      const b = document.createElement("button");
      b.textContent = id;
      b.onclick = () => runTrip(id, b);
      $("trips").appendChild(b);
    });
    status("Select a trip to run it through the API.");
  } catch (e) {
    status(`Cannot reach API at ${API} – ${e.message}. Check config.js and that Laptop 1 is running uvicorn.`, true);
  }
}
init();
