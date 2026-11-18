// PathSnap evaluator UI. Everything shown comes from the live API response.
// Optional URL parameters: ?api=<base url> overrides config.js, ?trip=<id> runs that trip on load,
// ?clf=1 also shows the classifier-only layer. Handy for demos and reproducible screenshots.
// Empty window.PATHSNAP_API means "same origin" (the API also serves this page).
const QS = new URLSearchParams(location.search);
const API = (QS.get("api") || window.PATHSNAP_API || "").replace(/\/+$/, "");
const API_LABEL = API || location.origin;  // for status messages only
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const map = L.map("map", { zoomControl: false, preferCanvas: true }).setView([30, 10], 2);
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
const layers = { gps: L.layerGroup().addTo(map), clf: L.layerGroup(), hmm: L.layerGroup().addTo(map) };

document.querySelectorAll("[data-layer]").forEach((cb) => {
  cb.addEventListener("change", () => {
    const l = layers[cb.dataset.layer];
    cb.checked ? l.addTo(map) : map.removeLayer(l);
    if (layers.hmm._map) layers.hmm.eachLayer((x) => x.bringToFront && x.bringToFront());
  });
});

function status(msg, { err = false, busy = false } = {}) {
  const s = $("status");
  s.className = "status" + (err ? " err" : "");
  s.innerHTML = (busy ? '<span class="spinner"></span>' : "") + `<span>${msg}</span>`;
}
function conn(ok, text) {
  $("conn").className = "pill " + (ok ? "on" : "off");
  $("conn").querySelector("span").textContent = text;
}

const pct = (v) => (100 * v).toFixed(1) + "%";
const num = (v) => Number(v).toLocaleString();
const ROWS = [
  ["top1_accuracy_classifier", "Top-1 accuracy · classifier", pct],
  ["route_continuity_classifier", "Route continuity · classifier", pct],
  ["invalid_transitions_hmm", "Invalid transitions · HMM", num],
  ["invalid_transitions_classifier", "Invalid transitions · classifier", num],
  ["route_precision_hmm", "Route arc precision · HMM", pct],
  ["route_recall_hmm", "Route arc recall · HMM", pct],
  ["candidate_miss_rate", "Candidate miss rate", (v) => (100 * v).toFixed(2) + "%"],
  ["labelled_fixes", "Labelled GPS fixes", num],
  ["unmatched_fixes_hmm", "Unmatched fixes · HMM", num],
];

function delta(a, b) {
  const d = 100 * (a - b);
  const cls = d > 0.05 ? "up" : d < -0.05 ? "down" : "";
  return `<div class="d ${cls}">${d >= 0 ? "+" : ""}${d.toFixed(1)} pts vs classifier</div>`;
}

function renderHero(m) {
  const kept = m.invalid_transitions_classifier - m.invalid_transitions_hmm;
  $("hero").innerHTML = `
    <div class="kpi"><div class="k">Top-1 accuracy</div><div class="v">${pct(m.top1_accuracy_hmm)}</div>${delta(m.top1_accuracy_hmm, m.top1_accuracy_classifier)}</div>
    <div class="kpi"><div class="k">Route continuity</div><div class="v">${pct(m.route_continuity_hmm)}</div>${delta(m.route_continuity_hmm, m.route_continuity_classifier)}</div>
    <div class="kpi"><div class="k">Route arc F1</div><div class="v">${pct(m.route_f1_hmm)}</div><div class="d">vs reference route</div></div>
    <div class="kpi"><div class="k">Broken links removed</div><div class="v">${num(kept)}</div><div class="d">${num(m.invalid_transitions_classifier)} → ${num(m.invalid_transitions_hmm)}</div></div>`;
}

function renderRows(id, entries) {
  $(id).innerHTML = entries.map(([k, v]) => `<div class="row"><span>${k}</span><span>${v}</span></div>`).join("");
}

function renderTimings(t) {
  const parts = [["candidates_features_ms", "#d9d9d9"], ["classifier_ms", "#ffb020"], ["hmm_ms", "#3d8bff"]];
  const total = parts.reduce((s, [k]) => s + (t[k] || 0), 0) || 1;
  const bar = `<div class="bar">${parts.map(([k, c]) => `<i style="width:${100 * (t[k] || 0) / total}%;background:${c}"></i>`).join("")}</div>`;
  const label = { candidates_features_ms: "Candidates + features", classifier_ms: "Classifier", hmm_ms: "HMM / Viterbi",
                  total_ms: "Server total", round_trip_ms: "Round trip (network incl.)" };
  $("timings").innerHTML = `<div style="padding-top:10px">${bar}</div>` +
    Object.entries(t).map(([k, v]) => `<div class="row"><span>${label[k] || k}</span><span>${Math.round(v).toLocaleString()} ms</span></div>`).join("");
}

async function runTrip(id, btn) {
  document.querySelectorAll(".trip").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  status(`Running the full pipeline for trip ${id}…`, { busy: true });
  Object.values(layers).forEach((l) => l.clearLayers());
  $("tripcard").classList.remove("show");
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/api/predict/${id}`, { method: "POST" });
    if (!r.ok) throw new Error(`${r.status} – ${(await r.json().catch(() => ({}))).detail || r.statusText}`);
    const d = await r.json();
    $("placeholder").style.display = "none";
    d.gps.forEach(([la, lo]) => L.circleMarker([la, lo], { radius: 2, color: css("--gps"), weight: 0, fillOpacity: .55 }).addTo(layers.gps));
    L.polyline(d.classifier_route, { color: css("--clf"), weight: 3, opacity: .9, dashArray: "6 6" }).addTo(layers.clf);
    L.polyline(d.hmm_route, { color: css("--hmm"), weight: 10, opacity: .18 }).addTo(layers.hmm);   // glow
    const hmm = L.polyline(d.hmm_route, { color: css("--hmm"), weight: 4.5, opacity: 1 }).addTo(layers.hmm);
    if (d.gps.length) {
      const s = d.gps[0], e = d.gps[d.gps.length - 1];
      L.circleMarker(s, { radius: 7, color: "#000", weight: 3, fillColor: "#fff", fillOpacity: 1 }).bindTooltip("Start").addTo(layers.hmm);
      L.circleMarker(e, { radius: 7, color: "#fff", weight: 3, fillColor: "#000", fillOpacity: 1 }).bindTooltip("End").addTo(layers.hmm);
    }
    const b = L.latLngBounds(d.gps);
    map.flyToBounds(b.isValid() ? b : hmm.getBounds(), { padding: [60, 60], duration: 0.8 });

    const m = d.metrics;
    renderHero(m);
    renderRows("metrics", ROWS.filter(([k]) => k in m).map(([k, l, f]) => [l, f(m[k])]));
    renderTimings({ ...d.timings_ms, round_trip_ms: performance.now() - t0 });
    $("tc-id").textContent = `Trip ${id}`;
    $("tc-meta").textContent = `${num(d.n_fixes)} GPS fixes · ${num(d.hmm_route.length)} road arcs matched`;
    $("tc-chips").innerHTML = `<span class="chip">Top-1 ${pct(m.top1_accuracy_hmm)}</span>
      <span class="chip">Continuity ${pct(m.route_continuity_hmm)}</span>
      <span class="chip">${Math.round(d.timings_ms.total_ms).toLocaleString()} ms</span>`;
    $("tripcard").classList.add("show");
    status(`Trip ${id} done – ${num(d.n_fixes)} fixes matched by the live API.`);
  } catch (e) {
    status(`Request failed: ${e.message}. Is the API running at ${API_LABEL}?`, { err: true });
  }
}

async function init() {
  try {
    const r = await fetch(`${API}/api/trips`);
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    conn(true, "API live");
    $("partition").textContent = `· ${d.trips.length} ${d.partition.replace("_", "-")}`;
    d.trips.forEach((id, i) => {
      const b = document.createElement("button");
      b.className = "trip";
      b.innerHTML = `<b>${id}</b><span>#${i + 1}</span>`;
      b.onclick = () => runTrip(id, b);
      $("trips").appendChild(b);
    });
    status("Select any trip – each click runs the full pipeline on the server.");
    if (QS.get("clf") === "1") document.querySelector('[data-layer="clf"]').click();
    const auto = QS.get("trip");
    if (auto) {
      const btn = [...document.querySelectorAll(".trip")].find((x) => x.querySelector("b").textContent === auto);
      if (btn) await runTrip(auto, btn);
      document.body.dataset.done = "1";
    }
  } catch (e) {
    conn(false, "Offline");
    status(`Cannot reach the API at ${API_LABEL} (${e.message}). Check that uvicorn is running and config.js points at it.`, { err: true });
  }
}
init();
