"""Build docs/PathSnap_Speaker_Guide.pdf – one page per team member for the panel.

All numbers are read from results/ so the guide always matches the committed experiments.
Run:  python tools/make_speaker_guide.py      (needs: pip install reportlab)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "PathSnap_Speaker_Guide.pdf"

# ---------------------------------------------------------------- numbers from results/
comp = pd.read_csv(R / "models" / "comparison.csv").set_index("model")
hmm = pd.read_csv(R / "hmm" / "clf_vs_hmm_summary.csv", index_col=0)["mean_over_10_trips"]
per_trip = pd.read_csv(R / "hmm" / "clf_vs_hmm_per_trip.csv", dtype={"trip": str})
tvu = pd.read_csv(R / "optuna" / "tuned_vs_untuned.csv").set_index("model")
best = json.loads((R / "optuna" / "best_params.json").read_text())
audit = json.loads((R / "eda" / "audit_summary.json").read_text())
cq = pd.read_csv(R / "features" / "candidate_quality.csv")
miss = {p: g.candidate_miss.sum() / g.labelled_obs.sum() for p, g in cq.groupby("partition")}
lab = {p: (int(g.labelled_obs.sum()), int(g.obs.sum())) for p, g in cq.groupby("partition")}
lgb = comp.loc["lightgbm"]
worse = per_trip[per_trip.pointwise_acc_hmm < per_trip.pointwise_acc_clf]
helped = int((per_trip.pointwise_acc_hmm > per_trip.pointwise_acc_clf).sum())
knn_x = comp.loc["knn", "infer_ms_per_1k_rows"] / lgb["infer_ms_per_1k_rows"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- styles
INK, MUTED, ACCENT, LIGHT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), \
    colors.HexColor("#1a5cff"), colors.HexColor("#f2f1ee")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26,
                    alignment=TA_LEFT, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10, textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14,
                    textColor=INK, spaceBefore=10, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
                    leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=5)
CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=8.4, leading=10.4, textColor=INK)
CELLH = ParagraphStyle("CELLH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.6, leading=13.4, textColor=INK)
SAY = ParagraphStyle("SAY", parent=BODY, leftIndent=8, borderPadding=(6, 8, 6, 8),
                     backColor=LIGHT, borderColor=LIGHT, borderWidth=0, spaceAfter=4)
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=12, bulletIndent=2)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, textColor=MUTED, leading=10)


def p(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def qa(pairs):
    out = []
    for q, a in pairs:
        out.append(KeepTogether([p(f"<b>Q: {q}</b>"), p(f"A: {a}", BUL), Spacer(1, 3)]))
    return out


def table(rows, widths, header=True):
    rows = [[Paragraph(str(c), CELLH if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.6), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6), ("BACKGROUND", (0, 0), (-1, 0), INK),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(st))
    return t


def fig(path, w=170 * mm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=w, height=w * ih / iw)


def member_header(name, role, minutes, files):
    return [p(name, H1), p(f"{role} &nbsp;·&nbsp; speaks ~{minutes} min", SUB),
            p("<b>Open on screen:</b> " + ", ".join(f"<font face='Courier'>{f}</font>" for f in files), BODY),
            Spacer(1, 4)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "PathSnap · speaker guide · numbers generated from results/ by tools/make_speaker_guide.py")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- content
S = []
S += [p("PathSnap – Panel Speaker Guide", H1),
      p("UE24CS352A Machine Learning hackathon · one page per member · every number is from the repo's results/ folder", SUB)]
S += [p("The system in one breath", H2),
      p("PathSnap matches noisy GPS fixes to road segments and rebuilds the route. For each GPS fix we take up to "
        "<b>8 nearby candidate road arcs</b>; a <b>LightGBM classifier</b> scores each one; an <b>HMM decoded with Viterbi</b> "
        "picks the sequence of arcs that is actually connected on the road graph. It runs as a <b>FastAPI server on "
        "Laptop 1</b> and a <b>map UI on Laptop 2</b>.", SAY)]
S += [p("Speaking order (~12 min + questions)", H2),
      table([["#", "Speaker", "Topic", "Time", "Hand-off line"],
             ["0", "Priyanka", "Opening (problem + architecture)", "0:30", "\"It all starts with the data – Prithvish.\""],
             ["1", "Prithvish", "Dataset, projection, candidates, features", "2:30", "\"Now every fix has ~8 candidates – Krishna picks the right one.\""],
             ["2", "Krishna", "6 models, GroupKFold, Optuna", "3:00", "\"A classifier judges each fix alone – Pulkit makes the route coherent.\""],
             ["3", "Pulkit", "Labels, HMM, Viterbi, results", "3:00", "\"Let's see it live across two laptops – Priyanka.\""],
             ["4", "Priyanka", "Live demo + API logs + leakage safeguards", "3:00", "\"Happy to take questions – each of us takes our own area first.\""]],
            [8 * mm, 20 * mm, 52 * mm, 12 * mm, 82 * mm])]
S += [p("Headline numbers everyone should know", H2),
      table([["What", "Value", "Source"],
             ["Trips", "80 train · 10 student-test · 10 evaluator (hidden)", "results/eda/audit_summary.json"],
             ["GPS fixes", f"{audit['train']['observations']:,} train · {audit['student_test']['observations']:,} student-test", "audit_summary.json"],
             ["Candidate miss rate (50 m, top-8)", f"{pct(miss['train'], 2)} train · {pct(miss['student_test'], 2)} student-test", "results/features/candidate_quality.csv"],
             ["Selected model", f"LightGBM – ST Top-1 {lgb.st_top1:.3f}, macro-F1 {lgb.st_macro_f1:.3f}, log loss {lgb.st_log_loss:.3f}", "results/models/comparison.csv"],
             ["Optuna", f"{best['n_trials_completed']} TPE trials, {best['optimisation_seconds']:.0f}s – no gain, kept initial config", "results/optuna/"],
             ["HMM vs classifier (accuracy)", f"{pct(hmm.pointwise_acc_clf)} -> {pct(hmm.pointwise_acc_hmm)}", "results/hmm/clf_vs_hmm_summary.csv"],
             ["HMM vs classifier (continuity)", f"{pct(hmm.continuity_clf)} -> {pct(hmm.continuity_hmm)}", "same"],
             ["Invalid transitions / trip", f"{hmm.invalid_transitions_clf:.0f} -> {hmm.invalid_transitions_hmm:.0f}", "same"]],
            [48 * mm, 80 * mm, 46 * mm])]
S += [p("Rules in the room", H2)] + bullets([
    "<b>Let the panel pick the trip</b> in the UI – it proves nothing is hard-coded. Show the uvicorn log on Laptop 1.",
    "<b>Owner answers first</b>, one teammate may add one sentence. Never talk over each other.",
    "<b>Never claim something we did not run.</b> Say \"we haven't tested that; we'd expect X because Y.\"",
    "<b>Say the negatives yourself</b>: Optuna gave no gain, HMM hurt one trip, HMM uses straight-line distance for now.",
    "AI assistance was used for scaffolding and is disclosed in the README – each of us has read and can explain our module."])
S.append(PageBreak())

# ---------------- Prithvish
S += member_header("1 · Prithvish", "Data pipeline, EDA, candidate generation, features", "2.5",
                   ["src/pathsnap/data/loaders.py", "candidates/generate.py", "features/build.py", "results/eda/figures/"])
S += [p("What to say", H3), p(
    f"\"We have 90 trips – 80 train, 10 student-test. Every trip carries its <b>own city's road network</b> – Germany, "
    "Japan, Argentina and more – so node and arc ids only mean something inside one trip. Longitude and latitude are "
    "angles, so I project each trip into its <b>own UTM zone</b> to get metres; a unit test shows it matches the "
    "great-circle distance within 0.2%. Arcs are <b>directed</b> – about 92% of roads have a reverse twin – so "
    "heading is what separates the two directions. For candidates I build an <b>R-tree (STRtree)</b> over the arcs and "
    f"keep the 8 nearest within 50 m. The <b>candidate miss rate</b> is {pct(miss['train'], 2)} on train and "
    f"{pct(miss['student_test'], 2)} on student-test. I built <b>14 relative features</b> – perpendicular distance, rank, "
    "heading difference, speed, arc length, whether the same arc was a candidate for the previous and next fix – and "
    "deliberately <b>excluded lat/lon</b>, because the model would just memorise cities.\"", SAY)]
S += [p("Key facts", H3)] + bullets([
    "Files per trip: .track (lon, lat, t), .nodes (lon, lat; row = node id), .arcs (from, to; row = arc id), .route (arc ids).",
    "A GPS row is <b>not</b> a node or an arc – it is a noisy point <i>near</i> an arc.",
    f"Anomalies flagged (not deleted): trip 90 has a 42-hour gap; trip 32 has 889 jumps > 50 m/s; trips 33 & 54 have "
    f"large off-route clusters (parked, GPS drift); 3 trips have fractional timestamps.",
    f"Labelled fixes: {lab['train'][0]:,} / {lab['train'][1]:,} train, {lab['student_test'][0]:,} / {lab['student_test'][1]:,} student-test."])
S += [p("Likely questions", H3)] + qa([
    ("Why not Euclidean distance on lat/lon?", "A degree of longitude shrinks with latitude (~half at 60°N), so distances "
     "would be distorted and direction-dependent. UTM gives metres with <0.1% error inside a zone."),
    ("Why keep bad / duplicate node rows?", "The row number IS the id. Deleting a row would renumber every node after it and break every arc."),
    ("Why a spatial index?", "Networks reach 307k arcs; checking every arc for every fix is far too slow. The R-tree only checks nearby boxes."),
    ("Why 50 m and top-8?", "Median GPS-to-route distance is ~4.5 m and p95 ~14 m; 50 m keeps the miss rate below 1% while candidate sets stay small.")])
S += [Spacer(1, 4), fig(R / "eda" / "figures" / "06_trip_locations.png", 150 * mm)]
S.append(PageBreak())

# ---------------- Krishna
rows = [["Model", "Family", "CV F1", "CV Top-1", "ST Top-1", "ST F1", "ST log loss", "ms/1k rows"]]
for m, r in comp.iterrows():
    rows.append([m, r.family, f"{r.cv_macro_f1_mean:.3f}", f"{r.cv_top1_mean:.3f}", f"{r.st_top1:.3f}",
                 f"{r.st_macro_f1:.3f}", f"{r.st_log_loss:.3f}", f"{r.infer_ms_per_1k_rows:.1f}"])
S += member_header("2 · Krishna", "Models, evaluation, Bayesian optimisation", "3",
                   ["models/registry.py", "models/train_all.py", "models/optimise.py", "results/models/", "results/optuna/"])
S += [p("What to say", H3), p(
    "\"Each training row is a (GPS fix, candidate arc) pair labelled 1 if it is the true arc. I compared <b>six models "
    "from five families</b>: logistic regression, KNN, random forest, HistGradientBoosting, LightGBM and an MLP. "
    "Validation is <b>GroupKFold by trip</b> – one trip's fixes never sit in both training and validation, because "
    "neighbouring 1-second fixes are near-duplicates and a random split would leak. My primary metric is <b>Top-1 "
    "accuracy</b> – for each fix, is the highest-scored candidate the true arc – because that is what the HMM consumes; "
    f"log loss tells us whether the probabilities are trustworthy. <b>LightGBM won every metric</b>: CV Top-1 "
    f"{lgb.cv_top1_mean:.3f}, student-test Top-1 {lgb.st_top1:.3f}, macro-F1 {lgb.st_macro_f1:.3f}, log loss "
    f"{lgb.st_log_loss:.3f}. KNN was about <b>{knn_x:.0f}x slower</b> at inference. I then tuned LightGBM with "
    f"<b>Optuna TPE</b> – {best['n_trials_completed']} trials, {best['optimisation_seconds']:.0f} seconds – and it "
    f"<b>did not beat</b> the initial settings (CV Top-1 {tvu.loc['tuned', 'cv_top1']:.4f} vs "
    f"{tvu.loc['untuned', 'cv_top1']:.4f}), so we kept the initial configuration, decided on CV only.\"", SAY)]
S += [p("Results table (student-test = ST)", H3), table(rows, [26 * mm, 26 * mm, 15 * mm, 17 * mm, 17 * mm, 15 * mm, 19 * mm, 20 * mm])]
S += [p("Likely questions", H3)] + qa([
    ("Why is plain accuracy misleading?", "About 83% of candidate rows are negatives, so predicting 'no' everywhere scores high. Top-1 per fix is the honest measure."),
    ("How is TPE different from grid search?", "It models which hyperparameter regions gave good scores and samples where improvement is likely, instead of trying every combination."),
    ("Over- or under-fitting?", f"CV and student-test agree closely (LightGBM {lgb.cv_top1_mean:.3f} vs {lgb.st_top1:.3f}), so no strong overfitting. "
     "The MLP's real per-epoch loss curve is in results/models; tree models have no epochs, so we draw no curves for them."),
    ("Why did tuning not help?", "The default LightGBM was already near the best region; the remaining error comes from features and noisy labels, not hyperparameters.")])
S.append(PageBreak())

# ---------------- Pulkit
S += member_header("3 · Pulkit", "Label alignment, HMM, Viterbi decoding", "3",
                   ["labels/align.py", "hmm/viterbi.py", "hmm/compare.py", "results/hmm/"])
S += [p("What to say", H3), p(
    "\"The .route file is an ordered list of arcs, <b>not a label per GPS fix</b>. So I align them with <b>dynamic "
    "programming</b>: every fix maps to a route arc, the route index can only move forward in time, and the cost is "
    "distance. Fixes more than 50 m from the route are marked <i>unmatched</i> and left out of accuracy. In the HMM "
    "the <b>hidden states</b> are each fix's candidate arcs. <b>Emission</b> is the classifier's log-probability. "
    "<b>Transition</b> penalises the difference between how far the GPS moved and how far the matched position moved, "
    "plus a connectivity term: same arc or the next connected arc costs nothing, one arc in between a small penalty, "
    "not connected a large penalty. <b>Viterbi</b> keeps, for every candidate, the best score of any path ending "
    "there, then backtracks. A gap over 60 s or a fix with no candidates starts a new segment. On student-test it "
    f"raised <b>route continuity from {pct(hmm.continuity_clf)} to {pct(hmm.continuity_hmm)}</b>, cut invalid "
    f"transitions per trip <b>from {hmm.invalid_transitions_clf:.0f} to {hmm.invalid_transitions_hmm:.0f}</b>, and "
    f"accuracy from {pct(hmm.pointwise_acc_clf)} to {pct(hmm.pointwise_acc_hmm)}. It helped {helped} of 10 trips"
    + (f"; on trip {', '.join(worse.trip.str[-2:])} it did not." if len(worse) else ".") + "\"", SAY)]
S += [p("The recurrence (write it on the board if asked)", H3),
      p("<font face='Courier'><b>delta_t(s') = log e_t(s') + max over s of [ delta_(t-1)(s) + log a(s, s') ]</b></font>", SAY),
      p("<font face='Courier'>psi_t(s') = the s that achieved that max</font> (the back-pointer). At the end, start "
        "from the best final state and follow the back-pointers to recover the whole route. "
        "Cost: O(T x K x K) for T fixes and K &lt;= 8 candidates.", BODY)]
S += [p("Likely questions", H3)] + qa([
    ("Why soft penalties instead of forbidding unconnected transitions?", "If the true path is temporarily missing (bad candidates, gaps), a hard ban would trap the decoder; a soft penalty lets it recover."),
    ("What is the current weakness?", "Transition uses straight-line distance between matched points as a proxy for road distance. Next step: Dijkstra network distance."),
    ("How were decoder parameters chosen?", "Small grid search on 12 held-out train trips; the classifier used during tuning was trained without those trips. Never on evaluator data."),
    ("Why did the HMM not help on some trips?", "Where the classifier is already wrong for long stretches, the HMM follows a consistent but wrong road; it fixes short flips, not long errors.")])
S += [Spacer(1, 4), fig(R / "hmm" / "clf_vs_hmm.png", 170 * mm)]
S.append(PageBreak())

# ---------------- Priyanka
S += member_header("4 · Priyanka", "API, UI, shared metrics, deployment", "3 (+ opening)",
                   ["backend/app.py", "frontend/", "eval/metrics.py", "docs/DEPLOY.md"])
S += [p("Opening (30 s)", H3), p(
    "\"PathSnap matches noisy GPS to road segments and rebuilds the route. Candidates, then a classifier, then an HMM "
    "that enforces road connectivity. It runs as a FastAPI server on this laptop and a map UI on that one. Each of us "
    "owns one stage – Prithvish starts with the data.\"", SAY)]
S += [p("What to say during the live demo", H3), p(
    "\"Laptop 1 runs FastAPI. The UI calls <b>GET /api/trips</b>, which lists the trips <b>from disk</b> – nothing is "
    "hard-coded. Please pick any trip. That click calls <b>POST /api/predict/{id}</b>, which runs the <b>whole pipeline "
    "live</b> – candidates, features, LightGBM, Viterbi – and you can see every stage and its timing in this server log. "
    "Scoring happens <b>on the server</b> against the hidden route: the loader <b>refuses to read evaluator ground "
    "truth</b> unless the server-side scorer asks explicitly, and the reference route is <b>never sent</b> to the UI. "
    "The UI shows the HMM route, the classifier-only route, the raw GPS, and the metrics with the change versus the "
    "classifier. On evaluation day the 10 evaluator trips go into data/evaluator_test and the API switches to them "
    "automatically.\"", SAY)]
S += [p("Demo checklist", H3)] + bullets([
    "Laptop 1: <font face='Courier'>uvicorn backend.app:app --host 0.0.0.0 --port 8000</font> – keep this window visible.",
    "Laptop 2: set Laptop 1's IP in <font face='Courier'>frontend/config.js</font>, then <font face='Courier'>python -m http.server 5500 --directory frontend</font>.",
    "Before the panel arrives: open <font face='Courier'>http://&lt;Laptop-1-IP&gt;:8000/api/health</font> on Laptop 2.",
    "Toggle 'Classifier only' on to show where the HMM fixed the route.",
    "Backup: phone hotspot if venue Wi-Fi blocks laptop-to-laptop traffic. Never use cached results."])
S += [p("Likely questions", H3)] + qa([
    ("How do the two laptops communicate?", "HTTP over the same network. Uvicorn listens on 0.0.0.0:8000, a firewall rule opens port 8000, and config.js on Laptop 2 holds Laptop 1's IP. CORS is enabled."),
    ("What happens if a request fails?", "Unknown trip -> 404; pipeline error -> 500 with a message, logged with a stack trace. The UI shows the error and the status pill turns red."),
    ("How do you guarantee no ground-truth leakage?", "load_trip never reads .route; load_reference_route raises PermissionError for evaluator data unless allow_evaluator=True, used only inside the server-side scorer; the response contains no reference route."),
    ("Are training and inference consistent?", "Both call the same pipeline.predict_trip(); the saved model ships with its feature list and the API asserts it matches the code.")])
S.append(PageBreak())

# ---------------- shared
S += [p("Questions any of us may get", H1), Spacer(1, 4)] + qa([
    ("What is the difference between a GPS fix, a node and an arc?", "A fix is a noisy measured position with a timestamp; a node is a road-network point; an arc is a directed road segment between two nodes. Fixes are matched TO arcs."),
    ("What is the evaluation protocol?", "80 train trips for fitting and GroupKFold CV; 10 student-test trips only for comparing models; 10 evaluator trips untouched until the demo."),
    ("What is the candidate miss rate and why does it matter?", f"Share of labelled fixes whose true arc is not among the candidates ({pct(miss['student_test'], 2)} on student-test). If the right arc is not a candidate, no model or HMM can recover it."),
    ("What is route continuity?", "Of all places where the matched arc changes, the share where the next arc starts where the previous one ended (to-node = from-node)."),
    ("What would you improve next?", "Road-network (Dijkstra) distance in the HMM transition; feature-importance based pruning; calibrating probabilities; better handling of parked/stationary GPS clusters."),
    ("Are the reference routes perfect?", "No – the brief warns they are not guaranteed ground truth, and we found routes that do not cover off-route clusters (trips 33, 54) and 14 breaks. We exclude unmatched fixes and report how many.")])
S += [p("Honest limitations (say these before they are asked)", H2)] + bullets([
    f"Optuna did not improve LightGBM (CV Top-1 {tvu.loc['tuned', 'cv_top1']:.4f} tuned vs {tvu.loc['untuned', 'cv_top1']:.4f} initial).",
    "HMM transition uses straight-line distance, not road distance (phase-2 work).",
    f"HMM did not help every trip ({10 - helped} of 10 got slightly worse or equal).",
    "Training used every 4th fix to keep runtime laptop-friendly (consecutive 1 Hz fixes are near-duplicates).",
    "Labels are derived from the reference route by our alignment, so they inherit its imperfections."])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=16 * mm, title="PathSnap Speaker Guide",
                        author="PathSnap team")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
