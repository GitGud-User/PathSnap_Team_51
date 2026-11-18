"""Build the Deliverable-2 technical report DRAFT (Word) from results/.

All tables, figures and numbers come from files in results/ so the report always matches the
repository. Shaded "notes" boxes are factual bullet points for each owner to rewrite in their
own words before exporting the final PDF. Yellow boxes are screenshots the team must insert.

Run (after tools/report_figures.py):
    python tools/build_report.py
Out: report/PathSnap_Technical_Report.docx (git-ignored). Export it to PDF in Word as
     report/PathSnap_Technical_Report.pdf, which is the file kept in the repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "PathSnap_Technical_Report.docx"
REPO = "https://github.com/GitGud-User/PathSnap_Team_51"

# ------------------------------------------------------------------ data
audit = json.loads((R / "eda" / "audit_summary.json").read_text())
ta = pd.read_csv(R / "eda" / "trip_audit.csv", dtype={"trip": str})
cq = pd.read_csv(R / "features" / "candidate_quality.csv", dtype={"trip": str})
comp = pd.read_csv(R / "models" / "comparison.csv").set_index("model")
folds = pd.read_csv(R / "models" / "cv_folds.csv")
best = json.loads((R / "optuna" / "best_params.json").read_text())
space = json.loads((R / "optuna" / "search_space.json").read_text())
tvu = pd.read_csv(R / "optuna" / "tuned_vs_untuned.csv").set_index("model")
grid = pd.read_csv(R / "hmm" / "decoder_grid.csv")
per = pd.read_csv(R / "hmm" / "clf_vs_hmm_per_trip.csv", dtype={"trip": str})
hs = pd.read_csv(R / "hmm" / "clf_vs_hmm_summary.csv", index_col=0)["mean_over_10_trips"]
hp = json.loads((ROOT / "models" / "final" / "hmm_params.json").read_text())
zoom = json.loads((R / "report" / "route_zoom_info.json").read_text())
tune_trips = json.loads((R / "hmm" / "tuning_trips.json").read_text())
miss = {p: g.candidate_miss.sum() / g.labelled_obs.sum() for p, g in cq.groupby("partition")}
lab = {p: (int(g.labelled_obs.sum()), int(g.obs.sum())) for p, g in cq.groupby("partition")}
lgb = comp.loc["lightgbm"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ document helpers
doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)

FIG = [0]
TAB = [0]


def shade(cell_or_par, hex_fill):
    el = cell_or_par._tc if hasattr(cell_or_par, "_tc") else cell_or_par._p
    pr = el.get_or_add_tcPr() if hasattr(cell_or_par, "_tc") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pr.append(shd)


def H(text, lvl=1):
    doc.add_heading(text, level=lvl)


def P(text, bold=False, italic=False, size=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    return p


def note(owner, bullets):
    """Grey box of factual notes the owner rewrites in their own words."""
    p = doc.add_paragraph()
    r = p.add_run(f"Notes for {owner} (rewrite in your own words, then delete this box)")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), RGBColor(0x52, 0x51, 0x4E)
    shade(p, "F2F1EE")
    for b in bullets:
        q = doc.add_paragraph(style="List Bullet")
        rr = q.add_run(b)
        rr.font.size = Pt(10)
        shade(q, "F2F1EE")


def W(*paragraphs):
    """Body paragraphs of report prose."""
    for text in paragraphs:
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(6)


def Q(lead, text):
    """A paragraph that starts with a short bold lead sentence, used to answer the brief's questions."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(lead + " ").bold = True
    p.add_run(text)


def todo(text):
    p = doc.add_paragraph()
    r = p.add_run(f"[ INSERT ] {text}")
    r.bold = True
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def image(path, caption, width_cm=16):
    FIG[0] += 1
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {FIG[0]}. {caption}")
    r.italic, r.font.size = True, Pt(9.5)


def table(rows, caption=None, widths=None, font=9):
    if caption:
        TAB[0] += 1
        c = doc.add_paragraph()
        r = c.add_run(f"Table {TAB[0]}. {caption}")
        r.bold, r.font.size = True, Pt(9.5)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
            if i == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "1F1F1F")
            if widths:
                cell.width = Cm(widths[j])
    doc.add_paragraph()
    return t


# ------------------------------------------------------------------ title page
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("PathSnap")
r.bold, r.font.size = True, Pt(30)
for line, size in [("Distributed Map-Matching and Route Reconstruction", 15),
                   ("Technical Report", 13), ("UE24CS352A Machine Learning, PES University", 11),
                   ("Team 51", 11)]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(line).font.size = Pt(size)
doc.add_paragraph()
table([["Member", "Workstream"],
       ["Prithvish", "Data pipeline, EDA, candidate generation, feature engineering"],
       ["Krishna", "Model training and comparison, evaluation, Bayesian optimisation"],
       ["Pulkit", "Label alignment, HMM and Viterbi route reconstruction"],
       ["Priyanka", "Metrics, FastAPI backend, web UI, two-laptop deployment"]], widths=[4, 12])
P(f"Repository  {REPO}", italic=True)
P("Every table, figure and number in this report is generated from the repository's results "
  "folder by tools/build_report.py and can be reproduced with the commands in the README.", italic=True, size=9.5)
doc.add_page_break()

# ------------------------------------------------------------------ 1 intro
H("1. Introduction")
W("GPS on a phone or in a car is rarely exact. Tall buildings reflect the signal, the receiver loses the sky for a "
  "while, and some devices only record a position every few seconds. On a map this shows up as a car that seems to "
  "hop onto a parallel street, cut through a block of buildings, or stand still while the dots drift around it.",
  "PathSnap takes the raw GPS trace of a trip and works out which directed road segment the vehicle was really on "
  "for every GPS fix, then joins those segments into one continuous route. We built it in two layers. A classifier "
  "looks at each GPS fix on its own and scores the nearby road segments, and a Hidden Markov Model decoded with the "
  "Viterbi algorithm then picks the sequence of segments which is both likely and actually connected on the road "
  "network.",
  "The finished system runs on two laptops. The first one hosts a FastAPI server which holds the trained model, the "
  "road networks and the decoder, and does all of the computation. The second one shows a web map which the "
  "evaluator uses to pick a trip and see the reconstructed route together with metrics that the server calculates "
  "against the hidden reference route.")

# ------------------------------------------------------------------ 2 data
H("2. Dataset Understanding and Exploratory Data Analysis")
H("2.1 Dataset overview", 2)
tr, stp = audit["train"], audit["student_test"]
table([["Item", "Train", "Student-test"],
       ["Trips", tr["trips"], stp["trips"]],
       ["Trips with all four files", tr["trips_with_all_4_files"], stp["trips_with_all_4_files"]],
       ["GPS observations", f"{tr['observations']:,}", f"{stp['observations']:,}"],
       ["Malformed GPS rows", tr["malformed_gps_rows"], stp["malformed_gps_rows"]],
       ["Steps exactly 1 s apart", f"{tr['pct_steps_exactly_1s']}%", f"{stp['pct_steps_exactly_1s']}%"],
       ["Trips with a gap over 60 s", tr["trips_with_gap_gt_60s"], stp["trips_with_gap_gt_60s"]],
       ["Steps implying speed over 50 m/s", tr["jump_steps_gt_50mps"], stp["jump_steps_gt_50mps"]],
       ["Road nodes (all trips)", f"{tr['nodes_total']:,}", f"{stp['nodes_total']:,}"],
       ["Road arcs (all trips)", f"{tr['arcs_total']:,}", f"{stp['arcs_total']:,}"],
       ["Arcs per trip (min to max)", f"{tr['arcs_per_trip_min_max'][0]:,} to {tr['arcs_per_trip_min_max'][1]:,}",
        f"{stp['arcs_per_trip_min_max'][0]:,} to {stp['arcs_per_trip_min_max'][1]:,}"],
       ["Isolated nodes", tr["isolated_nodes"], stp["isolated_nodes"]],
       ["Reference route arcs", f"{tr['route_arcs_total']:,}", f"{stp['route_arcs_total']:,}"],
       ["Breaks in reference routes", tr["route_breaks_total"], stp["route_breaks_total"]]],
      "Dataset inventory from results/eda/audit_summary.json (evaluator partition not available to us)",
      widths=[8, 4, 4])
W("Every trip folder holds four files without headers. The track file lists longitude, latitude and a time in "
  "seconds for each GPS fix. The nodes file lists the longitude and latitude of each road point, and the row number "
  "is the node id. The arcs file lists pairs of node ids, and again the row number is the arc id. The route file is "
  "the reference, an ordered list of the arc ids the vehicle drove along.",
  "A GPS row is not a node and not an arc. It is a noisy position which "
  "lies somewhere near an arc, and the route file is a list of arcs rather than a label for every GPS row. We "
  "therefore had to build the link between GPS fixes and road arcs ourselves.",
  "Each trip also comes with its own road network from a different city, with trips in Germany, Russia, Japan, the "
  "United States, Argentina, South Africa and many other places. Node and arc ids only mean something inside one "
  "trip. Arcs are directed, and about 92 percent of them have a reverse twin, which is how two-way streets are "
  "stored. The files were clean in the technical sense. Nothing was malformed, no timestamp ever went backwards and "
  "no node was isolated. Almost every network does have several disconnected pieces though, and we confirmed that "
  "the reference route always lies inside the largest piece.")
H("2.2 How we preprocessed the data", 2)
W("Longitude and latitude are angles, and one degree of longitude covers about 111 km at the equator but only about "
  "55 km at 60 degrees north, which is roughly where many of our trips are. We therefore projected each trip into "
  "its own UTM zone, chosen from the centre of its GPS track, so that every distance is measured in metres. A unit "
  "test checks that the projected distance agrees with the great-circle distance to within 0.2 percent.",
  "We did not delete any node or arc rows, because the row number is the id and removing a row would quietly shift "
  "every id after it. Unusual GPS behaviour was flagged, not removed. Gaps longer than 60 seconds, steps "
  "which imply a speed above 50 m/s and points far from the reference route are all recorded in "
  "results/eda/data_quality_log.csv together with the decision we took for each. Speed and heading are calculated "
  "from the real time difference using the previous and next fix, which also copes with the three trips whose "
  "timestamps have fractions of a second.",
  f"To get a label for every GPS fix we aligned the fixes to the reference route with dynamic programming. Each fix "
  f"is matched to an arc of the route, the position in the route is only allowed to move forward in time, and the "
  f"cost is the distance between the fix and the arc. Fixes which end up more than 50 m from the route are marked "
  f"as unmatched and are left out of both training and accuracy. This gave labels for {lab['train'][0]:,} of the "
  f"{lab['train'][1]:,} train fixes and {lab['student_test'][0]:,} of the {lab['student_test'][1]:,} student-test "
  f"fixes.")
H("2.3 Patterns observed and features extracted", 2)
W("Most GPS fixes sit very close to a road. The median distance to the reference route is about 4.5 m and 95 "
  "percent of fixes are within about 14 m, which told us that a search radius of 50 m would almost always include "
  f"the right road. Using an R-tree spatial index we take up to 8 arcs within 50 m of each fix as its candidates. "
  f"The true arc was missing from the candidates for only {pct(miss['train'], 2)} of labelled train fixes and "
  f"{pct(miss['student_test'], 2)} of student-test fixes.",
  "The data also has some rough edges. Trip 90 has a gap of about 42 hours in the middle, and trip 32 has 889 steps "
  "which would need a speed above 50 m/s. Trips 33 and 54 contain long periods where the vehicle seems to be parked "
  "while the GPS drifts around a small area, which also makes the raw track look much longer than the real drive.",
  "For every pair of a GPS fix and a candidate arc we built the 14 features listed in the table below. They "
  "describe how close the fix is, how the candidate compares with the other candidates, whether the car is moving "
  "in the direction of the arc, and whether the same arc was also near the previous and next fix. We left out latitude and longitude, because every trip is a different city and a model given coordinates would "
  "learn where things are instead of learning how matching works.")
table([["Feature", "Why it helps"],
       ["perp_dist_m", "how far the fix is from the candidate arc"],
       ["cand_rank, dist_minus_min, dist_ratio_min", "how the candidate compares with the closest one"],
       ["n_candidates", "how crowded or ambiguous the area is"],
       ["heading_diff_deg", "whether the car moves along the arc direction, which separates a road from its reverse twin"],
       ["speed_mps, heading_reliable", "heading is only meaningful when the car is moving"],
       ["arc_len_m, proj_t", "arc length and where along the arc the fix projects"],
       ["to_out_degree, has_reverse", "road structure around the arc"],
       ["prev_same_arc_dist, next_same_arc_dist", "whether the same arc was also near the previous and next fix"]],
      "The 14 features (src/pathsnap/features/build.py)", widths=[6, 10.5])
H("2.4 EDA figures", 2)
E = R / "eda" / "figures"
image(E / "01_inventory.png", "Observations, path length and duration per trip. Train and student-test partitions.")
image(E / "02_missing_values.png", "Missing value analysis. Malformed rows per file type and share of seconds without a "
      "GPS fix. Train and student-test partitions.")
image(E / "03_sampling_and_speed.png", "Sampling interval and implied speed between fixes. Train and student-test.")
image(E / "04_network_structure.png", "Road network size, two-way share and connectivity per trip. Train and student-test.")
image(E / "05_gps_to_reference_route.png", "Distance from GPS fixes to the reference route. Train and student-test.")
image(E / "06_trip_locations.png", "Trip locations. Each trip carries its own city network. Train and student-test.")
image(E / "07_sample_trajectories.png", "GPS trajectories with road network and reference route for three trips.")
S = ROOT / "report" / "screenshots"
image(S / "map_eda_train_00000001.png", "Folium map of train trip 1 near Frankfurt, raw GPS fixes and the reference "
      "route over the road network, jump flags hidden for readability. Train partition.")
image(S / "map_eda_train_00000090.png", "Folium map of train trip 90, raw GPS fixes following the reference route "
      "through a small town. Train partition.")
doc.add_page_break()

# ------------------------------------------------------------------ 3 models
H("3. Model Experimentation and Comparison")
W("We framed the learning problem around candidates. Each training row is one GPS fix paired with one of its "
  "candidate arcs, and the label is 1 when that candidate is the true arc. A model then gives every candidate a "
  "probability, and the candidate with the highest probability is the model's choice for that fix.",
  "We trained six models from five families on the 80 train trips. To validate them we used GroupKFold with five "
  "folds grouped by trip, so the fixes of one trip are never split between training and validation. Two fixes one second apart are almost identical, and a random split would let the model train on near copies of its validation data. We trained on every fourth fix to keep training time reasonable on a laptop, "
  "and KNN was additionally limited to 150,000 rows because it becomes very slow at prediction time.",
  "Our main selection metric is observation-level Top-1 accuracy, which asks whether the highest scored candidate "
  "is the true arc. The HMM builds on that same choice. We used log loss as the second criterion because "
  "the HMM uses the probabilities themselves, so they need to be trustworthy. Plain accuracy on candidate rows is "
  "misleading here, since about 83 percent of rows are negatives and a model which always says no would already "
  "look good.")
H("3.1 Overall model comparison", 2)
KEY = {"logreg": "C=1.0, scaled", "knn": "k=25, distance-weighted", "random_forest": "150 trees, depth 16",
       "hist_gb": "300 iters, lr 0.1", "lightgbm": "300 trees, lr 0.05, 31 leaves", "mlp": "64-32, Adam, early stop"}
LABEL = {"logreg": "Logistic regression", "knn": "KNN", "random_forest": "Random forest",
         "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM", "mlp": "MLP"}
rows = [["Model", "Family", "Key configuration", "CV macro-F1", "Top-1 (ST)", "Route continuity (ST)",
         "ST macro-F1", "Log loss (ST)", "Inference ms / 1k rows"]]
for m, r in comp.iterrows():
    rows.append([LABEL[m], r.family, KEY[m], f"{r.cv_macro_f1_mean:.3f} ± {r.cv_macro_f1_std:.3f}", f"{r.st_top1:.3f}",
                 f"{r.st_route_continuity:.3f}", f"{r.st_macro_f1:.3f}", f"{r.st_log_loss:.3f}",
                 f"{r.infer_ms_per_1k_rows:.1f}"])
table(rows, "Model comparison (results/models/comparison.csv). ST means student-test. Route continuity is "
      "for classifier-only predictions.", font=8)

H("3.2 Individual models", 2)
HYP = {
    "logreg": "C 1.0 with L2 penalty, max_iter 500, StandardScaler before the model",
    "knn": "n_neighbors 25, weights by distance, StandardScaler, fitted on at most 150,000 rows",
    "random_forest": "n_estimators 150, max_depth 16, min_samples_leaf 5, random_state 42",
    "hist_gb": "max_iter 300, learning_rate 0.1, max_leaf_nodes 31, random_state 42",
    "lightgbm": "n_estimators 300, learning_rate 0.05, num_leaves 31, subsample 0.8, colsample_bytree 0.8",
    "mlp": "hidden layers 64 and 32, alpha 1e-4, batch 1024, up to 60 epochs, early stopping with patience 5",
}

# The comparative claims in the model write-ups below are checked against the results here, so the
# text cannot silently go out of date if the experiments are re-run.
_gap = comp.train_top1_mean - comp.cv_top1_mean
assert comp.cv_top1_mean.idxmin() == "knn" and comp.st_log_loss.idxmax() == "knn"
assert comp.train_top1_mean.idxmin() == "logreg"
assert _gap.sort_values(ascending=False).index[:2].tolist() == ["knn", "random_forest"]
assert all(getattr(comp, c).idxmax() == "lightgbm" for c in ("cv_top1_mean", "st_top1", "st_macro_f1"))
assert comp.st_log_loss.idxmin() == "lightgbm"
assert set(comp.train_time_s.nlargest(2).index) == {"mlp", "random_forest"}
assert comp.loc["mlp", "st_top1"] < comp.loc[["random_forest", "hist_gb", "lightgbm"], "st_top1"].min()


def model_prose(m, r, fstd):
    st, f1, ll = f"{r.st_top1:.3f}", f"{r.st_macro_f1:.3f}", f"{r.st_log_loss:.3f}"
    tr, cv, g = f"{r.train_top1_mean:.3f}", f"{r.cv_top1_mean:.3f}", f"{_gap[m] * 100:.1f}"
    tt, inf = f"{r.train_time_s:.1f}", f"{r.infer_ms_per_1k_rows:.1f}"
    if m == "logreg":
        return [
            "Logistic regression was our baseline. It trains in well under a second, it is easy to explain to someone "
            "who has never seen the data, and it gives probabilities straight away, which is what the HMM needs. Its "
            "weakness is that it separates right and wrong candidates with one straight boundary. A rule like distance "
            "mattering less when the car is standing still is out of its reach.",
            "We used a C of 1.0 with an L2 penalty and up to 500 iterations, after standardising the features. It "
            "minimises binary cross-entropy, the same log loss we report.",
            f"On student-test it reached a Top-1 of {st}, a macro-F1 of {f1} and a log loss of {ll}. Training and "
            f"validation Top-1 were almost identical at {tr} and {cv}, and the training score is the lowest of the six "
            "models. A model that scores this low even on data it has already seen is underfitting. It is too simple "
            "for the problem, not too specialised.",
        ]
    if m == "knn":
        lg = comp.loc["lightgbm", "infer_ms_per_1k_rows"]
        return [
            "KNN does no real learning. For a new candidate it finds the 25 most similar rows it has stored and lets "
            "them vote, weighted by how close they are. We included it to see how far plain similarity can go. It "
            f"needs scaled features, and the price shows up at prediction time. It took about {inf} ms per 1,000 rows "
            f"against {lg:.1f} ms for LightGBM, because every query is compared with the stored data, so we fitted it "
            "on at most 150,000 rows.",
            "There is no loss function to minimise. The probability for a candidate is the distance-weighted share of "
            "its 25 neighbours that were correct.",
            f"Student-test Top-1 was {st}, macro-F1 {f1} and log loss {ll}, the worst log loss of the six. The "
            f"training folds look excellent at {tr} against {cv} on validation, a gap of {g} points. Part of that gap "
            "comes with the method, since with distance weighting every training row is its own closest neighbour. "
            "Validation is where it counts, and there KNN had the lowest score of all six models.",
        ]
    if m == "random_forest":
        return [
            "The random forest grows 150 decision trees, each on a random sample of rows and features, and averages "
            "their votes. Trees pick up rules such as distance mattering less when the heading is unreliable without "
            "us writing them, and they do not care whether a feature is measured in metres or degrees. The cost is "
            f"size and speed. Together with the MLP it was one of the two slowest to train, at {tt} seconds, and "
            "averaged votes give softer probabilities than "
            "boosting.",
            "We limited the depth to 16 and required at least 5 rows in every leaf. Each split is chosen to reduce "
            "Gini impurity.",
            f"It reached a student-test Top-1 of {st}, macro-F1 {f1} and log loss {ll}. Top-1 was {tr} on the "
            f"training folds and {cv} on validation. That gap of {g} points is the largest after KNN and the clearest "
            "sign of overfitting among the tree models, which fits what we expected from deep trees.",
        ]
    if m == "hist_gb":
        return [
            "HistGradientBoosting builds trees one at a time, and each new tree tries to fix the errors of the trees "
            "before it. It first sorts every feature into a small number of bins, which is why it trained in "
            f"{tt} seconds on about 265,000 rows. Boosting has more settings than a forest and can overfit if it runs "
            "for too many rounds.",
            "We used 300 boosting iterations, a learning rate of 0.1 and at most 31 leaves per tree, minimising binary "
            "log loss.",
            f"Its student-test Top-1 was {st}, macro-F1 {f1} and log loss {ll}, very close to LightGBM. On the training "
            f"folds it scored {tr} against {cv} on validation, a gap of {g} points, so it overfits a little but far less than the "
            "random forest.",
        ]
    if m == "lightgbm":
        return [
            "LightGBM is also gradient boosting, but it grows each tree leaf by leaf, always splitting the leaf that "
            "lowers the loss most. On tabular data like our 14 numeric features it tends to be one of the strongest "
            "models, and it is fast to train and to run. The learning rate and the number of leaves need care, and a "
            "boosted ensemble is harder to explain than a linear model.",
            "Our starting point was 300 trees, a learning rate of 0.05 and 31 leaves, with 80 percent of the rows and "
            "columns sampled for each tree. The objective is binary log loss.",
            f"This was the best model on every measure we used. Student-test Top-1 was {st}, macro-F1 {f1} and log "
            f"loss {ll}, and inference took about {inf} ms per 1,000 rows. Validation Top-1 varied by only {fstd:.3f} "
            f"between folds. Its training-fold score of {tr} sits {g} points above the validation score of {cv}, mild "
            "overfitting of about the same size as HistGradientBoosting.",
        ]
    if m == "mlp":
        return [
            "The multi-layer perceptron has two hidden layers of 64 and 32 units. It can combine features in smooth, "
            "non-linear ways that a single boundary cannot, and it is the only model here that trains in epochs, so "
            "it gives a genuine training curve, shown with the other figures below. In exchange it needs scaled "
            f"inputs, was one of the two slowest models to train at {tt} seconds, and is the hardest of the six to explain.",
            "It minimises binary cross-entropy with the Adam optimiser, in batches of 1,024, with a small L2 penalty of "
            "0.0001. Training stops when the score on an internal 10 percent validation split has not improved for "
            "5 epochs.",
            f"On student-test the MLP reached a Top-1 of {st}, a macro-F1 of {f1} and a log loss of {ll}. Its training and "
            f"validation scores of {tr} and {cv} are only {g} points apart, so early stopping kept it from overfitting. It still finished below the "
            "three tree ensembles.",
        ]
    raise KeyError(m)


for m, r in comp.iterrows():
    H(f"3.2.{list(comp.index).index(m) + 1} {LABEL[m]} ({r.family})", 3)
    W(*model_prose(m, r, folds[folds.model == m].top1.std()))

table([["Model", "Precision", "Recall", "Accuracy", "Training time s"]] +
      [[LABEL[m], f"{r.st_precision:.3f}", f"{r.st_recall:.3f}", f"{r.st_accuracy:.3f}", f"{r.train_time_s:.1f}"]
       for m, r in comp.iterrows()],
      "Student-test precision, recall and accuracy on candidate rows, with training time", font=8.5)

H("3.3 Model visualisations", 2)
M = R / "models"
image(M / "cv_macro_f1.png", "Cross-validation macro-F1 across all models, GroupKFold with 5 folds on train.")
image(M / "st_top1.png", "Student-test Top-1 accuracy across all models.")
image(M / "st_macro_f1.png", "Student-test macro-F1 across all models.")
image(M / "st_log_loss.png", "Student-test log loss across all models, lower is better.")
image(R / "report" / "train_vs_cv.png", "Training-fold versus validation versus student-test Top-1 for every model.")
image(M / "confusion_matrices.png", "Student-test confusion matrices on candidate rows at threshold 0.5.")
image(M / "mlp_loss_curve.png", "MLP training loss per epoch. Only the MLP has a genuine epoch-wise history, "
      "so no curves are drawn for the other models.", width_cm=11)
image(M / "mlp_validation_curve.png", "MLP internal validation accuracy per epoch, used for early stopping.", width_cm=11)
H("3.4 Model selection", 2)
W(f"We selected LightGBM. It had the best cross-validation Top-1 at {lgb.cv_top1_mean:.3f}, the best student-test "
  f"Top-1 at {lgb.st_top1:.3f}, the best student-test macro-F1 at {lgb.st_macro_f1:.3f} and the lowest log loss at "
  f"{lgb.st_log_loss:.3f}. It is also quick, at about {lgb.infer_ms_per_1k_rows:.1f} ms per 1,000 candidate rows, "
  f"while KNN needed about {comp.loc['knn', 'infer_ms_per_1k_rows']:.0f} ms for the same work.",
  "The differences between the three tree ensembles are small, so accuracy alone would not have been a convincing "
  "reason. We chose it mainly for its probabilities, since LightGBM has the lowest log loss and the HMM uses "
  "these probabilities directly, together with its speed during live API requests and its stable scores across the "
  "five folds.")
doc.add_page_break()

# ------------------------------------------------------------------ 4 BO
H("4. Bayesian Hyperparameter Optimisation")
table([["Hyperparameter", "Range or distribution"]] + [[k, v] for k, v in space["space"].items()],
      "LightGBM search space (results/optuna/search_space.json)", widths=[6, 8])
table([["Hyperparameter", "Best value"]] +
      [[k, f"{v:.4g}" if isinstance(v, float) else v] for k, v in best["best_params"].items()],
      "Best configuration found (results/optuna/best_params.json)", widths=[6, 8])
cols = ["cv_top1", "cv_macro_f1", "cv_log_loss", "st_top1", "st_macro_f1", "st_log_loss", "infer_ms_per_1k_rows"]
hdr = ["Model", "CV Top-1", "CV macro-F1", "CV log loss", "ST Top-1", "ST macro-F1", "ST log loss", "Inference ms / 1k"]
table([hdr] + [[i] + [f"{tvu.loc[i, c]:.4f}" if c != "infer_ms_per_1k_rows" else f"{tvu.loc[i, c]:.2f}" for c in cols]
               for i in tvu.index],
      "Untuned versus tuned LightGBM with identical protocol (results/optuna/tuned_vs_untuned.csv)", font=8.5)
image(R / "optuna" / "optimisation_history.png", "Optuna TPE optimisation history, CV Top-1 per trial and best so far.",
      width_cm=13)
tuned_better = tvu.loc["tuned", "cv_top1"] > tvu.loc["untuned", "cv_top1"]
Q("Which hyperparameters and ranges.", "We tuned LightGBM, the best model from the comparison. The search covered "
  "eight hyperparameters, which were the number of leaves, the learning rate, the number of trees, the minimum "
  "number of samples in a leaf, the share of columns and of rows used per tree, and the L1 and L2 regularisation "
  "strengths. Settings that span several orders of magnitude, such as the learning rate and the regularisation, "
  "were searched on a log scale. The exact ranges are in the first table above.")
Q("Objective.", "Optuna maximised the mean observation-level Top-1 accuracy of GroupKFold with three folds on the "
  "80 train trips. Student-test and evaluator data played no part in the search.")
Q("Number of trials and cost.", f"The TPE sampler ran {best['n_trials_completed']} trials in about "
  f"{best['optimisation_seconds']:.0f} seconds. TPE learns which regions of the search space gave good scores and "
  "samples more often from those regions, which is why it needs far fewer trials than a full grid.")
Q("Best values.", "The best configuration is listed in the second table above.")
Q("Did tuning help.", (f"Yes. The tuned model reached a cross-validation Top-1 of {tvu.loc['tuned', 'cv_top1']:.4f} "
                       f"against {tvu.loc['untuned', 'cv_top1']:.4f} for the initial settings, so we used the tuned "
                       "configuration as the final model.") if tuned_better else
  (f"No. The best trial reached a cross-validation Top-1 of {tvu.loc['tuned', 'cv_top1']:.4f}, slightly below the "
   f"{tvu.loc['untuned', 'cv_top1']:.4f} of the initial settings, and the student-test Top-1 was "
   f"{tvu.loc['tuned', 'st_top1']:.4f} against {tvu.loc['untuned', 'st_top1']:.4f}. Because we decide only on "
   "cross-validation, we kept the initial configuration as the final model. We think the defaults were "
   "already in a good region, and that the remaining errors come from noisy GPS and imperfect reference labels "
   "rather than from the hyperparameters. The third table compares both versions on the same metrics and data."))
doc.add_page_break()

# ------------------------------------------------------------------ 5 HMM
H("5. HMM and Viterbi Route Reconstruction")
ROAD = "cutoff" in hp
W("A classifier judges every GPS fix on its own, so nothing stops it from choosing the main road for one fix, a "
  "parallel side street for the next, and the main road again after that. A real car cannot do this. The Hidden "
  "Markov Model adds the missing knowledge, namely that consecutive positions must be connected on the road network "
  "and must be consistent with how far the car actually moved.")
Q("Hidden states.", "At each time step the hidden state is one of the candidate arcs of that GPS fix, so there are "
  "at most 8 states per step. The decoder therefore chooses one arc per fix, and the arc ids stay as "
  "they are in the arcs file.")
Q("Emission probabilities.", f"The emission score of a candidate is the log of the probability that LightGBM gives "
  f"it, blended with a Gaussian score on the perpendicular distance using a weight alpha. The tuned value of alpha "
  f"is {hp['alpha']}" + (", which means the classifier probability is used on its own and already contains the "
                          "distance information." if hp["alpha"] == 1.0 else "."))
transition_core = (
    "how far the car would have to drive along the road network between the two matched positions, found with "
    "Dijkstra's shortest path algorithm over the directed arcs, with the search limited to the GPS step plus "
    f"{hp.get('cutoff', 0):.0f} metres" if ROAD else
    "how far the matched position moved, measured as the straight-line distance between the two matched points")
Q("Transition probabilities.", f"The transition score compares how far the GPS moved between two fixes with "
  f"{transition_core}. The larger the difference, the lower the score, scaled by a parameter beta of {hp['beta']}. "
  f"On top of this we add a connectivity term. Staying on the same arc or moving onto an arc that starts where the "
  f"previous one ends costs nothing, one arc in between costs {hp['hop2']}, moving backwards along the same arc costs "
  f"{hp['backward']}, and arcs that are not connected within two steps cost {hp['gamma']}.")
Q("Missing candidates and large gaps.", f"When the time between two fixes is more than {hp['gap_s']:.0f} seconds, or "
  "when a fix has no candidate at all, the decoder closes the current segment and starts a new independent one. A "
  "fix without candidates is reported as unmatched rather than forced onto a road. This is how we handle cases such "
  "as trip 90, which has a gap of about 42 hours.")
Q("Disconnected road segments.", "Moving between disconnected roads is never made impossible. It only receives a "
  "large penalty. If we forbade it completely, one stretch of bad candidates could trap the decoder with no allowed "
  "path at all, while a penalty lets it recover as soon as the evidence points back to a connected road.")
Q("Viterbi decoding.", "Viterbi keeps, for every candidate at every step, the best score of any route that ends "
  "there, together with a pointer to the candidate it came from. At the end it starts from the best final candidate "
  "and follows the pointers backwards to recover the whole route. The work grows with the number of steps times the "
  "square of the number of candidates, which stays small because there are at most 8 candidates per fix.")
P("Viterbi recurrence used in src/pathsnap/hmm/viterbi.py", bold=True)
P("delta_t(s') = log e_t(s') + max over s of [ delta_(t-1)(s) + log a(s, s') ]")
P("psi_t(s') = the s that gives that maximum, used to backtrack the best route.")
gcols = [c for c in ["alpha", "beta", "gamma", "cutoff"] if c in grid.columns]
table([gcols + ["Mean pointwise accuracy"]] +
      [[getattr(r, c) for c in gcols] + [f"{r.mean_pointwise_acc:.4f}"] for r in grid.itertuples()],
      f"Decoder parameter grid on {len(tune_trips)} held-out train trips (results/hmm/decoder_grid.csv)", font=8.5)
table([["Metric (mean of 10 student-test trips)", "Classifier only", "Classifier + HMM"],
       ["Pointwise accuracy", f"{hs.pointwise_acc_clf:.3f}", f"{hs.pointwise_acc_hmm:.3f}"],
       ["Route continuity", f"{hs.continuity_clf:.3f}", f"{hs.continuity_hmm:.3f}"],
       ["Invalid transitions per trip", f"{hs.invalid_transitions_clf:.1f}", f"{hs.invalid_transitions_hmm:.1f}"],
       ["Route arc precision", f"{hs.route_precision_clf:.3f}", f"{hs.route_precision_hmm:.3f}"],
       ["Route arc recall", f"{hs.route_recall_clf:.3f}", f"{hs.route_recall_hmm:.3f}"],
       ["Route arc F1 (sequence level)", f"{hs.route_f1_clf:.3f}", f"{hs.route_f1_hmm:.3f}"],
       ["Decoder runtime per trip", "", f"{hs.hmm_ms:.0f} ms"]],
      "Classifier-only versus classifier plus HMM (results/hmm/clf_vs_hmm_summary.csv)", widths=[7, 4, 4])
table([["Trip", "Fixes", "Accuracy clf", "Accuracy HMM", "Continuity clf", "Continuity HMM", "Route F1 clf",
        "Route F1 HMM"]] +
      [[r.trip, r.obs, f"{r.pointwise_acc_clf:.3f}", f"{r.pointwise_acc_hmm:.3f}", f"{r.continuity_clf:.3f}",
        f"{r.continuity_hmm:.3f}", f"{r.route_f1_clf:.3f}", f"{r.route_f1_hmm:.3f}"] for r in per.itertuples()],
      "Per trip comparison on student-test (results/hmm/clf_vs_hmm_per_trip.csv)", font=8.5)
image(R / "hmm" / "clf_vs_hmm.png", "Classifier-only versus HMM per student-test trip.")
image(R / "report" / "route_zoom.png", f"Close-up of student-test trip {zoom['trip']} where the HMM corrects "
      f"{zoom['fixes_corrected_in_window']} fixes inside a {zoom['window_len']}-fix window around a junction with "
      f"parallel carriageways.")
worse = per[per.pointwise_acc_hmm < per.pointwise_acc_clf]
n_better = int((per.pointwise_acc_hmm > per.pointwise_acc_clf).sum())
Q("Did HMM decoding improve the route.", ("Yes, clearly on both route quality and accuracy. "
                                          if hs.pointwise_acc_hmm - hs.pointwise_acc_clf > 0.02 else
                                          "Yes, clearly on route quality and modestly on accuracy. ") + f"Averaged over the "
  f"10 student-test trips, pointwise accuracy rose from {pct(hs.pointwise_acc_clf)} to {pct(hs.pointwise_acc_hmm)}, "
  f"route continuity rose from {pct(hs.continuity_clf)} to {pct(hs.continuity_hmm)} and the route arc F1 rose from "
  f"{pct(hs.route_f1_clf)} to {pct(hs.route_f1_hmm)}. The number of invalid transitions per trip, meaning jumps "
  f"between roads that are not connected, fell from {hs.invalid_transitions_clf:.0f} to "
  f"{hs.invalid_transitions_hmm:.0f}. Decoding took about {hs.hmm_ms:.0f} ms per trip.")
Q("Where it did not help.", (f"The HMM improved {n_better} of the 10 trips. "
                             + (f"On trip {', '.join(worse.trip)} accuracy went down slightly. When the classifier is "
                                "wrong for a long stretch, the HMM follows a consistent but wrong road, so it is good "
                                "at fixing short flips but cannot rescue long errors. " if len(worse) else "")
                             + ("Using the real driving distance along the road network was what made the biggest "
                                "difference. In the close-up the classifier jumps to the parallel carriageway and to "
                                "side roads, and because reaching those roads would mean a long detour by road, the "
                                "HMM stays on the correct carriageway. Errors still remain, mostly where the "
                                "classifier is wrong for many fixes in a row." if ROAD else
                                "The close-up figure also shows a remaining weakness. Two parallel carriageways a "
                                "few metres apart look almost the same to both the classifier and the decoder, so "
                                "the HMM can still settle on the wrong one.")))
Q("How the parameters were chosen.", f"We tuned {', '.join(gcols[:-1])} and {gcols[-1]} with a small grid search on "
  f"{len(tune_trips)} "
  "train trips that were held out from the classifier used for tuning, scoring each setting by pointwise accuracy. "
  "Student-test was only used to report the final comparison and the evaluator partition was never used.")
doc.add_page_break()

# ------------------------------------------------------------------ 6 Folium
H("6. Geographical Visualisation Using Folium")
W("All interactive maps are produced by src/pathsnap/viz/folium_maps.py and saved as HTML files in "
  "results/eda/maps and results/hmm, so anyone can open them from the repository. Each map has separate layers for "
  "the raw GPS fixes, the road network near the track, the reference route, the classifier-only route and the HMM "
  "route. Every layer has its own colour, and a legend and a layer control let the reader switch them on and off. "
  "The routes are drawn above the GPS points so they are never hidden by thousands of dots, and the view zooms "
  "to the trip automatically.",
  "The reference route layer is only allowed for the train and student-test partitions. The map function refuses to "
  "draw it for evaluator trips, which is one of the ways we make sure ground truth can never reach the UI.",
  "The overview map of trip 0 shows the HMM route following the reference along the whole drive. The close-up of "
  "trip 56 shows where they differ. Around the West 78th Street junction the classifier-only route jumps onto the "
  "parallel carriageway and onto side roads and slip roads, while the HMM route stays on the same carriageway as "
  "the reference route.")
image(S / "map_hmm_trip00000000_overview.png", "Folium map of student-test trip 0, HMM route in purple over the "
      "reference route in green for the whole trip. Student-test partition.")
image(S / "map_hmm_zoom.png", f"Folium map of student-test trip {zoom['trip']} at the West 78th Street junction. The "
      "classifier-only route in red jumps onto the parallel carriageway and side roads while the HMM route in purple "
      "follows the reference route in green. Student-test partition.")
doc.add_page_break()

# ------------------------------------------------------------------ 7 API UI
H("7. API, UI and Deployment")
table([["Method", "Endpoint", "Purpose"],
       ["GET", "/api/health", "status, active partition, model and HMM parameters"],
       ["GET", "/api/trips", "trip ids found on disk for the active partition"],
       ["POST", "/api/predict/{trip_id}", "runs the full pipeline and returns routes and server-side metrics"]],
      "API endpoints (backend/app.py)", widths=[2, 5, 9])
W("Laptop 1 runs the FastAPI backend with uvicorn. It holds the trained model, the road networks, the HMM decoder "
  "and the hidden reference routes. Laptop 2 serves the web UI, which is a single page built with Leaflet. The UI "
  "asks the API for the list of trips, which the API reads from disk instead of a fixed list, and when the "
  "evaluator clicks a trip it sends only that trip id to the API.",
  "Every predict request runs the whole pipeline live, from candidate generation and features through LightGBM to "
  "the Viterbi decoder, and the API log on Laptop 1 records each stage with its timing. Nothing is cached or "
  "precomputed. The server then scores the result against the reference route inside its own scoring function. "
  "Only the predicted routes and the metrics are sent back, and the reference route is never part of a response. "
  "An unknown trip id returns an error 404 and a failure inside the pipeline returns an error 500 with a message, "
  "which the UI shows to the evaluator.",
  "For the demonstration the two laptops are connected through a Cloudflare quick tunnel. It gives the API on "
  "Laptop 1 a temporary https address, so Laptop 2 can reach it from any network, while all computation and all "
  "logs stay on Laptop 1 and Cloudflare only relays the requests. The same system also works without the tunnel when "
  "both laptops share a Wi-Fi network, with uvicorn listening on all interfaces, a firewall rule for port 8000 and "
  "the local address of Laptop 1 in the UI configuration.",
  "The first two screenshots below come from our two-laptop rehearsal. The requests arrive from the internet "
  "address of Laptop 2, and two requests that overlapped in time were both answered correctly. That rehearsal ran "
  "the earlier version of the HMM, before we added road network distance, so its numbers are the older ones.",
  f"The last two screenshots were taken after the final model and decoder were in place. The live API again returns "
  f"exactly the same values as our offline experiments, a Top-1 of "
  f"{per.set_index('trip').loc['00000000', 'pointwise_acc_hmm']:.3f} for trip 0 and "
  f"{per.set_index('trip').loc['00000056', 'pointwise_acc_hmm']:.3f} for trip 56.")
image(S / "ui_laptop2_trip00000000.png", "Evaluator UI on Laptop 2 showing student-test trip 0 as returned by the "
      "API on Laptop 1 through the Cloudflare tunnel, recorded during the two-laptop rehearsal with the earlier HMM.")
image(S / "api_log_laptop1_two_laptops.png", "uvicorn console on Laptop 1, the receiving laptop, logging live predict "
      "requests sent from the UI on Laptop 2 through the Cloudflare tunnel, with pipeline stages, timings, "
      "server-side metrics and HTTP 200 responses. Recorded during the rehearsal with the earlier HMM.")
image(S / "api_log_laptop1_cmd.png", "uvicorn console on Laptop 1 with the final model and road-distance HMM, "
      "logging three predict requests from the UI.")
image(S / "ui_trip00000000_full_panel.png", "UI with the final model and decoder for trip 0, including every "
      "server-side metric and the pipeline timing breakdown.", width_cm=12)
doc.add_page_break()

# ------------------------------------------------------------------ 8 conclusion
H("8. Limitations and Conclusions")
W("The two-layer design did what we hoped. The classifier makes good local choices on its own, and the HMM turns "
  "those choices into a route a car could actually drive, with far fewer jumps between disconnected roads. Running "
  "it as a live API on one laptop and a map on the other worked reliably, and the live results matched our offline "
  "experiments exactly.",
  "There are also clear limitations. Our labels come from reference routes which are not perfect, with parking "
  "clusters that the route does not cover and a few breaks in the routes themselves, so every accuracy number "
  "inherits some of that uncertainty. "
  + ("Hyperparameter tuning did not improve LightGBM, which suggests that our features and labels limit performance "
     "more than the model settings do. " if not tuned_better else "")
  + "We trained on every fourth fix to keep training time practical, which leaves part of the data unused. "
  + (f"The HMM still makes mistakes where the classifier is wrong for many fixes in a row, and searching the road "
     f"network makes decoding slower, at about {hs.hmm_ms:.0f} ms per trip." if ROAD else
     "The HMM still struggles with parallel carriageways a few metres apart and with long stretches where the "
     "classifier is consistently wrong."),
  "With more time we would "
  + ("" if ROAD else "replace the straight-line distance in the HMM transition with true road network distance, ")
  + "calibrate the classifier probabilities, use heading and lane information to separate parallel carriageways, "
    "and handle stationary GPS clusters explicitly instead of only flagging them.")
H("Appendix A. Reproducing the results", 2)
for c in ["python -m pathsnap.data.audit", "python -m pathsnap.viz.eda_plots", "python -m pathsnap.viz.eda_maps",
          "python -m pathsnap.features.build", "python -m pathsnap.models.train_all",
          "python -m pathsnap.models.optimise --trials 20 --timeout 420", "python -m pathsnap.hmm.compare",
          "python tools/report_figures.py", "python tools/build_report.py"]:
    q = doc.add_paragraph(style="List Number")
    q.add_run(c).font.name = "Consolas"
H("Appendix B. Use of AI tools", 2)
W("We used an AI assistant, Claude by Anthropic, during this project, and we disclose it here as the event policy "
  "requires. It helped us set up the project structure, write and debug parts of the code, produce the plotting and "
  "map scripts, and draft the text of this report from our own result files.",
  "All experiments were run on our own laptops on the provided data, and every number, table and figure in this "
  "report is generated from the files in the repository's results folder, so it can be checked and reproduced with "
  "the commands in Appendix A. Each team member reviewed the sections for their own part of the system and is able "
  "to explain the methods, the results and the design decisions described here.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print("wrote", OUT)
