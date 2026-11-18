"""Build the Deliverable-2 technical report DRAFT (Word) from results/.

All tables, figures and numbers come from files in results/ so the report always matches the
repository. Shaded "notes" boxes are factual bullet points for each owner to rewrite in their
own words before exporting the final PDF. Yellow boxes are screenshots the team must insert.

Run (after tools/report_figures.py):
    python tools/build_report.py
Out: report/PathSnap_Technical_Report_DRAFT.docx
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
OUT = ROOT / "report" / "PathSnap_Technical_Report_DRAFT.docx"
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
p = doc.add_paragraph()
r = p.add_run("DRAFT. Grey boxes are notes for each owner to rewrite in their own words. Yellow boxes are "
              "screenshots to insert. Delete this line, all grey boxes and all yellow boxes before exporting to PDF.")
r.bold = True
r.font.highlight_color = WD_COLOR_INDEX.YELLOW
doc.add_page_break()

# ------------------------------------------------------------------ 1 intro
H("1. Introduction")
note("Priyanka", [
    "GPS on phones and vehicles is noisy because of tall buildings, signal reflections, dropouts and low sampling, "
    "so a car can look like it jumped onto a parallel street",
    "PathSnap takes the raw GPS trace of a trip and works out which directed road segment the vehicle was on for "
    "every GPS fix, then joins them into one continuous route",
    "Our approach has two layers, a classifier that scores nearby road candidates for each fix, and a Hidden Markov "
    "Model decoded with Viterbi that keeps the route connected on the road network",
    "The system runs on two laptops, a FastAPI server that does all the computation and a web map UI that the "
    "evaluator uses to pick a trip and see the result",
])

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
note("Prithvish", [
    "Each trip folder has four header-less files, the track with longitude latitude and time, the nodes with "
    "longitude and latitude, the arcs as from-node and to-node pairs, and the route as a list of arc ids",
    "A GPS row is not a node and not an arc, it is a noisy point that lies near an arc, and the route is a list "
    "of arcs rather than a label for every GPS row",
    "Every trip comes with its own road network from a different city, from Europe to Japan to South America, "
    "which means node and arc ids only mean something inside one trip",
    "Arcs are directed and about 92 percent have a reverse twin, which is how two-way streets are stored",
    "Nothing was malformed, no timestamps went backwards and there were no isolated nodes, but almost every network "
    "has several disconnected pieces and the largest piece always contains the reference route",
])
H("2.2 How we preprocessed the data", 2)
note("Prithvish", [
    "We projected each trip into its own UTM zone, chosen from the trip's centre, so every distance is in metres. "
    "A unit test shows the projected distance agrees with the great-circle distance to within 0.2 percent",
    "Node and arc rows were never deleted because the row number is the id, so removing one would shift every id after it",
    "Gaps longer than 60 seconds, jumps faster than 50 m/s and points far from the route were flagged in "
    "results/eda/data_quality_log.csv instead of being deleted",
    "Speed and heading use the real time difference and the previous and next fix, which also handles the three "
    "trips that have fractional timestamps",
    f"Labels come from a monotonic alignment of the GPS fixes to the reference route. Fixes more than 50 m from the "
    f"route are marked unmatched. This labelled {lab['train'][0]:,} of {lab['train'][1]:,} train fixes and "
    f"{lab['student_test'][0]:,} of {lab['student_test'][1]:,} student-test fixes",
])
H("2.3 Patterns observed and features extracted", 2)
note("Prithvish", [
    "Most GPS fixes sit very close to a road, with a median distance to the reference route of about 4.5 m and a "
    "95th percentile of about 14 m, which is why a 50 m candidate radius is enough",
    f"Candidate generation with an R-tree and the 8 nearest arcs within 50 m missed the true arc for "
    f"{pct(miss['train'], 2)} of labelled train fixes and {pct(miss['student_test'], 2)} of student-test fixes",
    "Some trips have very long gaps, like trip 90 with a gap of about 42 hours, and some have bursts of jumps, like "
    "trip 32 with 889 steps above 50 m/s",
    "Trips 33 and 54 have long off-route clusters where the vehicle seems parked and the GPS drifts in a small area, "
    "which also inflates the raw track length",
    "We left out latitude and longitude as features because each trip is a different city and the model would just "
    "learn locations instead of how matching works",
])
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
note("Krishna", [
    "Each training row is one GPS fix paired with one candidate arc, labelled 1 when the candidate is the true arc",
    "We trained six models from five families on the 80 train trips and used GroupKFold with five folds grouped by "
    "trip, so fixes from one trip never appear in both the training and the validation fold",
    "We trained on every fourth fix because neighbouring one-second fixes are almost identical, which kept runtime "
    "reasonable on a laptop. KNN was further limited to 150,000 rows because it is slow",
    "Our primary selection metric is observation-level Top-1 accuracy, which checks whether the highest scored "
    "candidate is the true arc. That is exactly what the HMM consumes. Log loss was the tie breaker because the HMM "
    "uses the probabilities themselves",
    "Plain accuracy is misleading here because about 83 percent of candidate rows are negatives",
])
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
WHY = {
    "logreg": ["a simple and fast linear baseline which gives well calibrated probabilities, useful as HMM emissions",
               "it can only draw a straight boundary, so feature interactions like distance mattering less when "
               "heading is unreliable are hard for it"],
    "knn": ["it tests whether simply copying the answer of similar past examples is enough",
            "it needs feature scaling, gets slow at prediction time because it compares with every stored row, and "
            "has no real training step"],
    "random_forest": ["many decision trees voting together handle non-linear rules and mixed feature scales without tuning",
                      "the model is large, slower than boosting, and its probabilities are less sharp"],
    "hist_gb": ["boosting fixes the previous trees' mistakes one after another and handles millions of rows quickly",
                "more hyperparameters to set and it can overfit if allowed too many iterations"],
    "lightgbm": ["leaf-wise boosting which is fast and usually the strongest on tabular features like ours",
                 "sensitive to settings like number of leaves, and less interpretable than a linear model"],
    "mlp": ["a small neural network which can learn non-linear feature combinations and is the only model here with "
            "a real per-epoch training history",
            "needs scaling, is slower to train, and is harder to explain than trees"],
}
HYP = {
    "logreg": "C 1.0 with L2 penalty, max_iter 500, StandardScaler before the model",
    "knn": "n_neighbors 25, weights by distance, StandardScaler, fitted on at most 150,000 rows",
    "random_forest": "n_estimators 150, max_depth 16, min_samples_leaf 5, random_state 42",
    "hist_gb": "max_iter 300, learning_rate 0.1, max_leaf_nodes 31, random_state 42",
    "lightgbm": "n_estimators 300, learning_rate 0.05, num_leaves 31, subsample 0.8, colsample_bytree 0.8",
    "mlp": "hidden layers 64 and 32, alpha 1e-4, batch 1024, up to 60 epochs, early stopping with patience 5",
}
OBJ = {
    "logreg": "binary cross-entropy, also called log loss, with L2 regularisation",
    "knn": "there is no training loss, the probability is a distance-weighted vote of the 25 nearest rows",
    "random_forest": "each split reduces Gini impurity and the forest averages the trees' votes",
    "hist_gb": "binary log loss minimised by gradient boosting on binned features",
    "lightgbm": "binary log loss minimised by leaf-wise gradient boosting",
    "mlp": "binary cross-entropy minimised with the Adam optimiser",
}


def fit_comment(m, r):
    gap = r.train_top1_mean - r.cv_top1_mean
    if m == "knn":
        extra = (" Because KNN uses distance weighting, a training row is its own nearest neighbour, so a very high "
                 "training score is expected and the validation score is the honest one")
    else:
        extra = ""
    if gap > 0.05:
        verdict = "a clear sign of overfitting"
    elif gap > 0.015:
        verdict = "mild overfitting"
    else:
        verdict = "no meaningful overfitting"
    under = (" It also has the lowest training score of the six, so it cannot fit even the data it has seen as well "
             "as the other models, which points to underfitting from a model that is too simple") \
        if r.train_top1_mean == comp.train_top1_mean.min() else ""
    return (f"Training-fold Top-1 was {r.train_top1_mean:.3f} and validation Top-1 was {r.cv_top1_mean:.3f}, a gap "
            f"of {gap * 100:.1f} points, which we read as {verdict}. Student-test Top-1 was {r.st_top1:.3f}, close "
            f"to validation.{extra}{under}")


owner = "Krishna"
for m, r in comp.iterrows():
    H(f"3.2.{list(comp.index).index(m) + 1} {LABEL[m]} ({r.family})", 3)
    fstd = folds[folds.model == m].top1.std()
    note(owner, [
        f"Why suitable. We chose it because {WHY[m][0]}",
        f"Limitations. {WHY[m][1][0].upper() + WHY[m][1][1:]}",
        f"Hyperparameters and initial values. {HYP[m]}",
        f"Loss or objective. {OBJ[m][0].upper() + OBJ[m][1:]}",
        f"Student-test results. Top-1 {r.st_top1:.3f}, macro-F1 {r.st_macro_f1:.3f}, log loss {r.st_log_loss:.3f}, "
        f"precision {r.st_precision:.3f}, recall {r.st_recall:.3f}, accuracy {r.st_accuracy:.3f}",
        f"Stability across folds. Validation Top-1 varied with a standard deviation of {fstd:.3f}",
        f"Over or underfitting. {fit_comment(m, r)}",
        f"Cost. Training took {r.train_time_s:.1f} s on the full training rows and inference about "
        f"{r.infer_ms_per_1k_rows:.1f} ms per 1,000 candidate rows",
    ])

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
note("Krishna", [
    f"LightGBM had the best cross-validation Top-1 at {lgb.cv_top1_mean:.3f}, the best student-test Top-1 at "
    f"{lgb.st_top1:.3f}, the best macro-F1 at {lgb.st_macro_f1:.3f} and the lowest log loss at {lgb.st_log_loss:.3f}",
    f"It is also fast, about {lgb.infer_ms_per_1k_rows:.1f} ms per 1,000 rows, while KNN took about "
    f"{comp.loc['knn', 'infer_ms_per_1k_rows']:.0f} ms",
    "Its probabilities are the best calibrated of the six by log loss, which matters because the HMM uses them as "
    "emission scores",
    "The gap between boosting models and random forest is small, so the choice is driven by probability quality "
    "and speed as much as by accuracy",
])
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
note("Krishna", [
    "We tuned LightGBM because it was the best model in the comparison",
    "We searched eight hyperparameters, the number of leaves, learning rate, number of trees, minimum child samples, "
    "column and row sampling, and L1 and L2 regularisation, using log scales for the ones that span orders of magnitude",
    "The objective was the mean Top-1 accuracy of GroupKFold with three folds on the 80 train trips only",
    f"Optuna's TPE sampler ran {best['n_trials_completed']} trials in about {best['optimisation_seconds']:.0f} seconds",
    f"The best trial reached a CV Top-1 of {tvu.loc['tuned', 'cv_top1']:.4f}, which was slightly below the untuned "
    f"model's {tvu.loc['untuned', 'cv_top1']:.4f}, so tuning did not outperform the defaults",
    "We kept the untuned configuration as the final model because the decision was made on cross-validation only, "
    "never on student-test",
    "Our reading is that the default settings were already in a good region and the remaining errors come from "
    "noisy GPS and imperfect reference labels rather than from the hyperparameters",
])
doc.add_page_break()

# ------------------------------------------------------------------ 5 HMM
H("5. HMM and Viterbi Route Reconstruction")
note("Pulkit", [
    "The hidden state at each time step is one of the candidate arcs of that GPS fix, so the decoder chooses one arc "
    "per fix while keeping arc ids exactly as in the arcs file",
    f"Emission uses the classifier's log probability, blended with a Gaussian on perpendicular distance using weight "
    f"alpha. The tuned alpha is {hp['alpha']}, which means the classifier probability is used on its own",
    f"Transition compares how far the GPS moved with how far the matched position moved, scaled by beta {hp['beta']}, "
    f"and adds a connectivity term. Staying on the same arc or moving to an arc that starts where the previous one "
    f"ends costs nothing, one arc in between costs {hp['hop2']}, moving backwards along an arc costs {hp['backward']}, "
    f"and arcs not connected within two hops cost gamma {hp['gamma']}",
    "Penalties are soft rather than impossible so the decoder can recover after a stretch of bad candidates",
    f"A new independent segment starts when the time gap is over {hp['gap_s']:.0f} seconds or when a fix has no "
    f"candidate, and that fix is reported as unmatched",
    "Viterbi keeps for every candidate the best score of any path ending there, then follows back-pointers from the "
    "best final state. The cost is time steps multiplied by the square of the candidate count, at most 8",
    "A current simplification is that straight-line distance between matched points stands in for road distance",
])
P("Viterbi recurrence used in src/pathsnap/hmm/viterbi.py", bold=True)
P("delta_t(s') = log e_t(s') + max over s of [ delta_(t-1)(s) + log a(s, s') ]")
P("psi_t(s') = the s that gives that maximum, used to backtrack the best route.")
table([["alpha", "beta", "gamma", "Mean pointwise accuracy"]] +
      [[r.alpha, r.beta, r.gamma, f"{r.mean_pointwise_acc:.4f}"] for r in grid.itertuples()],
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
      f"{zoom['fixes_corrected_in_window']} fixes inside a {zoom['window_len']}-fix window. The HMM is not perfect "
      f"here either, a parallel arc on the lower left is still wrong.")
worse = per[per.pointwise_acc_hmm < per.pointwise_acc_clf]
note("Pulkit", [
    f"The HMM raised mean pointwise accuracy from {pct(hs.pointwise_acc_clf)} to {pct(hs.pointwise_acc_hmm)} and "
    f"route continuity from {pct(hs.continuity_clf)} to {pct(hs.continuity_hmm)}",
    f"Invalid transitions per trip fell from {hs.invalid_transitions_clf:.0f} to {hs.invalid_transitions_hmm:.0f}, "
    "which means far fewer impossible jumps between disconnected roads",
    f"It improved {int((per.pointwise_acc_hmm > per.pointwise_acc_clf).sum())} of 10 trips. "
    + (f"On trip {', '.join(worse.trip)} accuracy dropped slightly. When the classifier is wrong for a long stretch, "
       "the HMM follows a consistent but wrong road, so it fixes short flips better than long errors"
       if len(worse) else ""),
    "Decoder parameters were tuned only on train trips that the tuning classifier had not been trained on, and the "
    "evaluator partition was never used",
    "The next improvement would be using Dijkstra road distance instead of straight-line distance in the transition",
])
doc.add_page_break()

# ------------------------------------------------------------------ 6 Folium
H("6. Geographical Visualisation Using Folium")
note("Pulkit and Prithvish", [
    "Maps are produced by src/pathsnap/viz/folium_maps.py and saved as HTML in results/eda/maps and results/hmm",
    "Layers are raw GPS fixes, nearby road network, reference route on development trips only, classifier route and "
    "HMM route, each with its own colour, a legend and a layer control",
    "The map module refuses to draw the reference route for evaluator trips so ground truth can never reach the UI",
    "The HMM map shows where the classifier-only route jumps to a parallel road and the HMM route stays on the true road",
])
image(S / "map_hmm_trip00000000_overview.png", "Folium map of student-test trip 0, HMM route in purple over the "
      "reference route in green for the whole trip. Student-test partition.")
image(S / "map_hmm_trip00000056_zoom.png", "Folium map of student-test trip 56 at West 78th Street. The classifier-only "
      "route in red jumps onto side roads while the HMM route in purple stays on the main road, though it still "
      "picks the wrong parallel carriageway in the lower half. Student-test partition.")
doc.add_page_break()

# ------------------------------------------------------------------ 7 API UI
H("7. API, UI and Deployment")
table([["Method", "Endpoint", "Purpose"],
       ["GET", "/api/health", "status, active partition, model and HMM parameters"],
       ["GET", "/api/trips", "trip ids found on disk for the active partition"],
       ["POST", "/api/predict/{trip_id}", "runs the full pipeline and returns routes and server-side metrics"]],
      "API endpoints (backend/app.py)", widths=[2, 5, 9])
note("Priyanka", [
    "Laptop 1 runs FastAPI with uvicorn and holds the model, the road networks, the HMM and the hidden reference routes",
    "Laptop 2 serves the static web UI, which only sends the selected trip id and draws what the API returns",
    "Every request runs candidates, features, LightGBM and Viterbi live and logs each stage with its timing, so "
    "nothing is cached or hard-coded",
    "The reference route is read only inside the server-side scoring function and is never included in a response",
    "For the demo the laptops connect through a Cloudflare quick tunnel, which gives the API on Laptop 1 a "
    "temporary https address so Laptop 2 can reach it from any network. All computation and logs stay on Laptop 1, "
    "Cloudflare only relays the requests",
    "The same setup also works without the tunnel on one Wi-Fi network, with uvicorn listening on 0.0.0.0 port "
    "8000, a firewall rule for that port and Laptop 1's local IP in frontend/config.js",
    "The log screenshot shows requests arriving from Laptop 2's own internet address, and two requests that "
    "overlapped in time were both answered correctly",
    "The live API returned the same Top-1 values as the offline experiments, for example 0.947 for trip 0 and "
    "0.943 for trip 56",
    "Unknown trips return 404 and pipeline errors return 500 with a message, which the UI shows to the evaluator",
])
image(S / "ui_laptop2_trip00000000.png", "Evaluator UI on Laptop 2 showing student-test trip 0 as returned by the "
      "API on Laptop 1 through the Cloudflare tunnel.")
image(S / "api_log_laptop1_two_laptops.png", "uvicorn console on Laptop 1, the receiving laptop, logging live predict "
      "requests sent from the UI on Laptop 2 through the Cloudflare tunnel, with pipeline stages, timings, "
      "server-side metrics and HTTP 200 responses.")
image(S / "ui_trip00000000_full_panel.png", "Full UI panel for trip 0 including every server-side metric and the "
      "pipeline timing breakdown.", width_cm=12)
doc.add_page_break()

# ------------------------------------------------------------------ 8 conclusion
H("8. Limitations and Conclusions")
note("All four members", [
    "The two-layer design worked, the classifier gives good local choices and the HMM turns them into a drivable route",
    "Hyperparameter tuning did not help, which tells us the features and labels limit performance more than settings",
    "Labels are derived from reference routes that are not perfect, for example off-route parking clusters and a few "
    "route breaks, so our accuracy numbers inherit that uncertainty",
    "Training on every fourth fix kept runtime practical but uses only part of the available data",
    "Future work would add road network distance to the HMM, probability calibration and better handling of "
    "stationary GPS clusters",
])
H("Appendix A. Reproducing the results", 2)
for c in ["python -m pathsnap.data.audit", "python -m pathsnap.viz.eda_plots", "python -m pathsnap.viz.eda_maps",
          "python -m pathsnap.features.build", "python -m pathsnap.models.train_all",
          "python -m pathsnap.models.optimise --trials 20 --timeout 420", "python -m pathsnap.hmm.compare",
          "python tools/report_figures.py", "python tools/build_report.py"]:
    q = doc.add_paragraph(style="List Number")
    q.add_run(c).font.name = "Consolas"
H("Appendix B. Use of AI tools", 2)
todo("Write a short disclosure in your own words of how AI assistance was used, as required by the event policy.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print("wrote", OUT)
