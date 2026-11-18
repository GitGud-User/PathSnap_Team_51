# PathSnap – Stage 1 Dataset Audit (preliminary)

**Partition used:** `train` (80 trips) + `student_test` (10 trips). The evaluator partition is **not** in the archive.
**How it was produced:** `tools/audit/Audit.cs`, a throwaway C# script. This laptop has no Python yet, so the official, reproducible audit still has to be rewritten in Python (`src/pathsnap/data/audit.py`, owned by Prithvish) and must reproduce these numbers.
**Raw output:** [`docs/audit/trip_audit.csv`](audit/trip_audit.csv) has one row per trip and 48 columns.

> The `.route` files were read **only to audit them** (are they valid, do they line up with the GPS). Nothing here feeds a prediction.

---

## 1. Inventory

| Item | Documentation says | Archive actually has |
|---|---|---|
| Training trips | 80 | **80** ✅ |
| Student-test trips | 10 | **10** ✅ (`00000000, 07, 09, 40, 52, 56, 65, 76, 77, 91`) |
| Evaluator trips | 10 (held back) | **not included**. The trip IDs missing from 0–99 are `15, 17, 23, 30, 49, 59, 67, 73, 87, 94`, which are presumably the evaluator trips (this is our inference) |
| Files per trip | `.track .nodes .arcs .route` | All 4 present in all 90 trips, and none are empty ✅ |
| Layout | – | `Student folder/{train,student_test}/<8-digit id>/<id>.<ext>` |
| Total size | – | 89.7 MB |
| GPS observations | – | 208,417 (train) + 19,160 (student_test) = **227,577** |
| GPS path length | – | ~2,978 km (train) + ~297 km (student_test) |

## 2. File-format checks

| Check | Result |
|---|---|
| Malformed / non-numeric rows (all 4 file types) | **0** |
| `.track` columns | lon, lat, t. The order is confirmed: values like `8.595 49.844` sit in Germany, which is plausible. Timestamps are relative seconds starting at 1.0, not Unix epoch |
| Non-increasing timestamps | **0** in every trip |
| Steps exactly 1 s apart | 96.7 % overall, which matches the documented "≈97 %". The per-trip minimum is **53 %** |
| Arc endpoints that point to missing nodes | **0** |
| Route arc IDs outside the arc table | **0** |
| Isolated nodes (degree 0) | **0** |
| Self-loop arcs | 24 in total (max 8 in one trip) |
| Duplicate (from,to) arcs | 1,903 in total (median 3 per trip) |
| Zero-length arcs (< 1 cm) | 106 in total |
| Exactly repeated consecutive GPS points | 6 in total |

## 3. Key structural findings

1. **Each trip ships its own road network, and the trips are spread around the world.** Locations run from Buenos Aires (−34.6°, −58.5°) to Japan (36.7°, 137.2°), with Moscow, Alabama, Philippines, South Africa and more in between. Node and arc IDs are **local to their trip**: arc `5` in trip 1 and arc `5` in trip 2 are unrelated.
   → Raw lon/lat must **never** be a model feature (it would just learn "which city"). Every feature must be relative: distances, angles, ranks, speeds.
   → Distances must come from a per-trip projection (a local UTM zone, chosen automatically from the trip centroid). Haversine is fine for point-to-point distances.
2. **Network size varies by about 500×.** Trips range from 299 nodes / 649 arcs up to 155k nodes / **307k arcs** (trip 13). Brute-force candidate search is therefore not viable; we need a spatial index (STRtree / R-tree on arc segments).
3. **Arcs are directed.** About 92 % of arcs have a reverse twin, i.e. two-way streets are stored as two arcs. Trip 80 (35 %) and trip 35 (67 %) have many one-way streets. So the label is a *directed* arc, and **heading versus arc bearing is the only thing that separates an arc from its reverse twin.**
4. **The networks are not fully connected.** 89 of 90 trips have more than one weakly connected component (median 19, max 421). The largest component holds about 97 % of nodes on average, but only 69 % in the worst trip. The HMM has to cope with "no path exists" between candidates.
5. **Arc lengths:** median ~26 m, p95 ~126 m, max 9.3 km. Arcs are short relative to 1 Hz GPS, so a car often stays on one arc for many observations, and consecutive fixes may also skip over very short arcs.

## 4. GPS quality findings

| Finding | Evidence | Planned treatment (not applied yet) |
|---|---|---|
| Large time gaps | 23 trips have a gap > 60 s. Six trips have a gap > 10 min: 11, 21, 42, 69, 78 (**14.7 h**) and 90 (**42.5 h**) | Split the trip into independent HMM segments when Δt or distance exceeds a threshold. Keep every point and drop nothing silently |
| Irregular sampling | Some trips are only 53 % 1 Hz; trip 98 has 1,268 non-1 s steps | Base speed and heading features on the true Δt, never assume 1 s |
| Position jumps (> 50 m/s implied speed) | 1,532 steps in total, concentrated in trip 32 (889), trip 1 (340) and trip 13 (112, max 141 m/s) | Flag them as outliers. Keep them in evaluation; down-weight them in emission/transition rather than deleting |
| Noise level | Median GPS-to-reference-route distance is **~4.5 m**, p95 ~14 m (typical trip) | Suggests a candidate radius of about 50 m. This will be tuned with Candidate Miss Rate |

## 5. Reference-route findings

| Finding | Evidence | Consequence |
|---|---|---|
| Routes are arc sequences, **not per-GPS labels** | 227,577 GPS points in total versus 34,813 route arcs (median 277 per trip), so many points map to the same arc | We must **derive** a label for each observation by aligning the GPS sequence to the route arcs in order (monotonic DP alignment). This is the biggest hidden task in the project |
| Routes are mostly continuous | `to(arc_i) == from(arc_i+1)` holds everywhere except **14 breaks in 11 trips** (2, 19, 33, 41, 42, 69, 75, 80, 86, 89, student-test 76) | Treat the breaks as allowed gaps in the alignment and document them |
| Large **off-route clusters** (corrected in Stage 2: these are *not* routes that end early) | Trip 54: 16,849 of 19,313 fixes are > 50 m from the route, all inside a ~1 × 1 km patch. Trip 33: 3,445 of 5,844 inside a ~0.35 × 1 km patch. The pattern fits a long stop or off-network area (car park) with GPS drift. The drift also inflates their "GPS path length" (169 km and 48 km). Trips 2 and 84 have smaller clusters | These fixes get the label **"unmatched / no label"**. They are left out of training labels and accuracy denominators, and we report the count. In Stage 3 we will also flag them as *stationary* (low displacement over a time window) |
| No consecutive duplicate arcs in routes | 0 | – |

## 5b. Cross-check with Pulkit's independent analysis

Pulkit audited the same 90 trips independently. Every claim we hadn't measured was re-run with `python -m pathsnap.data.extra_checks`, which writes `results/eda/extra_checks.csv`.

| Claim | Our measurement | Verdict |
|---|---|---|
| 3 trips have fractional-second timestamps (2, 71, 97) | same 3 trips | ✅ new finding, now recorded. Speed must use the true Δt |
| The route always lies in the largest weak component | true for all 90 | ✅ new and useful: the HMM's shortest-path search can live inside that component |
| 92 duplicate node coordinates | 92 | ✅ |
| Arcs under 1 m: 4,976 | 4,988 (UTM projection; difference comes from the distance method) | ✅ |
| Zero-length arcs: 177 | **106** exactly zero (matches our earlier count) | ⚠️ different threshold; we use 106 = exactly 0 m |
| 10 trips revisit route arcs (loops) | 10 | ✅ |
| 13 route breaks across 10 trips | **14 across 11** (train 13 + student_test `00000076` 1), confirmed by both the C# and Python audits | ⚠️ the 14th break is in student-test trip 76 |
| Arcs within 20 m: median 2–8; within 50 m: 4–32 | median of trip medians 4 (range 0–8) and 8 (range 0–32). On average 3.2 % of fixes have **no** arc within 20 m | ✅ supports a candidate radius of 30–50 m |
| Route start/end within ~10–200 m of the track | median 33 m (start) / 27 m (end), **max 457 / 548 m** | ⚠️ wider than stated |
| Trip 54: parked about 175 m from the route end for the last ~4.3 h | far fixes run from the 2nd decile of the trip to the end, inside a ~1 km patch | ✅ consistent with our "off-route cluster" |
| Trip 33: the route ends ~30 % in and the vehicle continues elsewhere | % of far fixes per time decile: `0, 41, 28, 44, 100, 100, 100, 100, 77, 0`. The GPS **returns to the route in the last decile** | ❌ The route doesn't end early. The vehicle leaves the route mid-trip (deciles 2–9, a ~0.35 × 1 km area) and comes back |
| GPS noise inflates track length (trip 2: 56 km over 5 km displacement) | agrees with our gps_path_m vs route_len_m | ✅ never use track length as trip length |

## 6. Decisions still open (team to confirm)

- Choose the candidate radius and top-K from the miss-rate curve on **train only**.
- Choose the unmatched-label threshold (proposed: point-to-route distance > 50 m, or no feasible alignment).
- Choose the HMM segment-break threshold (proposed: Δt > 60 s or jump > 500 m).
- Decide whether trip 54 is kept in training. Proposal: keep only its labelled points and report this.
