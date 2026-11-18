import numpy as np
import pandas as pd

from pathsnap.eval.metrics import pointwise_accuracy, route_continuity, route_f1
from pathsnap.hmm.viterbi import HMMParams, decode, route_from_fixes

# tiny network: 0 -a0-> 1 -a1-> 2 ; a2 is a parallel arc 3 -> 4 nowhere near connected
ARCS = np.array([[0, 1], [1, 2], [3, 4]])


def test_route_continuity():
    assert route_continuity(np.array([0, 0, 1, 1]), ARCS)["continuity"] == 1.0
    r = route_continuity(np.array([0, 2, 1]), ARCS)
    assert r["invalid_transitions"] == 2


def test_pointwise_and_route_f1():
    assert pointwise_accuracy(np.array([0, 1, 1]), np.array([0, 1, -1])) == 1.0
    assert route_f1(np.array([0, 1]), np.array([0, 1]))["route_f1"] == 1.0


def test_viterbi_prefers_connected_route():
    # 3 fixes moving east; middle fix's classifier slightly prefers the disconnected arc a2
    gps = np.array([[5.0, 0], [15.0, 0], [25.0, 0]])
    rows = [  # obs, rank, arc, dist, t, px, py, prob
        (0, 0, 0, 1.0, 0.5, 5, 0, 0.9),
        (1, 0, 2, 1.0, 0.5, 15, 1, 0.55), (1, 1, 0, 2.0, 1.0, 10, 0, 0.45),
        (2, 0, 1, 1.0, 0.5, 25, 0, 0.9),
    ]
    tab = pd.DataFrame(rows, columns=["obs_idx", "cand_rank", "arc_id", "perp_dist_m", "proj_t",
                                      "proj_x", "proj_y", "p"])
    out, _ = decode(tab, tab.p.to_numpy(), gps, np.array([0.0, 1, 2]), ARCS, 5, HMMParams())
    assert list(out) == [0, 0, 1]
    assert list(route_from_fixes(out)) == [0, 1]


def test_gap_and_no_candidate_are_unmatched():
    gps = np.array([[0.0, 0], [10.0, 0]])
    tab = pd.DataFrame([(0, 0, 0, 1.0, 0.5, 0, 0, 0.9)],
                       columns=["obs_idx", "cand_rank", "arc_id", "perp_dist_m", "proj_t",
                                "proj_x", "proj_y", "p"])
    out, _ = decode(tab, tab.p.to_numpy(), gps, np.array([0.0, 1]), ARCS, 5, HMMParams())
    assert list(out) == [0, -1]
