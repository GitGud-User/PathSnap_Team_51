"""Raw-file loaders for PathSnap trips.

Folder layout (verified in Stage 1):
    <DATA_DIR>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}

File formats (no headers, whitespace separated):
    .track  lon lat t          one GPS fix per row, t in seconds (relative, starts ~1.0)
    .nodes  lon lat            row index = node_id (0-based)
    .arcs   from_node to_node  row index = arc_id  (0-based, DIRECTED)
    .route  arc_id             chronological reference arcs (ground truth)

IMPORTANT relationships
  * A .track row is NOT a node and NOT an arc: it is a noisy position that lies *near* an arc.
  * node_id / arc_id are local to one trip – every trip has its own road network.
  * The reference route is an arc sequence, not a per-GPS-point label.

Leakage guard: `load_trip` never reads .route. The route is read through
`load_reference_route`, which refuses the evaluator partition unless the caller is the
server-side evaluator and passes `allow_evaluator=True` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import DATA_DIR, PARTITIONS
from pathsnap.geo import Projector


@dataclass
class ParseReport:
    """How many rows were read / rejected per file. Nothing is dropped silently."""
    rows: dict = field(default_factory=dict)
    bad_rows: dict = field(default_factory=dict)


@dataclass
class Trip:
    trip_id: str
    partition: str
    path: Path
    gps: pd.DataFrame          # columns: lon, lat, t   (index = obs_idx, 0-based)
    nodes: np.ndarray          # (M, 2) float  lon, lat
    arcs: np.ndarray           # (A, 2) int    from_node, to_node
    parse: ParseReport

    @property
    def n_obs(self) -> int:
        return len(self.gps)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    @cached_property
    def projector(self) -> Projector:
        """One UTM projection per trip, chosen from the GPS centroid."""
        return Projector(float(self.gps["lon"].mean()), float(self.gps["lat"].mean()))

    @cached_property
    def gps_xy(self) -> np.ndarray:
        x, y = self.projector.to_xy(self.gps["lon"].to_numpy(), self.gps["lat"].to_numpy())
        return np.column_stack([x, y])

    @cached_property
    def nodes_xy(self) -> np.ndarray:
        x, y = self.projector.to_xy(self.nodes[:, 0], self.nodes[:, 1])
        return np.column_stack([x, y])


# --------------------------------------------------------------------------- helpers

def _read_numeric(path: Path, names: list[str], kind: str) -> tuple[pd.DataFrame, int]:
    """Read a header-less whitespace file; coerce to numbers and count unparseable rows."""
    if not path.exists():
        raise FileNotFoundError(f"missing {kind} file: {path}")
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=names, dtype=float), 0
    raw =pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="c")
    if raw.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, found {raw.shape[1]}")
    raw.columns = names
    df = raw.apply(pd.to_numeric, errors="coerce")
    bad = int(df.isna().any(axis=1).sum())
    return df, bad


def trip_dir(trip_id: str, partition: str, data_dir: Path | None = None) -> Path:
    root = Path(data_dir) if data_dir else DATA_DIR
    return root / PARTITIONS[partition] / trip_id


def list_trips(partition: str, data_dir: Path | None = None) -> list[str]:
    """Trip IDs found on disk for a partition (sorted). Never hard-coded."""
    root = (Path(data_dir) if data_dir else DATA_DIR) / PARTITIONS[partition]
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- public API

def load_trip(trip_id: str, partition: str, data_dir: Path | None = None) -> Trip:
    """Load GPS + road network for one trip. Does NOT read the reference route."""
    d = trip_dir(trip_id, partition, data_dir)
    rep = ParseReport()

    gps, bad = _read_numeric(d / f"{trip_id}.track", ["lon", "lat", "t"], "track")
    rep.rows["track"], rep.bad_rows["track"] = len(gps), bad
    # Malformed GPS rows are reported, then excluded (they cannot be located on a map).
    gps = gps.dropna().reset_index(drop=True)

    nodes, bad = _read_numeric(d / f"{trip_id}.nodes", ["lon", "lat"], "nodes")
    rep.rows["nodes"], rep.bad_rows["nodes"] = len(nodes), bad
    # Nodes are NOT dropped: the row index is the node_id, dropping would shift every id.

    arcs, bad = _read_numeric(d / f"{trip_id}.arcs", ["u", "v"], "arcs")
    rep.rows["arcs"], rep.bad_rows["arcs"] = len(arcs), bad
    # Same for arcs: row index = arc_id. Bad arcs are kept as -1 and flagged by the audit.
    arcs_arr = arcs.fillna(-1).to_numpy(dtype=np.int64)

    return Trip(
        trip_id=trip_id,
        partition=partition,
        path=d,
        gps=gps,
        nodes=nodes.to_numpy(dtype=float),
        arcs=arcs_arr,
        parse=rep,
    )


def load_reference_route(trip_id: str, partition: str, data_dir: Path | None = None,
                         allow_evaluator: bool = False) -> np.ndarray:
    """Reference arc_id sequence. Training labels / evaluation only – never a model input."""
    if partition == "evaluator_test" and not allow_evaluator:
        raise PermissionError(
            "Evaluator ground truth may only be read by the server-side evaluator "
            "(pass allow_evaluator=True there and nowhere else)."
        )
    p = trip_dir(trip_id, partition, data_dir) / f"{trip_id}.route"
    df, bad = _read_numeric(p, ["arc_id"], "route")
    if bad:
        raise ValueError(f"{p.name}: {bad} non-integer route rows")
    return df["arc_id"].to_numpy(dtype=np.int64)
