"""Stage 2 tests. Run: pytest -q"""
import numpy as np
import pytest

from pathsnap.data.loaders import list_trips, load_reference_route, load_trip
from pathsnap.geo import Projector, angle_diff_deg, bearing_deg, haversine_m


def test_haversine_known_distance():
    # 1 degree of latitude ≈ 111.2 km
    assert abs(haversine_m(0, 0, 0, 1) - 111_195) < 50


def test_projection_matches_haversine():
    p = Projector(8.59, 49.84)
    lon = np.array([8.595203, 8.605203])
    lat = np.array([49.844053, 49.848053])
    x, y = p.to_xy(lon, lat)
    d_proj = np.hypot(x[1] - x[0], y[1] - y[0])
    d_hav = haversine_m(lon[0], lat[0], lon[1], lat[1])
    assert abs(d_proj - d_hav) / d_hav < 0.002  # < 0.2 %


def test_bearing_and_angle_diff():
    assert bearing_deg(0, 1) == pytest.approx(0)
    assert bearing_deg(1, 0) == pytest.approx(90)
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(0, 180) == pytest.approx(180)


def test_partitions_on_disk():
    assert len(list_trips("train")) == 80
    assert len(list_trips("student_test")) == 10


def test_load_trip_shapes_and_ids():
    t = load_trip("00000001", "train")
    assert list(t.gps.columns) == ["lon", "lat", "t"]
    assert t.arcs.dtype == np.int64 and t.arcs.shape[1] == 2
    assert t.arcs.max() < t.n_nodes            # arc endpoints are valid node ids
    assert np.all(np.diff(t.gps.t.to_numpy()) > 0)
    r = load_reference_route("00000001", "train")
    assert r.max() < t.n_arcs                  # route ids are valid arc ids


def test_evaluator_route_is_guarded():
    with pytest.raises(PermissionError):
        load_reference_route("00000015", "evaluator_test")
