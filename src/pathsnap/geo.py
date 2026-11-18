"""Geographic helpers.

Why projection?  Longitude/latitude are angles. One degree of longitude is ~111 km at the
equator but only ~55 km at 60°N (our dataset spans -34° to +62° latitude), so Euclidean
distance in degrees is wrong by up to 2x and direction-dependent.

We project every trip into its own UTM zone (metres, conformal, <0.1 % scale error inside a
zone). Each trip covers at most a few thousand km², so a single zone per trip is adequate.
Haversine is kept for quick point-to-point checks and for validating the projection.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def utm_crs_for(lon: float, lat: float) -> CRS:
    """UTM CRS (WGS84) for the zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


class Projector:
    """Converts lon/lat <-> metric x/y for one trip."""

    def __init__(self, lon0: float, lat0: float):
        self.crs = utm_crs_for(lon0, lat0)
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a vector given in projected x(east)/y(north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d
