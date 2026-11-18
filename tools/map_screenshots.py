"""Screenshot the generated Folium HMM maps for the report (Playwright + installed Edge).

Usage (serve the results folder first, e.g. python -m http.server 5603 --directory results):
    python tools/map_screenshots.py --base http://127.0.0.1:5603
Out: report/screenshots/map_hmm_trip00000000_overview.png, map_hmm_zoom.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "screenshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5603")
    a = ap.parse_args()
    z = json.loads((ROOT / "results" / "report" / "route_zoom_info.json").read_text())
    shots = [("hmm/map_student_test_00000000.html", "map_hmm_trip00000000_overview.png"),
             (f"hmm/map_student_test_{z['trip']}_zoom.html", "map_hmm_zoom.png")]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        for page_path, name in shots:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"{a.base}/{page_path}")
            page.wait_for_load_state("networkidle", timeout=90_000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT / name))
            print("wrote", OUT / name)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
