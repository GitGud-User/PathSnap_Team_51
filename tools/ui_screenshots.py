"""Capture report screenshots of the live UI (Playwright driving the installed Microsoft Edge).

The page calls the running API for real, so every screenshot also produces genuine API log
lines on the server. Nothing is mocked.

Usage:
    python tools/ui_screenshots.py --ui http://127.0.0.1:5502 --api http://127.0.0.1:8001
Out: report/screenshots/ui_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "report" / "screenshots"

SHOTS = [  # (trip, show classifier layer, width, height, file name)
    ("00000000", True, 1600, 1000, "ui_trip00000000_classifier_on.png"),
    ("00000000", False, 1600, 1750, "ui_trip00000000_full_panel.png"),
    ("00000056", True, 1600, 1000, "ui_trip00000056_classifier_on.png"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="http://127.0.0.1:5500")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        for trip, clf, w, h, name in SHOTS:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"{a.ui}/?api={a.api}&trip={trip}&clf={int(clf)}")
            page.wait_for_selector("body[data-done='1']", timeout=120_000)   # API answered
            page.wait_for_load_state("networkidle", timeout=60_000)          # map tiles loaded
            page.wait_for_timeout(1500)                                     # fly-to animation
            page.screenshot(path=str(OUT / name))
            print("wrote", OUT / name)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
