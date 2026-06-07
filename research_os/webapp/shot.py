#!/usr/bin/env python3
"""Headless screenshot helper for verifying the web app renders.

Drives the cached Playwright chromium against the running backend. Explicit waits
(wait_until='commit' + fixed delay) avoid the networkidle trap — the app holds a
long-lived SSE connection that never goes idle.

Usage: shot.py <path-after-host> <out.png> [wait_ms] [full]
       shot.py / /tmp/ro_shots/overview.png 3500
"""
import glob
import sys
from playwright.sync_api import sync_playwright

CHROME = sorted(glob.glob("/home/girish/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"))[-1]
BASE = "http://127.0.0.1:8138"

path = sys.argv[1] if len(sys.argv) > 1 else "/"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ro_shots/shot.png"
wait_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 3500
full = len(sys.argv) > 4 and sys.argv[4] == "full"

with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=CHROME,
        headless=True,
        args=[
            "--no-sandbox", "--disable-gpu", "--disable-software-rasterizer",
            "--disable-background-networking", "--disable-dev-shm-usage",
        ],
    )
    page = b.new_page(viewport={"width": 1600, "height": 1040}, device_scale_factor=2)
    page.goto(BASE + path, wait_until="commit", timeout=20000)
    page.wait_for_timeout(wait_ms)
    page.screenshot(path=out, full_page=full)
    b.close()
    print(f"shot: {out}")
