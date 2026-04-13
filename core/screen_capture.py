"""
core/screen_capture.py
----------------------
Native Windows desktop screen capture module for Aria.

Captures the full desktop compositor output every N seconds using the
mss library, which reads directly from the Windows GDI display layer.
This is identical to how OBS captures the screen — no process injection,
no memory reading, no interaction with any running application.

Screenshots are saved to data/captures/ as a rolling buffer.
The latest screenshot is always available at data/captures/latest.png
for downstream analysis modules (Stage 2: Gemini vision).

Usage:
    capture = ScreenCapture()
    capture.start()   # begins background capture thread
    capture.stop()    # stops cleanly
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import mss
import mss.tools

# ── Configuration ─────────────────────────────────────────────────────────────
CAPTURE_DIR     = Path("data/captures")
LATEST_PATH     = CAPTURE_DIR / "latest.png"
BUFFER_SIZE     = 10        # Maximum number of screenshots to keep on disk
CAPTURE_INTERVAL = 5.0      # Seconds between captures


class ScreenCapture:
    """Manages periodic full-desktop screenshot capture.

    Runs in a background daemon thread. Saves screenshots to a rolling
    buffer in data/captures/. The latest frame is always available at
    data/captures/latest.png for downstream analysis.

    Attributes:
        running: True if the capture loop is currently active.
        frame_count: Total number of frames captured this session.
    """

    def __init__(self, interval: float = CAPTURE_INTERVAL):
        """Initialise the screen capture module.

        Args:
            interval: Seconds between each screenshot. Default 5.0.
        """
        self.interval   = interval
        self.running    = False
        self.frame_count = 0
        self._thread: threading.Thread | None = None

        # Ensure capture directory exists
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the background capture thread.

        Safe to call multiple times — will not start a second thread
        if one is already running.
        """
        if self.running:
            print("[Capture] Already running.")
            return

        self.running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="ScreenCaptureThread",
        )
        self._thread.start()
        print(f"[Capture] Screen capture started — every {self.interval}s → {CAPTURE_DIR}/")

    def stop(self) -> None:
        """Stop the capture thread gracefully."""
        self.running = False
        print("[Capture] Screen capture stopped.")

    def get_latest_path(self) -> Path | None:
        """Return the path to the most recent screenshot.

        Returns:
            Path to latest.png if it exists, otherwise None.
        """
        return LATEST_PATH if LATEST_PATH.exists() else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture loop — runs in background thread.

        Takes a screenshot every self.interval seconds.
        Saves to a timestamped file and updates latest.png.
        Prunes old files to maintain the rolling buffer.
        """
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Full desktop (all monitors combined)

            while self.running:
                try:
                    self._take_screenshot(sct, monitor)
                    time.sleep(self.interval)

                except Exception as e:
                    print(f"[Capture] ERROR during capture: {e}")
                    time.sleep(self.interval)

    def _take_screenshot(self, sct: mss.mss, monitor: dict) -> None:
        """Capture a single screenshot and save it to disk.

        Saves a timestamped copy and overwrites latest.png.
        Prunes the buffer if it exceeds BUFFER_SIZE.

        Args:
            sct: Active mss instance.
            monitor: Monitor dict from mss.monitors.
        """
        self.frame_count += 1
        timestamp = int(time.time())
        filename  = CAPTURE_DIR / f"frame_{timestamp}.png"

        # Capture and save timestamped frame
        screenshot = sct.grab(monitor)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filename))

        # Always keep latest.png up to date for Stage 2 analysis
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(LATEST_PATH))

        print(f"[Capture] Frame {self.frame_count} saved → {filename.name}")

        # Prune rolling buffer
        self._prune_buffer()

    def _prune_buffer(self) -> None:
        """Delete oldest screenshots if buffer exceeds BUFFER_SIZE.

        Excludes latest.png from the count — it is always retained.
        """
        frames = sorted(
            [f for f in CAPTURE_DIR.glob("frame_*.png")],
            key=lambda f: f.stat().st_mtime,
        )

        while len(frames) > BUFFER_SIZE:
            oldest = frames.pop(0)
            oldest.unlink()
            print(f"[Capture] Buffer pruned — deleted {oldest.name}")
