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

import io
import os
import threading
import time
from pathlib import Path

import mss
import mss.tools
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
CAPTURE_DIR      = Path("data/captures")
LATEST_PATH      = CAPTURE_DIR / "latest.png"
BUFFER_SIZE      = 10         # Maximum number of screenshots to keep on disk
CAPTURE_INTERVAL = 5.0        # Seconds between captures

# Compression — downscale full desktop to a size Gemini Flash can swallow cheaply.
# 1280x720 preserves enough detail for gameplay/UI recognition while keeping
# payloads small (~150–300 KB per frame vs ~8–9 MB raw).
COMPRESS_MAX_WIDTH  = 1280
COMPRESS_MAX_HEIGHT = 720
PNG_COMPRESS_LEVEL  = 6       # 0 (none) – 9 (max). 6 is PIL's sweet spot.


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
        """Capture a single screenshot, compress it, and save to disk.

        Saves a timestamped copy and overwrites latest.png. Screenshots
        are downscaled with PIL (LANCZOS) to COMPRESS_MAX_WIDTH x
        COMPRESS_MAX_HEIGHT before saving, keeping per-frame payloads
        small enough for Gemini Flash vision calls.

        latest.png is updated atomically — written to a .tmp file first
        and then renamed via os.replace(). This prevents Gemini's
        vision_analyzer reader from ever seeing a partially-written PNG
        (the cause of "image file is truncated" errors).

        Args:
            sct: Active mss instance.
            monitor: Monitor dict from mss.monitors.
        """
        self.frame_count += 1
        timestamp = int(time.time())
        filename  = CAPTURE_DIR / f"frame_{timestamp}.png"

        # Grab the raw frame from the GDI compositor
        screenshot = sct.grab(monitor)

        # Compress: mss returns BGRA; PIL wants RGB
        compressed = self._compress_screenshot(screenshot)

        # Timestamped frame — direct write is fine, nothing else reads these
        compressed.save(filename, format="PNG", optimize=True, compress_level=PNG_COMPRESS_LEVEL)

        # latest.png — atomic write-then-rename to avoid race with vision_analyzer
        latest_tmp = LATEST_PATH.with_suffix(".png.tmp")
        compressed.save(latest_tmp, format="PNG", optimize=True, compress_level=PNG_COMPRESS_LEVEL)
        os.replace(latest_tmp, LATEST_PATH)  # atomic on Windows + Unix

        size_kb = filename.stat().st_size / 1024
        print(f"[Capture] Frame {self.frame_count} saved → {filename.name} ({size_kb:.0f} KB)")

        # Prune rolling buffer
        self._prune_buffer()

    def _compress_screenshot(self, screenshot) -> Image.Image:
        """Convert an mss screenshot to a downscaled RGB PIL image.

        Uses LANCZOS resampling for high-quality downscaling and
        preserves aspect ratio — the frame is fit inside a
        COMPRESS_MAX_WIDTH x COMPRESS_MAX_HEIGHT bounding box.

        Args:
            screenshot: The raw frame returned by mss.grab().

        Returns:
            A PIL Image in RGB mode, ready to save as PNG.
        """
        # mss gives us BGRA bytes; PIL reads them via 'raw' decoder
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # Downscale in place, preserving aspect ratio
        img.thumbnail((COMPRESS_MAX_WIDTH, COMPRESS_MAX_HEIGHT), Image.LANCZOS)
        return img

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
