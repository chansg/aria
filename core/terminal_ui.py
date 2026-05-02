"""
core/terminal_ui.py
-------------------
Rich terminal dashboard for Aria.

Renders a live-updating panel layout in PowerShell 7 / Windows Terminal.
Runs in the main thread (blocks via run() until KeyboardInterrupt).
Other subsystems update shared state via thread-safe methods which the
UI re-reads at 4Hz.

Layout:
    ┌──────────────────────┬─────────────────────┐
    │  Status              │  Last Response       │
    ├──────────────────────┴─────────────────────┤
    │  Recent Activity (last 20 lines)            │
    └─────────────────────────────────────────────┘

Colour scheme:
    Panel borders:  navy_blue (dim)
    Labels:         gold
    States:         green / amber / cyan / dim
    Module tags:    per-module colour map (VTS purple, Router cyan,
                    Analyst gold, Speaker green, Vision blue, etc.)

Graceful fallback:
    All `rich` imports are guarded by RICH_AVAILABLE. If rich isn't
    installed, AriaUI is still importable but methods are no-ops and
    run() returns immediately. main.py checks RICH_AVAILABLE at startup
    and falls back to plain print/log output if False.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Optional, Tuple

# ── Rich imports (optional dependency) ────────────────────────────────────────
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── Module colour map ─────────────────────────────────────────────────────────
MODULE_COLOURS = {
    "VTS":         "medium_purple",
    "Router":      "cyan",
    "Analyst":     "gold1",
    "Market":      "spring_green2",
    "Speaker":     "green",
    "Vision":      "cornflower_blue",
    "WebSearch":   "orange1",
    "Capture":     "grey50",
    "Brain":       "white",
    "Aria":        "white",
    "Chan":        "bright_white",
    "Memory":      "steel_blue1",
    "Personality": "thistle1",
    "Transcriber": "pale_green1",
    "Listener":    "pale_green1",
    "Wake":        "pale_green1",
    "Chime":       "grey50",
    "Scheduler":   "wheat1",
    "Avatar":      "medium_purple",
    "Logger":      "grey50",
    "ERROR":       "bold red",
    "WARNING":     "yellow",
}

# ── State colour map ──────────────────────────────────────────────────────────
STATE_COLOURS = {
    "LISTENING": "bright_green",
    "THINKING":  "yellow",
    "SPEAKING":  "cyan",
    "IDLE":      "dim white",
    "DORMANT":   "grey50",
    "STARTING":  "dim white",
    "ERROR":     "bold red",
}

GOLD = "color(220)"
NAVY = "navy_blue"


class AriaUI:
    """Live terminal dashboard for Aria.

    Thread-safe state updates from any subsystem thread.
    Renders at 4Hz via rich.Live in the main thread.

    Usage:
        ui = AriaUI()
        ui.set_state("LISTENING")
        ui.set_last_response("The weather in Slough is 19°C.")
        ui.log("Router", "Tier 2 matched: weather")
        ui.run()   # blocks — call from main thread
    """

    def __init__(self, max_log_lines: int = 20) -> None:
        self._lock          = threading.Lock()
        self._stop_event    = threading.Event()
        self._state         = "STARTING"
        self._last_response = ""
        self._conv_mode     = True
        self._analysis_mode = False
        self._vts_status    = "Connecting..."
        self._gemini_status = "Ready"
        # Each entry: (timestamp_str, module_str, message_str, level_str)
        self._log: Deque[Tuple[str, str, str, str]] = deque(maxlen=max_log_lines)
        self._console: Optional[Console] = Console() if RICH_AVAILABLE else None
        self._live: Optional[Live]       = None

    # ── State setters (thread-safe) ───────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """Update Aria's current state. Safe to call from any thread."""
        with self._lock:
            self._state = state.upper()

    def set_last_response(self, text: str) -> None:
        """Update the Last Response panel. Safe to call from any thread."""
        with self._lock:
            self._last_response = text

    def set_conv_mode(self, enabled: bool) -> None:
        with self._lock:
            self._conv_mode = enabled

    def set_analysis_mode(self, enabled: bool) -> None:
        with self._lock:
            self._analysis_mode = enabled

    def set_vts_status(self, status: str) -> None:
        with self._lock:
            self._vts_status = status

    def log(self, module: str, message: str, level: str = "INFO") -> None:
        """Add a line to the activity log.

        Args:
            module:  Module display name e.g. 'Router', 'VTS'.
            message: The log message text.
            level:   'INFO', 'WARNING', or 'ERROR'.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._log.append((timestamp, module, message, level.upper()))

    # ── Render ────────────────────────────────────────────────────────────────

    def _build_status_panel(self) -> "Panel":
        with self._lock:
            state = self._state
            conv  = self._conv_mode

        # Poll VTS connection state from the controller live — saves the
        # controller from having to push updates explicitly.
        try:
            from avatar.renderer import _controller as _vts_ctrl
            if _vts_ctrl and _vts_ctrl.connected:
                vts = "Connected — Hiyori_A"
            elif _vts_ctrl:
                vts = "Connecting..."
            else:
                vts = "Not started"
        except Exception:
            vts = "Unknown"

        # Poll analysis-mode flag from the registered ProactiveAnalyst —
        # same pattern as VTS so brain.py / analyst don't need a callback.
        try:
            from core.proactive_analyst import get_instance as _get_analyst
            analyst = _get_analyst()
            analysis = bool(analyst and analyst.enabled)
        except Exception:
            analysis = False

        state_colour = STATE_COLOURS.get(state, "white")

        t = Table.grid(padding=(0, 1))
        t.add_column(style=GOLD, width=14)
        t.add_column()

        t.add_row("State",       Text(state, style=state_colour))
        t.add_row("",            "")
        t.add_row("Conv Mode",   Text("ON" if conv else "OFF",
                                      style="bright_green" if conv else "dim white"))
        t.add_row("Analysis",    Text("ON" if analysis else "OFF",
                                      style="gold1" if analysis else "dim white"))
        t.add_row("",            "")
        t.add_row("VTS",         Text(vts, style="medium_purple"))
        t.add_row("Gemini",      Text(self._gemini_status, style="cornflower_blue"))

        return Panel(
            t,
            title=Text("ARIA", style=f"bold {GOLD}"),
            border_style=NAVY,
            box=box.ROUNDED,
        )

    def _build_response_panel(self) -> "Panel":
        with self._lock:
            resp = self._last_response

        text = Text(resp or "—", style="white", overflow="fold")
        return Panel(
            text,
            title=Text("Last Response", style=GOLD),
            border_style=NAVY,
            box=box.ROUNDED,
        )

    def _build_log_panel(self) -> "Panel":
        with self._lock:
            entries = list(self._log)

        t = Table.grid(padding=(0, 1))
        t.add_column(style="grey50", width=8, no_wrap=True)   # timestamp
        t.add_column(width=14, no_wrap=True)                   # module tag
        t.add_column(overflow="fold")                          # message

        for timestamp, module, message, level in entries:
            if level == "ERROR":
                mod_colour = MODULE_COLOURS["ERROR"]
                msg_style  = MODULE_COLOURS["ERROR"]
            elif level == "WARNING":
                mod_colour = MODULE_COLOURS["WARNING"]
                msg_style  = MODULE_COLOURS["WARNING"]
            else:
                mod_colour = MODULE_COLOURS.get(module, "white")
                msg_style  = "white"

            t.add_row(
                timestamp,
                Text(f"[{module}]", style=mod_colour),
                Text(message, style=msg_style),
            )

        return Panel(
            t,
            title=Text("Recent Activity", style=GOLD),
            border_style=NAVY,
            box=box.ROUNDED,
        )

    def _build_layout(self) -> "Layout":
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="log"),
        )
        layout["top"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="response", ratio=2),
        )
        layout["status"].update(self._build_status_panel())
        layout["response"].update(self._build_response_panel())
        layout["log"].update(self._build_log_panel())
        return layout

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the live display. Blocks until stop() or KeyboardInterrupt.

        Call from the main thread after all subsystems are started. If
        rich is not installed, returns immediately so main.py can fall
        back to its previous blocking pattern.
        """
        if not RICH_AVAILABLE:
            return

        try:
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=4,
                screen=True,
            ) as live:
                self._live = live
                while not self._stop_event.is_set():
                    live.update(self._build_layout())
                    time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self._live = None

    def stop(self) -> None:
        """Signal the live display to exit. Safe from any thread."""
        self._stop_event.set()
