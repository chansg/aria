"""
core/logger.py
--------------
Centralised logging configuration for Aria.

All modules obtain their logger via:
    from core.logger import get_logger
    log = get_logger(__name__)
    log.info("Tier 2 matched: weather")
    # → console: 17:23:41 [Router] Tier 2 matched: weather
    # → file:    2026-04-28 17:23:41 [INFO ] [Router] Tier 2 matched: weather

Output:
    Console: INFO and above, clean format (HH:MM:SS [Module] message)
    File:    DEBUG and above, full timestamped format with level
             logs/aria.log — daily rotation, 7 days retained

Design notes:
    Uses a logging.Filter rather than a LoggerAdapter to inject the
    [Module] prefix into every record. The Filter approach:
      - Returns a real Logger (preserves exc_info, stack_info, extra)
      - Doesn't override stdlib internals
      - Plays nicely with custom handlers (e.g. UILogHandler from the
        rich-terminal-UI work that follows in the next PR)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "aria.log"

# Module-name → display-prefix mapping. Modules not listed get a default
# prefix derived from the last dotted component (capitalized).
_PREFIX_MAP = {
    "core.brain":             "Brain",
    "core.router":            "Router",
    "core.web_search":        "WebSearch",
    "core.vision_analyzer":   "Vision",
    "core.screen_capture":    "Capture",
    "core.proactive_analyst": "Analyst",
    "core.memory":            "Memory",
    "core.personality":       "Personality",
    "core.scheduler":         "Scheduler",
    "core.logger":            "Logger",
    "avatar.vts_controller":  "VTS",
    "avatar.renderer":        "Avatar",
    "voice.speaker":          "Speaker",
    "voice.listener":         "Listener",
    "voice.transcriber":      "Transcriber",
    "voice.trainer":          "Trainer",
    "voice.wake":             "Wake",
    "voice.chime":            "Chime",
    "main":                   "Aria",
    "__main__":               "Aria",
}

_initialised = False
_root_logger: logging.Logger | None = None


def _initialise() -> None:
    """Set up the root 'aria' logger with console + file handlers. Idempotent."""
    global _initialised, _root_logger
    if _initialised:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("aria")
    root.setLevel(logging.DEBUG)
    root.propagate = False  # don't bubble to Python's root logger

    # ── File handler — full detail ────────────────────────────────────────
    file_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] [%(prefix)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # ── Console handler — INFO and above, clean format ───────────────────
    console_fmt = logging.Formatter(
        fmt="%(asctime)s [%(prefix)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    _initialised = True
    _root_logger = root


class _PrefixFilter(logging.Filter):
    """Injects a [prefix] attribute into every log record.

    The formatters expect %(prefix)s — without this filter the format
    string would raise. Idempotent per (logger, prefix) pair.
    """

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        record.prefix = self.prefix
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a Logger for a module or explicit prefix label.

    Args:
        name: Either a Python module name (use __name__) or an explicit
              display label (e.g. 'Aria') if you want a prefix that
              doesn't match any module name.

    Returns:
        A standard logging.Logger that emits with the correct
        [ModuleName] prefix on both console and file.

    Examples:
        log = get_logger(__name__)
        log.info("Listening...")
        log.warning("Stale screenshot — skipping")
        log.error("Gemini failed: %s", exc, exc_info=True)
    """
    _initialise()

    # Look up display prefix; otherwise derive from last dotted component
    if name in _PREFIX_MAP:
        prefix = _PREFIX_MAP[name]
    elif "." not in name:
        # Caller passed an explicit label like "Aria" rather than __name__
        prefix = name
    else:
        prefix = name.split(".")[-1].capitalize()

    logger = logging.getLogger(f"aria.{name}")

    # Idempotent — only add the filter once per (logger, prefix) pair
    if not any(isinstance(f, _PrefixFilter) and f.prefix == prefix for f in logger.filters):
        logger.addFilter(_PrefixFilter(prefix))

    return logger


def attach_ui(ui) -> None:
    """Attach a UILogHandler so all log records also flow to the rich UI.

    Stub for the next PR (rich terminal UI). Call after the AriaUI is
    constructed but before AriaUI.run(). Currently a no-op if the
    UILogHandler module isn't present yet.

    Args:
        ui: The AriaUI instance (rich terminal dashboard).
    """
    _initialise()
    try:
        from core.ui_log_handler import UILogHandler  # type: ignore
    except ImportError:
        return  # UI feature not installed yet — silently skip

    root    = logging.getLogger("aria")
    handler = UILogHandler(ui)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
