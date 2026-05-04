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
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / ("aria_test.log" if "pytest" in sys.modules else "aria.log")

# Module-name → display-prefix mapping. Modules not listed get a default
# prefix derived from the last dotted component (capitalized).
_PREFIX_MAP = {
    "core.brain":             "Brain",
    "core.router":            "Router",
    "core.web_search":        "WebSearch",
    "core.vision_analyzer":   "Vision",
    "core.screen_capture":    "Capture",
    "core.proactive_analyst": "Analyst",
    "core.market_analyst":    "Market",
    "core.brokers.trading212": "T212",
    "core.diagnostics":       "Diagnostics",
    "core.memory":            "Memory",
    "core.personality":       "Personality",
    "core.scheduler":         "Scheduler",
    "core.logger":            "Logger",
    "avatar.renderer":        "Avatar",
    "voice.speaker":          "Speaker",
    "voice.tts.piper_provider":  "Piper",
    "voice.tts.kokoro_provider": "Kokoro",
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
_error_hooks_installed = False
_original_excepthook = sys.excepthook
_original_threading_excepthook = getattr(threading, "excepthook", None)
_original_unraisablehook = getattr(sys, "unraisablehook", None)
_original_stderr = sys.stderr


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


class _StderrToLogger:
    """File-like object that forwards stderr lines into aria.log."""

    def __init__(self, logger: logging.Logger, original_stream) -> None:
        self._logger = logger
        self._original_stream = original_stream
        self._buffer = ""
        self._lock = threading.Lock()
        self.encoding = getattr(original_stream, "encoding", "utf-8")
        self.errors = getattr(original_stream, "errors", "replace")

    def write(self, message: str) -> int:
        if not message:
            return 0

        with self._lock:
            self._buffer += message
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    self._logger.error("%s", line)

        return len(message)

    def flush(self) -> None:
        with self._lock:
            line = self._buffer.rstrip("\r\n")
            self._buffer = ""
        if line.strip():
            self._logger.error("%s", line)

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._original_stream.fileno()


def install_error_logging(capture_stderr: bool = True) -> None:
    """Capture uncaught process, thread, and terminal errors in aria.log."""
    global _error_hooks_installed
    if _error_hooks_installed:
        return

    terminal_log = get_logger("Terminal")

    def _handle_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            terminal_log.info("KeyboardInterrupt received.")
            _original_excepthook(exc_type, exc_value, exc_traceback)
            return

        if issubclass(exc_type, SystemExit):
            terminal_log.info("SystemExit: %s", exc_value)
            return

        terminal_log.critical(
            "Unhandled exception on main thread.",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _handle_thread_exception(args) -> None:
        if issubclass(args.exc_type, SystemExit):
            return

        thread_name = args.thread.name if args.thread else "unknown"
        terminal_log.critical(
            "Unhandled exception in thread %s.",
            thread_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _handle_unraisable(args) -> None:
        terminal_log.error(
            "Unraisable exception: %s object=%r",
            args.err_msg,
            args.object,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle_exception
    if _original_threading_excepthook is not None:
        threading.excepthook = _handle_thread_exception
    if _original_unraisablehook is not None:
        sys.unraisablehook = _handle_unraisable

    if capture_stderr and not isinstance(sys.stderr, _StderrToLogger):
        sys.stderr = _StderrToLogger(terminal_log, _original_stderr)

    _error_hooks_installed = True
    terminal_log.debug("Terminal/error logging hooks installed.")


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
