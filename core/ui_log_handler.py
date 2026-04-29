"""
core/ui_log_handler.py
----------------------
Custom logging handler that forwards log records to the AriaUI dashboard.

Bridges the Python logging system with the rich terminal UI so all
subsystem messages appear in the activity log panel in real time.

Wired up by core.logger.attach_ui(ui) — call once after the AriaUI
instance is created and before AriaUI.run().
"""

from __future__ import annotations

import logging


class UILogHandler(logging.Handler):
    """Forwards log records to AriaUI.log() for live display.

    Reads the [prefix] attribute set by core.logger._PrefixFilter to
    derive the module tag for the activity log column. Falls back to
    the last dotted component of record.name if no prefix is set
    (e.g. when records bypass our logger factory).

    Args:
        ui: The AriaUI instance to forward records to.
    """

    def __init__(self, ui) -> None:
        super().__init__()
        self._ui = ui

    def emit(self, record: logging.LogRecord) -> None:
        try:
            module  = getattr(record, "prefix", None) or record.name.split(".")[-1].capitalize()
            # record.getMessage() handles printf-style args (log.info("x %s", y)).
            message = record.getMessage()
            level   = record.levelname
            self._ui.log(module, message, level)
        except Exception:
            # Never let logging crash the application.
            self.handleError(record)
