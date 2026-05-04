"""
Fast per-session conversation state for Aria.

This module is deliberately not long-term memory. It stores the small amount of
context needed to keep live voice follow-ups snappy, such as the last finance
instrument Aria discussed. Durable memory still lives in core.memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from typing import Any

from core.logger import get_logger

log = get_logger(__name__)


DEFAULT_CONTEXT_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class FinanceContext:
    """Short-lived state for the most recent finance answer."""

    symbol: str
    display_name: str
    price: float | None
    previous_close: float | None
    daily_change_pct: float | None
    as_of_date: str | None
    source: str | None
    captured_at: str
    user_text: str | None = None


_lock = threading.Lock()
_last_finance_context: FinanceContext | None = None


def remember_finance_quote(quote: dict[str, Any], user_text: str | None = None) -> None:
    """Store the latest successful quote in volatile session state."""
    if quote.get("error"):
        return

    symbol = str(quote.get("ticker") or "").strip().upper()
    if not symbol:
        return

    context = FinanceContext(
        symbol=symbol,
        display_name=str(quote.get("display_name") or symbol),
        price=quote.get("price"),
        previous_close=quote.get("previous_close"),
        daily_change_pct=quote.get("daily_change_pct"),
        as_of_date=quote.get("as_of_date"),
        source=quote.get("source"),
        captured_at=datetime.now().isoformat(timespec="seconds"),
        user_text=user_text,
    )

    global _last_finance_context
    with _lock:
        _last_finance_context = context

    log.debug("Finance session context updated: %s", context)


def get_last_finance_context(
    max_age_seconds: int | None = DEFAULT_CONTEXT_TTL_SECONDS,
) -> FinanceContext | None:
    """Return recent finance context, or None when missing/expired."""
    with _lock:
        context = _last_finance_context

    if context is None or max_age_seconds is None:
        return context

    try:
        captured_at = datetime.fromisoformat(context.captured_at)
    except ValueError:
        return None

    if datetime.now() - captured_at > timedelta(seconds=max_age_seconds):
        return None

    return context


def has_recent_finance_context() -> bool:
    """Return True when a finance follow-up can be answered locally."""
    return get_last_finance_context() is not None


def reset_conversation_state() -> None:
    """Clear volatile state. Intended for tests and fresh session resets."""
    global _last_finance_context
    with _lock:
        _last_finance_context = None
