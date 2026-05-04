"""
tests.test_conversation_state
-----------------------------
Fast volatile session-state tests.
"""

from __future__ import annotations

from core.conversation_state import (
    get_last_finance_context,
    remember_finance_quote,
    reset_conversation_state,
)


def test_remembers_successful_finance_quote() -> None:
    reset_conversation_state()

    remember_finance_quote({
        "ticker": "AAPL",
        "display_name": "AAPL",
        "price": 277.85,
        "previous_close": 280.14,
        "daily_change_pct": -0.82,
        "as_of_date": "Monday 04 May 2026",
        "source": "yahoo-chart",
    }, user_text="what is the price of apple")

    context = get_last_finance_context()

    assert context is not None
    assert context.symbol == "AAPL"
    assert context.price == 277.85
    assert context.as_of_date == "Monday 04 May 2026"


def test_failed_quote_does_not_update_finance_context() -> None:
    reset_conversation_state()

    remember_finance_quote({"ticker": "BROKEN", "error": "no data"})

    assert get_last_finance_context() is None
