"""
tests.test_router
-----------------
Intent routing regression tests.
"""

from __future__ import annotations

import pytest

from core.conversation_state import remember_finance_quote, reset_conversation_state
from core.router import classify


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_conversation_state()
    yield
    reset_conversation_state()


def test_stock_quote_intent_handles_latest_price_request() -> None:
    route = classify("give me the latest stock price for GME")

    assert route == {"intent": "stock_quote", "tier": 1}


def test_stock_quote_intent_handles_last_close_request() -> None:
    route = classify("what price did GME last end with?")

    assert route == {"intent": "stock_quote", "tier": 1}


def test_stock_quote_intent_handles_ticker_without_stock_word() -> None:
    route = classify("current price of GME")

    assert route == {"intent": "stock_quote", "tier": 1}


def test_non_stock_price_query_stays_web_search() -> None:
    route = classify("price of bitcoin")

    assert route == {"intent": "web_search", "tier": 2}


def test_market_update_still_routes_to_market_snapshot() -> None:
    route = classify("give me the market update")

    assert route == {"intent": "market", "tier": 1}


def test_trading212_demo_account_routes_locally() -> None:
    route = classify("check my Trading 212 demo account")

    assert route == {"intent": "broker_account", "tier": 1}


def test_paper_positions_route_locally() -> None:
    route = classify("show my paper positions")

    assert route == {"intent": "broker_account", "tier": 1}


def test_finance_recency_followup_requires_context() -> None:
    route = classify("is this recent?")

    assert route == {"intent": "claude", "tier": 3}


def test_finance_recency_followup_routes_locally_with_context() -> None:
    remember_finance_quote({
        "ticker": "AAPL",
        "price": 277.85,
        "as_of_date": "Monday 04 May 2026",
    })

    route = classify("is this recent?")

    assert route == {"intent": "finance_followup", "tier": 1}


def test_finance_symbol_followup_routes_locally_with_context() -> None:
    remember_finance_quote({
        "ticker": "AAPL",
        "price": 277.85,
        "as_of_date": "Monday 04 May 2026",
    })

    route = classify("what about Nvidia?")

    assert route == {"intent": "finance_followup", "tier": 1}
