"""
tests.test_brain_finance_followup
---------------------------------
Local finance follow-up handling.
"""

from __future__ import annotations

import core.brain as brain
from core.conversation_state import remember_finance_quote, reset_conversation_state


def test_finance_recency_followup_uses_session_context() -> None:
    reset_conversation_state()
    remember_finance_quote({
        "ticker": "AAPL",
        "display_name": "AAPL",
        "price": 277.85,
        "previous_close": 280.14,
        "daily_change_pct": -0.82,
        "as_of_date": "Monday 04 May 2026",
        "source": "yahoo-chart",
    })

    response = brain._handle_finance_followup("Is this recent?")

    assert "AAPL" in response
    assert "Monday 04 May 2026" in response
    assert "not live intraday" in response


def test_finance_symbol_followup_fetches_new_quote(monkeypatch) -> None:
    reset_conversation_state()
    remember_finance_quote({
        "ticker": "AAPL",
        "price": 277.85,
        "as_of_date": "Monday 04 May 2026",
    })

    class FakeMarketAnalyst:
        def fetch_quote(self, ticker: str) -> dict:
            assert ticker == "NVDA"
            return {
                "ticker": "NVDA",
                "display_name": "NVDA",
                "price": 900.0,
                "previous_close": 850.0,
                "daily_change_pct": 5.88,
                "as_of_date": "Monday 04 May 2026",
            }

    import core.market_analyst as market_analyst
    monkeypatch.setattr(market_analyst, "MarketAnalyst", FakeMarketAnalyst)

    response = brain._handle_finance_followup("What about Nvidia?")

    assert "NVDA closed at $900.00" in response
    assert "up 5.9%" in response


def test_finance_performance_followup_uses_last_symbol(monkeypatch) -> None:
    reset_conversation_state()
    remember_finance_quote({
        "ticker": "AAPL",
        "price": 277.85,
        "as_of_date": "Monday 04 May 2026",
    })

    calls = {}

    class FakeMarketAnalyst:
        def spoken_performance(self, ticker: str, period_label: str, range_: str) -> str:
            calls["ticker"] = ticker
            calls["period_label"] = period_label
            calls["range"] = range_
            return "performance summary"

    import core.market_analyst as market_analyst
    monkeypatch.setattr(market_analyst, "MarketAnalyst", FakeMarketAnalyst)

    response = brain._handle_finance_followup("How has it performed in the last six months?")

    assert response == "performance summary"
    assert calls == {"ticker": "AAPL", "period_label": "six months", "range": "6mo"}
