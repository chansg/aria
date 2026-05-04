"""
tests.test_broker_brain
-----------------------
Brain dispatch tests for broker account intent.
"""

from __future__ import annotations

import core.brain as brain
from core.brokers.trading212 import Trading212ConfigError


class _FakeTrading212Client:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def spoken_account_summary(self) -> str:
        return "account summary"

    def spoken_positions_summary(self) -> str:
        return "positions summary"

    def spoken_pending_orders_summary(self) -> str:
        return "orders summary"


def test_broker_account_handler_uses_account_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.brokers.trading212.Trading212Client.from_runtime",
        lambda: _FakeTrading212Client(),
    )

    assert brain._handle_broker_account("check my Trading 212 demo account") == "account summary"


def test_broker_account_handler_uses_positions_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.brokers.trading212.Trading212Client.from_runtime",
        lambda: _FakeTrading212Client(),
    )

    assert brain._handle_broker_account("show my paper positions") == "positions summary"


def test_broker_account_handler_reports_missing_config(monkeypatch) -> None:
    def raise_config_error():
        raise Trading212ConfigError("missing")

    monkeypatch.setattr(
        "core.brokers.trading212.Trading212Client.from_runtime",
        raise_config_error,
    )

    out = brain._handle_broker_account("check my broker account")

    assert "not configured" in out
