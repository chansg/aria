"""
tests.test_trading212_client
----------------------------
Network-free tests for the Trading 212 demo adapter.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from core.brokers.trading212 import (
    DEMO_BASE_URL,
    LIVE_BASE_URL,
    Trading212ApiError,
    Trading212Client,
    Trading212Config,
    Trading212ConfigError,
    Trading212LiveModeBlocked,
)


def _config(tmp_path: Path) -> Trading212Config:
    return Trading212Config(
        base_url=DEMO_BASE_URL,
        api_key="demo-key",
        api_secret="demo-secret",
        environment="demo",
        audit_log_path=tmp_path / "audit.jsonl",
        training_log_path=tmp_path / "training.jsonl",
    )


def _client(tmp_path: Path, handler) -> Trading212Client:
    http_client = httpx.Client(
        base_url=DEMO_BASE_URL,
        auth=httpx.BasicAuth("demo-key", "demo-secret"),
        transport=httpx.MockTransport(handler),
    )
    return Trading212Client(_config(tmp_path), http_client=http_client)


def _audit_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_get_account_cash_uses_basic_auth_and_audits_without_secrets(tmp_path) -> None:
    expected_auth = "Basic " + base64.b64encode(b"demo-key:demo-secret").decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/equity/account/cash"
        assert request.headers["authorization"] == expected_auth
        return httpx.Response(
            200,
            json={"free": 1000.0, "currency": "GBP"},
            headers={"x-ratelimit-remaining": "49"},
        )

    client = _client(tmp_path, handler)

    assert client.get_account_cash() == {"free": 1000.0, "currency": "GBP"}

    audit_text = _config(tmp_path).audit_log_path.read_text(encoding="utf-8")
    assert "demo-key" not in audit_text
    assert "demo-secret" not in audit_text
    events = _audit_events(_config(tmp_path).audit_log_path)
    assert events[0]["action"] == "account_cash"
    assert events[0]["status_code"] == 200
    assert events[0]["rate_limit"]["x-ratelimit-remaining"] == "49"


def test_live_environment_is_blocked(tmp_path) -> None:
    config = Trading212Config(
        base_url=LIVE_BASE_URL,
        api_key="key",
        api_secret="secret",
        environment="live",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    with pytest.raises(Trading212LiveModeBlocked):
        Trading212Client(config)


def test_non_get_requests_are_refused(tmp_path) -> None:
    client = _client(tmp_path, lambda _request: httpx.Response(200, json={}))

    with pytest.raises(Trading212ConfigError):
        client._request(
            "POST",
            "/equity/orders/market",
            json_body={"ticker": "AAPL_US_EQ", "quantity": 1},
            action="blocked_order",
        )


def test_capture_account_state_fetches_read_only_account_data(tmp_path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/account/summary"):
            return httpx.Response(200, json={"total": 1500.0, "currency": "GBP"})
        if request.url.path.endswith("/account/cash"):
            return httpx.Response(200, json={"free": 750.0, "currency": "GBP"})
        if request.url.path.endswith("/positions"):
            return httpx.Response(200, json=[{"ticker": "AAPL_US_EQ", "quantity": 2}])
        if request.url.path.endswith("/orders"):
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"error": "not found"})

    client = _client(tmp_path, handler)

    state = client.capture_account_state()

    assert state["account_summary"]["total"] == 1500.0
    assert state["cash"]["free"] == 750.0
    assert state["positions"] == [{"ticker": "AAPL_US_EQ", "quantity": 2}]
    assert calls == [
        "/api/v0/equity/account/summary",
        "/api/v0/equity/account/cash",
        "/api/v0/equity/positions",
        "/api/v0/equity/orders",
    ]

    events = _audit_events(_config(tmp_path).audit_log_path)
    assert events[-2]["action"] == "account_state_snapshot"
    assert events[-2]["positions_count"] == 1
    assert events[-1]["action"] == "training_account_state_saved"

    training_events = _audit_events(_config(tmp_path).training_log_path)
    assert training_events[0]["broker"] == "trading212"
    assert training_events[0]["positions"] == [{"ticker": "AAPL_US_EQ", "quantity": 2}]


def test_spoken_account_summary_is_concise(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/account/summary"):
            return httpx.Response(200, json={"total": 1500.0, "invested": 500.0, "currency": "GBP"})
        if request.url.path.endswith("/account/cash"):
            return httpx.Response(200, json={"free": 1000.0, "currency": "GBP"})
        if request.url.path.endswith("/positions"):
            return httpx.Response(200, json=[{"ticker": "AAPL_US_EQ", "quantity": 2}])
        if request.url.path.endswith("/orders"):
            return httpx.Response(200, json=[{"ticker": "MSFT_US_EQ", "quantity": 1}])
        return httpx.Response(404)

    client = _client(tmp_path, handler)

    out = client.spoken_account_summary()

    assert "Trading 212 demo account is connected" in out
    assert "£1,500.00" in out
    assert "1 open position" in out
    assert "1 pending order" in out


def test_iter_paginated_follows_next_page_path(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "next":
            return httpx.Response(200, json={"items": [{"id": 2}], "nextPagePath": None})
        return httpx.Response(
            200,
            json={
                "items": [{"id": 1}],
                "nextPagePath": "/api/v0/equity/history/orders?limit=50&cursor=next",
            },
        )

    client = _client(tmp_path, handler)

    assert list(client.iter_paginated("/equity/history/orders")) == [{"id": 1}, {"id": 2}]


def test_api_errors_raise_and_are_audited(tmp_path) -> None:
    client = _client(tmp_path, lambda _request: httpx.Response(401, json={"error": "bad auth"}))

    with pytest.raises(Trading212ApiError) as exc:
        client.get_positions()

    assert exc.value.status_code == 401
    events = _audit_events(_config(tmp_path).audit_log_path)
    assert events[0]["status_code"] == 401
