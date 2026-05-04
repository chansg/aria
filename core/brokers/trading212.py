"""
core/brokers/trading212.py
--------------------------
Read-only Trading 212 demo adapter for Aria.

This client deliberately refuses non-GET requests. Paper order placement belongs
in a later branch with explicit user approval, duplicate protection, and risk
limits.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from core.logger import get_logger

log = get_logger(__name__)

DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
LIVE_BASE_URL = "https://live.trading212.com/api/v0"
DEFAULT_AUDIT_LOG_PATH = Path("data") / "broker" / "trading212_audit.jsonl"
DEFAULT_TRAINING_LOG_PATH = Path("data") / "broker" / "trading212_account_state.jsonl"


class Trading212Error(Exception):
    """Base Trading 212 adapter error."""


class Trading212ConfigError(Trading212Error):
    """Raised when the local Trading 212 config is missing or unsafe."""


class Trading212LiveModeBlocked(Trading212ConfigError):
    """Raised when config attempts to use the live-money API."""


class Trading212ApiError(Trading212Error):
    """Raised when Trading 212 returns a non-successful response."""

    def __init__(self, status_code: int, message: str, *, path: str) -> None:
        super().__init__(f"Trading 212 API error {status_code} on {path}: {message}")
        self.status_code = status_code
        self.path = path
        self.message = message


@dataclass(frozen=True)
class Trading212Config:
    """Runtime configuration for the read-only Trading 212 client."""

    base_url: str
    api_key: str
    api_secret: str
    environment: str = "demo"
    timeout_seconds: float = 10.0
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH
    training_log_path: Path = DEFAULT_TRAINING_LOG_PATH

    @classmethod
    def from_runtime(cls) -> "Trading212Config":
        """Build config from config.py with environment-variable fallback."""
        env = str(_setting("TRADING212_ENV", "demo")).strip().lower()
        base_url = str(_setting("TRADING212_BASE_URL", "")).strip()
        if not base_url:
            base_url = DEMO_BASE_URL if env == "demo" else LIVE_BASE_URL

        api_key = str(_setting("TRADING212_API_KEY", "")).strip()
        api_secret = str(_setting("TRADING212_API_SECRET", "")).strip()
        if not api_key or not api_secret:
            raise Trading212ConfigError(
                "Trading 212 demo credentials are not configured. Set "
                "TRADING212_API_KEY and TRADING212_API_SECRET in .env or config.py."
            )

        timeout_seconds = float(_setting("TRADING212_TIMEOUT_SECONDS", 10.0))
        audit_path = Path(str(_setting("TRADING212_AUDIT_LOG_PATH", DEFAULT_AUDIT_LOG_PATH)))
        training_path = Path(str(_setting("TRADING212_TRAINING_LOG_PATH", DEFAULT_TRAINING_LOG_PATH)))

        config = cls(
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
            environment=env,
            timeout_seconds=timeout_seconds,
            audit_log_path=audit_path,
            training_log_path=training_path,
        )
        config.validate_demo_only()
        return config

    def validate_demo_only(self) -> None:
        """Reject live-money settings for the Phase 1 adapter."""
        normalized = self.base_url.rstrip("/")
        host = urlparse(normalized).netloc.lower()
        if self.environment != "demo" or "live.trading212.com" in host:
            raise Trading212LiveModeBlocked(
                "Trading 212 live mode is blocked. This adapter only permits "
                f"the demo API: {DEMO_BASE_URL}"
            )
        if "demo.trading212.com" not in host:
            raise Trading212ConfigError(
                "Trading 212 base URL must point to demo.trading212.com in this phase."
            )


class Trading212Client:
    """Read-only Trading 212 Public API client for the demo environment."""

    def __init__(
        self,
        config: Trading212Config,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        config.validate_demo_only()
        self.config = config
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            auth=httpx.BasicAuth(config.api_key, config.api_secret),
            timeout=config.timeout_seconds,
            headers={"Accept": "application/json", "User-Agent": "Aria/1.0"},
        )
        log.info("Trading 212 demo client ready — base_url=%s", config.base_url)

    @classmethod
    def from_runtime(cls) -> "Trading212Client":
        """Construct a client from config.py / environment settings."""
        return cls(Trading212Config.from_runtime())

    def close(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "Trading212Client":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    # ── Public read-only API ────────────────────────────────────────────────

    def get_account_cash(self) -> dict[str, Any]:
        """Fetch account cash breakdown."""
        return self._request("GET", "/equity/account/cash", action="account_cash")

    def get_account_summary(self) -> dict[str, Any]:
        """Fetch account summary."""
        return self._request("GET", "/equity/account/summary", action="account_summary")

    def get_positions(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """Fetch all open positions, optionally filtered by Trading 212 ticker."""
        params = {"ticker": ticker} if ticker else None
        data = self._request("GET", "/equity/positions", params=params, action="positions")
        return _as_list(data)

    def get_pending_orders(self) -> list[dict[str, Any]]:
        """Fetch active pending orders."""
        data = self._request("GET", "/equity/orders", action="pending_orders")
        return _as_list(data)

    def get_exchanges(self) -> list[dict[str, Any]]:
        """Fetch exchange metadata."""
        data = self._request("GET", "/equity/metadata/exchanges", action="exchanges")
        return _as_list(data)

    def get_instruments(self) -> list[dict[str, Any]]:
        """Fetch available instrument metadata."""
        data = self._request("GET", "/equity/metadata/instruments", action="instruments")
        return _as_list(data)

    def get_order_history(self, *, limit: int = 20, cursor: str | None = None) -> dict[str, Any]:
        """Fetch one page of historical orders."""
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        data = self._request(
            "GET",
            "/equity/history/orders",
            params=params,
            action="history_orders",
        )
        return data if isinstance(data, dict) else {"items": _as_list(data), "nextPagePath": None}

    def iter_paginated(self, path: str, *, limit: int = 50, max_pages: int = 10) -> Iterable[dict[str, Any]]:
        """Yield items from a Trading 212 paginated list endpoint."""
        page_path = path
        params: dict[str, Any] | None = {"limit": limit}
        for _ in range(max_pages):
            data = self._request("GET", page_path, params=params, action=f"paginate:{path}")
            if not isinstance(data, dict):
                break

            for item in _as_list(data.get("items", [])):
                yield item

            next_page = data.get("nextPagePath")
            if not next_page:
                break
            page_path = str(next_page)
            params = None

    def resolve_instrument(self, symbol_or_name: str) -> dict[str, Any] | None:
        """Resolve a simple symbol/name to a Trading 212 instrument record."""
        query = symbol_or_name.strip().upper()
        if not query:
            return None

        for instrument in self.get_instruments():
            ticker = str(instrument.get("ticker", "")).upper()
            short_name = str(instrument.get("shortName", "")).upper()
            name = str(instrument.get("name", "")).upper()
            if ticker == query or ticker.startswith(f"{query}_"):
                return instrument
            if short_name == query or name == query:
                return instrument
        return None

    def capture_account_state(self, *, persist_training: bool = True) -> dict[str, Any]:
        """Fetch a compact account state snapshot for local training logs."""
        state = {
            "captured_at": _utc_now(),
            "environment": self.config.environment,
            "broker": "trading212",
            "account_summary": self.get_account_summary(),
            "cash": self.get_account_cash(),
            "positions": self.get_positions(),
            "pending_orders": self.get_pending_orders(),
        }
        self._audit(
            "account_state_snapshot",
            {
                "positions_count": len(state["positions"]),
                "pending_orders_count": len(state["pending_orders"]),
                "summary_keys": sorted(state["account_summary"].keys()),
                "cash_keys": sorted(state["cash"].keys()),
            },
        )
        if persist_training:
            _append_jsonl(self.config.training_log_path, state)
            self._audit(
                "training_account_state_saved",
                {
                    "path": str(self.config.training_log_path),
                    "positions_count": len(state["positions"]),
                    "pending_orders_count": len(state["pending_orders"]),
                },
            )
        return state

    # ── Spoken summaries ────────────────────────────────────────────────────

    def spoken_account_summary(self) -> str:
        """Return a concise account summary suitable for Aria's TTS."""
        state = self.capture_account_state()
        summary = state["account_summary"]
        cash = state["cash"]
        positions = state["positions"]
        pending = state["pending_orders"]

        total = _first_number(summary, cash, keys=("total", "totalValue", "accountValue", "portfolioValue"))
        free = _first_number(cash, summary, keys=("free", "available", "availableFunds", "cash"))
        invested = _first_number(summary, cash, keys=("invested", "investedValue", "blocked"))
        currency = _first_text(summary, cash, keys=("currency", "accountCurrency")) or ""

        fragments = ["Trading 212 demo account is connected"]
        if total is not None:
            fragments.append(f"total value {_format_money(total, currency)}")
        if free is not None:
            fragments.append(f"available cash {_format_money(free, currency)}")
        if invested is not None:
            fragments.append(f"invested {_format_money(invested, currency)}")

        return (
            f"{', '.join(fragments)}, with {len(positions)} open position"
            f"{'' if len(positions) == 1 else 's'} and {len(pending)} pending order"
            f"{'' if len(pending) == 1 else 's'}."
        )

    def spoken_positions_summary(self) -> str:
        """Return a concise open-position summary."""
        positions = self.get_positions()
        if not positions:
            return "Your Trading 212 demo account has no open positions, Chan."

        top = positions[:3]
        names = []
        for position in top:
            ticker = _instrument_ticker(position) or "unknown ticker"
            quantity = _first_number(position, keys=("quantity", "ownedQuantity", "filledQuantity"))
            if quantity is None:
                names.append(ticker)
            else:
                names.append(f"{ticker} {quantity:g} shares")

        extra = len(positions) - len(top)
        suffix = f", plus {extra} more" if extra > 0 else ""
        return f"You have {len(positions)} open demo position{'' if len(positions) == 1 else 's'}: {', '.join(names)}{suffix}."

    def spoken_pending_orders_summary(self) -> str:
        """Return a concise pending-order summary."""
        orders = self.get_pending_orders()
        if not orders:
            return "There are no pending Trading 212 demo orders, Chan."

        top = orders[:3]
        labels = []
        for order in top:
            ticker = _instrument_ticker(order) or "unknown ticker"
            side = str(order.get("side", "")).lower() or "order"
            quantity = _first_number(order, keys=("quantity", "filledQuantity"))
            labels.append(f"{side} {quantity:g} {ticker}" if quantity is not None else f"{side} {ticker}")

        extra = len(orders) - len(top)
        suffix = f", plus {extra} more" if extra > 0 else ""
        return f"You have {len(orders)} pending demo order{'' if len(orders) == 1 else 's'}: {', '.join(labels)}{suffix}."

    # ── Internal HTTP / audit ───────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        action: str,
    ) -> Any:
        """Execute one read-only request and audit metadata."""
        method_upper = method.upper()
        if method_upper != "GET":
            raise Trading212ConfigError(
                "Trading 212 adapter is read-only in this phase and refuses non-GET requests."
            )

        response = self._client.request(method_upper, path, params=params, json=json_body)
        rate_limit = _rate_limit_headers(response)
        self._audit(
            action,
            {
                "method": method_upper,
                "path": path,
                "params": params or {},
                "status_code": response.status_code,
                "rate_limit": rate_limit,
            },
        )

        if response.status_code >= 400:
            body = response.text[:300] if response.text else response.reason_phrase
            raise Trading212ApiError(response.status_code, body, path=path)

        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise Trading212ApiError(response.status_code, f"Invalid JSON: {exc}", path=path) from exc

    def _audit(self, action: str, payload: dict[str, Any]) -> None:
        """Append a redacted JSONL event for development/debugging."""
        event = {
            "timestamp": _utc_now(),
            "broker": "trading212",
            "environment": self.config.environment,
            "base_url_host": urlparse(self.config.base_url).netloc,
            "action": action,
            **payload,
        }
        _append_jsonl(self.config.audit_log_path, event)


def _setting(name: str, default: Any) -> Any:
    """Read a setting from config.py, then environment variables."""
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [item for item in data["items"] if isinstance(item, dict)]
    return []


def _rate_limit_headers(response: httpx.Response) -> dict[str, str]:
    keys = (
        "x-ratelimit-limit",
        "x-ratelimit-period",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-used",
    )
    return {key: response.headers[key] for key in keys if key in response.headers}


def _first_number(*dicts: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for data in dicts:
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _first_text(*dicts: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for data in dicts:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _format_money(value: float, currency: str = "") -> str:
    symbol = {"GBP": "£", "USD": "$", "EUR": "€"}.get(currency.upper(), "")
    suffix = "" if symbol else (f" {currency}" if currency else "")
    return f"{symbol}{value:,.2f}{suffix}"


def _instrument_ticker(item: dict[str, Any]) -> str | None:
    instrument = item.get("instrument")
    if isinstance(instrument, dict):
        ticker = instrument.get("ticker")
        if isinstance(ticker, str) and ticker:
            return ticker
    ticker = item.get("ticker")
    return ticker if isinstance(ticker, str) and ticker else None
