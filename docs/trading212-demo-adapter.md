# Trading 212 Demo Adapter

Aria's Trading 212 integration is intentionally read-only in this phase.

The adapter connects only to:

```text
https://demo.trading212.com/api/v0
```

Live-money API access is blocked in code. Order placement is not implemented in
this branch.

## Configuration

Add these values to `.env` or `config.py`:

```env
TRADING212_ENV=demo
TRADING212_BASE_URL=https://demo.trading212.com/api/v0
TRADING212_API_KEY=your_demo_api_key
TRADING212_API_SECRET=your_demo_api_secret
TRADING212_TIMEOUT_SECONDS=10
TRADING212_AUDIT_LOG_PATH=data/broker/trading212_audit.jsonl
TRADING212_TRAINING_LOG_PATH=data/broker/trading212_account_state.jsonl
```

Do not commit credentials. `config.py`, `.env`, and `data/broker/` are ignored.

## Voice Commands

Examples:

```text
Aria, check my Trading 212 demo account.
Aria, what is my paper account cash balance?
Aria, show my demo positions.
Aria, any pending paper orders?
```

## Current Scope

Implemented:

- account cash
- account summary
- open positions
- pending orders
- exchange metadata
- instrument metadata
- one-page historical orders
- paginated list helper
- local JSONL audit metadata
- local JSONL account-state training snapshots

Not implemented:

- market orders
- limit orders
- stop orders
- live environment support
- autonomous trade execution

## Safety Rules

- Non-GET requests are refused by the client.
- `live.trading212.com` is rejected.
- `TRADING212_ENV` must be `demo`.
- Audit logs never include API key, API secret, or Authorization headers.
- Training snapshots are local/private under `data/broker/` and are ignored by Git.

## Next Phase

The next branch can add paper-order execution, but only with:

- explicit voice confirmation
- max order size
- max position size
- duplicate order prevention
- immutable order-intent IDs
- full proposal -> approval -> request -> response audit trail
