"""
core/market_analyst.py
----------------------
Stock market snapshot generator for Aria.

Fetches daily OHLCV data for a configurable list of tickers via yfinance,
computes a small set of indicators (moving averages, daily change,
30-day volatility, volume vs average), surfaces unusual moves
(>2σ price moves, volume spikes, MA crossovers), and emits both:

  - structured snapshot:  data/market/YYYY-MM-DD.json
  - spoken summary:       1-3 sentence string for Aria's TTS

Voice trigger:  "Aria, market update" → core.router 'market' intent
                → core.brain._handle_market_query()

MVP scope: on-demand only (no scheduled background fetch),
anomalies surfaced inside the spoken summary (no proactive comment).

Design split:
  - Pure functions (no network):
        _compute_indicators(df, ticker) -> dict
        _detect_anomalies(snapshot)     -> list[dict]
        _build_summary(snapshot, short) -> str
    Tested directly in tests/test_market_analyst.py.
  - I/O layer (network + disk):
        _fetch_ticker_data(ticker)      -> pd.DataFrame
        fetch_snapshot()                -> dict
        save_snapshot(snapshot)         -> Path
    Network code is kept thin so failures degrade per-ticker.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.logger import get_logger

log = get_logger(__name__)

# ── Defaults (overridable via config.py) ──────────────────────────────────────

DEFAULT_TICKERS              = ["AAPL", "MSFT", "NVDA", "TSLA"]
DEFAULT_VOLUME_SPIKE_PCT     = 30.0   # flag if today's volume > 30-day avg + 30%
DEFAULT_VOLATILITY_SIGMA     = 2.0    # flag if today's return > 2σ of last 30 days
DEFAULT_HISTORY_DAYS         = 60     # request 60 calendar days to safely cover 30 trading days

SNAPSHOT_DIR = Path("data/market")


def _load_config() -> dict:
    """Read market settings from config.py with safe defaults.

    Returns:
        dict with keys: tickers, volume_spike_pct, volatility_sigma.
    """
    try:
        import config
        return {
            "tickers":          getattr(config, "MARKET_TICKERS", DEFAULT_TICKERS),
            "volume_spike_pct": getattr(config, "MARKET_VOLUME_SPIKE_PCT", DEFAULT_VOLUME_SPIKE_PCT),
            "volatility_sigma": getattr(config, "MARKET_VOLATILITY_SIGMA", DEFAULT_VOLATILITY_SIGMA),
        }
    except ImportError:
        return {
            "tickers":          DEFAULT_TICKERS,
            "volume_spike_pct": DEFAULT_VOLUME_SPIKE_PCT,
            "volatility_sigma": DEFAULT_VOLATILITY_SIGMA,
        }


# ── Pure indicator math (network-free) ────────────────────────────────────────

def _compute_indicators(df, ticker: str) -> dict[str, Any]:
    """Compute the indicator dict for a single ticker.

    Args:
        df: pandas DataFrame from yfinance.history() with columns
            'Close' and 'Volume', indexed by date, sorted ascending.
            Must contain at least 2 rows for daily change to be meaningful;
            longer history is needed for the moving averages.
        ticker: Ticker symbol — included in the output dict for downstream
                consumers.

    Returns:
        dict with all indicators. Keys present on every successful call:
            ticker, current_price, previous_close, daily_change_pct,
            ma_7, ma_30, above_ma_7, above_ma_30, ma_crossover,
            volatility_30d, volume, volume_30d_avg, volume_change_pct,
            data_points.

        Values default to None when there isn't enough history for that
        particular indicator (e.g. ma_30 is None with < 30 rows). Callers
        check for None before formatting into the spoken summary.
    """
    n = len(df)
    closes  = df["Close"]
    volumes = df["Volume"]

    current_price  = float(closes.iloc[-1])
    previous_close = float(closes.iloc[-2]) if n >= 2 else None

    daily_change_pct = (
        ((current_price - previous_close) / previous_close) * 100.0
        if previous_close
        else None
    )

    ma_7  = float(closes.iloc[-7:].mean())  if n >= 7  else None
    ma_30 = float(closes.iloc[-30:].mean()) if n >= 30 else None

    above_ma_7  = (current_price > ma_7)  if ma_7  is not None else None
    above_ma_30 = (current_price > ma_30) if ma_30 is not None else None

    # MA crossover detection: compare today's MAs vs yesterday's MAs.
    # Only meaningful when both MAs are computable for both days
    # (so we need at least 31 rows of history).
    ma_crossover = None
    if n >= 31:
        ma_7_yest  = float(closes.iloc[-8:-1].mean())
        ma_30_yest = float(closes.iloc[-31:-1].mean())
        if ma_7_yest <= ma_30_yest and ma_7 > ma_30:
            ma_crossover = "bullish"  # 7-day crossed above 30-day
        elif ma_7_yest >= ma_30_yest and ma_7 < ma_30:
            ma_crossover = "bearish"  # 7-day crossed below 30-day

    # 30-day volatility = stddev of daily returns over last 30 trading days.
    volatility_30d = None
    if n >= 31:
        recent_closes = closes.iloc[-31:]
        returns       = recent_closes.pct_change().dropna()
        volatility_30d = float(returns.std())

    volume          = int(volumes.iloc[-1])
    volume_30d_avg  = float(volumes.iloc[-30:].mean()) if n >= 30 else None
    volume_change_pct = (
        ((volume - volume_30d_avg) / volume_30d_avg) * 100.0
        if volume_30d_avg
        else None
    )

    return {
        "ticker":            ticker,
        "current_price":     round(current_price, 2),
        "previous_close":    round(previous_close, 2) if previous_close is not None else None,
        "daily_change_pct":  round(daily_change_pct, 2) if daily_change_pct is not None else None,
        "ma_7":              round(ma_7, 2)  if ma_7  is not None else None,
        "ma_30":             round(ma_30, 2) if ma_30 is not None else None,
        "above_ma_7":        above_ma_7,
        "above_ma_30":       above_ma_30,
        "ma_crossover":      ma_crossover,
        "volatility_30d":    round(volatility_30d, 4) if volatility_30d is not None else None,
        "volume":            volume,
        "volume_30d_avg":    round(volume_30d_avg, 0) if volume_30d_avg is not None else None,
        "volume_change_pct": round(volume_change_pct, 1) if volume_change_pct is not None else None,
        "data_points":       n,
    }


def _detect_anomalies(
    snapshot: dict,
    volume_spike_pct: float = DEFAULT_VOLUME_SPIKE_PCT,
    volatility_sigma: float = DEFAULT_VOLATILITY_SIGMA,
) -> list[dict[str, Any]]:
    """Surface unusual movements across the snapshot.

    Args:
        snapshot: snapshot dict produced by fetch_snapshot().
        volume_spike_pct: volume must exceed 30-day avg by this percentage
                          (default 30.0).
        volatility_sigma: today's daily change in absolute % must exceed
                          this many σ of the 30-day return distribution.
                          σ comes from `volatility_30d` which is in
                          fractional units (0.024 = 2.4%), so we multiply
                          by 100 to compare with daily_change_pct.

    Returns:
        List of anomaly dicts with keys: ticker, kind, detail.
        Empty list when nothing is unusual.
    """
    anomalies: list[dict[str, Any]] = []

    for ticker, data in snapshot.get("tickers", {}).items():
        change_pct = data.get("daily_change_pct")
        sigma_pct  = data.get("volatility_30d")
        sigma_pct_in_pct = (sigma_pct * 100.0) if sigma_pct is not None else None

        # >Nσ price move
        if (
            change_pct is not None
            and sigma_pct_in_pct is not None
            and sigma_pct_in_pct > 0
            and abs(change_pct) >= volatility_sigma * sigma_pct_in_pct
        ):
            direction = "up" if change_pct > 0 else "down"
            anomalies.append({
                "ticker": ticker,
                "kind":   "large_move",
                "detail": (
                    f"{ticker} moved {direction} {abs(change_pct):.1f}%, "
                    f"more than {volatility_sigma:g}σ of its 30-day daily range "
                    f"(σ ≈ {sigma_pct_in_pct:.1f}%)."
                ),
            })

        # Volume spike
        vol_change_pct = data.get("volume_change_pct")
        if vol_change_pct is not None and vol_change_pct >= volume_spike_pct:
            anomalies.append({
                "ticker": ticker,
                "kind":   "volume_spike",
                "detail": (
                    f"{ticker} volume is {vol_change_pct:.0f}% above its "
                    f"30-day average."
                ),
            })

        # MA crossover
        crossover = data.get("ma_crossover")
        if crossover == "bullish":
            anomalies.append({
                "ticker": ticker,
                "kind":   "bullish_crossover",
                "detail": f"{ticker} 7-day moving average crossed above its 30-day — bullish signal.",
            })
        elif crossover == "bearish":
            anomalies.append({
                "ticker": ticker,
                "kind":   "bearish_crossover",
                "detail": f"{ticker} 7-day moving average crossed below its 30-day — bearish signal.",
            })

    return anomalies


def _format_ticker_sentence(data: dict) -> str:
    """One-sentence ticker line for the full-mode spoken summary.

    Falls back to a minimal price-only sentence when MAs / change %
    aren't available (e.g. very fresh ticker with < 7 days of history).

    Args:
        data: per-ticker indicator dict from _compute_indicators.

    Returns:
        A single-sentence plain-English string.
    """
    ticker = data["ticker"]
    price  = data.get("current_price")
    change = data.get("daily_change_pct")
    ma_7   = data.get("ma_7")
    ma_30  = data.get("ma_30")
    vol_change = data.get("volume_change_pct")

    if change is None or price is None:
        return f"{ticker} is at {price}." if price is not None else f"{ticker} data unavailable."

    direction = "up" if change >= 0 else "down"
    parts = [f"{ticker} is {direction} {abs(change):.1f}% at {price:.2f}"]

    if ma_7 is not None and ma_30 is not None:
        if price > ma_7 and price > ma_30:
            parts.append("trading above both its 7-day and 30-day moving averages")
        elif price < ma_7 and price < ma_30:
            parts.append("trading below both its 7-day and 30-day moving averages")
        elif price > ma_7:
            parts.append("above the 7-day but below the 30-day moving average")
        else:
            parts.append("below the 7-day but above the 30-day moving average")

    if vol_change is not None:
        if vol_change >= 10:
            parts.append(f"volume {vol_change:.0f}% above its monthly average")
        elif vol_change <= -10:
            parts.append(f"volume {abs(vol_change):.0f}% below its monthly average")

    return ", ".join(parts) + "."


def _build_summary(snapshot: dict, short: bool = True) -> str:
    """Compose the spoken summary string.

    Short mode (default) leads with anomalies if any, then the single
    biggest gainer + loser. Aimed at ~5–10 seconds of TTS.

    Full mode emits one sentence per ticker. Aimed at ~30 seconds.

    Args:
        snapshot: snapshot dict from fetch_snapshot().
        short:    True = short mode, False = full per-ticker breakdown.

    Returns:
        Plain text suitable for Aria's TTS — no markdown, no newlines.
    """
    tickers   = snapshot.get("tickers", {})
    anomalies = snapshot.get("anomalies", [])
    errors    = snapshot.get("errors", {})

    if not tickers and not errors:
        return "I couldn't fetch any market data right now, Chan."

    fragments: list[str] = []

    # Errors first — flag them so Chan knows the picture is incomplete.
    if errors:
        n = len(errors)
        names = ", ".join(errors.keys())
        fragments.append(
            f"Heads up Chan, I couldn't get data for {n} ticker{'s' if n > 1 else ''}: {names}."
        )

    if not tickers:
        return " ".join(fragments)

    # Anomalies always lead — they're the whole point of the module.
    if anomalies:
        if short and len(anomalies) > 3:
            # Cap to top 3 in short mode to keep TTS bounded.
            for a in anomalies[:3]:
                fragments.append(a["detail"])
            fragments.append(f"Plus {len(anomalies) - 3} more flagged movements.")
        else:
            for a in anomalies:
                fragments.append(a["detail"])

    if short:
        # Biggest gainer + loser (excluding tickers without daily_change_pct).
        scored = [
            (t, d.get("daily_change_pct", 0) or 0)
            for t, d in tickers.items()
            if d.get("daily_change_pct") is not None
        ]
        if scored:
            biggest_gain = max(scored, key=lambda x: x[1])
            biggest_loss = min(scored, key=lambda x: x[1])
            if biggest_gain[0] != biggest_loss[0]:
                # Distinct tickers — mention both
                fragments.append(
                    f"Biggest mover up: {biggest_gain[0]} at {biggest_gain[1]:+.1f}%. "
                    f"Biggest down: {biggest_loss[0]} at {biggest_loss[1]:+.1f}%."
                )
            else:
                # Single ticker is both — shouldn't happen with > 1 ticker but defensive.
                fragments.append(
                    f"{biggest_gain[0]} moved {biggest_gain[1]:+.1f}% today."
                )
    else:
        # Full mode — one sentence per ticker.
        for ticker, data in tickers.items():
            fragments.append(_format_ticker_sentence(data))

    return " ".join(fragments)


# ── I/O layer (network + disk) ────────────────────────────────────────────────

class MarketAnalyst:
    """Fetches and analyses daily stock data for Aria's ticker list.

    On-demand only in this MVP — call fetch_snapshot() to get a fresh
    snapshot, save_snapshot() to persist, spoken_summary() to get TTS text.

    Attributes:
        config: dict with tickers + thresholds, loaded from config.py.
    """

    def __init__(self) -> None:
        self.config = _load_config()
        log.info(
            "MarketAnalyst ready — %d ticker(s): %s",
            len(self.config["tickers"]), ", ".join(self.config["tickers"]),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_snapshot(self, tickers: list[str] | None = None) -> dict[str, Any]:
        """Fetch + analyse all tickers, returning the structured snapshot.

        Never raises — yfinance failures, network errors, missing data, and
        rate limits are caught per-ticker. Failed tickers appear in the
        snapshot's `errors` dict so the spoken summary can flag them.

        Args:
            tickers: Optional override of the config ticker list.

        Returns:
            dict with keys:
                date            (str, YYYY-MM-DD)
                fetched_at      (ISO timestamp)
                tickers         (dict[str, indicator_dict])
                anomalies       (list of anomaly dicts)
                errors          (dict[str, error_message])
        """
        target = tickers if tickers is not None else self.config["tickers"]

        ticker_data: dict[str, dict] = {}
        errors:      dict[str, str]  = {}

        for ticker in target:
            try:
                df = self._fetch_ticker_data(ticker)
                if df is None or df.empty:
                    errors[ticker] = "no data returned"
                    log.warning("Ticker %s: no data returned from yfinance.", ticker)
                    continue
                indicators = _compute_indicators(df, ticker)
                ticker_data[ticker] = indicators
                log.info(
                    "Ticker %s: price %s, daily change %s%%",
                    ticker,
                    indicators.get("current_price"),
                    indicators.get("daily_change_pct"),
                )
            except Exception as e:
                errors[ticker] = str(e)
                log.error("Ticker %s fetch failed: %s", ticker, e)

        snapshot = {
            "date":       datetime.now().strftime("%Y-%m-%d"),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "tickers":    ticker_data,
            "errors":     errors,
        }
        snapshot["anomalies"] = _detect_anomalies(
            snapshot,
            volume_spike_pct=self.config["volume_spike_pct"],
            volatility_sigma=self.config["volatility_sigma"],
        )

        log.info(
            "Snapshot complete — %d ticker(s) ok, %d failed, %d anomalies.",
            len(ticker_data), len(errors), len(snapshot["anomalies"]),
        )
        return snapshot

    def spoken_summary(
        self,
        snapshot: dict | None = None,
        short: bool = True,
    ) -> str:
        """Produce a TTS-ready summary of the snapshot.

        If `snapshot` is None, fetches a fresh one first.

        Args:
            snapshot: optional pre-fetched snapshot.
            short:    True = anomalies + biggest movers only (~5–10s TTS),
                      False = one sentence per ticker (~30s).

        Returns:
            Plain-text spoken summary string.
        """
        if snapshot is None:
            snapshot = self.fetch_snapshot()
        return _build_summary(snapshot, short=short)

    def save_snapshot(self, snapshot: dict) -> Path:
        """Persist a snapshot to data/market/YYYY-MM-DD.json.

        Overwrites any existing snapshot for the same date so multiple
        intra-day fetches keep only the latest.

        Args:
            snapshot: snapshot dict from fetch_snapshot().

        Returns:
            Path to the file written.
        """
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        date_str = snapshot.get("date") or datetime.now().strftime("%Y-%m-%d")
        path     = SNAPSHOT_DIR / f"{date_str}.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        log.info("Snapshot saved: %s", path)
        return path

    def detect_anomalies(self, snapshot: dict) -> list[dict[str, Any]]:
        """Public anomaly-detection entry point — wraps the pure helper."""
        return _detect_anomalies(
            snapshot,
            volume_spike_pct=self.config["volume_spike_pct"],
            volatility_sigma=self.config["volatility_sigma"],
        )

    # ── Network layer ─────────────────────────────────────────────────────────

    def _fetch_ticker_data(self, ticker: str):
        """Fetch the recent daily history for a ticker via yfinance.

        Kept narrow so unit tests can monkeypatch this single method to
        bypass the network entirely.

        Args:
            ticker: Ticker symbol e.g. 'AAPL'.

        Returns:
            pandas DataFrame with at least Close + Volume columns,
            sorted ascending by date. None on hard failure.
        """
        import yfinance as yf
        # auto_adjust=True applies splits + dividends so price math is consistent.
        df = yf.Ticker(ticker).history(period=f"{DEFAULT_HISTORY_DAYS}d", auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
