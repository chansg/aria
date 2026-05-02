"""
tests/test_market_analyst.py
----------------------------
Unit tests for core.market_analyst.

Network-free: yfinance is never called. Tests exercise the pure
indicator math, anomaly detection, and summary text generation
directly with hand-crafted pandas DataFrames and snapshot dicts.

Run from project root:
    pytest tests/test_market_analyst.py -v
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.market_analyst import (
    MarketAnalyst,
    _build_summary,
    _compute_indicators,
    _detect_anomalies,
    _format_ticker_sentence,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic ticker data
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """Build a yfinance-shaped DataFrame from a list of closes.

    Volume defaults to a flat 1_000_000 per day if not supplied.
    """
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n
    dates = pd.date_range(end="2026-04-30", periods=n, freq="D")
    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_indicators
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIndicators:
    """Indicator math is the heart of the module — verify per-field."""

    def test_basic_two_day(self) -> None:
        """With only 2 rows, only daily change is computable. MAs are None."""
        df = _make_df([100.0, 110.0])
        out = _compute_indicators(df, "TEST")

        assert out["ticker"]            == "TEST"
        assert out["current_price"]     == 110.0
        assert out["previous_close"]    == 100.0
        assert out["daily_change_pct"]  == 10.0
        assert out["ma_7"]              is None
        assert out["ma_30"]             is None
        assert out["above_ma_7"]        is None
        assert out["volatility_30d"]    is None
        assert out["data_points"]       == 2

    def test_ma_7_with_exactly_7_rows(self) -> None:
        """Average of 7 closes from 100 to 106 is 103."""
        df = _make_df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        out = _compute_indicators(df, "TEST")
        assert out["ma_7"]        == 103.0
        assert out["ma_30"]       is None  # only 7 rows
        assert out["above_ma_7"]  is True   # 106 > 103

    def test_full_30_day_history(self) -> None:
        """Flat history at 100 → MAs both 100, change 0, volume change 0."""
        df = _make_df([100.0] * 31)
        out = _compute_indicators(df, "FLAT")
        assert out["ma_7"]              == 100.0
        assert out["ma_30"]             == 100.0
        assert out["above_ma_7"]        is False  # equal, not above
        assert out["above_ma_30"]       is False
        assert out["daily_change_pct"]  == 0.0
        assert out["volume_change_pct"] == 0.0
        # All daily returns are 0 → volatility is 0
        assert out["volatility_30d"]    == 0.0

    def test_volatility_nonzero(self) -> None:
        """Alternating ±1% returns → non-zero volatility."""
        # 31 closes that alternate around 100
        closes = [100.0]
        for i in range(30):
            closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
        df = _make_df(closes)
        out = _compute_indicators(df, "VOL")
        assert out["volatility_30d"] is not None
        assert out["volatility_30d"] > 0.005  # ~0.01 expected

    def test_volume_spike(self) -> None:
        """Today's volume 2× the average → 100% volume change."""
        volumes  = [1_000_000] * 30 + [2_000_000]
        closes   = [100.0] * 31
        df       = _make_df(closes, volumes)
        out      = _compute_indicators(df, "VS")
        assert out["volume"]          == 2_000_000
        # 30-day avg over the LAST 30 rows = (29*1M + 1*2M)/30 ≈ 1.033M
        assert math.isclose(out["volume_30d_avg"], (29 * 1_000_000 + 2_000_000) / 30, rel_tol=1e-6)
        assert out["volume_change_pct"] is not None
        assert out["volume_change_pct"] > 90  # ~93.5%

    def test_bullish_crossover(self) -> None:
        """Long flat history then a sharp climb → 7-day MA crosses above 30-day."""
        # 30 days at 100, then sharp climb on day 31
        closes = [100.0] * 30 + [200.0]
        df = _make_df(closes)
        out = _compute_indicators(df, "X")
        # Today: ma_7 = (5*100 + 200)/7? Wait — last 7 closes are [100,100,100,100,100,100,200]
        # ma_7 today = (6*100 + 200)/7 = 114.3
        # ma_30 today = (29*100 + 200)/30 = 103.3
        # Yesterday (closes 1..30, 30 rows): ma_7_yest = 100 (last 7 of those 30 are all 100), ma_30_yest = 100
        # So 100 ≤ 100 (yesterday) and ma_7 (114.3) > ma_30 (103.3) (today) → bullish
        assert out["ma_crossover"] == "bullish"

    def test_bearish_crossover(self) -> None:
        """Long flat history then a sharp drop → 7-day MA crosses below 30-day."""
        closes = [100.0] * 30 + [50.0]
        df = _make_df(closes)
        out = _compute_indicators(df, "X")
        assert out["ma_crossover"] == "bearish"

    def test_no_crossover_in_steady_uptrend(self) -> None:
        """If 7-day was already above 30-day yesterday, today's still-above isn't a crossover."""
        # Steady uptrend means the 7-day has been above the 30-day for many days
        closes = [100.0 + i for i in range(40)]  # 100, 101, ..., 139
        df = _make_df(closes)
        out = _compute_indicators(df, "UP")
        # 7-day MA averages the most recent values; 30-day averages older ones
        assert out["ma_7"] > out["ma_30"]  # confirms 7d > 30d today
        assert out["ma_crossover"] is None  # no flip — was already above


# ─────────────────────────────────────────────────────────────────────────────
# _detect_anomalies
# ─────────────────────────────────────────────────────────────────────────────

def _ticker_with(**overrides) -> dict:
    """Build a minimal indicator dict, overriding fields per test."""
    base = {
        "ticker":            "TST",
        "current_price":     100.0,
        "daily_change_pct":  0.5,
        "volatility_30d":    0.01,   # 1% daily σ
        "volume":            1_000_000,
        "volume_30d_avg":    1_000_000,
        "volume_change_pct": 0.0,
        "ma_crossover":      None,
    }
    base.update(overrides)
    return base


class TestDetectAnomalies:
    """Anomaly detection — independent thresholds verified one at a time."""

    def test_no_anomalies_when_quiet(self) -> None:
        snapshot = {"tickers": {"TST": _ticker_with()}}
        assert _detect_anomalies(snapshot) == []

    def test_large_move_above_2_sigma(self) -> None:
        """Daily change of 3% with σ of 1% → 3σ → flagged."""
        snapshot = {"tickers": {"TST": _ticker_with(daily_change_pct=3.0)}}
        anomalies = _detect_anomalies(snapshot, volatility_sigma=2.0)
        assert len(anomalies) == 1
        assert anomalies[0]["kind"] == "large_move"
        assert "TST" in anomalies[0]["detail"]

    def test_large_move_below_threshold_not_flagged(self) -> None:
        """1.5% change with σ of 1% → 1.5σ → NOT flagged at 2σ threshold."""
        snapshot = {"tickers": {"TST": _ticker_with(daily_change_pct=1.5)}}
        assert _detect_anomalies(snapshot, volatility_sigma=2.0) == []

    def test_volume_spike(self) -> None:
        snapshot = {"tickers": {"TST": _ticker_with(volume_change_pct=50.0)}}
        anomalies = _detect_anomalies(snapshot, volume_spike_pct=30.0)
        assert any(a["kind"] == "volume_spike" for a in anomalies)

    def test_volume_change_below_threshold_not_flagged(self) -> None:
        snapshot = {"tickers": {"TST": _ticker_with(volume_change_pct=15.0)}}
        anomalies = _detect_anomalies(snapshot, volume_spike_pct=30.0)
        assert not any(a["kind"] == "volume_spike" for a in anomalies)

    def test_bullish_crossover_flagged(self) -> None:
        snapshot = {"tickers": {"TST": _ticker_with(ma_crossover="bullish")}}
        anomalies = _detect_anomalies(snapshot)
        assert any(a["kind"] == "bullish_crossover" for a in anomalies)

    def test_bearish_crossover_flagged(self) -> None:
        snapshot = {"tickers": {"TST": _ticker_with(ma_crossover="bearish")}}
        anomalies = _detect_anomalies(snapshot)
        assert any(a["kind"] == "bearish_crossover" for a in anomalies)

    def test_multiple_anomalies_per_ticker(self) -> None:
        """A single ticker can trip multiple anomaly types in the same snapshot."""
        snapshot = {"tickers": {"TST": _ticker_with(
            daily_change_pct=3.0,
            volume_change_pct=50.0,
            ma_crossover="bullish",
        )}}
        anomalies = _detect_anomalies(snapshot)
        kinds = {a["kind"] for a in anomalies}
        assert kinds == {"large_move", "volume_spike", "bullish_crossover"}

    def test_missing_volatility_does_not_raise(self) -> None:
        """If volatility_30d is None (insufficient history), large_move check skips."""
        snapshot = {"tickers": {"TST": _ticker_with(
            volatility_30d=None,
            daily_change_pct=10.0,
        )}}
        anomalies = _detect_anomalies(snapshot)
        assert not any(a["kind"] == "large_move" for a in anomalies)


# ─────────────────────────────────────────────────────────────────────────────
# Summary text generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryGeneration:
    """Spoken summary — verifies content, not exact phrasing."""

    def test_empty_snapshot(self) -> None:
        snapshot = {"tickers": {}, "anomalies": [], "errors": {}}
        out = _build_summary(snapshot, short=True)
        assert "couldn't fetch" in out.lower()

    def test_short_mode_leads_with_anomaly(self) -> None:
        snapshot = {
            "tickers": {
                "AAPL": _ticker_with(ticker="AAPL", daily_change_pct=0.5),
                "NVDA": _ticker_with(ticker="NVDA", daily_change_pct=-1.0),
            },
            "anomalies": [{
                "ticker": "NVDA",
                "kind":   "large_move",
                "detail": "NVDA moved down 5%, more than 2σ.",
            }],
            "errors": {},
        }
        out = _build_summary(snapshot, short=True)
        # Anomaly appears, biggest movers section also appears
        assert "NVDA moved down" in out
        assert "Biggest mover up" in out

    def test_full_mode_one_sentence_per_ticker(self) -> None:
        snapshot = {
            "tickers": {
                "AAPL": _ticker_with(ticker="AAPL", current_price=150.0, daily_change_pct=1.0,
                                     ma_7=148.0, ma_30=145.0),
                "MSFT": _ticker_with(ticker="MSFT", current_price=300.0, daily_change_pct=-0.5,
                                     ma_7=302.0, ma_30=298.0),
            },
            "anomalies": [],
            "errors": {},
        }
        out = _build_summary(snapshot, short=False)
        assert "AAPL is up 1.0%" in out
        assert "MSFT is down 0.5%" in out

    def test_short_mode_caps_anomalies_at_three(self) -> None:
        """When 4+ anomalies fire, short mode lists 3 + 'plus N more' summary."""
        anomalies = [
            {"ticker": f"T{i}", "kind": "large_move", "detail": f"T{i} detail."}
            for i in range(5)
        ]
        snapshot = {
            "tickers":   {f"T{i}": _ticker_with(ticker=f"T{i}", daily_change_pct=1.0) for i in range(5)},
            "anomalies": anomalies,
            "errors":    {},
        }
        out = _build_summary(snapshot, short=True)
        # First 3 detailed, the rest collapsed
        assert "T0 detail." in out
        assert "T1 detail." in out
        assert "T2 detail." in out
        assert "T3 detail." not in out
        assert "Plus 2 more flagged movements." in out

    def test_errors_flagged_in_summary(self) -> None:
        snapshot = {
            "tickers":   {"AAPL": _ticker_with(ticker="AAPL", daily_change_pct=1.0)},
            "anomalies": [],
            "errors":    {"ZZZZ": "no data returned"},
        }
        out = _build_summary(snapshot, short=True)
        assert "ZZZZ" in out
        assert "couldn't get data" in out.lower() or "couldn t get data" in out.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _format_ticker_sentence
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatTickerSentence:
    """Per-ticker sentence formatting."""

    def test_above_both_mas(self) -> None:
        data = _ticker_with(ticker="NVDA", current_price=500.32, daily_change_pct=2.3,
                            ma_7=495.20, ma_30=480.10, volume_change_pct=18.0)
        sent = _format_ticker_sentence(data)
        assert "NVDA is up 2.3%" in sent
        assert "above both" in sent
        assert "18% above" in sent

    def test_below_both_mas(self) -> None:
        data = _ticker_with(ticker="X", current_price=90.0, daily_change_pct=-3.0,
                            ma_7=95.0, ma_30=100.0, volume_change_pct=-15.0)
        sent = _format_ticker_sentence(data)
        assert "is down 3.0%" in sent
        assert "below both" in sent
        assert "15% below" in sent

    def test_no_mas_falls_back_to_price_only(self) -> None:
        data = _ticker_with(ticker="NEW", current_price=42.0, daily_change_pct=1.0,
                            ma_7=None, ma_30=None, volume_change_pct=None)
        sent = _format_ticker_sentence(data)
        # Should still include the price + change
        assert "NEW is up 1.0%" in sent
        assert "42.00" in sent

    def test_missing_change_pct(self) -> None:
        data = _ticker_with(ticker="T", current_price=10.0, daily_change_pct=None)
        sent = _format_ticker_sentence(data)
        assert "T is at 10.0" in sent


# ─────────────────────────────────────────────────────────────────────────────
# MarketAnalyst end-to-end (with mocked network)
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketAnalyst:
    """Verify the public API. Network is monkeypatched; never called for real."""

    def test_fetch_snapshot_returns_expected_shape(self, monkeypatch) -> None:
        """fetch_snapshot orchestrates fetch + compute + anomaly detection."""
        analyst = MarketAnalyst()

        # Stub _fetch_ticker_data on this instance only
        df = _make_df([100.0] * 31)
        monkeypatch.setattr(analyst, "_fetch_ticker_data", lambda ticker: df)

        snap = analyst.fetch_snapshot(tickers=["AAPL", "MSFT"])

        assert "date" in snap
        assert "fetched_at" in snap
        assert set(snap["tickers"].keys()) == {"AAPL", "MSFT"}
        assert snap["errors"] == {}
        assert snap["anomalies"] == []  # flat history, no anomalies

    def test_fetch_snapshot_handles_per_ticker_failure(self, monkeypatch) -> None:
        """One bad ticker shouldn't poison the whole snapshot."""
        analyst = MarketAnalyst()

        good_df = _make_df([100.0] * 31)

        def fake_fetch(ticker: str):
            if ticker == "BROKEN":
                raise RuntimeError("simulated network failure")
            return good_df

        monkeypatch.setattr(analyst, "_fetch_ticker_data", fake_fetch)

        snap = analyst.fetch_snapshot(tickers=["AAPL", "BROKEN", "MSFT"])

        assert set(snap["tickers"].keys())  == {"AAPL", "MSFT"}
        assert "BROKEN" in snap["errors"]
        assert "simulated network failure" in snap["errors"]["BROKEN"]

    def test_fetch_snapshot_handles_empty_data(self, monkeypatch) -> None:
        """Empty DataFrame should be reported as a no-data error."""
        analyst = MarketAnalyst()

        monkeypatch.setattr(analyst, "_fetch_ticker_data", lambda ticker: None)

        snap = analyst.fetch_snapshot(tickers=["AAPL"])
        assert snap["tickers"] == {}
        assert "AAPL" in snap["errors"]

    def test_save_snapshot_writes_json(self, tmp_path, monkeypatch) -> None:
        """save_snapshot writes valid JSON to data/market/<date>.json."""
        # Redirect SNAPSHOT_DIR to a tmp directory for the test
        import core.market_analyst as ma
        monkeypatch.setattr(ma, "SNAPSHOT_DIR", tmp_path)

        analyst = MarketAnalyst()
        snap = {
            "date": "2026-04-30",
            "fetched_at": "2026-04-30T12:00:00",
            "tickers": {"AAPL": _ticker_with(ticker="AAPL")},
            "anomalies": [],
            "errors": {},
        }
        path = analyst.save_snapshot(snap)

        assert path.exists()
        assert path.name == "2026-04-30.json"
        # Round-trips
        import json
        loaded = json.loads(path.read_text())
        assert loaded["date"] == "2026-04-30"
        assert "AAPL" in loaded["tickers"]

    def test_spoken_summary_does_not_raise_on_empty_fetch(self, monkeypatch) -> None:
        """spoken_summary must NEVER raise — degrades to a graceful message."""
        analyst = MarketAnalyst()
        monkeypatch.setattr(analyst, "_fetch_ticker_data", lambda ticker: None)
        out = analyst.spoken_summary()
        assert isinstance(out, str)
        assert len(out) > 0
