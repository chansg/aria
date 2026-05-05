"""
Project Aria validation harness.

Runs deterministic, non-interactive health checks and writes a structured
session-review report under data/session_reviews/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPORT_DIR = ROOT / "data" / "session_reviews"


@dataclass
class ValidationItem:
    """One validation check result."""

    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def _item(name: str, status: str, summary: str, **details: Any) -> ValidationItem:
    return ValidationItem(name=name, status=status, summary=summary, details=details)


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception as exc:
        return f"unavailable ({exc})"
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"unavailable ({output or completed.returncode})"
    return output


def run_pytest(timeout_seconds: int = 180) -> ValidationItem:
    """Run the local test suite."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _item(
            "pytest",
            "fail",
            f"pytest timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    except Exception as exc:
        return _item("pytest", "fail", f"pytest could not run: {exc}")

    output = (completed.stdout or "").strip()
    error_output = (completed.stderr or "").strip()
    status = "pass" if completed.returncode == 0 else "fail"
    summary = "pytest passed" if status == "pass" else "pytest failed"
    return _item(
        "pytest",
        status,
        summary,
        returncode=completed.returncode,
        stdout=output[-4000:],
        stderr=error_output[-4000:],
    )


def check_config() -> ValidationItem:
    """Validate required local config shape without exposing secrets."""
    try:
        config = importlib.import_module("config")
    except Exception as exc:
        return _item("config", "fail", f"config.py could not be imported: {exc}")

    required = {
        "TTS_PROVIDER": "kokoro",
        "TTS_CONVERSATION_PROVIDER": "kokoro",
        "TTS_FALLBACK_PROVIDER": "",
        "TTS_FAIL_LOUD": True,
        "KOKORO_ONNX_PROVIDER": "CUDAExecutionProvider",
        "KOKORO_DISABLE_PROVIDER_FALLBACK": True,
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for field_name, expected in required.items():
        actual = getattr(config, field_name, None)
        if actual != expected:
            mismatches[field_name] = {"expected": expected, "actual": actual}

    path_fields = ("KOKORO_ONNX_MODEL_PATH", "KOKORO_ONNX_VOICES_PATH", "PIPER_MODEL_PATH")
    paths = {
        field_name: {
            "path": str(getattr(config, field_name, "")),
            "exists": Path(str(getattr(config, field_name, ""))).exists(),
        }
        for field_name in path_fields
        if hasattr(config, field_name)
    }

    missing_assets = [name for name, meta in paths.items() if not meta["exists"]]
    status = "pass"
    summary = "config shape is valid"
    if mismatches or missing_assets:
        status = "warn"
        summary = "config has non-blocking warnings"

    return _item(
        "config",
        status,
        summary,
        mismatches=mismatches,
        paths=paths,
        secrets={
            "ANTHROPIC_API_KEY_present": bool(getattr(config, "ANTHROPIC_API_KEY", "")),
            "GEMINI_API_KEY_present": bool(getattr(config, "GEMINI_API_KEY", "")),
            "TRADING212_API_KEY_present": bool(getattr(config, "TRADING212_API_KEY", "")),
            "TRADING212_API_SECRET_present": bool(getattr(config, "TRADING212_API_SECRET", "")),
        },
        voice={
            "KOKORO_SPEED": getattr(config, "KOKORO_SPEED", None),
            "TTS_MAX_CHUNK_CHARS": getattr(config, "TTS_MAX_CHUNK_CHARS", None),
            "TTS_TRIM_SILENCE": getattr(config, "TTS_TRIM_SILENCE", None),
            "TTS_SILENCE_THRESHOLD": getattr(config, "TTS_SILENCE_THRESHOLD", None),
            "TTS_SILENCE_PADDING_MS": getattr(config, "TTS_SILENCE_PADDING_MS", None),
        },
    )


def check_tts_provider(*, synthesize: bool = False) -> ValidationItem:
    """Check TTS provider routing, optionally running synthesis."""
    try:
        from voice.speaker import _provider_chain, _synthesize_with_fallback
    except Exception as exc:
        return _item("tts_provider", "fail", f"TTS provider imports failed: {exc}")

    chain = _provider_chain("kokoro")
    if chain != ["kokoro-onnx"]:
        return _item(
            "tts_provider",
            "fail",
            "conversation provider chain is not Kokoro-only",
            provider_chain=chain,
        )

    if not synthesize:
        return _item(
            "tts_provider",
            "pass",
            "Kokoro provider chain is configured without fallback",
            provider_chain=chain,
            synthesized=False,
        )

    try:
        result, provider = _synthesize_with_fallback("Validation voice check.", preferred_name="kokoro")
    except Exception as exc:
        return _item("tts_provider", "fail", f"Kokoro synthesis failed: {exc}", provider_chain=chain)

    sample_count = int(result.samples.size)
    duration = sample_count / int(result.sample_rate) if result.sample_rate else 0.0
    return _item(
        "tts_provider",
        "pass",
        "Kokoro synthesis succeeded",
        provider_chain=chain,
        provider=provider,
        sample_rate=result.sample_rate,
        sample_count=sample_count,
        duration_seconds=round(duration, 3),
        synthesized=True,
    )


def check_finance_routes() -> ValidationItem:
    """Check local finance routing and ticker extraction."""
    from core.conversation_state import remember_finance_quote, reset_conversation_state
    from core.market_analyst import extract_ticker_symbol
    from core.router import classify

    reset_conversation_state()
    route_cases = [
        ("What is the price of Apple stock?", {"intent": "stock_quote", "tier": 1}),
        ("What is the price of the S&P 500?", {"intent": "stock_quote", "tier": 1}),
        ("price of bitcoin", {"intent": "web_search", "tier": 2}),
        ("is this recent?", {"intent": "claude", "tier": 3}),
    ]
    route_results = []
    failures = []
    for text, expected in route_cases:
        actual = classify(text)
        route_results.append({"text": text, "expected": expected, "actual": actual})
        if actual != expected:
            failures.append({"text": text, "expected": expected, "actual": actual})

    remember_finance_quote({
        "ticker": "AAPL",
        "display_name": "AAPL",
        "price": 277.85,
        "as_of_date": "Monday 04 May 2026",
    })
    followup = classify("is this recent?")
    if followup != {"intent": "finance_followup", "tier": 1}:
        failures.append({
            "text": "is this recent? with finance context",
            "expected": {"intent": "finance_followup", "tier": 1},
            "actual": followup,
        })

    ticker_cases = {
        "What is the price of Apple stock?": "AAPL",
        "What is the price of the S&P 500?": "^GSPC",
        "What about Nvidia?": "NVDA",
    }
    ticker_results = []
    for text, expected in ticker_cases.items():
        actual = extract_ticker_symbol(text)
        ticker_results.append({"text": text, "expected": expected, "actual": actual})
        if actual != expected:
            failures.append({"text": text, "expected_ticker": expected, "actual_ticker": actual})

    reset_conversation_state()
    if failures:
        return _item(
            "finance_routes",
            "fail",
            "finance route checks failed",
            failures=failures,
            route_results=route_results,
            ticker_results=ticker_results,
        )
    return _item(
        "finance_routes",
        "pass",
        "finance routes and ticker extraction are valid",
        route_results=route_results,
        ticker_results=ticker_results,
    )


def check_trading212_safety() -> ValidationItem:
    """Check Trading 212 adapter is demo-only and read-only."""
    try:
        from core.brokers.trading212 import (
            DEMO_BASE_URL,
            LIVE_BASE_URL,
            Trading212Client,
            Trading212Config,
            Trading212ConfigError,
            Trading212LiveModeBlocked,
        )
    except Exception as exc:
        return _item("trading212_safety", "fail", f"Trading 212 imports failed: {exc}")

    failures = []
    safe_config = Trading212Config(
        base_url=DEMO_BASE_URL,
        api_key="demo-key",
        api_secret="demo-secret",
        environment="demo",
    )
    try:
        safe_config.validate_demo_only()
    except Exception as exc:
        failures.append(f"demo config rejected unexpectedly: {exc}")

    live_config = Trading212Config(
        base_url=LIVE_BASE_URL,
        api_key="key",
        api_secret="secret",
        environment="live",
    )
    try:
        live_config.validate_demo_only()
        failures.append("live config was not blocked")
    except Trading212LiveModeBlocked:
        pass

    client = Trading212Client(safe_config)
    try:
        try:
            client._request("POST", "/equity/orders/market", action="blocked_order")
            failures.append("non-GET request was not blocked")
        except Trading212ConfigError:
            pass
    finally:
        client.close()

    if failures:
        return _item("trading212_safety", "fail", "Trading 212 safety checks failed", failures=failures)
    return _item(
        "trading212_safety",
        "pass",
        "Trading 212 adapter is demo-only and refuses non-GET requests",
        demo_base_url=DEMO_BASE_URL,
    )


def build_report(items: list[ValidationItem]) -> dict[str, Any]:
    """Build the JSON report object."""
    failed = [item for item in items if item.status == "fail"]
    warned = [item for item in items if item.status == "warn"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "fail" if failed else "pass",
        "summary": {
            "passed": sum(1 for item in items if item.status == "pass"),
            "warned": len(warned),
            "failed": len(failed),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(ROOT),
            "git_branch": _run_git(["branch", "--show-current"]),
            "git_commit": _run_git(["rev-parse", "--short", "HEAD"]),
            "git_dirty": _run_git(["status", "--short"]),
        },
        "checks": [asdict(item) for item in items],
    }


def write_report(report: dict[str, Any], report_dir: Path = DEFAULT_REPORT_DIR) -> Path:
    """Write a validation report and return its path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = report_dir / f"validation-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def run_validation(
    *,
    skip_pytest: bool = False,
    pytest_timeout: int = 180,
    tts_synthesize: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Run all validation checks and write the report."""
    items: list[ValidationItem] = []
    if skip_pytest:
        items.append(_item("pytest", "warn", "pytest skipped by request"))
    else:
        items.append(run_pytest(timeout_seconds=pytest_timeout))

    items.extend([
        check_config(),
        check_tts_provider(synthesize=tts_synthesize),
        check_finance_routes(),
        check_trading212_safety(),
    ])

    report = build_report(items)
    return report, write_report(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Project Aria validation checks.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip the pytest suite.")
    parser.add_argument("--pytest-timeout", type=int, default=180, help="pytest timeout in seconds.")
    parser.add_argument(
        "--tts-synthesize",
        action="store_true",
        help="Run a Kokoro synthesis smoke test instead of config-only TTS validation.",
    )
    args = parser.parse_args()

    report, path = run_validation(
        skip_pytest=args.skip_pytest,
        pytest_timeout=args.pytest_timeout,
        tts_synthesize=args.tts_synthesize,
    )

    print(f"Validation status: {report['status'].upper()}")
    print(f"Summary: {report['summary']}")
    print(f"Report: {path}")

    for check in report["checks"]:
        print(f"- {check['name']}: {check['status']} — {check['summary']}")

    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
