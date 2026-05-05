"""
tests.test_validation_harness
-----------------------------
Validation harness checks.
"""

from __future__ import annotations

import json

from tools.run_validation import (
    ValidationItem,
    build_report,
    check_finance_routes,
    check_trading212_safety,
    check_tts_provider,
    write_report,
)


def test_build_report_marks_failures() -> None:
    report = build_report([
        ValidationItem("one", "pass", "ok"),
        ValidationItem("two", "warn", "warning"),
        ValidationItem("three", "fail", "broken"),
    ])

    assert report["status"] == "fail"
    assert report["summary"] == {"passed": 1, "warned": 1, "failed": 1}


def test_write_report_creates_json_file(tmp_path) -> None:
    report = build_report([ValidationItem("one", "pass", "ok")])

    path = write_report(report, report_dir=tmp_path)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "pass"
    assert loaded["checks"][0]["name"] == "one"


def test_finance_route_validation_passes() -> None:
    result = check_finance_routes()

    assert result.status == "pass"


def test_trading212_safety_validation_passes() -> None:
    result = check_trading212_safety()

    assert result.status == "pass"


def test_tts_provider_config_validation_passes() -> None:
    result = check_tts_provider(synthesize=False)

    assert result.status == "pass"
    assert result.details["provider_chain"] == ["kokoro-onnx"]
    assert result.details["synthesized"] is False
