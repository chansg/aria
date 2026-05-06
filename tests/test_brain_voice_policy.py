"""
tests.test_brain_voice_policy
-----------------------------
Voice response length policy tests.
"""

from __future__ import annotations

import pytest

import core.brain as brain


def test_voice_policy_caps_normal_conversation(monkeypatch) -> None:
    monkeypatch.setattr(brain, "_spoken_response_max_chars", lambda: 80)
    response = (
        "First useful sentence. Second useful sentence. "
        "Third sentence should not be spoken in this short live reply."
    )

    out = brain._limit_spoken_response(response, "How do I become wealthy?")

    assert out == "First useful sentence. Second useful sentence."
    assert len(out) <= 80


def test_voice_policy_allows_requested_detail(monkeypatch) -> None:
    monkeypatch.setattr(brain, "_spoken_response_max_chars", lambda: 40)
    response = "Sentence one. Sentence two. Sentence three."

    out = brain._limit_spoken_response(response, "Explain this in detail")

    assert out == response


def test_voice_policy_trims_long_single_sentence(monkeypatch) -> None:
    monkeypatch.setattr(brain, "_spoken_response_max_chars", lambda: 45)
    response = "This single sentence is much too long for the configured cap."

    out = brain._limit_spoken_response(response, "Quick answer")

    assert out.endswith(".")
    assert len(out) <= 45


def test_complete_sentence_guard_appends_safe_missing_period() -> None:
    out = brain._ensure_complete_sentence(
        "Hey Chan, you've got your project notes",
        source="test",
    )

    assert out == "Hey Chan, you've got your project notes."


def test_complete_sentence_guard_rejects_truncated_external_access() -> None:
    with pytest.raises(brain.IncompleteModelResponseError):
        brain._ensure_complete_sentence(
            "Chan, I still don't have access to external",
            source="test",
            strict=True,
        )


def test_complete_sentence_guard_uses_fallback_when_not_strict() -> None:
    out = brain._ensure_complete_sentence(
        "Chan, I couldn't find the market capitalization for",
        source="test",
        fallback_message="Fallback response.",
    )

    assert out == "Fallback response."
