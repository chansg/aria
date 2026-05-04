"""
tests.test_speaker_text_normalization
-------------------------------------
TTS text cleanup tests.
"""

from __future__ import annotations

from voice.speaker import _clean_for_speech


def test_mm_dash_is_rewritten_for_kokoro_pronunciation() -> None:
    out = _clean_for_speech("Mm—so you want the bigger picture.")

    assert out == "Mmm, so you want the bigger picture."


def test_plain_mm_is_rewritten_for_kokoro_pronunciation() -> None:
    out = _clean_for_speech("Mm. I see.")

    assert out == "Mmm. I see."
