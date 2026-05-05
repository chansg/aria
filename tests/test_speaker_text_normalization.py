"""
tests.test_speaker_text_normalization
-------------------------------------
TTS text cleanup tests.
"""

from __future__ import annotations

import numpy as np

from voice.speaker import _clean_for_speech, _split_into_sentences, _trim_edge_silence


def test_mm_dash_is_rewritten_for_kokoro_pronunciation() -> None:
    out = _clean_for_speech("Mm—so you want the bigger picture.")

    assert out == "Mmm, so you want the bigger picture."


def test_plain_mm_is_rewritten_for_kokoro_pronunciation() -> None:
    out = _clean_for_speech("Mm. I see.")

    assert out == "Mmm. I see."


def test_split_into_sentences_honours_shorter_chunks() -> None:
    text = "First short sentence. Second short sentence. Third short sentence."

    chunks = _split_into_sentences(text, max_chunk=45)

    assert chunks == [
        "First short sentence. Second short sentence.",
        "Third short sentence.",
    ]


def test_trim_edge_silence_preserves_padding() -> None:
    sample_rate = 1000
    samples = np.concatenate([
        np.zeros(100, dtype=np.float32),
        np.ones(200, dtype=np.float32) * 0.1,
        np.zeros(100, dtype=np.float32),
    ])

    trimmed = _trim_edge_silence(samples, sample_rate, threshold=0.01, padding_ms=10)

    assert len(trimmed) == 220
    assert np.allclose(trimmed[:10], 0.0)
    assert np.allclose(trimmed[-10:], 0.0)


def test_trim_edge_silence_keeps_all_silence_audio() -> None:
    samples = np.zeros(100, dtype=np.float32)

    trimmed = _trim_edge_silence(samples, 1000, threshold=0.01, padding_ms=10)

    assert trimmed is samples
