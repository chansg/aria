"""
tests/test_tts_provider_selection.py
------------------------------------
Provider selection tests for Aria's TTS facade.
"""

from __future__ import annotations

import numpy as np

from voice import speaker
from voice.tts.base import TTSResult


class _BrokenProvider:
    name = "kokoro-onnx"

    def synthesize(self, text: str) -> TTSResult:
        raise FileNotFoundError("missing kokoro assets")


class _WorkingProvider:
    name = "piper"

    def synthesize(self, text: str) -> TTSResult:
        return TTSResult(samples=np.zeros(240, dtype=np.float32), sample_rate=24000)


def test_provider_name_aliases() -> None:
    assert speaker._normalise_provider_name("kokoro") == "kokoro-onnx"
    assert speaker._normalise_provider_name("kokoro_onnx") == "kokoro-onnx"
    assert speaker._normalise_provider_name("piper-tts") == "piper"


def test_provider_chain_keeps_piper_fallback(monkeypatch) -> None:
    monkeypatch.setattr(speaker.config, "TTS_PROVIDER", "kokoro")
    monkeypatch.setattr(speaker.config, "TTS_FALLBACK_PROVIDER", "piper")

    assert speaker._provider_chain() == ["kokoro-onnx", "piper"]


def test_provider_chain_allows_conversation_override(monkeypatch) -> None:
    monkeypatch.setattr(speaker.config, "TTS_PROVIDER", "kokoro")
    monkeypatch.setattr(speaker.config, "TTS_FALLBACK_PROVIDER", "piper")

    assert speaker._provider_chain("piper") == ["piper"]


def test_provider_chain_allows_no_fallback(monkeypatch) -> None:
    monkeypatch.setattr(speaker.config, "TTS_PROVIDER", "kokoro")
    monkeypatch.setattr(speaker.config, "TTS_FALLBACK_PROVIDER", "")

    assert speaker._provider_chain() == ["kokoro-onnx"]


def test_synthesize_falls_back_to_piper(monkeypatch) -> None:
    monkeypatch.setattr(speaker.config, "TTS_PROVIDER", "kokoro")
    monkeypatch.setattr(speaker.config, "TTS_FALLBACK_PROVIDER", "piper")

    def fake_load_provider(name: str):
        if speaker._normalise_provider_name(name) == "kokoro-onnx":
            return _BrokenProvider()
        return _WorkingProvider()

    monkeypatch.setattr(speaker, "_load_provider", fake_load_provider)

    result, provider_name = speaker._synthesize_with_fallback("hello")

    assert provider_name == "piper"
    assert result.sample_rate == 24000
    assert result.samples.shape == (240,)
