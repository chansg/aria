"""
voice.tts.piper_provider
------------------------
Piper TTS backend. This preserves Aria's existing local ONNX voice as the
fallback provider while the project trials Kokoro.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from config import PIPER_MODEL_PATH
from core.logger import get_logger
from voice.tts.base import TTSResult

log = get_logger(__name__)


class PiperProvider:
    """Synthesise speech with Piper."""

    name = "piper"

    def __init__(self, output_wav: str | Path) -> None:
        self.output_wav = Path(output_wav)
        self._voice: Any = None

    def _load_voice(self):
        if self._voice is not None:
            return self._voice

        if not os.path.exists(PIPER_MODEL_PATH):
            raise FileNotFoundError(
                f"Piper voice model not found at {PIPER_MODEL_PATH}. "
                "Download the hfc_female medium ONNX voice into assets/voices/."
            )

        from piper.voice import PiperVoice

        log.info("Loading Piper voice model: %s", PIPER_MODEL_PATH)
        self._voice = PiperVoice.load(PIPER_MODEL_PATH)
        log.info("Piper voice model loaded (%dHz).", self._voice.config.sample_rate)
        return self._voice

    def synthesize(self, text: str) -> TTSResult:
        voice = self._load_voice()

        self.output_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.output_wav), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        with wave.open(str(self.output_wav), "rb") as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw_data = wf.readframes(n_frames)

        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)

        return TTSResult(samples=samples, sample_rate=sample_rate)
