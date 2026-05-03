"""
voice.tts.base
--------------
Small provider contract shared by Aria's text-to-speech backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TTSResult:
    """Synthesised audio ready for playback."""

    samples: np.ndarray
    sample_rate: int


class TTSProvider(Protocol):
    """Minimal interface implemented by TTS backends."""

    name: str

    def synthesize(self, text: str) -> TTSResult:
        """Convert text into normalised float32 audio samples."""
        ...
