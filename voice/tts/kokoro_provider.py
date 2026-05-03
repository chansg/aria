"""
voice.tts.kokoro_provider
-------------------------
Kokoro ONNX backend for Aria's spoken voice.
"""

from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    KOKORO_DISABLE_PROVIDER_FALLBACK,
    KOKORO_LANG,
    KOKORO_ONNX_MODEL_PATH,
    KOKORO_ONNX_PROVIDER,
    KOKORO_ONNX_VOICES_PATH,
    KOKORO_SPEED,
    KOKORO_VOICE,
)
from core.logger import get_logger
from voice.tts.base import TTSResult

log = get_logger(__name__)


class KokoroOnnxProvider:
    """Synthesise speech with kokoro-onnx."""

    name = "kokoro-onnx"

    def __init__(self) -> None:
        self.model_path = Path(KOKORO_ONNX_MODEL_PATH)
        self.voices_path = Path(KOKORO_ONNX_VOICES_PATH)
        self.voice = KOKORO_VOICE
        self.speed = float(KOKORO_SPEED)
        self.lang = KOKORO_LANG
        self.provider = KOKORO_ONNX_PROVIDER
        self.disable_provider_fallback = bool(KOKORO_DISABLE_PROVIDER_FALLBACK)
        self._kokoro: Any = None

    def _add_nvidia_dll_dirs(self) -> None:
        """Expose pip-installed NVIDIA runtime DLLs to ONNX Runtime on Windows."""
        if os.name != "nt":
            return

        for site_dir in site.getsitepackages():
            nvidia_dir = Path(site_dir) / "nvidia"
            if not nvidia_dir.is_dir():
                continue
            for bin_dir in nvidia_dir.glob("*\\bin"):
                if not bin_dir.is_dir():
                    continue
                bin_path = str(bin_dir)
                os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(bin_path)
                except OSError:
                    pass

    def _load_model(self):
        if self._kokoro is not None:
            return self._kokoro

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Kokoro ONNX model not found at {self.model_path}. "
                "Download kokoro-v1.0.onnx into assets/voices/kokoro/."
            )
        if not self.voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro voices file not found at {self.voices_path}. "
                "Download voices-v1.0.bin into assets/voices/kokoro/."
            )

        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise ImportError(
                "kokoro-onnx is not installed. Install project dependencies "
                "after reviewing requirements.txt."
            ) from exc

        available_providers = ort.get_available_providers()
        log.info("ONNX Runtime providers available for Kokoro: %s", available_providers)
        if self.provider:
            if self.provider not in available_providers:
                raise RuntimeError(
                    f"Requested Kokoro ONNX provider {self.provider!r} is not available. "
                    f"Available providers: {available_providers}"
                )
            if self.provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
                self._add_nvidia_dll_dirs()
                ort.preload_dlls(directory="")

            os.environ["ONNX_PROVIDER"] = self.provider

        log.info(
            "Loading Kokoro ONNX model: model=%s voices=%s provider=%s voice=%s lang=%s speed=%.2f",
            self.model_path,
            self.voices_path,
            self.provider or "default",
            self.voice,
            self.lang,
            self.speed,
        )
        self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        if self.disable_provider_fallback and hasattr(self._kokoro.sess, "disable_fallback"):
            self._kokoro.sess.disable_fallback()
        log.info("Kokoro ONNX model loaded.")
        return self._kokoro

    def synthesize(self, text: str) -> TTSResult:
        kokoro = self._load_model()
        samples, sample_rate = kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )

        audio = np.asarray(samples, dtype=np.float32)
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 1.0:
                audio = audio / peak

        return TTSResult(samples=audio, sample_rate=int(sample_rate))
