"""
Aria Transcriber Module
=======================
Transcribes audio to text using faster-whisper running locally.
Uses CUDA (RTX 3080) by default for fast inference.
"""

import os
import sys
import site
import threading
import numpy as np

# Add NVIDIA CUDA DLL directories to PATH before ctranslate2 loads.
# pip installs cuBLAS/cuDNN DLLs inside the nvidia package tree, but
# ctranslate2 searches PATH at runtime, not os.add_dll_directory().
if sys.platform == "win32":
    _site_dirs = site.getsitepackages()
    for _site_dir in _site_dirs:
        _nvidia_dir = os.path.join(_site_dir, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _subdir in ("cublas", "cudnn", "cuda_runtime"):
                _bin = os.path.join(_nvidia_dir, _subdir, "bin")
                if os.path.isdir(_bin):
                    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
                    os.add_dll_directory(_bin)

from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, SAMPLE_RATE, TRANSCRIPTION_TIMEOUT
from core.logger import get_logger

log = get_logger(__name__)

# Biases Whisper transcription toward Aria's expected vocabulary.
# Include the name variants and common command words Chan uses.
WHISPER_INITIAL_PROMPT = (
    "Aria, hey Aria, weather, temperature, reminder, schedule, "
    "appointment, time, date, what's, tell me, can you, please, "
    "Birmingham, morning, evening, today, tomorrow"
)

# Module-level model instance (loaded once, reused)
_model: WhisperModel = None


def load_model() -> WhisperModel:
    """Load the Whisper model into memory.

    Uses the model size, device, and compute type from config.py.
    The model is cached at module level so it only loads once.

    Returns:
        The loaded WhisperModel instance.
    """
    global _model
    if _model is None:
        log.info("Loading Whisper model %r on %s...", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        log.info("Whisper model loaded successfully.")
    return _model


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe raw PCM audio bytes to text.

    Args:
        audio_bytes: Raw 16-bit PCM audio data (mono, 16kHz).

    Returns:
        The transcribed text string. Empty string if no speech detected.
    """
    if not audio_bytes:
        return ""

    model = load_model()

    # Convert raw bytes to float32 numpy array (what faster-whisper expects)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    log.info("Transcribing...")

    # Run transcription with a timeout to prevent hangs
    result_holder = [None, None]

    def _run_transcribe():
        try:
            result_holder[0], result_holder[1] = model.transcribe(
                audio_array,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
                initial_prompt=WHISPER_INITIAL_PROMPT,
            )
        except Exception as e:
            log.error("Whisper error: %s", e)

    t = threading.Thread(target=_run_transcribe, daemon=True)
    t.start()
    t.join(timeout=TRANSCRIPTION_TIMEOUT)

    if t.is_alive():
        log.warning("Transcription timed out after %ds.", TRANSCRIPTION_TIMEOUT)
        return ""

    segments, info = result_holder
    if segments is None:
        return ""

    # Collect all segment texts
    full_text = " ".join(segment.text.strip() for segment in segments)

    if full_text:
        # Apply learned corrections from voice training
        try:
            from voice.trainer import apply_corrections
            corrected = apply_corrections(full_text)
            if corrected != full_text.lower().strip():
                log.info("Transcription: %r -> %r", full_text, corrected)
                full_text = corrected
            else:
                log.info("Transcription: %r", full_text)
        except ImportError:
            print(f"[Aria] Transcription: \"{full_text}\"")
    else:
        log.debug("No speech detected in audio.")

    return full_text


if __name__ == "__main__":
    print("=== Aria Transcriber Test ===")
    print("Loading model...")
    load_model()
    print("Model ready. (Run main.py for full pipeline test)")
