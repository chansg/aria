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
import time
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
_transcribe_lock = threading.Lock()


def _release_transcribe_lock_after(worker: threading.Thread) -> None:
    """Release the transcription lock after a timed-out worker finishes."""
    worker.join()
    try:
        _transcribe_lock.release()
    except RuntimeError:
        # Defensive only: the lock should still be held by the caller's timeout
        # path, but avoid crashing a daemon cleanup thread if state changes.
        pass


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

    if not _transcribe_lock.acquire(blocking=False):
        log.warning("Previous transcription still running — skipping this audio clip.")
        return ""

    release_in_cleanup = False

    try:
        model = load_model()

        # Convert raw bytes to float32 numpy array (what faster-whisper expects)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio_bytes) / (SAMPLE_RATE * 2)

        started_at = time.time()
        log.info("Transcribing %.1fs audio...", duration)

        # Run transcription with a timeout to prevent hangs. faster-whisper returns
        # a lazy segment generator, so consume it inside the worker thread; otherwise
        # generator-time exceptions escape the try/except and can kill the voice loop.
        result_holder = {"text": "", "error": None}

        def _run_transcribe():
            try:
                segments, _info = model.transcribe(
                    audio_array,
                    beam_size=1,
                    best_of=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200,
                    ),
                    initial_prompt=WHISPER_INITIAL_PROMPT,
                    condition_on_previous_text=False,
                )
                result_holder["text"] = " ".join(
                    segment.text.strip() for segment in segments
                ).strip()
            except Exception as e:
                result_holder["error"] = e
                log.error("Whisper error: %s", e, exc_info=True)

        t = threading.Thread(target=_run_transcribe, name="WhisperTranscriber", daemon=True)
        t.start()
        t.join(timeout=TRANSCRIPTION_TIMEOUT)
        elapsed = time.time() - started_at

        if t.is_alive():
            log.warning(
                "Transcription timed out after %ds (elapsed %.1fs); waiting for Whisper to finish before accepting another clip.",
                TRANSCRIPTION_TIMEOUT,
                elapsed,
            )
            cleanup = threading.Thread(
                target=_release_transcribe_lock_after,
                args=(t,),
                name="WhisperTimeoutCleanup",
                daemon=True,
            )
            cleanup.start()
            release_in_cleanup = True
            return ""

        if result_holder["error"] is not None:
            log.debug("Transcription failed after %.1fs.", elapsed)
            return ""

        full_text = result_holder["text"]

        if full_text:
            # Apply learned corrections from voice training
            try:
                from voice.trainer import apply_corrections
                corrected = apply_corrections(full_text)
                if corrected != full_text.lower().strip():
                    log.info("Transcription completed in %.1fs: %r -> %r", elapsed, full_text, corrected)
                    full_text = corrected
                else:
                    log.info("Transcription completed in %.1fs: %r", elapsed, full_text)
            except ImportError:
                print(f"[Aria] Transcription: \"{full_text}\"")
        else:
            log.debug("No speech detected in audio after %.1fs.", elapsed)

        return full_text
    finally:
        if not release_in_cleanup:
            _transcribe_lock.release()


if __name__ == "__main__":
    print("=== Aria Transcriber Test ===")
    print("Loading model...")
    load_model()
    print("Model ready. (Run main.py for full pipeline test)")
