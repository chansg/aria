"""
Aria Transcriber Module
=======================
Transcribes audio to text using faster-whisper running locally.
Uses CUDA (RTX 3080) by default for fast inference.
"""

import os
import sys
import site
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
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, SAMPLE_RATE

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
        print(f"[Aria] Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[Aria] Whisper model loaded successfully.")
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

    print("[Aria] Transcribing...")
    segments, info = model.transcribe(
        audio_array,
        beam_size=5,
        language="en",
        vad_filter=True,  # Filter out non-speech segments
    )

    # Collect all segment texts
    full_text = " ".join(segment.text.strip() for segment in segments)

    if full_text:
        # Apply learned corrections from voice training
        try:
            from voice.trainer import apply_corrections
            corrected = apply_corrections(full_text)
            if corrected != full_text.lower().strip():
                print(f"[Aria] Transcription: \"{full_text}\" → \"{corrected}\"")
                full_text = corrected
            else:
                print(f"[Aria] Transcription: \"{full_text}\"")
        except ImportError:
            print(f"[Aria] Transcription: \"{full_text}\"")
    else:
        print("[Aria] No speech detected in audio.")

    return full_text


if __name__ == "__main__":
    print("=== Aria Transcriber Test ===")
    print("Loading model...")
    load_model()
    print("Model ready. (Run main.py for full pipeline test)")
