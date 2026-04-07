"""
Aria Speaker Module
===================
Text-to-speech output using Piper TTS running locally.
Generates WAV audio via ONNX inference and plays it through
the default output device using pygame.mixer.
Integrates with the avatar renderer for speaking state and lip-sync.
"""

import wave
import tempfile
import os
import time
import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from config import PIPER_MODEL_PATH, PIPER_SPEAKING_RATE


# Module-level voice instance (loaded once, reused)
_voice: PiperVoice = None


def _load_voice() -> PiperVoice:
    """Load the Piper TTS voice model.

    Returns:
        The loaded PiperVoice instance.
    """
    global _voice
    if _voice is None:
        print(f"[Aria] Loading Piper TTS model...")
        _voice = PiperVoice.load(PIPER_MODEL_PATH)
        print(f"[Aria] Piper TTS loaded ({_voice.config.sample_rate}Hz).")
    return _voice


def speak(text: str) -> None:
    """Convert text to speech and play it aloud.

    Updates the avatar state to 'speaking' during playback with
    amplitude-synced lip movement, then back to 'idle' when finished.

    Args:
        text: The text for Aria to speak.
    """
    if not text:
        return

    # Update avatar state (imported here to avoid circular imports)
    try:
        from avatar.renderer import set_speaking, set_idle, set_amplitude
        set_speaking()
    except ImportError:
        set_speaking = set_idle = set_amplitude = None

    _speak_sync(text)

    try:
        from avatar.renderer import set_idle, set_amplitude
        set_amplitude(0.0)
        set_idle()
    except ImportError:
        pass


def _speak_sync(text: str) -> None:
    """Synchronous implementation of text-to-speech with lip-sync.

    Generates a WAV via Piper TTS, then plays it using pygame.mixer
    while feeding amplitude data to the avatar for lip-sync.

    Args:
        text: The text to speak.
    """
    import pygame

    voice = _load_voice()

    # Generate speech audio to a temp WAV file
    syn_config = SynthesisConfig(length_scale=PIPER_SPEAKING_RATE)
    tmp_path = os.path.join(tempfile.gettempdir(), "aria_speech.wav")
    with wave.open(tmp_path, "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=syn_config)

    # Read the WAV data for amplitude analysis
    with wave.open(tmp_path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        raw_data = wf.readframes(n_frames)

    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    # Play the audio
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    # Feed amplitude to the avatar during playback
    chunk_samples = int(sample_rate * 0.05)  # 50ms chunks
    start_time = time.time()

    try:
        from avatar.renderer import set_amplitude
    except ImportError:
        set_amplitude = None

    while pygame.mixer.music.get_busy():
        if set_amplitude and len(audio_array) > 0:
            elapsed = time.time() - start_time
            sample_pos = int(elapsed * sample_rate)

            if sample_pos < len(audio_array):
                chunk_end = min(sample_pos + chunk_samples, len(audio_array))
                chunk = audio_array[sample_pos:chunk_end]
                rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0
                # Normalise to 0.0-1.0 range (typical speech RMS ~2000-8000)
                amplitude = min(1.0, rms / 6000.0)
                set_amplitude(amplitude)
            else:
                set_amplitude(0.0)

        time.sleep(0.05)

    if set_amplitude:
        set_amplitude(0.0)

    pygame.mixer.music.unload()
