"""
Aria Wake Chime
===============
Generates and plays a pleasant two-tone ascending chime when
the wake word "Hey Aria" is detected. The chime is generated
programmatically (no external audio files needed) and cached
to a WAV file on first use.
"""

import os
import wave
import struct
import math
import tempfile
import numpy as np
import sounddevice as sd

# Chime parameters
CHIME_SAMPLE_RATE = 44100
CHIME_DURATION = 0.3    # Duration of each tone in seconds
CHIME_FREQ_1 = 523.25   # C5
CHIME_FREQ_2 = 659.25   # E5 (major third above — pleasant and bright)
CHIME_VOLUME = 0.4       # 0.0 to 1.0

_chime_path: str = None


def _generate_tone(frequency: float, duration: float, volume: float) -> list[int]:
    """Generate a sine wave tone with fade in/out.

    Args:
        frequency: Frequency in Hz.
        duration: Duration in seconds.
        volume: Volume from 0.0 to 1.0.

    Returns:
        List of 16-bit PCM samples.
    """
    num_samples = int(CHIME_SAMPLE_RATE * duration)
    samples = []
    fade_samples = int(CHIME_SAMPLE_RATE * 0.05)  # 50ms fade

    for i in range(num_samples):
        # Sine wave
        value = math.sin(2 * math.pi * frequency * i / CHIME_SAMPLE_RATE)

        # Fade in
        if i < fade_samples:
            value *= i / fade_samples
        # Fade out
        elif i > num_samples - fade_samples:
            value *= (num_samples - i) / fade_samples

        sample = int(value * volume * 32767)
        samples.append(max(-32768, min(32767, sample)))

    return samples


def _generate_chime_wav() -> str:
    """Generate the two-tone chime WAV file.

    Creates the file in the assets/sounds directory if possible,
    otherwise in the temp directory.

    Returns:
        Path to the generated WAV file.
    """
    # Try assets/sounds first, fall back to temp
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sounds")
    if os.path.isdir(assets_dir):
        path = os.path.join(assets_dir, "wake_chime.wav")
    else:
        path = os.path.join(tempfile.gettempdir(), "aria_wake_chime.wav")

    # Generate two tones
    tone_1 = _generate_tone(CHIME_FREQ_1, CHIME_DURATION, CHIME_VOLUME)
    silence = [0] * int(CHIME_SAMPLE_RATE * 0.05)  # 50ms gap
    tone_2 = _generate_tone(CHIME_FREQ_2, CHIME_DURATION, CHIME_VOLUME)

    all_samples = tone_1 + silence + tone_2

    # Write WAV file
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CHIME_SAMPLE_RATE)
        for sample in all_samples:
            wf.writeframes(struct.pack("<h", sample))

    return path


def play_chime() -> None:
    """Play the wake chime sound.

    Generates the chime file on first call, then reuses it.
    Uses sounddevice for playback (no pygame dependency).
    """
    global _chime_path

    if _chime_path is None or not os.path.exists(_chime_path):
        _chime_path = _generate_chime_wav()
        print(f"[Aria] Wake chime generated: {_chime_path}")

    # Read WAV and play via sounddevice
    with wave.open(_chime_path, "rb") as wf:
        raw_data = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()

    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Block until chime finishes
