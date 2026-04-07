"""
Aria Wake Word Detection
=========================
Listens continuously for "Hey Aria" (or just "Aria") using
voice activity detection + quick Whisper keyword spotting.

Flow:
1. Monitor mic for speech (using existing silence detection)
2. Capture short utterance
3. Quick Whisper transcription
4. Check if "aria" is in the text
5. If yes → return True
6. If no → discard and keep listening

No API keys or external services needed — runs entirely local.
"""

import threading
import numpy as np
import pyaudio
import time
from voice.listener import is_silent
from voice.transcriber import transcribe_audio
from config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
)

# Wake word variants to match (case-insensitive)
WAKE_PHRASES = {"aria", "arya", "hey aria", "hey arya", "a]ria"}

# Short recording settings for wake word detection
WAKE_MAX_DURATION = 3.0     # Max seconds to record for wake word check
WAKE_SILENCE_SECS = 0.8     # Shorter silence window for quick detection


def listen_for_wake_word(
    device_index: int = None,
    stop_event: threading.Event = None,
) -> bool:
    """Block until the wake word "Aria" is detected.

    Continuously monitors the microphone. When speech is detected,
    captures a short clip and checks if it contains the wake word.

    Args:
        device_index: Audio input device index (None for default).
        stop_event: If set, the listener exits early and returns False.
                    Used to interrupt sleep mode when conversation mode
                    is toggled back on.

    Returns:
        True when the wake word is detected. False if interrupted
        by stop_event or an error.
    """
    try:
        audio = pyaudio.PyAudio()
    except Exception as e:
        print(f"[Aria] Failed to initialise PyAudio for wake word: {e}")
        return False

    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": CHANNELS,
        "rate": SAMPLE_RATE,
        "input": True,
        "frames_per_buffer": CHUNK_SIZE,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    try:
        stream = audio.open(**stream_kwargs)
    except Exception as e:
        print(f"[Aria] Failed to open audio stream for wake word: {e}")
        audio.terminate()
        return False

    chunks_for_silence = int(WAKE_SILENCE_SECS * SAMPLE_RATE / CHUNK_SIZE)

    try:
        while True:
            # Check for external stop signal (mode toggled back to ON)
            if stop_event and stop_event.is_set():
                return False

            # Wait for speech to start
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if is_silent(data):
                continue

            # Speech detected — capture a short clip
            frames = [data]
            silent_chunks = 0
            start_time = time.time()

            while True:
                if stop_event and stop_event.is_set():
                    return False

                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                elapsed = time.time() - start_time
                frames.append(data)

                if elapsed > WAKE_MAX_DURATION:
                    break

                if is_silent(data):
                    silent_chunks += 1
                    if silent_chunks >= chunks_for_silence:
                        break
                else:
                    silent_chunks = 0

            # Quick transcription
            raw_audio = b"".join(frames)
            duration = len(raw_audio) / (SAMPLE_RATE * 2)

            if duration < 0.3:
                continue

            text = transcribe_audio(raw_audio)
            if not text:
                continue

            # Check for wake word
            text_lower = text.strip().lower()
            text_lower = text_lower.rstrip(".,!?")

            if _contains_wake_word(text_lower):
                print(f"[Aria] Wake word detected: \"{text}\"")
                return True

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def _contains_wake_word(text: str) -> bool:
    """Check if the transcribed text contains the wake word.

    Args:
        text: Lowercased transcription text.

    Returns:
        True if any wake phrase variant is found.
    """
    for phrase in WAKE_PHRASES:
        if phrase in text:
            return True
    return False
