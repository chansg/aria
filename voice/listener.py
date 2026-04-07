"""
Aria Listener Module
====================
Captures audio from the microphone with automatic silence detection.
Records until the user stops speaking (silence threshold exceeded),
then returns the raw audio data for transcription.
"""

import numpy as np
import pyaudio
import wave
import io
import time
from config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MAX_RECORDING_DURATION,
    MIN_RECORDING_DURATION,
)

# Calibrated silence threshold (set by calibrate_silence)
_silence_threshold: float = SILENCE_THRESHOLD


def get_audio_devices():
    """List all available audio input devices.

    Returns:
        list[dict]: List of dicts with 'index', 'name', and 'channels' for each input device.
    """
    audio = pyaudio.PyAudio()
    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index": i,
                "name": info["name"],
                "channels": info["maxInputChannels"],
            })
    audio.terminate()
    return devices


def calibrate_silence(device_index: int = None, duration: float = 2.0) -> float:
    """Measure ambient noise level to auto-set the silence threshold.

    Records a few seconds of silence and sets the threshold to
    1.5x the average RMS, so speech reliably rises above it.

    Args:
        device_index: Audio input device index (None for default).
        duration: Seconds to sample ambient noise.

    Returns:
        The calibrated silence threshold.
    """
    global _silence_threshold
    audio = pyaudio.PyAudio()

    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": CHANNELS,
        "rate": SAMPLE_RATE,
        "input": True,
        "frames_per_buffer": CHUNK_SIZE,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    stream = audio.open(**stream_kwargs)

    print("[Aria] Calibrating microphone — stay quiet for 2 seconds...")
    rms_values = []
    chunks_needed = int(duration * SAMPLE_RATE / CHUNK_SIZE)

    for _ in range(chunks_needed):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        rms_values.append(rms)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    avg_rms = np.mean(rms_values)
    _silence_threshold = max(avg_rms * 1.5, 200)  # Floor of 200 to avoid ultra-sensitivity
    print(f"[Aria] Ambient noise level: {avg_rms:.0f} RMS — threshold set to {_silence_threshold:.0f}")
    return _silence_threshold


def is_silent(audio_chunk: bytes) -> bool:
    """Check if an audio chunk is below the silence threshold.

    Args:
        audio_chunk: Raw audio bytes (16-bit PCM).

    Returns:
        True if the chunk's RMS amplitude is below the calibrated threshold.
    """
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    return rms < _silence_threshold


def record_audio(device_index: int = None) -> bytes:
    """Record audio from the microphone until silence is detected.

    Starts recording when sound is detected, then stops after
    SILENCE_DURATION seconds of continuous silence or when
    MAX_RECORDING_DURATION is reached.

    Args:
        device_index: Optional index of the audio input device to use.
                      If None, uses the system default.

    Returns:
        Raw audio bytes (16-bit PCM, mono, 16kHz) suitable for Whisper.
    """
    audio = pyaudio.PyAudio()

    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": CHANNELS,
        "rate": SAMPLE_RATE,
        "input": True,
        "frames_per_buffer": CHUNK_SIZE,
    }
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    stream = audio.open(**stream_kwargs)

    print("[Aria] Listening... speak now.")

    frames = []
    silent_chunks = 0
    has_speech = False
    start_time = time.time()
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            elapsed = time.time() - start_time

            if elapsed > MAX_RECORDING_DURATION:
                print(f"[Aria] Max recording duration ({MAX_RECORDING_DURATION}s) reached.")
                break

            if is_silent(data):
                silent_chunks += 1
                if has_speech:
                    frames.append(data)
                if has_speech and silent_chunks >= chunks_for_silence:
                    print("[Aria] Silence detected — stopping recording.")
                    break
            else:
                silent_chunks = 0
                has_speech = True
                frames.append(data)

        if not has_speech:
            print("[Aria] No speech detected.")
            return b""

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    raw_audio = b"".join(frames)
    duration = len(raw_audio) / (SAMPLE_RATE * 2)  # 2 bytes per sample (int16)
    print(f"[Aria] Captured {duration:.1f}s of audio.")

    if duration < MIN_RECORDING_DURATION:
        print(f"[Aria] Too short ({duration:.1f}s < {MIN_RECORDING_DURATION}s) — ignoring.")
        return b""

    return raw_audio


def audio_bytes_to_wav(audio_bytes: bytes) -> bytes:
    """Convert raw PCM audio bytes to WAV format in memory.

    Args:
        audio_bytes: Raw 16-bit PCM audio data.

    Returns:
        WAV-formatted audio bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    return buffer.getvalue()


if __name__ == "__main__":
    print("=== Aria Listener Test ===")
    print("\nAvailable audio devices:")
    for device in get_audio_devices():
        print(f"  [{device['index']}] {device['name']} (channels: {device['channels']})")

    print("\nRecording with default device...")
    audio_data = record_audio()
    if audio_data:
        wav_data = audio_bytes_to_wav(audio_data)
        print(f"[Aria] WAV data ready: {len(wav_data)} bytes")
    else:
        print("[Aria] No audio captured.")
