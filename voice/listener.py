"""
Aria Listener Module
====================
Captures audio from the microphone with automatic silence detection.
Records until the user stops speaking (silence threshold exceeded),
then returns the raw audio data for transcription.
"""

import numpy as np
import noisereduce as nr
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
from core.logger import get_logger

log = get_logger(__name__)

# Calibrated silence threshold (set by calibrate_silence)
_silence_threshold: float = SILENCE_THRESHOLD


def _chunk_rms(audio_chunk: bytes) -> float:
    """Return RMS amplitude for a raw int16 PCM audio chunk."""
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    if audio_data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))


def get_audio_devices():
    """List all available audio input devices.

    Returns:
        list[dict]: List of dicts with 'index', 'name', and 'channels' for each input device.
    """
    try:
        audio = pyaudio.PyAudio()
    except Exception as e:
        log.error("Failed to initialise PyAudio while listing devices: %s", e, exc_info=True)
        return []

    try:
        try:
            default_input = audio.get_default_input_device_info()
            default_input_index = int(default_input.get("index"))
        except Exception:
            default_input_index = None

        devices = []
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                host_api_name = None
                try:
                    host_api = audio.get_host_api_info_by_index(info.get("hostApi"))
                    host_api_name = host_api.get("name")
                except Exception:
                    host_api_name = None

                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "host_api": host_api_name,
                    "default_sample_rate": info.get("defaultSampleRate"),
                    "is_default": i == default_input_index,
                })
        return devices
    finally:
        audio.terminate()


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

    log.debug(
        "Opening calibration stream: device_index=%s rate=%s channels=%s chunk=%s",
        device_index,
        SAMPLE_RATE,
        CHANNELS,
        CHUNK_SIZE,
    )

    try:
        stream = audio.open(**stream_kwargs)
    except Exception as e:
        audio.terminate()
        log.error("Failed to open calibration stream: %s", e, exc_info=True)
        raise

    log.info("Calibrating microphone — stay quiet for 2 seconds...")
    rms_values = []
    chunks_needed = int(duration * SAMPLE_RATE / CHUNK_SIZE)

    for _ in range(chunks_needed):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        rms_values.append(_chunk_rms(data))

    stream.stop_stream()
    stream.close()
    audio.terminate()

    avg_rms = np.mean(rms_values)
    _silence_threshold = max(avg_rms * 1.5, 200)  # Floor of 200 to avoid ultra-sensitivity
    log.info("Ambient noise level: %.0f RMS — threshold set to %.0f", avg_rms, _silence_threshold)
    return _silence_threshold


def is_silent(audio_chunk: bytes) -> bool:
    """Check if an audio chunk is below the silence threshold.

    Args:
        audio_chunk: Raw audio bytes (16-bit PCM).

    Returns:
        True if the chunk's RMS amplitude is below the calibrated threshold.
    """
    return _chunk_rms(audio_chunk) < _silence_threshold


def preprocess_audio(audio_bytes: bytes) -> bytes:
    """Apply noise reduction to raw audio before transcription.

    Reduces background noise and improves Whisper accuracy in home
    environments. Uses a stationary noise estimation approach.

    Args:
        audio_bytes: Raw 16-bit PCM audio bytes.

    Returns:
        Noise-reduced audio as raw bytes.
    """
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        reduced = nr.reduce_noise(
            y=audio_data,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.75,
        )
        log.debug("Noise reduction applied.")
        return reduced.astype(np.int16).tobytes()
    except Exception as e:
        log.warning("Noise reduction failed (%s), using raw audio.", e)
        return audio_bytes


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
    try:
        audio = pyaudio.PyAudio()
    except Exception as e:
        log.error("Failed to initialise PyAudio: %s", e)
        return b""

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
        log.debug(
            "Opening audio stream: device_index=%s rate=%s channels=%s chunk=%s",
            device_index,
            SAMPLE_RATE,
            CHANNELS,
            CHUNK_SIZE,
        )
        stream = audio.open(**stream_kwargs)
    except Exception as e:
        log.error("Failed to open audio stream: %s", e, exc_info=True)
        audio.terminate()
        return b""

    log.info("Listening... speak now.")

    frames = []
    silent_chunks = 0
    has_speech = False
    start_time = time.time()
    peak_rms = 0.0
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            elapsed = time.time() - start_time
            rms = _chunk_rms(data)
            peak_rms = max(peak_rms, rms)

            if elapsed > MAX_RECORDING_DURATION:
                log.info("Max recording duration (%ds) reached.", MAX_RECORDING_DURATION)
                break

            if rms < _silence_threshold:
                silent_chunks += 1
                if has_speech:
                    frames.append(data)
                if has_speech and silent_chunks >= chunks_for_silence:
                    log.info("Silence detected — stopping recording.")
                    break
            else:
                silent_chunks = 0
                if not has_speech:
                    log.debug(
                        "Speech detected — RMS %.0f above threshold %.0f.",
                        rms,
                        _silence_threshold,
                    )
                has_speech = True
                frames.append(data)

        if not has_speech:
            log.debug(
                "No speech detected (peak RMS %.0f, threshold %.0f).",
                peak_rms,
                _silence_threshold,
            )
            return b""

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    raw_audio = b"".join(frames)
    duration = len(raw_audio) / (SAMPLE_RATE * 2)  # 2 bytes per sample (int16)
    log.info("Captured %.1fs of audio (peak RMS %.0f).", duration, peak_rms)

    if duration < MIN_RECORDING_DURATION:
        log.debug("Too short (%.1fs < %ds) — ignoring.", duration, MIN_RECORDING_DURATION)
        return b""

    # Apply noise reduction before returning
    return preprocess_audio(raw_audio)


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
