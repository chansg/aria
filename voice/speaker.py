"""
Aria Speaker Module
===================
Text-to-speech output using Piper TTS with the hfc_female medium
ONNX voice model. Runs fully locally with no API calls.

Generates WAV audio via ONNX inference and plays it through
the default output device using sounddevice.
Integrates with the avatar renderer for speaking state and lip-sync.
"""

import wave
import os
import time
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
import re
from config import PIPER_MODEL_PATH, PIPER_SPEAKING_RATE


# Output WAV path (reused each utterance — overwritten, not accumulated)
OUTPUT_WAV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sounds", "aria_response.wav")

# Module-level voice instance (loaded once, reused)
_voice: PiperVoice = None


def _load_voice() -> PiperVoice:
    """Load and cache the Piper voice model.

    Returns:
        PiperVoice: Loaded voice instance.

    Raises:
        FileNotFoundError: If the ONNX model file is missing.
    """
    global _voice
    if _voice is None:
        if not os.path.exists(PIPER_MODEL_PATH):
            raise FileNotFoundError(
                f"[Aria] Voice model not found at {PIPER_MODEL_PATH}. "
                "Download the hfc_female medium ONNX from HuggingFace "
                "(rhasspy/piper-voices) into assets/voices/."
            )
        print(f"[Aria] Loading voice model: {PIPER_MODEL_PATH}")
        _voice = PiperVoice.load(PIPER_MODEL_PATH)
        print(f"[Aria] Voice model loaded ({_voice.config.sample_rate}Hz).")
    return _voice


def _clean_for_speech(text: str) -> str:
    """Remove markdown formatting characters before TTS synthesis.

    LLMs return markdown-formatted text. Piper TTS reads symbols
    literally — asterisks become 'asterisk', hashes become 'hash'.
    This function strips all markdown to produce clean speakable text.

    Args:
        text: Raw LLM response potentially containing markdown.

    Returns:
        Clean plain text suitable for TTS synthesis.
    """
    # Remove bold/italic asterisks and underscores
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    # Remove inline code backticks
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bullet point dashes and asterisks at line start
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    # Remove any remaining lone asterisks
    text = text.replace('*', '').replace('#', '')
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def speak(text: str) -> None:
    """Synthesise text to speech and play it via sounddevice.

    Strips markdown formatting and speaks the full response.
    Updates avatar state to 'speaking' during playback with
    amplitude-synced lip movement, then back to 'idle' when finished.
    Falls back to terminal print if TTS or playback fails.

    Args:
        text: The text string for Aria to speak aloud.
    """
    if not text or not text.strip():
        return

    # Warn on very long responses but still speak the full text
    if len(text) > 1500:
        print(f"[Aria] Long response ({len(text)} chars) — speaking full text.")

    # Strip markdown formatting before synthesis
    text = _clean_for_speech(text)

    # Update avatar state (imported here to avoid circular imports)
    try:
        from avatar.renderer import set_speaking, set_idle, set_amplitude
        set_speaking()
    except ImportError:
        set_speaking = set_idle = set_amplitude = None

    try:
        _speak_sync(text)
    except FileNotFoundError as e:
        print(f"[Aria] ERROR: {e}")
        print(f"[Aria] (text): {text}")
    except Exception as e:
        print(f"[Aria] ERROR: TTS failed — {e}")
        print(f"[Aria] (text): {text}")

    try:
        from avatar.renderer import set_idle, set_amplitude
        set_amplitude(0.0)
        set_idle()
    except ImportError:
        pass


def _speak_sync(text: str) -> None:
    """Synchronous implementation of text-to-speech with lip-sync.

    Generates a WAV via Piper TTS, then plays it using sounddevice
    while feeding amplitude data to the avatar for lip-sync.

    Args:
        text: The text to speak.
    """
    voice = _load_voice()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)

    # Synthesise to WAV file (synthesize_wav handles WAV headers)
    with wave.open(OUTPUT_WAV, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    # Read the WAV data for playback and amplitude analysis
    with wave.open(OUTPUT_WAV, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw_data = wf.readframes(n_frames)

    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    # If stereo, take first channel for amplitude analysis
    if n_channels > 1:
        audio_array = audio_array[::n_channels]

    # Normalise to float32 [-1.0, 1.0] for sounddevice
    audio_playback = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio_playback = audio_playback.reshape(-1, n_channels)

    # Start non-blocking playback
    sd.play(audio_playback, samplerate=sample_rate)

    # Feed amplitude to the avatar during playback
    chunk_samples = int(sample_rate * 0.05)  # 50ms chunks
    start_time = time.time()

    try:
        from avatar.renderer import set_amplitude
    except ImportError:
        set_amplitude = None

    while sd.get_stream().active:
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

    sd.wait()  # Ensure playback is fully complete

    if set_amplitude:
        set_amplitude(0.0)
