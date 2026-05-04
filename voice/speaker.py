"""
Aria Speaker Module
===================
Public text-to-speech surface for Aria.

Callers use speak(text). The selected synthesis backend is configurable:
Kokoro ONNX is the preferred provider, while Piper remains the local fallback.
Playback, chunking, markdown cleanup, and visual amplitude hooks stay here so
the rest of Aria does not need to know which voice engine is active.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

import config
from core.logger import get_logger
from voice.tts.base import TTSProvider, TTSResult

log = get_logger(__name__)


OUTPUT_WAV = Path(__file__).resolve().parents[1] / "assets" / "sounds" / "aria_response.wav"

_provider_cache: dict[str, TTSProvider] = {}
_speak_lock = threading.Lock()
_speaking_event = threading.Event()


def _normalise_provider_name(name: str | None) -> str:
    provider = (name or "piper").strip().lower().replace("_", "-")
    aliases = {
        "kokoro": "kokoro-onnx",
        "kokoro-82m": "kokoro-onnx",
        "kokoroonnx": "kokoro-onnx",
        "piper-tts": "piper",
    }
    return aliases.get(provider, provider)


def _provider_chain(preferred_name: str | None = None) -> list[str]:
    preferred = _normalise_provider_name(preferred_name or getattr(config, "TTS_PROVIDER", "piper"))
    raw_fallback = getattr(config, "TTS_FALLBACK_PROVIDER", "piper")
    fallback = _normalise_provider_name(raw_fallback) if raw_fallback else ""

    chain = [preferred]
    if fallback and fallback != preferred:
        chain.append(fallback)
    return chain


def is_speaking() -> bool:
    """Return True while Aria is synthesising or playing speech."""
    return _speaking_event.is_set()


def _load_provider(name: str) -> TTSProvider:
    provider_name = _normalise_provider_name(name)
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    if provider_name == "piper":
        from voice.tts.piper_provider import PiperProvider

        provider = PiperProvider(output_wav=OUTPUT_WAV)
    elif provider_name == "kokoro-onnx":
        from voice.tts.kokoro_provider import KokoroOnnxProvider

        provider = KokoroOnnxProvider()
    else:
        raise ValueError(f"Unknown TTS provider: {name!r}")

    _provider_cache[provider_name] = provider
    log.info("TTS provider initialised: %s", provider.name)
    return provider


def _clean_for_speech(text: str) -> str:
    """Remove markdown formatting characters before TTS synthesis."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("*", "").replace("#", "")
    text = _normalise_pronunciation_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalise_pronunciation_tokens(text: str) -> str:
    """Rewrite short tokens that Kokoro tends to spell aloud."""
    text = re.sub(r"(?i)\bmm\s*[—–-]\s*", "Mmm, ", text)
    text = re.sub(r"(?i)\bmm\b", "Mmm", text)
    text = re.sub(r"(?i)\bhmm\s*[—–-]\s*", "Hmm, ", text)
    return text


def _split_into_sentences(text: str, max_chunk: int = 300) -> list[str]:
    """Split text into speakable chunks at sentence and clause boundaries."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in raw:
        if len(current) + len(sentence) <= max_chunk:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            if len(sentence) > max_chunk:
                sub = re.split(r"(?<=,)\s+", sentence)
                sub_chunk = ""
                for part in sub:
                    if len(sub_chunk) + len(part) <= max_chunk:
                        sub_chunk += (" " if sub_chunk else "") + part
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = part
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _synthesize_with_fallback(text: str, preferred_name: str | None = None) -> tuple[TTSResult, str]:
    chain = _provider_chain(preferred_name)
    errors: list[str] = []

    for index, provider_name in enumerate(chain):
        try:
            provider = _load_provider(provider_name)
            start = time.time()
            result = provider.synthesize(text)
            elapsed = time.time() - start

            if result.samples.size == 0:
                raise RuntimeError(f"{provider.name} returned empty audio")

            log.debug(
                "TTS synthesis complete: provider=%s sample_rate=%s samples=%s elapsed=%.2fs",
                provider.name,
                result.sample_rate,
                result.samples.shape,
                elapsed,
            )
            return result, provider.name
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")
            if index + 1 < len(chain):
                log.warning(
                    "TTS provider %s failed: %s. Falling back to %s.",
                    provider_name,
                    exc,
                    chain[index + 1],
                    exc_info=True,
                )
            else:
                if bool(getattr(config, "TTS_FAIL_LOUD", False)):
                    log.critical("TTS provider %s failed with no fallback: %s", provider_name, exc, exc_info=True)
                else:
                    log.error("TTS provider %s failed: %s", provider_name, exc, exc_info=True)

    raise RuntimeError("All TTS providers failed: " + " | ".join(errors))


def speak(text: str, provider: str | None = None) -> None:
    """Synthesise text to speech and play it via sounddevice."""
    if not text or not text.strip():
        return

    text = _clean_for_speech(text)
    chunks = _split_into_sentences(text)
    chain = _provider_chain(provider)

    if _speak_lock.locked():
        log.info("TTS busy — waiting for current speech to finish.")

    with _speak_lock:
        _speaking_event.set()
        log.info(
            "Speaking %d chunk(s), %d total chars. provider=%s fallback=%s",
            len(chunks),
            len(text),
            chain[0],
            chain[1] if len(chain) > 1 else "none",
        )

        try:
            from avatar.renderer import set_speaking

            set_speaking()
        except Exception:
            pass

        try:
            for i, chunk in enumerate(chunks):
                try:
                    log.info(
                        "Synthesising TTS chunk %d/%d via %s (%d chars).",
                        i + 1,
                        len(chunks),
                        chain[0],
                        len(chunk),
                    )
                    result, provider_name = _synthesize_with_fallback(chunk, preferred_name=provider)
                    _play_audio(result, provider_name)
                except Exception as exc:
                    log.error("TTS failed on chunk %d/%d: %s", i + 1, len(chunks), exc, exc_info=True)
                    log.error("(text): %s", chunk)
                    break
        finally:
            _speaking_event.clear()
            try:
                from avatar.renderer import set_amplitude, set_idle

                set_amplitude(0.0)
                set_idle()
            except Exception:
                pass


def _mono_for_amplitude(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2 and samples.shape[1] > 0:
        return samples[:, 0]
    return samples.reshape(-1)


def _normalise_playback(samples: np.ndarray) -> np.ndarray:
    playback = np.asarray(samples, dtype=np.float32)
    if playback.size == 0:
        return playback

    peak = float(np.max(np.abs(playback)))
    if peak > 1.0:
        playback = playback / peak
    return playback


def _play_audio(result: TTSResult, provider_name: str) -> None:
    """Play normalised float32 audio and feed amplitude to the visual facade."""
    audio_playback = _normalise_playback(result.samples)
    audio_mono = _mono_for_amplitude(audio_playback)
    sample_rate = int(result.sample_rate)
    duration = len(audio_mono) / sample_rate if sample_rate else 0

    log.debug("Playing TTS audio: provider=%s duration=%.2fs sample_rate=%s", provider_name, duration, sample_rate)
    sd.play(audio_playback, samplerate=sample_rate)

    chunk_samples = int(sample_rate * 0.05)
    start_time = time.time()

    try:
        from avatar.renderer import set_amplitude
    except Exception:
        set_amplitude = None

    while sd.get_stream().active:
        if set_amplitude and len(audio_mono) > 0:
            elapsed = time.time() - start_time
            sample_pos = int(elapsed * sample_rate)

            if sample_pos < len(audio_mono):
                chunk_end = min(sample_pos + chunk_samples, len(audio_mono))
                chunk = audio_mono[sample_pos:chunk_end]
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
                amplitude = min(1.0, rms / 0.20)
                set_amplitude(amplitude)
            else:
                set_amplitude(0.0)

        time.sleep(0.05)

    sd.wait()

    if set_amplitude:
        set_amplitude(0.0)
