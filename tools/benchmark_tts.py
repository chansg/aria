"""
Benchmark Aria's configured TTS providers.

Usage:
    python tools/benchmark_tts.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice.speaker import (
    _clean_for_speech,
    _mono_for_amplitude,
    _normalise_playback,
    _normalise_provider_name,
    _provider_chain,
    _synthesize_with_fallback,
    _trim_edge_silence,
)


DEFAULT_TEXTS = (
    "Warm up.",
    "Hello Chan. Kokoro is online and using the configured provider.",
    "Analysis mode is on. I will log anything important without interrupting your conversation.",
)


def _run_once(text: str, provider: str | None) -> None:
    text = _clean_for_speech(text)
    start = time.perf_counter()
    result, provider_name = _synthesize_with_fallback(text, preferred_name=provider)
    elapsed = time.perf_counter() - start
    playback = _normalise_playback(result.samples)
    trimmed = _trim_edge_silence(playback, int(result.sample_rate))
    duration = len(_mono_for_amplitude(playback)) / result.sample_rate if result.sample_rate else 0
    trimmed_duration = len(_mono_for_amplitude(trimmed)) / result.sample_rate if result.sample_rate else 0
    realtime_factor = elapsed / trimmed_duration if trimmed_duration else 0
    print(
        f"provider={provider_name} chars={len(text)} "
        f"elapsed={elapsed:.2f}s audio={duration:.2f}s "
        f"trimmed_audio={trimmed_duration:.2f}s rtf={realtime_factor:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider to benchmark. Defaults to configured TTS_PROVIDER.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Optional custom text. If omitted, runs a warmup and two sample phrases.",
    )
    args = parser.parse_args()

    provider = _normalise_provider_name(args.provider) if args.provider else None
    print(f"provider_chain={_provider_chain(provider)}")

    texts = (args.text,) if args.text else DEFAULT_TEXTS
    for text in texts:
        _run_once(text, provider)


if __name__ == "__main__":
    main()
