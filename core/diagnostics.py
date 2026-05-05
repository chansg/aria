"""
core/diagnostics.py
-------------------
Structured startup and runtime diagnostics for aria.log.

The goal is to make logs shareable with Codex/Claude without exposing secrets.
Only configuration shape, device metadata, package versions, and environment
facts are written here.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from core.logger import get_logger

log = get_logger(__name__)

_PACKAGE_NAMES = (
    "anthropic",
    "faster-whisper",
    "ctranslate2",
    "numpy",
    "noisereduce",
    "PyAudio",
    "piper-tts",
    "rich",
    "sounddevice",
    "soundfile",
    "yfinance",
    "google-generativeai",
    "httpx",
    "kokoro-onnx",
    "onnxruntime",
)

_SAFE_CONFIG_FIELDS = (
    "USE_LOCAL_MODEL",
    "USE_LOCAL_FALLBACK",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MODEL_LITE",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "WHISPER_MODEL_SIZE",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "WAKE_WORD",
    "SAMPLE_RATE",
    "CHANNELS",
    "CHUNK_SIZE",
    "MIN_RECORDING_DURATION",
    "SILENCE_DURATION",
    "MAX_RECORDING_DURATION",
    "TRANSCRIPTION_TIMEOUT",
    "NOTIFICATIONS_PATH",
    "PIPER_SPEAKING_RATE",
    "TTS_PROVIDER",
    "TTS_CONVERSATION_PROVIDER",
    "TTS_FALLBACK_PROVIDER",
    "TTS_FAIL_LOUD",
    "TTS_MAX_CHUNK_CHARS",
    "TTS_TRIM_SILENCE",
    "TTS_SILENCE_THRESHOLD",
    "TTS_SILENCE_PADDING_MS",
    "KOKORO_ONNX_PROVIDER",
    "KOKORO_DISABLE_PROVIDER_FALLBACK",
    "KOKORO_VOICE",
    "KOKORO_LANG",
    "KOKORO_SPEED",
    "PROACTIVE_ANALYST_SPEAK_INSIGHTS",
    "SCREEN_CAPTURE_ENABLED",
    "SCREEN_CAPTURE_INTERVAL",
    "CONVERSATION_MODE_DEFAULT",
    "VISUAL_PLACEHOLDER_ENABLED",
    "TRADING212_ENV",
    "TRADING212_BASE_URL",
    "TRADING212_TIMEOUT_SECONDS",
    "TRADING212_AUDIT_LOG_PATH",
    "TRADING212_TRAINING_LOG_PATH",
    "MARKET_DATA_TIMEOUT_SECONDS",
    "MARKET_QUOTE_TIMEOUT_SECONDS",
)

_SECRET_FIELDS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "PICOVOICE_API_KEY",
    "OPENWEATHER_API_KEY",
    "TRADING212_API_KEY",
    "TRADING212_API_SECRET",
)


def _run_git(args: list[str], cwd: Path) -> str:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=creationflags,
        )
    except Exception as exc:
        return f"unavailable ({exc})"

    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"unavailable ({output or completed.returncode})"
    return output


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"
    except Exception as exc:
        return f"unknown ({exc})"


def _format_config_value(value) -> str:
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, (list, tuple, set)):
        return repr(list(value))
    return f"<{type(value).__name__}>"


def log_startup_diagnostics() -> None:
    """Write a compact diagnostic snapshot to aria.log."""
    cwd = Path.cwd()

    log.info(
        "Runtime: python=%s executable=%s platform=%s pid=%s cwd=%s",
        platform.python_version(),
        sys.executable,
        platform.platform(),
        os.getpid(),
        cwd,
    )
    log.debug("Process argv: %r", sys.argv)
    log.debug("Virtual env: %s", os.environ.get("VIRTUAL_ENV") or "not-set")

    branch = _run_git(["branch", "--show-current"], cwd)
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    status = _run_git(["status", "--short"], cwd)
    dirty_count = 0 if not status or status.startswith("unavailable") else len(status.splitlines())
    log.info("Git: branch=%s commit=%s dirty_files=%s", branch, commit, dirty_count)
    if status and not status.startswith("unavailable"):
        log.debug("Git dirty files: %s", "; ".join(status.splitlines()))

    package_versions = ", ".join(
        f"{package}={_package_version(package)}" for package in _PACKAGE_NAMES
    )
    log.info("Dependency versions: %s", package_versions)

    try:
        import config
    except Exception as exc:
        log.error("Could not import config for diagnostics: %s", exc, exc_info=True)
        return

    safe_values = []
    for field in _SAFE_CONFIG_FIELDS:
        if hasattr(config, field):
            safe_values.append(f"{field}={_format_config_value(getattr(config, field))}")
    log.info("Config snapshot: %s", ", ".join(safe_values))

    secret_presence = []
    for field in _SECRET_FIELDS:
        if hasattr(config, field):
            secret_presence.append(f"{field}_present={bool(getattr(config, field))}")
    if secret_presence:
        log.info("Secret presence: %s", ", ".join(secret_presence))

    for label, field in (
        ("Piper model path", "PIPER_MODEL_PATH"),
        ("Kokoro ONNX model path", "KOKORO_ONNX_MODEL_PATH"),
        ("Kokoro voices path", "KOKORO_ONNX_VOICES_PATH"),
    ):
        path_value = getattr(config, field, None)
        if path_value:
            model_path = Path(path_value)
            log.info("%s: %s exists=%s", label, model_path, model_path.exists())


def log_audio_devices(devices: Iterable[dict]) -> None:
    """Write detailed audio input device metadata to aria.log."""
    device_list = list(devices)
    log.info("Audio input devices detected: %d", len(device_list))
    for device in device_list:
        log.debug(
            "Audio input device: index=%s name=%r channels=%s host_api=%r "
            "default_rate=%r is_default=%s",
            device.get("index"),
            device.get("name"),
            device.get("channels"),
            device.get("host_api"),
            device.get("default_sample_rate"),
            device.get("is_default"),
        )


def log_selected_audio_device(device: dict | None) -> None:
    """Write the selected audio device in a single easy-to-find log line."""
    if device is None:
        log.info("Selected audio input: system default")
        return

    log.info(
        "Selected audio input: index=%s name=%r channels=%s host_api=%r "
        "default_rate=%r is_default=%s",
        device.get("index"),
        device.get("name"),
        device.get("channels"),
        device.get("host_api"),
        device.get("default_sample_rate"),
        device.get("is_default"),
    )
