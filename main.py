"""
Aria — Personal AI Desktop Assistant
=====================================
Main entry point. Runs the avatar on the main thread and the
voice pipeline in a background thread.

Two modes controlled by a toggle button on the avatar overlay:

  MODE ON  — Always listening. Transcribes everything, responds when
             "Aria" (or a variant) is detected. Full conversation.
  MODE OFF — Sleep mode. Dormant until "Aria" is heard via wake word
             detection. Wakes for one question, then returns to sleep.

Avatar uses VTube Studio via pyvts WebSocket connection.
VTS handles all rendering, lip sync via VB-Audio Virtual Cable.

Usage:
    python main.py
"""

import sys
import threading

from voice.listener import record_audio, get_audio_devices, calibrate_silence
from voice.transcriber import load_model, transcribe_audio
from voice.wake import listen_for_wake_word
from core.brain import think
from core.memory import init_memory
from core.personality import load_personality
from core.scheduler import init_scheduler, shutdown_scheduler
from voice.speaker import speak
from avatar.renderer import (
    create_avatar,
    set_idle      as _renderer_set_idle,
    set_listening as _renderer_set_listening,
    set_thinking  as _renderer_set_thinking,
    set_dormant   as _renderer_set_dormant,
)
from voice.trainer import run_calibration, get_profile_summary
from core.screen_capture import ScreenCapture
from core.proactive_analyst import ProactiveAnalyst, register as register_analyst
from core.logger import get_logger, attach_ui
from core.terminal_ui import AriaUI, RICH_AVAILABLE

log        = get_logger(__name__)        # [Aria]
chan_log   = get_logger("Chan")          # [Chan]   — user speech transcript
capture_log = get_logger("core.screen_capture")  # [Capture] — for capture-init messages

# Name variants Whisper might produce (case-insensitive matching)
# Expanded for large-v2 + British accent phonetic variants
ARIA_VARIANTS = {
    "aria", "arya", "area", "ariya", "ariia",
    "ari", "ariah", "aeria", "era", "aaria",
    "harry", "maria",  # Common misheard variants
}

# ── Conversation mode flag ─────────────────────────────────────────
# ON (set)  = always-listening conversation mode
# OFF (clear) = sleep mode, wake-word only
_conversation_mode = threading.Event()
_conversation_mode.set()  # ON by default


def is_conversation_mode() -> bool:
    """Check whether conversation mode is active.

    Returns:
        True if conversation mode is ON, False if in sleep mode.
    """
    return _conversation_mode.is_set()


def toggle_conversation_mode() -> bool:
    """Toggle between conversation mode and sleep mode.

    Returns:
        The new state — True if conversation mode is now ON.
    """
    if _conversation_mode.is_set():
        _conversation_mode.clear()
        log.info("Conversation mode OFF — entering sleep.")
        set_dormant()
    else:
        _conversation_mode.set()
        log.info("Conversation mode ON.")
        set_idle()
    return _conversation_mode.is_set()


# ── Free conversation mode flag ───────────────────────────────────
# ON (set)  = Aria responds to all speech without name detection
# OFF (clear) = Aria requires her name in every message
_free_conversation = threading.Event()

# Load default from config
try:
    import config as _cfg
    if getattr(_cfg, 'CONVERSATION_MODE_DEFAULT', True):
        _free_conversation.set()   # ON by default
except Exception:
    _free_conversation.set()       # Default to ON if config missing


def is_free_conversation() -> bool:
    """Check whether free conversation mode is active.

    When True, Aria responds to all speech without name detection.
    When False, Aria requires her name in every message.

    Returns:
        True if free conversation mode is ON.
    """
    return _free_conversation.is_set()


def toggle_free_conversation() -> bool:
    """Toggle free conversation mode on or off.

    Returns:
        The new state — True if now ON.
    """
    if _free_conversation.is_set():
        _free_conversation.clear()
        log.info("Conversation mode: OFF — name required.")
        if _ui is not None:
            _ui.set_conv_mode(False)
        speak("Conversation mode off. Say my name to get my attention.")
    else:
        _free_conversation.set()
        log.info("Conversation mode: ON — listening freely.")
        if _ui is not None:
            _ui.set_conv_mode(True)
        speak("Conversation mode on. I'm listening.")
    return _free_conversation.is_set()


# ── Module-level screen capture instance ──────────────────────────
_screen_capture: ScreenCapture | None = None

# ── Module-level proactive analyst instance (Stage 3a) ────────────
# Accessed by core.brain._handle_analysis_toggle to flip analysis mode
# from a voice command. Stays None if Gemini isn't configured.
_proactive_analyst: ProactiveAnalyst | None = None

# ── Module-level UI instance (rich terminal dashboard) ───────────
# Stays None when rich is unavailable or the interactive setup is
# still running. Subsystems push state updates through helper
# wrappers below; the activity log is fed automatically by the
# UILogHandler attached via core.logger.attach_ui().
_ui: AriaUI | None = None


def _set_ui_state(state: str) -> None:
    """Push an avatar state name into the dashboard if the UI is up."""
    if _ui is not None:
        _ui.set_state(state.upper())


# ── State setter wrappers ─────────────────────────────────────────────────────
# Voice pipeline calls these instead of avatar.renderer.set_X directly so each
# state change drives BOTH the VTS avatar AND the rich dashboard's Status panel
# from a single call site. Existing voice_pipeline code is unmodified.

def set_idle() -> None:
    _renderer_set_idle()
    _set_ui_state("IDLE")


def set_listening() -> None:
    _renderer_set_listening()
    _set_ui_state("LISTENING")


def set_thinking() -> None:
    _renderer_set_thinking()
    _set_ui_state("THINKING")


def set_dormant() -> None:
    _renderer_set_dormant()
    _set_ui_state("DORMANT")


def _analyst_set_state(state: str) -> None:
    """Forward a state-name string to the avatar renderer's setter.

    Used by ProactiveAnalyst — keeps the analyst module decoupled from
    the renderer's state-specific function names. Also mirrors the
    state into the rich dashboard so the Status panel updates in step
    with the avatar.

    Args:
        state: One of 'idle', 'listening', 'thinking', 'speaking', 'dormant'.
    """
    # The set_X wrappers below already mirror to the UI, so no extra push needed.
    if state == "idle":
        set_idle()
    elif state == "thinking":
        set_thinking()
    elif state == "listening":
        set_listening()
    elif state == "dormant":
        set_dormant()
    # 'speaking' is handled inside speak() — analyst doesn't need to fire it


def _analyst_trigger_mood(mood: str) -> None:
    """Trigger a VTS mood hotkey from the analyst, if VTS is connected.

    Args:
        mood: Mood tag (e.g. 'THINKING', 'HAPPY').
    """
    try:
        from avatar.renderer import _controller
        if _controller:
            _controller.trigger_mood(mood)
    except Exception:
        pass  # VTS not connected — continue silently


def print_banner():
    """Print the Aria startup banner."""
    print("=" * 50)
    print("  ARIA — Personal AI Desktop Assistant")
    print("  v1.0 — All Systems")
    print("=" * 50)
    print()


def select_audio_device() -> int | None:
    """Let the user pick an audio input device.

    Returns:
        The selected device index, or None for the system default.
    """
    devices = get_audio_devices()
    if not devices:
        log.error("No audio input devices found!")
        sys.exit(1)

    print("Available microphones:")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
    print(f"  [d] Use system default")
    print()

    choice = input("Select device (number or 'd' for default): ").strip().lower()
    if choice == "d" or choice == "":
        log.info("Using system default microphone.")
        return None

    try:
        index = int(choice)
        valid_indices = [d["index"] for d in devices]
        if index in valid_indices:
            selected = next(d for d in devices if d["index"] == index)
            log.info("Using: %s", selected['name'])
            return index
        else:
            log.warning("Invalid device index. Using system default.")
            return None
    except ValueError:
        log.warning("Invalid input. Using system default.")
        return None


def is_addressed_to_aria(text: str) -> bool:
    """Check if the transcribed text is addressed to Aria.

    Args:
        text: The transcribed text.

    Returns:
        True if any Aria name variant is found in the text.
    """
    words = text.strip().lower().replace(",", " ").replace(".", " ").split()
    return any(word in ARIA_VARIANTS for word in words)


def announce_reminder(text: str) -> None:
    """Callback for the scheduler to make Aria speak a reminder aloud.

    Args:
        text: The reminder text to announce.
    """
    speak(text)


def voice_pipeline(device_index: int | None, avatar) -> None:
    """Run the voice pipeline loop in a background thread.

    Branches on conversation mode:
    - ON:  Always listens, transcribes everything, responds to name.
    - OFF: Waits for wake word, handles one query, returns to sleep.

    Args:
        device_index: The audio input device index (None for default).
        avatar: The AvatarWindow instance to update and close on exit.
    """
    # Pre-load Whisper model
    load_model()
    print()

    # Verbal startup greeting
    log.info("All systems online.")
    speak("Hello Chan. I'm online and ready.")
    log.info("Say my name to get my attention, or press Ctrl+C to quit.")
    print()

    set_idle()

    try:
        while True:
            try:
                if is_conversation_mode():
                    # ── CONVERSATION MODE ─────────────────────────────
                    set_listening()
                    audio_data = record_audio(device_index=device_index)

                    if not audio_data:
                        set_idle()
                        continue

                    # Transcribe everything
                    set_thinking()
                    text = transcribe_audio(audio_data)

                    if not text:
                        set_idle()
                        continue

                    # Check for conversation mode toggle commands
                    text_lower = text.strip().lower()
                    if any(phrase in text_lower for phrase in (
                        "conversation mode on", "free conversation on",
                    )):
                        if not is_free_conversation():
                            toggle_free_conversation()
                        set_idle()
                        continue

                    if any(phrase in text_lower for phrase in (
                        "conversation mode off", "free conversation off",
                    )):
                        if is_free_conversation():
                            toggle_free_conversation()
                        set_idle()
                        continue

                    # Name check — bypassed in free conversation mode
                    if not is_free_conversation() and not is_addressed_to_aria(text):
                        set_idle()
                        continue

                    chan_log.info("%s", text)

                    # Check for exit commands
                    if any(cmd in text_lower for cmd in (
                        "quit", "exit", "goodbye", "shut down", "shutdown",
                    )):
                        log.info("Goodbye, Chan. Talk soon.")
                        speak("Goodbye, Chan. Talk soon.")
                        break

                    # Think and respond
                    set_thinking()
                    log.info("Thinking...")
                    response = think(text)
                    log.info("%s", response)

                    # Mirror to UI's Last Response panel before speaking
                    if _ui is not None and response:
                        _ui.set_last_response(response)

                    # Speak (avatar state handled inside speak())
                    speak(response)
                    set_idle()

                else:
                    # ── SLEEP MODE ────────────────────────────────────
                    set_dormant()

                    # Block until "Aria" is heard, or mode toggles back ON
                    detected = listen_for_wake_word(
                        device_index=device_index,
                        stop_event=_conversation_mode,
                    )

                    if not detected:
                        # Mode was toggled back to ON — loop will catch it
                        continue

                    # Briefly wake up for one question
                    set_listening()
                    log.info("I'm listening...")
                    audio_data = record_audio(device_index=device_index)

                    if not audio_data:
                        set_dormant()
                        continue

                    set_thinking()
                    text = transcribe_audio(audio_data)

                    if text:
                        chan_log.info("%s", text)
                        log.info("Thinking...")
                        response = think(text)
                        log.info("%s", response)
                        if _ui is not None and response:
                            _ui.set_last_response(response)
                        speak(response)

                    # Return to sleep after answering
                    set_dormant()

            except KeyboardInterrupt:
                log.info("Interrupted. Goodbye, Chan.")
                break
    finally:
        shutdown_scheduler()
        if _screen_capture:
            _screen_capture.stop()
        if _proactive_analyst:
            _proactive_analyst.stop()
        if _ui is not None:
            _ui.stop()
        avatar.close()


def run_aria():
    """Initialise all systems, then launch avatar + voice pipeline.

    The avatar connects to VTube Studio in a background thread.
    The voice pipeline runs in a daemon background thread.
    """
    log.info("Initialising systems...")
    print()

    # Initialise core systems
    init_memory()
    load_personality()
    init_scheduler(announce_fn=announce_reminder)
    print()

    # Initialise screen capture if enabled
    global _screen_capture
    try:
        import config
        if getattr(config, 'SCREEN_CAPTURE_ENABLED', False):
            _screen_capture = ScreenCapture(
                interval=getattr(config, 'SCREEN_CAPTURE_INTERVAL', 5.0)
            )
            _screen_capture.start()
        else:
            capture_log.info("Screen capture disabled — set SCREEN_CAPTURE_ENABLED = True in config.py to enable.")
    except Exception as e:
        capture_log.error("Could not start screen capture — %s", e)
    print()

    # Log conversation mode status
    if is_free_conversation():
        log.info("Conversation mode: ON — responding to all speech.")
    else:
        log.info("Conversation mode: OFF — name required.")
    print()

    # Select microphone (must happen before threading)
    device_index = select_audio_device()
    print()

    # Calibrate microphone
    calibrate_silence(device_index=device_index)
    print()

    # Voice profile status and optional calibration
    print(get_profile_summary())
    cal_choice = input("Run voice calibration? (y/N): ").strip().lower()
    if cal_choice == "y":
        run_calibration(device_index=device_index)
    print()

    # Create avatar (connects to VTube Studio in background)
    avatar = create_avatar(on_mode_toggle=toggle_conversation_mode)

    # Initialise proactive analyst (Stage 3a)
    # Defaults to OFF — user opts in via "Aria, analysis mode on".
    global _proactive_analyst
    try:
        _proactive_analyst = ProactiveAnalyst(
            speak_fn=speak,
            set_state_fn=_analyst_set_state,
            trigger_mood_fn=_analyst_trigger_mood,
        )
        _proactive_analyst.start()
        # Register with the module-level singleton so core.brain can reach
        # the live instance via core.proactive_analyst.get_instance().
        register_analyst(_proactive_analyst)
        # Definitive confirmation — if this line appears, the analyst object
        # exists, the daemon thread is running, and brain.py can reach it.
        analyst_log = get_logger("core.proactive_analyst")
        analyst_log.info("Startup confirmed.")
    except Exception as e:
        # Loud and unambiguous so the line cannot be lost in startup noise.
        analyst_log = get_logger("core.proactive_analyst")
        analyst_log.error("FAILED to start: %s", e, exc_info=True)
        _proactive_analyst = None
    print()

    # Initialise the rich terminal dashboard (Prompt 3 — Phase 17).
    # The UI mirrors logger output into an activity panel, exposes Aria's
    # current state + last response, and polls VTS / analyst status. If
    # rich isn't installed we fall back to the previous avatar.run() which
    # blocks the main thread on a quiet sleep loop. set_conv_mode is
    # seeded so the panel matches the actual flag at startup.
    global _ui
    if RICH_AVAILABLE:
        _ui = AriaUI()
        _ui.set_conv_mode(is_free_conversation())
        attach_ui(_ui)
        log.info("Rich terminal dashboard ready — press Ctrl+C to quit.")
    else:
        log.warning("rich not installed — running with plain terminal output.")
        _ui = None
    print()

    # Launch voice pipeline in a background thread
    pipeline_thread = threading.Thread(
        target=voice_pipeline,
        args=(device_index, avatar),
        daemon=True,
    )
    pipeline_thread.start()

    # Block the main thread on the dashboard until Ctrl+C. With no UI we
    # fall back to avatar.run() which keeps the VTS connection alive on
    # a sleep loop until interrupted — same behaviour as before this PR.
    if _ui is not None:
        _ui.run()
    else:
        avatar.run()

    log.info("Shutdown complete.")


def main():
    """Main entry point for Aria."""
    print_banner()
    run_aria()


if __name__ == "__main__":
    main()
