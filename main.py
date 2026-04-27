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
    set_idle,
    set_listening,
    set_thinking,
    set_dormant,
)
from voice.trainer import run_calibration, get_profile_summary
from core.screen_capture import ScreenCapture
from core.proactive_analyst import ProactiveAnalyst

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
        print("[Aria] Conversation mode OFF — entering sleep.")
        set_dormant()
    else:
        _conversation_mode.set()
        print("[Aria] Conversation mode ON.")
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
        print("[Aria] Conversation mode: OFF — name required.")
        speak("Conversation mode off. Say my name to get my attention.")
    else:
        _free_conversation.set()
        print("[Aria] Conversation mode: ON — listening freely.")
        speak("Conversation mode on. I'm listening.")
    return _free_conversation.is_set()


# ── Module-level screen capture instance ──────────────────────────
_screen_capture: ScreenCapture | None = None

# ── Module-level proactive analyst instance (Stage 3a) ────────────
# Accessed by core.brain._handle_analysis_toggle to flip analysis mode
# from a voice command. Stays None if Gemini isn't configured.
_proactive_analyst: ProactiveAnalyst | None = None


def _analyst_set_state(state: str) -> None:
    """Forward a state-name string to the avatar renderer's setter.

    Used by ProactiveAnalyst — keeps the analyst module decoupled from
    the renderer's state-specific function names.

    Args:
        state: One of 'idle', 'listening', 'thinking', 'speaking', 'dormant'.
    """
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
        print("[Aria] ERROR: No audio input devices found!")
        sys.exit(1)

    print("Available microphones:")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
    print(f"  [d] Use system default")
    print()

    choice = input("Select device (number or 'd' for default): ").strip().lower()
    if choice == "d" or choice == "":
        print("[Aria] Using system default microphone.")
        return None

    try:
        index = int(choice)
        valid_indices = [d["index"] for d in devices]
        if index in valid_indices:
            selected = next(d for d in devices if d["index"] == index)
            print(f"[Aria] Using: {selected['name']}")
            return index
        else:
            print(f"[Aria] Invalid device index. Using system default.")
            return None
    except ValueError:
        print("[Aria] Invalid input. Using system default.")
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
    print("[Aria] All systems online.")
    speak("Hello Chan. I'm online and ready.")
    print("[Aria] Say my name to get my attention, or press Ctrl+C to quit.")
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

                    print(f"\n[Chan] {text}")

                    # Check for exit commands
                    if any(cmd in text_lower for cmd in (
                        "quit", "exit", "goodbye", "shut down", "shutdown",
                    )):
                        print("\n[Aria] Goodbye, Chan. Talk soon.")
                        speak("Goodbye, Chan. Talk soon.")
                        break

                    # Think and respond
                    set_thinking()
                    print("[Aria] Thinking...")
                    response = think(text)
                    print(f"\n[Aria] {response}\n")

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
                    print("[Aria] I'm listening...")
                    audio_data = record_audio(device_index=device_index)

                    if not audio_data:
                        set_dormant()
                        continue

                    set_thinking()
                    text = transcribe_audio(audio_data)

                    if text:
                        print(f"\n[Chan] {text}")
                        print("[Aria] Thinking...")
                        response = think(text)
                        print(f"\n[Aria] {response}\n")
                        speak(response)

                    # Return to sleep after answering
                    set_dormant()

            except KeyboardInterrupt:
                print("\n[Aria] Interrupted. Goodbye, Chan.")
                break
    finally:
        shutdown_scheduler()
        if _screen_capture:
            _screen_capture.stop()
        if _proactive_analyst:
            _proactive_analyst.stop()
        avatar.close()


def run_aria():
    """Initialise all systems, then launch avatar + voice pipeline.

    The avatar connects to VTube Studio in a background thread.
    The voice pipeline runs in a daemon background thread.
    """
    print("[Aria] Initialising systems...")
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
            print("[Capture] Screen capture disabled — set SCREEN_CAPTURE_ENABLED = True in config.py to enable.")
    except Exception as e:
        print(f"[Capture] ERROR: Could not start screen capture — {e}")
    print()

    # Log conversation mode status
    if is_free_conversation():
        print("[Aria] Conversation mode: ON — responding to all speech.")
    else:
        print("[Aria] Conversation mode: OFF — name required.")
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
    except Exception as e:
        print(f"[Analyst] WARNING: Could not start proactive analyst — {e}")
        _proactive_analyst = None
    print()

    # Launch voice pipeline in a background thread
    pipeline_thread = threading.Thread(
        target=voice_pipeline,
        args=(device_index, avatar),
        daemon=True,
    )
    pipeline_thread.start()

    # Block main thread until exit (VTS handles its own window)
    avatar.run()

    print("[Aria] Shutdown complete.")


def main():
    """Main entry point for Aria."""
    print_banner()
    run_aria()


if __name__ == "__main__":
    main()
