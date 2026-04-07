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

Avatar uses a pygame sprite-based renderer with Win32 transparency.
Sprites live in assets/sprites/.

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

# Name variants Whisper might produce (case-insensitive matching)
ARIA_VARIANTS = {"aria", "arya", "area", "ariya", "ariia"}

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

                    # Check if it's addressed to Aria
                    if not is_addressed_to_aria(text):
                        set_idle()
                        continue

                    print(f"\n[Chan] {text}")

                    # Check for exit commands
                    text_lower = text.strip().lower()
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
        avatar.close()


def run_aria():
    """Initialise all systems, then launch avatar + voice pipeline.

    The avatar runs on the main thread (pygame requirement).
    The voice pipeline runs in a daemon background thread.
    """
    print("[Aria] Initialising systems...")
    print()

    # Initialise core systems
    init_memory()
    load_personality()
    init_scheduler(announce_fn=announce_reminder)
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

    # Create avatar window (main thread) with mode toggle callback
    avatar = create_avatar(on_mode_toggle=toggle_conversation_mode)

    # Launch voice pipeline in a background thread
    pipeline_thread = threading.Thread(
        target=voice_pipeline,
        args=(device_index, avatar),
        daemon=True,
    )
    pipeline_thread.start()

    # Run the avatar mainloop on the main thread (blocks until closed)
    avatar.run()

    print("[Aria] Shutdown complete.")


def main():
    """Main entry point for Aria."""
    print_banner()
    run_aria()


if __name__ == "__main__":
    main()
