"""
avatar/renderer.py
------------------
Visual-state facade for Aria.

The previous external avatar integration has been removed. This module
keeps the public surface used by main.py, speaker.py, and brain.py while
backing it with a lightweight local placeholder. That lets the core voice
and reasoning pipeline keep emitting state/mood signals without depending
on any external visual model.

Future visual layers should implement the same small interface:
    create_avatar()  - initialise and return a handle
    set_idle()       - set visual state to idle
    set_listening()  - set visual state to listening
    set_thinking()   - set visual state to thinking
    set_speaking()   - set visual state to speaking
    set_dormant()    - set visual state to dormant
    trigger_mood()   - record or render a mood cue
"""

from __future__ import annotations

import time

from core.logger import get_logger

log = get_logger(__name__)

_controller: "PlaceholderAvatar | None" = None


def create_avatar(on_mode_toggle=None) -> "AvatarHandle":
    """Initialise the local visual placeholder.

    Args:
        on_mode_toggle: Optional callback for mode toggle (retained for
                        compatibility with main.py).

    Returns:
        An AvatarHandle instance with a run() and close() method.
    """
    global _controller
    _controller = PlaceholderAvatar()
    _controller.start()
    return AvatarHandle(_controller, on_mode_toggle=on_mode_toggle)


def set_idle() -> None:
    """Set avatar to idle state."""
    if _controller:
        _controller.set_state("idle")


def set_listening() -> None:
    """Set avatar to listening state."""
    if _controller:
        _controller.set_state("listening")


def set_thinking() -> None:
    """Set avatar to thinking state."""
    if _controller:
        _controller.set_state("thinking")


def set_speaking() -> None:
    """Set avatar to speaking state."""
    if _controller:
        _controller.set_state("speaking")


def set_dormant() -> None:
    """Set avatar to dormant/sleeping state."""
    if _controller:
        _controller.set_state("dormant")


def set_amplitude(amplitude: float) -> None:
    """Accept an audio amplitude signal for future visual layers.

    The placeholder intentionally does nothing with amplitude. Keeping this
    hook avoids coupling speaker.py to a specific future renderer.

    Args:
        amplitude: Normalised amplitude (0.0 to 1.0). Ignored.
    """
    if _controller:
        _controller.set_amplitude(amplitude)


def trigger_mood(mood: str) -> None:
    """Record a mood cue for the active placeholder.

    Args:
        mood: Mood tag from a model response, e.g. HAPPY or THINKING.
    """
    if _controller:
        _controller.trigger_mood(mood)


def get_status() -> str:
    """Return a short status string for the terminal dashboard."""
    if not _controller:
        return "Not started"
    return _controller.status


class PlaceholderAvatar:
    """Minimal in-process placeholder for Aria's future visual layer.

    It stores the current state and last mood, logs startup once, and avoids
    all external connections. This is deliberately boring: the visual system
    is now an adapter boundary instead of a runtime dependency.
    """

    def __init__(self) -> None:
        self.state = "idle"
        self.last_mood = "NEUTRAL"
        self.running = False

    @property
    def status(self) -> str:
        return f"Placeholder - {self.state}"

    def start(self) -> None:
        self.running = True
        log.info("Visual placeholder active.")

    def stop(self) -> None:
        if self.running:
            log.info("Visual placeholder stopped.")
        self.running = False

    def set_state(self, state: str) -> None:
        self.state = state

    def set_amplitude(self, amplitude: float) -> None:
        return

    def trigger_mood(self, mood: str) -> None:
        self.last_mood = mood.upper()


class AvatarHandle:
    """Lightweight handle returned by create_avatar().

    The handle manages placeholder lifecycle and provides a blocking run()
    fallback when the rich dashboard is unavailable.

    Attributes:
        controller: The underlying PlaceholderAvatar instance.
    """

    def __init__(self, controller: PlaceholderAvatar, on_mode_toggle=None):
        """Initialise the handle with the placeholder controller.

        Args:
            controller: The active PlaceholderAvatar instance.
            on_mode_toggle: Optional mode toggle callback from main.py.
        """
        self.controller = controller
        self._on_mode_toggle = on_mode_toggle

    def run(self) -> None:
        """Block the main thread until Aria exits.

        Keeps the process alive while the voice pipeline runs in the
        background thread when the rich dashboard is not active.
        """
        log.info("Visual placeholder active. Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutting down...")

    def close(self) -> None:
        """Stop the placeholder controller."""
        if self.controller:
            self.controller.stop()
