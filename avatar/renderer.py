"""
avatar/renderer.py
------------------
Avatar interface for Aria. Delegates all rendering to VTube Studio
via VTSController. Pygame has been removed entirely.

Provides the same public interface as the old Pygame renderer so
main.py requires minimal changes:
    create_avatar()  — initialises and starts the VTS controller
    set_idle()       — sets avatar to idle state
    set_listening()  — sets avatar to listening state
    set_thinking()   — sets avatar to thinking state
    set_dormant()    — sets avatar to dormant/sleeping state
"""

from avatar.vts_controller import VTSController

_controller: VTSController = None


def create_avatar(on_mode_toggle=None) -> "AvatarHandle":
    """Initialise the VTS controller and connect to VTube Studio.

    Args:
        on_mode_toggle: Optional callback for mode toggle (retained for
                        compatibility with main.py — not used with VTS).

    Returns:
        An AvatarHandle instance with a run() and close() method.
    """
    global _controller
    _controller = VTSController()
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
    """Set the current audio amplitude for lip-sync animation.

    With VTube Studio, lip sync is handled by VB-Audio Virtual Cable
    routing directly to VTS. This function is retained for interface
    compatibility but is a no-op.

    Args:
        amplitude: Normalised amplitude (0.0 to 1.0). Ignored.
    """
    pass


class AvatarHandle:
    """Lightweight handle returned by create_avatar().

    Replaces the old AvatarWindow that ran the Pygame main loop.
    VTS handles its own window — this class just manages lifecycle.

    Attributes:
        controller: The underlying VTSController instance.
    """

    def __init__(self, controller: VTSController, on_mode_toggle=None):
        """Initialise the handle with a VTS controller.

        Args:
            controller: The active VTSController instance.
            on_mode_toggle: Optional mode toggle callback from main.py.
        """
        self.controller = controller
        self._on_mode_toggle = on_mode_toggle

    def run(self) -> None:
        """Block the main thread until Aria exits.

        Previously this ran the Pygame event loop. Now it simply keeps
        the main thread alive while the voice pipeline runs in the
        background thread.
        """
        print("[Avatar] VTube Studio mode active. Press Ctrl+C to quit.")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Aria] Shutting down...")

    def close(self) -> None:
        """Stop the VTS controller and close the connection."""
        if self.controller:
            self.controller.stop()
