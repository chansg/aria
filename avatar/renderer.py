"""
Aria Avatar Renderer
====================
High-level interface for controlling the avatar's visual state.
Wraps the AvatarWindow and provides simple methods for the
voice pipeline to update what the avatar is doing.
"""

from avatar.window import (
    AvatarWindow,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_THINKING,
    STATE_SPEAKING,
    STATE_DORMANT,
)

# Module-level avatar instance
_avatar: AvatarWindow = None


def create_avatar(on_mode_toggle=None) -> AvatarWindow:
    """Create the avatar window instance.

    Must be called from the main thread before starting the mainloop.

    Args:
        on_mode_toggle: Optional callback that toggles conversation mode.
                        Called when the user clicks the mode button on
                        the avatar overlay. Should return the new state
                        (True = ON, False = OFF).

    Returns:
        The AvatarWindow instance.
    """
    global _avatar
    _avatar = AvatarWindow(on_mode_toggle=on_mode_toggle)
    _avatar.create()
    return _avatar


def get_avatar() -> AvatarWindow:
    """Get the current avatar window instance.

    Returns:
        The AvatarWindow instance, or None if not created.
    """
    return _avatar


def set_idle() -> None:
    """Set the avatar to idle state (default resting state)."""
    if _avatar:
        _avatar.set_state(STATE_IDLE)


def set_listening() -> None:
    """Set the avatar to listening state (mic is active)."""
    if _avatar:
        _avatar.set_state(STATE_LISTENING)


def set_thinking() -> None:
    """Set the avatar to thinking state (waiting for AI response)."""
    if _avatar:
        _avatar.set_state(STATE_THINKING)


def set_speaking() -> None:
    """Set the avatar to speaking state (TTS is playing)."""
    if _avatar:
        _avatar.set_state(STATE_SPEAKING)


def set_dormant() -> None:
    """Set the avatar to dormant state (sleep mode)."""
    if _avatar:
        _avatar.set_state(STATE_DORMANT)


def set_amplitude(amplitude: float) -> None:
    """Set the current audio amplitude for lip-sync animation.

    Args:
        amplitude: Normalised amplitude (0.0 to 1.0).
    """
    if _avatar:
        _avatar.set_amplitude(amplitude)
