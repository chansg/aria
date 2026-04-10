"""
avatar/animations.py
--------------------
Animation state definitions for Aria's VTube Studio avatar.

Previously drove Pygame sprite swapping. Now maps Aria's internal
states to descriptive labels used by vts_controller.py for hotkey
triggering. Actual animations are handled by VTube Studio.
"""

# Avatar state constants — used across the codebase
STATE_IDLE      = "idle"
STATE_LISTENING = "listening"
STATE_THINKING  = "thinking"
STATE_DORMANT   = "dormant"

# Mood tag constants — returned by brain.py alongside response text
MOOD_HAPPY     = "HAPPY"
MOOD_NEUTRAL   = "NEUTRAL"
MOOD_THINKING  = "THINKING"
MOOD_SURPRISED = "SURPRISED"
MOOD_SAD       = "SAD"
