"""
avatar/animations.py
--------------------
Animation state definitions for Aria's visual placeholder.

The current renderer is intentionally local and lightweight. These
constants remain as the stable vocabulary for any future visual layer.
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
