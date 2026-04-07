"""
Aria Avatar Animations
======================
Tick-based animation system for the avatar. Each state has its own
animation behaviour:
- Idle: gentle breathing (scale pulse) + periodic eye blinks
- Listening: soft bounce + ear glow pulse
- Thinking: orbital dots + slow colour cycle
- Speaking: mouth open/close cycling (lip sync) + body pulse

All animations are driven by a frame counter updated every tick.
"""

import math
import random

# Animation speed (ms between frames)
TICK_INTERVAL = 50  # 20 FPS

# Blink timing
BLINK_CHANCE = 0.02  # 2% chance per frame to start a blink
BLINK_DURATION = 4   # Frames the eyes stay closed


class AnimationState:
    """Tracks animation frame counters and transient effects."""

    def __init__(self):
        """Initialise all animation counters."""
        self.frame = 0
        self.blink_timer = 0     # Counts down during a blink
        self.mouth_open = False  # Toggles for lip sync
        self.mouth_frame = 0     # Counts frames for mouth cycle

    def tick(self) -> None:
        """Advance all animation counters by one frame."""
        self.frame += 1

        # Blink logic
        if self.blink_timer > 0:
            self.blink_timer -= 1
        elif random.random() < BLINK_CHANCE:
            self.blink_timer = BLINK_DURATION

        # Mouth cycle for speaking (toggles every few frames)
        self.mouth_frame += 1
        if self.mouth_frame >= 4:
            self.mouth_open = not self.mouth_open
            self.mouth_frame = 0

    @property
    def is_blinking(self) -> bool:
        """Whether the avatar's eyes are currently closed."""
        return self.blink_timer > 0

    def breathing_scale(self) -> float:
        """Return a scale factor for the idle breathing animation.

        Returns:
            A float oscillating gently between ~0.97 and ~1.03.
        """
        return 1.0 + 0.03 * math.sin(self.frame * 0.08)

    def bounce_offset(self) -> float:
        """Return a vertical offset for the listening bounce.

        Returns:
            A float oscillating between ~-4 and ~4 pixels.
        """
        return 4.0 * math.sin(self.frame * 0.15)

    def thinking_angle(self) -> float:
        """Return the current angle for orbital thinking dots.

        Returns:
            Angle in radians, continuously rotating.
        """
        return self.frame * 0.1

    def glow_alpha(self) -> float:
        """Return a pulsing intensity for glow effects.

        Returns:
            A float oscillating between 0.3 and 1.0.
        """
        return 0.65 + 0.35 * math.sin(self.frame * 0.12)

    def speaking_mouth_height(self) -> int:
        """Return the mouth opening height for lip sync.

        Returns:
            Pixel height for the mouth oval (0 = closed, varies when open).
        """
        if self.mouth_open:
            # Vary the mouth size for more natural look
            return int(8 + 6 * abs(math.sin(self.frame * 0.2)))
        return 3  # Nearly closed
