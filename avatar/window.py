"""
Aria Avatar Window (Pygame)
===========================
Desktop overlay using pygame with two rendering modes:

  Transparent mode (AVATAR_USE_TRANSPARENT_BG = True):
    Win32 colour-key transparency — magenta pixels become invisible.

  Dark-card mode (AVATAR_USE_TRANSPARENT_BG = False):
    Solid dark background with window-level alpha transparency.
    Reliable across all Windows 10/11 DWM configurations.

Renders sprite-based chibi Kurisu Makise with state-driven animations:
idle (breathing + blink), listening (bounce), thinking (glow pulse),
speaking (lip-sync sprite swap), dormant (sleep sprite, dimmed).

Includes a clickable mode toggle button (top-right corner) that
switches between conversation mode and sleep mode.
"""

import os
import sys
import math
import ctypes
import ctypes.wintypes
import pygame
from avatar.animations import AnimationState, TICK_INTERVAL
from config import (
    AVATAR_USE_TRANSPARENT_BG,
    AVATAR_BG_COLOUR,
    AVATAR_ALPHA,
)

# Avatar states
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
STATE_DORMANT = "dormant"

# Window dimensions
AVATAR_SIZE = 256
PADDING = 20
FPS = 20

# Transparency colour key (magenta — never appears in artwork)
COLORKEY = (255, 0, 255)

# Toggle button rect (top-right corner)
BTN_W, BTN_H = 50, 20
BTN_X1 = AVATAR_SIZE - BTN_W - 8
BTN_Y1 = 6
BTN_X2 = BTN_X1 + BTN_W
BTN_Y2 = BTN_Y1 + BTN_H

# Win32 constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
LWA_COLORKEY = 0x00000001
LWA_ALPHA = 0x00000002
HWND_TOPMOST = ctypes.wintypes.HWND(-1)
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

# Fallback sprite colours per state (used when no PNGs are available)
FALLBACK_COLOURS = {
    "idle":   (32, 178, 170),    # Teal
    "talk_1": (91, 158, 255),    # Blue
    "talk_2": (70, 130, 220),    # Darker blue
    "blink":  (32, 150, 150),    # Muted teal
    "sleep":  (100, 100, 120),   # Grey
    "wake":   (95, 224, 122),    # Green
}


def _hex_to_rgb(hex_colour: str) -> tuple:
    """Convert a hex colour string to an RGB tuple.

    Args:
        hex_colour: Colour string like '#0D0D1A'.

    Returns:
        Tuple of (R, G, B) integers.
    """
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _setup_layered_window(hwnd: int, use_transparent_bg: bool) -> bool:
    """Configure Win32 layered window for transparency.

    Args:
        hwnd: The window handle.
        use_transparent_bg: If True, uses colour-key transparency.
                           If False, uses window-level alpha.

    Returns:
        True if all Win32 calls succeeded, False otherwise.
    """
    user32 = ctypes.windll.user32
    hwnd_handle = ctypes.wintypes.HWND(hwnd)

    # Add layered + toolwindow (hide from taskbar)
    print("[Aria] DEBUG: Setting WS_EX_LAYERED | WS_EX_TOOLWINDOW...")
    ex_style = user32.GetWindowLongW(hwnd_handle, GWL_EXSTYLE)
    ex_style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW
    result = user32.SetWindowLongW(hwnd_handle, GWL_EXSTYLE, ex_style)
    if result == 0:
        error = ctypes.get_last_error()
        print(f"[Aria] WARNING: SetWindowLongW failed (error={error})")

    if use_transparent_bg:
        # Colour-key: magenta pixels become invisible
        colorref = COLORKEY[0] | (COLORKEY[1] << 8) | (COLORKEY[2] << 16)
        print(f"[Aria] DEBUG: Setting colour-key transparency (colorref={colorref:#x})...")
        ok = user32.SetLayeredWindowAttributes(
            hwnd_handle, ctypes.wintypes.COLORREF(colorref), 255,
            LWA_COLORKEY,
        )
        if not ok:
            print(f"[Aria] WARNING: SetLayeredWindowAttributes (colorkey) failed")
            return False
    else:
        # Window-level alpha: entire window is semi-transparent
        alpha = int(AVATAR_ALPHA * 255)
        print(f"[Aria] DEBUG: Setting window alpha={alpha}/255...")
        ok = user32.SetLayeredWindowAttributes(
            hwnd_handle, 0, alpha, LWA_ALPHA,
        )
        if not ok:
            print(f"[Aria] WARNING: SetLayeredWindowAttributes (alpha) failed")
            return False

    # Force always-on-top and ensure window is shown
    print("[Aria] DEBUG: Setting HWND_TOPMOST + SWP_SHOWWINDOW...")
    ok = user32.SetWindowPos(
        hwnd_handle, HWND_TOPMOST,
        0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )
    if not ok:
        print(f"[Aria] WARNING: SetWindowPos (topmost) failed")
        return False

    print("[Aria] DEBUG: Win32 transparency setup complete.")
    return True


def _generate_fallback_sprites(bg_colour: tuple) -> dict:
    """Generate simple coloured-circle placeholder sprites.

    Called when no PNG sprite files are found. Ensures the avatar
    always displays something visible so the window can be confirmed
    working.

    Args:
        bg_colour: The (R, G, B) background fill colour.

    Returns:
        Dict of {name: pygame.Surface} for all 6 sprite slots.
    """
    print("[Aria] Generating fallback placeholder sprites...")
    sprites = {}

    try:
        label_font = pygame.font.SysFont("Segoe UI", 14, bold=True)
    except Exception:
        label_font = pygame.font.Font(None, 18)

    labels = {
        "idle": "IDLE", "talk_1": "TALK", "talk_2": "TALK",
        "blink": "BLINK", "sleep": "SLEEP", "wake": "WAKE",
    }

    for name, colour in FALLBACK_COLOURS.items():
        surf = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE))
        surf.fill(bg_colour)
        cx, cy = AVATAR_SIZE // 2, AVATAR_SIZE // 2

        # Draw a coloured circle as the "avatar"
        pygame.draw.circle(surf, colour, (cx, cy), 70)
        # Dark outline
        pygame.draw.circle(surf, (20, 20, 20), (cx, cy), 70, 3)

        # State label inside the circle
        label = labels.get(name, name.upper())
        label_surf = label_font.render(label, True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=(cx, cy))
        surf.blit(label_surf, label_rect)

        sprites[name] = surf

    print(f"[Aria] Generated {len(sprites)} fallback sprites.")
    return sprites


class AvatarWindow:
    """Pygame-based desktop overlay window for the Aria avatar."""

    def __init__(self, on_mode_toggle=None):
        """Initialise the avatar window.

        Args:
            on_mode_toggle: Callback that toggles conversation mode.
                           Returns the new state (True = ON, False = OFF).
        """
        self.screen = None
        self.clock = None
        self.hwnd = None
        self._state = STATE_IDLE
        self._anim = AnimationState()
        self._running = False
        self._sprites = {}         # name -> pygame.Surface
        self._amplitude = 0.0      # Current audio amplitude for lip sync
        self._on_mode_toggle = on_mode_toggle
        self._mode_on = True       # Current display state for button label
        self._bg_colour = _hex_to_rgb(AVATAR_BG_COLOUR) if not AVATAR_USE_TRANSPARENT_BG else COLORKEY
        self._dragging = False
        self._drag_offset = (0, 0)
        self._press_pos = (0, 0)   # For distinguishing clicks from drags

    def create(self) -> None:
        """Create and configure the pygame window.

        Loads sprites, positions the window in the bottom-right corner,
        and applies Win32 transparency settings. Falls back to placeholder
        sprites if PNGs are missing.
        Must be called from the main thread.
        """
        pos_str = self._bottom_right_pos()
        os.environ["SDL_VIDEO_WINDOW_POS"] = pos_str
        print(f"[Aria] DEBUG: SDL_VIDEO_WINDOW_POS = {pos_str}")

        print("[Aria] DEBUG: Initialising pygame...")
        pygame.init()
        print(f"[Aria] DEBUG: pygame.init() complete — display driver: {pygame.display.get_driver()}")

        print(f"[Aria] DEBUG: Creating {AVATAR_SIZE}x{AVATAR_SIZE} NOFRAME window...")
        self.screen = pygame.display.set_mode(
            (AVATAR_SIZE, AVATAR_SIZE),
            pygame.NOFRAME | pygame.SHOWN,
        )
        pygame.display.set_caption("Aria")
        self.clock = pygame.time.Clock()
        print(f"[Aria] DEBUG: Display surface created: {self.screen.get_size()}")

        # Get Win32 window handle and apply transparency
        info = pygame.display.get_wm_info()
        self.hwnd = info.get("window")
        print(f"[Aria] DEBUG: Win32 HWND = {self.hwnd}")

        if self.hwnd:
            ok = _setup_layered_window(self.hwnd, AVATAR_USE_TRANSPARENT_BG)
            if not ok:
                print("[Aria] WARNING: Win32 transparency setup had failures — "
                      "avatar may not render correctly.")
        else:
            print("[Aria] WARNING: Could not get Win32 HWND — "
                  "transparency and always-on-top will not work.")

        # Load sprites (PNG files or fallback placeholders)
        self._load_sprites()

        self._running = True

        # Report final position
        parts = pos_str.split(",")
        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        print(f"[Aria] Avatar window created at +{parts[0]}+{parts[1]} "
              f"(screen: {sw}x{sh}, mode: "
              f"{'transparent' if AVATAR_USE_TRANSPARENT_BG else 'dark-card'})")

    def _bottom_right_pos(self) -> str:
        """Calculate the bottom-right screen position above the taskbar.

        Uses SPI_GETWORKAREA to get the usable screen area (excluding
        the taskbar), ensuring the avatar is never hidden behind it.

        Returns:
            Position string in 'x,y' format for SDL_VIDEO_WINDOW_POS.
        """
        user32 = ctypes.windll.user32

        # Get work area (screen minus taskbar)
        work_area = ctypes.wintypes.RECT()
        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)
        work_w = work_area.right - work_area.left
        work_h = work_area.bottom - work_area.top

        x = work_area.left + work_w - AVATAR_SIZE - PADDING
        y = work_area.top + work_h - AVATAR_SIZE - PADDING

        print(f"[Aria] DEBUG: Work area: {work_w}x{work_h} "
              f"(top-left: {work_area.left},{work_area.top})")
        return f"{x},{y}"

    def _load_sprites(self) -> None:
        """Load all sprite images from assets/sprites/.

        Each sprite is pre-composited onto the background colour so that
        RGBA transparency renders correctly on the non-alpha display
        surface. If any sprites are missing, generates coloured-circle
        fallback placeholders.
        """
        from config import ASSETS_DIR

        sprite_dir = os.path.join(ASSETS_DIR, "sprites")
        sprite_files = {
            "idle": "aria_idle.png",
            "talk_1": "aria_talk_1.png",
            "talk_2": "aria_talk_2.png",
            "blink": "aria_blink.png",
            "sleep": "aria_sleep.png",
            "wake": "aria_wake.png",
        }

        print(f"[Aria] DEBUG: Loading sprites from {sprite_dir}...")

        for name, filename in sprite_files.items():
            path = os.path.join(sprite_dir, filename)
            if os.path.exists(path):
                try:
                    raw_surf = pygame.image.load(path).convert_alpha()
                    # Pre-composite onto the background colour so alpha
                    # blending works correctly on the RGB display surface
                    composited = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE))
                    composited.fill(self._bg_colour)
                    composited.blit(raw_surf, (0, 0))
                    self._sprites[name] = composited
                    print(f"[Aria] DEBUG: Loaded {filename} ({raw_surf.get_size()})")
                except Exception as e:
                    print(f"[Aria] WARNING: Failed to load {filename}: {e}")
            else:
                print(f"[Aria] WARNING: Missing sprite: {filename}")

        if not self._sprites:
            # No sprites loaded at all — generate fallback placeholders
            print("[Aria] WARNING: No sprite PNGs found — generating fallback placeholders.")
            self._sprites = _generate_fallback_sprites(self._bg_colour)
        elif len(self._sprites) < 6:
            # Some sprites loaded, fill gaps with fallback
            fallbacks = _generate_fallback_sprites(self._bg_colour)
            for name in sprite_files:
                if name not in self._sprites:
                    self._sprites[name] = fallbacks[name]
                    print(f"[Aria] DEBUG: Using fallback for missing sprite: {name}")
            print(f"[Aria] Loaded {len(self._sprites)} sprites (some fallback).")
        else:
            print(f"[Aria] Loaded {len(self._sprites)} sprites.")

    def run(self) -> None:
        """Run the pygame event loop (blocks — call from main thread).

        Processes events (including mouse clicks for the toggle button),
        updates animations, renders sprites, and maintains framerate.
        """
        if not self.screen:
            return

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_press(event)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._on_release(event)
                elif event.type == pygame.MOUSEMOTION and self._dragging:
                    self._on_drag(event)

            self._anim.tick()
            self._draw()
            self.clock.tick(FPS)

        pygame.quit()
        print("[Aria] Avatar window closed.")

    def set_state(self, state: str) -> None:
        """Update the avatar's visual state (thread-safe).

        Args:
            state: One of STATE_IDLE, STATE_LISTENING, STATE_THINKING,
                   STATE_SPEAKING, STATE_DORMANT.
        """
        self._state = state

    def set_amplitude(self, amplitude: float) -> None:
        """Set the current audio amplitude for lip-sync.

        Args:
            amplitude: Normalised amplitude (0.0 to 1.0).
        """
        self._amplitude = max(0.0, min(1.0, amplitude))

    def close(self) -> None:
        """Close the avatar window (thread-safe)."""
        self._running = False

    # ── Mouse handling ─────────────────────────────────────────────

    def _on_press(self, event) -> None:
        """Handle mouse button press — start potential drag or click.

        Args:
            event: The pygame MOUSEBUTTONDOWN event.
        """
        self._press_pos = (event.pos[0], event.pos[1])
        self._dragging = True
        if self.hwnd:
            self._drag_offset = (event.pos[0], event.pos[1])

    def _on_release(self, event) -> None:
        """Handle mouse button release — detect click vs drag.

        Args:
            event: The pygame MOUSEBUTTONUP event.
        """
        dx = abs(event.pos[0] - self._press_pos[0])
        dy = abs(event.pos[1] - self._press_pos[1])
        self._dragging = False

        # Only treat as a click (not a drag) if movement was tiny
        if dx < 5 and dy < 5:
            self._check_button_click(event.pos[0], event.pos[1])

    def _on_drag(self, event) -> None:
        """Handle mouse drag — move the avatar window.

        Args:
            event: The pygame MOUSEMOTION event.
        """
        if not self.hwnd:
            return

        # Get current mouse position on screen
        cursor = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))

        hwnd_handle = ctypes.wintypes.HWND(self.hwnd)
        new_x = cursor.x - self._drag_offset[0]
        new_y = cursor.y - self._drag_offset[1]

        ctypes.windll.user32.SetWindowPos(
            hwnd_handle, HWND_TOPMOST,
            new_x, new_y, 0, 0,
            SWP_NOSIZE | SWP_NOACTIVATE,
        )

    def _check_button_click(self, x: int, y: int) -> None:
        """Check if the click landed on the mode toggle button.

        Args:
            x: Click x coordinate within the window.
            y: Click y coordinate within the window.
        """
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            if self._on_mode_toggle:
                self._mode_on = self._on_mode_toggle()

    # ── Drawing ────────────────────────────────────────────────────

    def _draw(self) -> None:
        """Draw the current sprite frame with state-based effects."""
        if not self.screen:
            return

        # Clear with background colour
        self.screen.fill(self._bg_colour)

        # Select the sprite to display (always returns a surface now —
        # fallback sprites guarantee we never get None)
        sprite = self._select_sprite()
        if sprite is not None:
            # Calculate position with animation offsets
            x, y = 0, 0
            cx = AVATAR_SIZE // 2
            cy = AVATAR_SIZE // 2

            # State-based position offsets
            if self._state == STATE_LISTENING:
                y += int(self._anim.bounce_offset())
            elif self._state == STATE_IDLE:
                scale = self._anim.breathing_scale()
                if abs(scale - 1.0) > 0.005:
                    new_w = int(AVATAR_SIZE * scale)
                    new_h = int(AVATAR_SIZE * scale)
                    sprite = pygame.transform.smoothscale(sprite, (new_w, new_h))
                    x = (AVATAR_SIZE - new_w) // 2
                    y = (AVATAR_SIZE - new_h) // 2

            # Draw the sprite
            self.screen.blit(sprite, (x, y))

        cx = AVATAR_SIZE // 2
        cy = AVATAR_SIZE // 2

        # Draw glow ring
        self._draw_glow_ring(cx, cy)

        # Draw thinking dots
        if self._state == STATE_THINKING:
            self._draw_thinking_dots(cx, cy)

        # Draw state label
        self._draw_label(cx)

        # Draw mode toggle button (always on top)
        self._draw_toggle_button()

        pygame.display.flip()

    def _select_sprite(self) -> pygame.Surface | None:
        """Select the appropriate sprite for the current state and frame.

        Returns:
            The pygame Surface to render, or None if no sprites loaded.
        """
        if self._state == STATE_DORMANT:
            return self._sprites.get("sleep")

        if self._state == STATE_SPEAKING:
            if self._amplitude > 0.15:
                return self._sprites.get("talk_1")
            else:
                return self._sprites.get("talk_2")

        if self._anim.is_blinking:
            return self._sprites.get("blink")

        if self._state == STATE_LISTENING:
            return self._sprites.get("wake")

        return self._sprites.get("idle")

    def _draw_glow_ring(self, cx: int, cy: int) -> None:
        """Draw a pulsing glow ring around the avatar.

        Args:
            cx: Centre x coordinate.
            cy: Centre y coordinate.
        """
        glow_colours = {
            STATE_IDLE:      (32, 178, 170),   # Teal
            STATE_LISTENING: (95, 224, 122),    # Green
            STATE_THINKING:  (168, 85, 247),    # Purple
            STATE_SPEAKING:  (91, 158, 255),    # Blue
            STATE_DORMANT:   (119, 119, 119),   # Grey
        }

        colour = glow_colours.get(self._state, glow_colours[STATE_IDLE])
        intensity = self._anim.glow_alpha()
        width = max(1, int(3 * intensity))

        pygame.draw.circle(self.screen, colour, (cx, cy), 110, width)

    def _draw_thinking_dots(self, cx: int, cy: int) -> None:
        """Draw orbiting dots around the avatar during thinking state.

        Args:
            cx: Centre x coordinate.
            cy: Centre y coordinate.
        """
        angle = self._anim.thinking_angle()
        colour = (168, 85, 247)  # Purple

        for i in range(3):
            a = angle + i * (2 * math.pi / 3)
            dx = int(100 * math.cos(a))
            dy = int(100 * math.sin(a))
            pygame.draw.circle(self.screen, colour, (cx + dx, cy + dy), 6)

    def _draw_label(self, cx: int) -> None:
        """Draw the state label text below the avatar.

        Args:
            cx: Centre x coordinate for text alignment.
        """
        labels = {
            STATE_IDLE: "Aria",
            STATE_LISTENING: "Listening...",
            STATE_THINKING: "Thinking...",
            STATE_SPEAKING: "Speaking...",
            STATE_DORMANT: "Sleeping...",
        }

        text = labels.get(self._state, "Aria")

        try:
            font = pygame.font.SysFont("Segoe UI", 13, bold=True)
        except Exception:
            font = pygame.font.Font(None, 16)

        shadow_surf = font.render(text, True, (30, 30, 30))
        shadow_rect = shadow_surf.get_rect(center=(cx + 1, AVATAR_SIZE - 14))
        self.screen.blit(shadow_surf, shadow_rect)

        label_surf = font.render(text, True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=(cx, AVATAR_SIZE - 15))
        self.screen.blit(label_surf, label_rect)

    def _draw_toggle_button(self) -> None:
        """Draw the conversation mode toggle button in the top-right corner."""
        if self._mode_on:
            btn_colour = (32, 178, 170)   # Teal = ON
            btn_label = "ON"
        else:
            btn_colour = (85, 85, 85)     # Grey = OFF
            btn_label = "OFF"

        btn_rect = pygame.Rect(BTN_X1, BTN_Y1, BTN_W, BTN_H)
        pygame.draw.rect(self.screen, btn_colour, btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), btn_rect, width=1, border_radius=4)

        try:
            btn_font = pygame.font.SysFont("Segoe UI", 10, bold=True)
        except Exception:
            btn_font = pygame.font.Font(None, 14)

        label_surf = btn_font.render(btn_label, True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=btn_rect.center)
        self.screen.blit(label_surf, label_rect)
