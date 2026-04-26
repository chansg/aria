"""
avatar/vts_controller.py
------------------------
VTube Studio WebSocket controller for Aria.

Manages the connection to VTube Studio via pyvts, handles authentication
token storage, and provides a clean interface for triggering avatar
states and emotional expressions.

Architecture:
    - Single persistent async connection to VTS on ws://localhost:8001
    - Auth token stored in data/vts_token.json after first approval
    - Avatar states mapped to VTS hotkeys defined in config.py
    - Mood tags from brain.py trigger expression hotkeys
    - Runs in a background asyncio event loop alongside the voice pipeline

Usage:
    controller = VTSController()
    controller.start()           # Start background event loop
    controller.set_state("idle")
    controller.trigger_mood("HAPPY")
    controller.stop()
"""

from __future__ import annotations
import asyncio
import os
import threading
from typing import Optional

import pyvts

# ── Token storage ─────────────────────────────────────────────────────────────
TOKEN_FILE = "data/vts_token.txt"
PLUGIN_NAME = "Aria"
PLUGIN_DEVELOPER = "chansg"
VTS_HOST = "localhost"
VTS_PORT = 8001

# ── State to hotkey mapping — Hiyori_A model ─────────────────────────────────
# These map Aria's internal states to VTS hotkey names.
# Override in config.py if using a different Live2D model.
DEFAULT_STATE_HOTKEYS = {
    "idle":      None,           # Base state — Hiyori rests naturally
    "listening": "hiyori_m05",   # Active listening expression
    "thinking":  "hiyori_m03",   # Pensive, looking to the side
    "speaking":  "hiyori_m01",   # Engaged/cheerful while talking (was None)
    "dormant":   None,           # Sleeping — no hotkey configured yet
}

# ── Mood tag to hotkey mapping — Hiyori_A model ──────────────────────────────
# Maps mood tags returned by brain.py to VTS hotkey names.
# Hotkey names must match exactly what is configured in VTube Studio.
# Override in config.py if using a different Live2D model.
DEFAULT_MOOD_HOTKEYS = {
    "HAPPY":     "hiyori_m01",   # Bright cheerful smile
    "NEUTRAL":   None,           # Base state — no hotkey fired
    "THINKING":  "hiyori_m03",   # Pensive expression
    "SURPRISED": "hiyori_m02",   # Wide eyes, open mouth
    "SAD":       "hiyori_m04",   # Downcast, saddened
}


class VTSController:
    """Manages the VTube Studio WebSocket connection for Aria.

    Runs an asyncio event loop in a background thread so VTS commands
    can be issued from the synchronous voice pipeline without blocking.

    Attributes:
        connected: True if the WebSocket connection to VTS is active.
        state_hotkeys: Mapping of Aria states to VTS hotkey names.
        mood_hotkeys: Mapping of mood tags to VTS hotkey names.
    """

    def __init__(self):
        """Initialise the VTS controller with default hotkey mappings."""
        self._vts: Optional[pyvts.vts] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.connected: bool = False

        # Load hotkey mappings from config if available.
        # User's config.py values override defaults — but missing keys (e.g. a
        # newly-added "speaking" state) fall back to DEFAULT_STATE_HOTKEYS so
        # we don't force a config.py edit every time a new state is introduced.
        try:
            import config
            user_states = getattr(config, "VTS_STATE_HOTKEYS", {}) or {}
            user_moods  = getattr(config, "VTS_MOOD_HOTKEYS", {}) or {}
            self.state_hotkeys = {**DEFAULT_STATE_HOTKEYS, **user_states}
            self.mood_hotkeys  = {**DEFAULT_MOOD_HOTKEYS,  **user_moods}
        except ImportError:
            self.state_hotkeys = dict(DEFAULT_STATE_HOTKEYS)
            self.mood_hotkeys  = dict(DEFAULT_MOOD_HOTKEYS)

    def start(self) -> None:
        """Start the background asyncio event loop and connect to VTS.

        Runs the event loop in a daemon thread so it does not block
        Aria's main thread or the voice pipeline thread.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="VTSEventLoop",
        )
        self._thread.start()
        print("[VTS] Controller started — connecting to VTube Studio...")

    def stop(self) -> None:
        """Gracefully close the VTS connection and stop the event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    def set_state(self, state: str) -> None:
        """Trigger a VTS hotkey corresponding to an Aria avatar state.

        Intentionally-None states (idle, dormant) are silent — they exist
        in the mapping but produce no log noise on every interaction.
        Only logs when a state name is genuinely unknown or the
        WebSocket isn't connected yet.

        Args:
            state: One of 'idle', 'listening', 'thinking', 'speaking', 'dormant'.
        """
        if state not in self.state_hotkeys:
            print(f"[VTS] Unknown state '{state}' — not in state_hotkeys map.")
            return

        hotkey = self.state_hotkeys[state]
        if hotkey is None:
            return  # Intentionally unmapped — silent

        if not self.connected:
            return  # Avatar not connected — silent (set_state fires often)

        self._submit(self._trigger_hotkey(hotkey))

    def trigger_mood(self, mood: str) -> None:
        """Trigger a VTS hotkey for a mood tag from brain.py.

        Intentionally-None moods (NEUTRAL) are silent — they exist
        in the mapping but produce no log noise.

        Args:
            mood: One of 'HAPPY', 'NEUTRAL', 'THINKING', 'SURPRISED', 'SAD'.
        """
        key = mood.upper()
        if key not in self.mood_hotkeys:
            print(f"[VTS] Unknown mood '{mood}' — not in mood_hotkeys map.")
            return

        hotkey = self.mood_hotkeys[key]
        if hotkey is None or not self.connected:
            return  # Intentionally unmapped or not connected — silent

        self._submit(self._trigger_hotkey(hotkey))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _submit(self, coro) -> None:
        """Submit a coroutine to the background event loop.

        Args:
            coro: An awaitable coroutine to run on the VTS event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_loop(self) -> None:
        """Entry point for the background event loop thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run())

    async def _connect_and_run(self) -> None:
        """Connect to VTS, authenticate, and keep the connection alive."""
        try:
            # Ensure token directory exists
            os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)

            self._vts = pyvts.vts(
                plugin_info={
                    "plugin_name": PLUGIN_NAME,
                    "developer": PLUGIN_DEVELOPER,
                    "plugin_icon": None,
                    "authentication_token_path": TOKEN_FILE,
                },
                vts_api_info={
                    "version": "1.0",
                    "name": "VTubeStudioPublicAPI",
                    "host": VTS_HOST,
                    "port": VTS_PORT,
                },
            )

            await self._vts.connect()
            print("[VTS] WebSocket connected.")

            await self._vts.request_authenticate_token()
            await self._vts.request_authenticate()
            self.connected = True
            print("[VTS] Authenticated. Aria is connected to VTube Studio.")

            # Keep alive — VTS connection stays open
            while True:
                await asyncio.sleep(1)

        except ConnectionRefusedError:
            print(
                "[VTS] ERROR: Could not connect to VTube Studio.\n"
                "       Make sure VTube Studio is running and the API is enabled on port 8001.\n"
                "       Aria will continue without avatar."
            )
            self.connected = False
        except Exception as e:
            print(f"[VTS] ERROR: Connection failed — {e}")
            self.connected = False

    async def _disconnect(self) -> None:
        """Close the VTS WebSocket connection."""
        if self._vts:
            await self._vts.close()
            self.connected = False
            print("[VTS] Disconnected.")

    async def _trigger_hotkey(self, hotkey_name: str) -> None:
        """Trigger a named hotkey in VTube Studio.

        Args:
            hotkey_name: Exact hotkey name as configured in VTS.
                         If None or empty, returns immediately.
        """
        if not hotkey_name or not self._vts or not self.connected:
            return
        try:
            response = await self._vts.request(
                self._vts.vts_request.requestHotKeyList()
            )
            hotkeys = response.get("data", {}).get("availableHotkeys", [])
            match = next(
                (h for h in hotkeys if h.get("name") == hotkey_name), None
            )
            if match:
                hotkey_id = match.get("hotkeyID")
                await self._vts.request(
                    self._vts.vts_request.requestTriggerHotKey(hotkey_id)
                )
                print(f"[VTS] Triggered hotkey: {hotkey_name}")
            else:
                print(f"[VTS] Hotkey '{hotkey_name}' not found in model.")
        except Exception as e:
            print(f"[VTS] ERROR triggering hotkey '{hotkey_name}': {e}")

