"""
avatar/vts_controller.py
------------------------
VTube Studio WebSocket controller for Aria.

Manages the connection to VTube Studio via pyvts, handles authentication
token storage, and provides a clean interface for triggering avatar
states and emotional expressions.

Architecture:
    - Single persistent async connection to VTS on ws://localhost:8001
    - Auth token stored in data/vts_token.txt (managed by pyvts)
    - Avatar states + mood tags mapped to VTS hotkeys via config.py
    - All hotkey requests serialised through an asyncio.Queue — a single
      consumer coroutine processes them one at a time so two concurrent
      hotkey calls cannot race on the same WebSocket and trigger the
      "cannot call recv while another coroutine is already running recv"
      error.
    - Runs in a background asyncio event loop alongside the voice pipeline

Hotkey naming:
    config.py is the SINGLE SOURCE OF TRUTH for hotkey names. The
    DEFAULT_*_HOTKEYS dicts here document which states/moods exist but
    map every value to None — there are no fallback hotkey strings, so
    a missing config entry stays silent rather than firing a hotkey
    that doesn't exist in the user's VTS model.

Usage:
    controller = VTSController()
    controller.start()                      # background event loop
    controller.set_state("speaking")        # enqueued, fires N01
    controller.trigger_mood("HAPPY")        # enqueued, fires N01
    controller.stop()
"""

from __future__ import annotations
import asyncio
import os
import threading
from typing import Optional

import pyvts

from core.logger import get_logger

log = get_logger(__name__)

# ── Token storage ─────────────────────────────────────────────────────────────
TOKEN_FILE       = "data/vts_token.txt"
PLUGIN_NAME      = "Aria"
PLUGIN_DEVELOPER = "chansg"
VTS_HOST         = "localhost"
VTS_PORT         = 8001

# ── State / mood key catalogues — values are None on purpose ─────────────────
# These dicts document which state and mood keys exist in the system. Every
# value is None — config.py is the single source of truth for actual hotkey
# names. The merge in __init__ means any state/mood missing from the user's
# config stays silently None rather than firing an invented default that
# doesn't exist in the user's VTS model.
DEFAULT_STATE_HOTKEYS = {
    "idle":      None,
    "listening": None,
    "thinking":  None,
    "speaking":  None,
    "dormant":   None,
}

DEFAULT_MOOD_HOTKEYS = {
    "HAPPY":     None,
    "NEUTRAL":   None,
    "THINKING":  None,
    "SURPRISED": None,
    "SAD":       None,
}


class VTSController:
    """Manages the VTube Studio WebSocket connection for Aria.

    Uses an asyncio.Queue to serialise all hotkey requests — a single
    consumer coroutine processes the queue one item at a time so two
    overlapping hotkey calls from the speaking state and a mood tag
    cannot race on the same WebSocket.

    Attributes:
        connected: True if the WebSocket connection to VTS is active.
        state_hotkeys: Mapping of Aria states to VTS hotkey names.
        mood_hotkeys: Mapping of mood tags to VTS hotkey names.
    """

    def __init__(self):
        """Initialise the VTS controller with hotkey mappings from config."""
        self._vts:    Optional[pyvts.vts] = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._queue:  Optional[asyncio.Queue] = None
        self.connected: bool = False

        # Load hotkey mappings from config — config wins over None defaults.
        # Defaults exist only as a documented key list, so a state/mood
        # missing from config stays silently None instead of firing an
        # invented hotkey name.
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
        log.info("Controller started — connecting to VTube Studio...")

    def stop(self) -> None:
        """Gracefully close the VTS connection and stop the event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    def set_state(self, state: str) -> None:
        """Enqueue a VTS hotkey for an Aria avatar state.

        Intentionally-None states (idle, dormant) are silent — they exist
        in the mapping but produce no log noise on every interaction.

        Args:
            state: One of 'idle', 'listening', 'thinking', 'speaking', 'dormant'.
        """
        if state not in self.state_hotkeys:
            log.warning("Unknown state %r — not in state_hotkeys map.", state)
            return

        hotkey = self.state_hotkeys[state]
        if hotkey is None or not self.connected:
            return  # Intentionally unmapped or not connected — silent

        self._enqueue(hotkey)

    def trigger_mood(self, mood: str) -> None:
        """Enqueue a VTS hotkey for a mood tag from brain.py.

        Intentionally-None moods (NEUTRAL) are silent.

        Args:
            mood: One of 'HAPPY', 'NEUTRAL', 'THINKING', 'SURPRISED', 'SAD'.
        """
        key = mood.upper()
        if key not in self.mood_hotkeys:
            log.warning("Unknown mood %r — not in mood_hotkeys map.", mood)
            return

        hotkey = self.mood_hotkeys[key]
        if hotkey is None or not self.connected:
            return

        self._enqueue(hotkey)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enqueue(self, hotkey_name: str) -> None:
        """Push a hotkey name onto the serialised queue.

        Thread-safe — can be called from any thread. The single
        _queue_consumer coroutine pulls items one at a time, so two
        overlapping hotkey calls cannot race on the WebSocket.

        Args:
            hotkey_name: Exact hotkey name as configured in VTS.
        """
        if self._loop and self._loop.is_running() and self._queue:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(hotkey_name),
                self._loop,
            )

    def _run_loop(self) -> None:
        """Entry point for the background event loop thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run())

    async def _connect_and_run(self) -> None:
        """Connect to VTS, authenticate, then run the queue consumer forever."""
        # The asyncio.Queue must be created on the loop that will await it.
        self._queue = asyncio.Queue()

        try:
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
            log.info("WebSocket connected.")

            await self._vts.request_authenticate_token()
            await self._vts.request_authenticate()
            self.connected = True
            log.info("Authenticated. Aria is connected to VTube Studio.")

            # Run queue consumer — one hotkey at a time, no concurrent recv
            await self._queue_consumer()

        except ConnectionRefusedError:
            log.error(
                "Could not connect to VTube Studio. "
                "Make sure VTube Studio is running and the API is enabled on port 8001. "
                "Aria will continue without avatar."
            )
            self.connected = False
        except Exception as e:
            log.error("Connection failed — %s", e)
            self.connected = False

    async def _queue_consumer(self) -> None:
        """Drain the hotkey queue forever, processing one item at a time.

        Awaits each _trigger_hotkey() to completion before pulling the
        next item — this is what guarantees no two hotkey requests ever
        share the WebSocket concurrently.
        """
        while True:
            hotkey_name = await self._queue.get()
            try:
                await self._trigger_hotkey(hotkey_name)
            except Exception as e:
                log.error("triggering hotkey %r: %s", hotkey_name, e)
            finally:
                self._queue.task_done()

    async def _disconnect(self) -> None:
        """Close the VTS WebSocket connection."""
        if self._vts:
            await self._vts.close()
            self.connected = False
            log.info("Disconnected.")

    async def _trigger_hotkey(self, hotkey_name: str) -> None:
        """Trigger a named hotkey in VTube Studio.

        Args:
            hotkey_name: Exact hotkey name as configured in VTS.
                         If None or empty, returns immediately.
        """
        if not hotkey_name or not self._vts or not self.connected:
            return

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
            log.info("Triggered hotkey: %s", hotkey_name)
        else:
            log.warning("Hotkey %r not found in model.", hotkey_name)
