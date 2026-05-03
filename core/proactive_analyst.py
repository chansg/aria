"""
core/proactive_analyst.py
-------------------------
Stage 3a: Proactive Analyst Loop for Aria.

Runs as a background daemon thread. Every 60 seconds, reads the latest
desktop screenshot and sends it to Gemini with an insight-filtering prompt.

Response handling:
    'IDLE'        -> do nothing, stay silent
    anything else -> queue the insight for Stage 3b review by default

Design principles:
    - IDLE is the default. Gemini must be highly confident to return insight.
    - 5-minute cooldown prevents Aria from being intrusive.
    - Analysis mode defaults to OFF — user opts in via voice command.
    - Thread is non-blocking — voice pipeline operates independently.

Stage roadmap:
    3a (this):  Direct speech, general desktop context
    3b (current): Notification state — N05 alert, queued insights
    3c (future): Finance specialisation — tickers, sentiment, SEC filings
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from core.logger import get_logger

log = get_logger(__name__)
spoken_log = get_logger("Aria")  # for [Aria] confirmation lines

LATEST_SCREENSHOT  = Path("data/captures/latest.png")
ANALYSIS_INTERVAL  = 60          # Seconds between analysis cycles
COOLDOWN_MINUTES   = 5           # Minimum minutes between proactive comments
MAX_SCREENSHOT_AGE = 90          # Seconds — skip if latest.png is too stale

# ── Module-level registry ─────────────────────────────────────────────────────
# main.py calls register(analyst) once on startup. Other modules
# (e.g. core.brain._handle_analysis_toggle) reach the live instance via
# get_instance() — replaces the fragile `from main import _proactive_analyst`
# pattern, which forced Python to re-import main.py as a separate module
# under a different name, returning None instead of the real instance.
_instance: "ProactiveAnalyst | None" = None


def register(analyst: "ProactiveAnalyst") -> None:
    """Register the global ProactiveAnalyst instance.

    Called by main.py after initialisation. Replaces the fragile
    `from main import _proactive_analyst` pattern.

    Args:
        analyst: The live ProactiveAnalyst instance.
    """
    global _instance
    _instance = analyst


def get_instance() -> "ProactiveAnalyst | None":
    """Return the registered ProactiveAnalyst instance.

    Returns:
        The live instance, or None if not yet initialised.
    """
    return _instance

# Use the same Gemini model as vision_analyzer.py for consistency.
# Override via config.GEMINI_MODEL if needed.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Gemini prompt — negatively reinforced, IDLE is the default response
ANALYST_PROMPT = """You are Aria, an observant AI desktop assistant for Chan.

Review this desktop screenshot carefully.

Your job is to identify HIGH-VALUE insights only. You must be at least 80% confident
the insight is genuinely useful before returning it.

Return IDLE (and nothing else) if:
- The screen looks normal and there is nothing actionable to flag
- You are just describing what you see with no insight attached
- You have low confidence about what is on screen
- The screen has not changed significantly from a typical working state

Return a 1-2 sentence spoken insight ONLY if you observe:
- An error message, warning, or failed process the user may have missed
- A logical inconsistency, contradiction, or potential mistake in visible work
- A pattern, trend, or correlation across multiple visible sources
- A complex problem on screen where a quick observation could save time
- Something that directly requires the user's attention right now

If returning an insight, write it as Aria would speak it — concise, direct,
addressed to Chan. Do not use markdown. Do not prefix with IDLE.

Your response must be either exactly 'IDLE' or a 1-2 sentence spoken insight.
Nothing else."""


class ProactiveAnalyst:
    """Background analyst that monitors the desktop and queues insights.

    Reads latest.png on a 60-second cycle and sends it to Gemini with
    a strict insight-filtering prompt. Non-IDLE responses are queued by
    default so they do not interrupt live conversation.

    Attributes:
        enabled: Whether analysis mode is currently active.
        running: Whether the background thread is alive.
        last_comment_time: Timestamp of the last proactive comment spoken.
    """

    def __init__(
        self,
        speak_fn: Callable[[str], None],
        set_state_fn: Callable[[str], None],
        trigger_mood_fn: Callable[[str], None],
        can_speak_fn: Callable[[], bool] | None = None,
    ) -> None:
        """Initialise the proactive analyst.

        Args:
            speak_fn: Callable — the speak() function from voice/speaker.py.
                      Called to deliver proactive insights aloud.
            set_state_fn: Callable taking a state name ('idle', 'thinking', etc.).
                          Forwards to the visual facade's state setter.
            trigger_mood_fn: Callable taking a mood tag (e.g. 'THINKING')
                             which forwards a cue to the optional visual layer.
            can_speak_fn: Optional gate. Returns False while conversation
                          audio is busy, so proactive speech cannot collide
                          with a user turn.
        """
        self._speak        = speak_fn
        self._can_speak    = can_speak_fn or (lambda: True)
        self._set_state    = set_state_fn
        self._trigger_mood = trigger_mood_fn

        self.enabled: bool                       = False  # OFF by default
        self.running: bool                       = False
        self.last_comment_time: Optional[datetime] = None

        self._thread: Optional[threading.Thread] = None
        self._lock                               = threading.Lock()

        # Initialise Gemini client lazily — None means analysis disabled
        self._client: object | None = None
        self._model_name: str       = DEFAULT_GEMINI_MODEL
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialise the Gemini client.

        Uses the same SDK pattern as vision_analyzer.py
        (`from google import genai; genai.Client(api_key=...)`).
        Sets self._client to None on any failure — the analyst then
        runs but cycles produce no output.
        """
        try:
            from google import genai
            import config

            api_key = getattr(config, "GEMINI_API_KEY", "") or ""
            if not api_key or api_key.startswith("your-"):
                log.warning("GEMINI_API_KEY not set — proactive analysis disabled.")
                return

            self._client     = genai.Client(api_key=api_key)
            self._model_name = getattr(config, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            log.info("Gemini ready — model: %s", self._model_name)
        except ImportError:
            log.warning("google-genai not installed — proactive analysis disabled.")
        except Exception as e:
            log.error("Could not initialise Gemini — %s", e)

    # ── Public control ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background analyst thread.

        Safe to call multiple times — will not start a second thread.
        """
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ProactiveAnalystThread",
        )
        self._thread.start()
        log.info(
            "Proactive analyst started — %ds cycle, %dmin cooldown. "
            "Analysis mode: OFF (say 'Aria, analysis mode on' to enable).",
            ANALYSIS_INTERVAL, COOLDOWN_MINUTES,
        )

    def stop(self) -> None:
        """Stop the analyst thread gracefully."""
        self.running = False
        log.info("Proactive analyst stopped.")

    def enable(self) -> None:
        """Enable analysis mode — insights are logged or spoken per config.

        Prints the spoken confirmation explicitly. The analyst speaks
        via the injected speak_fn (bypassing the normal voice-pipeline
        print path), so without this print the terminal shows a bare
        '[Aria]' followed by a blank line.
        """
        with self._lock:
            self.enabled = True
        log.info("Analysis mode: ON")
        if self._speech_enabled():
            confirmation = "Analysis mode on, Chan. I'll let you know if I spot anything worth flagging."
        else:
            confirmation = "Analysis mode on, Chan. I'll queue anything worth flagging without interrupting you."
        spoken_log.info(confirmation)
        self._speak(confirmation)

    def disable(self) -> None:
        """Disable analysis mode — Aria stays silent proactively.

        Prints the spoken confirmation explicitly (see enable() for why).
        """
        with self._lock:
            self.enabled = False
        log.info("Analysis mode: OFF")
        confirmation = "Analysis mode off, Chan. I'll stay quiet unless you ask me something."
        spoken_log.info(confirmation)
        self._speak(confirmation)

    def toggle(self) -> bool:
        """Toggle analysis mode on or off.

        Returns:
            True if now enabled, False if now disabled.
        """
        if self.enabled:
            self.disable()
            return False
        self.enable()
        return True

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main analysis loop — runs in background daemon thread.

        Sleeps for ANALYSIS_INTERVAL seconds between cycles.
        Skips analysis if disabled, on cooldown, or screenshot is stale.
        Never raises — exceptions are logged and the loop continues.
        """
        while self.running:
            try:
                time.sleep(ANALYSIS_INTERVAL)

                if not self.enabled:
                    continue

                if self._on_cooldown():
                    remaining = self._cooldown_remaining()
                    log.debug("On cooldown — %.0fs remaining.", remaining)
                    continue

                if not self._screenshot_fresh():
                    log.warning("Screenshot too stale — skipping cycle.")
                    continue

                self._analyse()

            except Exception as e:
                log.error("in analysis loop: %s", e)
                # Never crash the thread

    def _analyse(self) -> None:
        """Run one analysis cycle — read screenshot, query Gemini, speak if insight."""
        if self._client is None:
            return

        log.info("Running analysis cycle...")

        try:
            # Signal avatar is thinking
            self._set_state("thinking")

            # Read latest.png as raw bytes (defensive: catch partial writes)
            image_bytes = LATEST_SCREENSHOT.read_bytes()
            if len(image_bytes) < 1024:
                log.warning("Screenshot too small (%d bytes) — likely partial write, skipping.", len(image_bytes))
                self._set_state("idle")
                return

            # Decode via PIL — same pattern vision_analyzer uses, plays well with Gemini SDK
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # force decode now so any truncation surfaces here

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[ANALYST_PROMPT, img],
            )

            result = (getattr(response, "text", "") or "").strip()
            if not result:
                log.info("Empty Gemini response — treating as IDLE.")
                self._set_state("idle")
                return

            log.info("Gemini response: %r", result[:80])

            # IDLE — stay silent (handle bare 'IDLE' or 'IDLE.' or 'IDLE\n' etc.)
            if result.upper().startswith("IDLE"):
                log.info("IDLE — no insight to share.")
                self._set_state("idle")
                return

            if not self._speech_enabled():
                notification = self._queue_insight(result)
                log.info(
                    "Insight queued — speech disabled (%s, %d chars): %s",
                    notification.get("id"),
                    len(result),
                    result,
                )
                self.last_comment_time = datetime.now()
                self._set_state("idle")
                return

            if not self._can_speak():
                notification = self._queue_insight(result)
                log.info(
                    "Insight queued — voice pipeline busy (%s, %d chars): %s",
                    notification.get("id"),
                    len(result),
                    result,
                )
                self._set_state("idle")
                return

            # Non-IDLE — speak the insight
            log.info("Insight detected — speaking (%d chars).", len(result))
            self.last_comment_time = datetime.now()
            self._trigger_mood("THINKING")
            self._speak(result)
            self._set_state("idle")

        except Exception as e:
            log.error("during analysis: %s", e)
            self._set_state("idle")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _on_cooldown(self) -> bool:
        """Return True if Aria should not make a proactive comment yet."""
        if self.last_comment_time is None:
            return False
        return datetime.now() - self.last_comment_time < timedelta(minutes=COOLDOWN_MINUTES)

    def _cooldown_remaining(self) -> float:
        """Return seconds remaining on the cooldown."""
        if self.last_comment_time is None:
            return 0.0
        elapsed = (datetime.now() - self.last_comment_time).total_seconds()
        return max(0.0, (COOLDOWN_MINUTES * 60) - elapsed)

    def _screenshot_fresh(self) -> bool:
        """Return True if latest.png is under MAX_SCREENSHOT_AGE seconds old."""
        if not LATEST_SCREENSHOT.exists():
            return False
        age = time.time() - LATEST_SCREENSHOT.stat().st_mtime
        return age < MAX_SCREENSHOT_AGE

    def _speech_enabled(self) -> bool:
        """Return whether proactive insights are allowed to speak aloud."""
        try:
            import config

            return bool(getattr(config, "PROACTIVE_ANALYST_SPEAK_INSIGHTS", False))
        except Exception:
            return False

    def _queue_insight(self, text: str) -> dict:
        """Persist a proactive analyst insight for Stage 3b review."""
        from core.notifications import enqueue_notification

        return enqueue_notification(
            text,
            source="proactive_analyst",
            priority="N05",
            metadata={
                "model": self._model_name,
                "screenshot": str(LATEST_SCREENSHOT),
            },
        )
