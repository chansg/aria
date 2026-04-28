"""
core/proactive_analyst.py
-------------------------
Stage 3a: Proactive Analyst Loop for Aria.

Runs as a background daemon thread. Every 60 seconds, reads the latest
desktop screenshot and sends it to Gemini with an insight-filtering prompt.

Response handling:
    'IDLE'        → do nothing, stay silent
    anything else → speak the insight via the voice pipeline

Design principles:
    - IDLE is the default. Gemini must be highly confident to return insight.
    - 5-minute cooldown prevents Aria from being intrusive.
    - Analysis mode defaults to OFF — user opts in via voice command.
    - Thread is non-blocking — voice pipeline operates independently.

Stage roadmap:
    3a (this):  Direct speech, general desktop context
    3b (future): Notification state — N05 alert, queued insights
    3c (future): Finance specialisation — tickers, sentiment, SEC filings
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

LATEST_SCREENSHOT  = Path("data/captures/latest.png")
ANALYSIS_INTERVAL  = 60          # Seconds between analysis cycles
COOLDOWN_MINUTES   = 5           # Minimum minutes between proactive comments
MAX_SCREENSHOT_AGE = 90          # Seconds — skip if latest.png is too stale

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
    """Background analyst that monitors the desktop and speaks unprompted.

    Reads latest.png on a 60-second cycle and sends it to Gemini with
    a strict insight-filtering prompt. Only non-IDLE responses are spoken.

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
    ) -> None:
        """Initialise the proactive analyst.

        Args:
            speak_fn: Callable — the speak() function from voice/speaker.py.
                      Called to deliver proactive insights aloud.
            set_state_fn: Callable taking a state name ('idle', 'thinking', etc.).
                          Forwards to the avatar renderer's state setter.
            trigger_mood_fn: Callable taking a mood tag (e.g. 'THINKING')
                             which fires a VTS hotkey via vts_controller.
        """
        self._speak        = speak_fn
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
                print("[Analyst] WARNING: GEMINI_API_KEY not set — proactive analysis disabled.")
                return

            self._client     = genai.Client(api_key=api_key)
            self._model_name = getattr(config, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            print(f"[Analyst] Gemini ready — model: {self._model_name}")
        except ImportError:
            print("[Analyst] WARNING: google-genai not installed — proactive analysis disabled.")
        except Exception as e:
            print(f"[Analyst] ERROR: Could not initialise Gemini — {e}")

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
        print(
            f"[Analyst] Proactive analyst started — "
            f"{ANALYSIS_INTERVAL}s cycle, {COOLDOWN_MINUTES}min cooldown. "
            f"Analysis mode: OFF (say 'Aria, analysis mode on' to enable)."
        )

    def stop(self) -> None:
        """Stop the analyst thread gracefully."""
        self.running = False
        print("[Analyst] Proactive analyst stopped.")

    def enable(self) -> None:
        """Enable analysis mode — Aria will speak proactive insights.

        Prints the spoken confirmation explicitly. The analyst speaks
        via the injected speak_fn (bypassing the normal voice-pipeline
        print path), so without this print the terminal shows a bare
        '[Aria]' followed by a blank line.
        """
        with self._lock:
            self.enabled = True
        print("[Analyst] Analysis mode: ON")
        confirmation = "Analysis mode on, Chan. I'll let you know if I spot anything worth flagging."
        print(f"[Aria] {confirmation}")
        self._speak(confirmation)

    def disable(self) -> None:
        """Disable analysis mode — Aria stays silent proactively.

        Prints the spoken confirmation explicitly (see enable() for why).
        """
        with self._lock:
            self.enabled = False
        print("[Analyst] Analysis mode: OFF")
        confirmation = "Analysis mode off, Chan. I'll stay quiet unless you ask me something."
        print(f"[Aria] {confirmation}")
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
                    print(f"[Analyst] On cooldown — {remaining:.0f}s remaining.")
                    continue

                if not self._screenshot_fresh():
                    print("[Analyst] Screenshot too stale — skipping cycle.")
                    continue

                self._analyse()

            except Exception as e:
                print(f"[Analyst] ERROR in analysis loop: {e}")
                # Never crash the thread

    def _analyse(self) -> None:
        """Run one analysis cycle — read screenshot, query Gemini, speak if insight."""
        if self._client is None:
            return

        print("[Analyst] Running analysis cycle...")

        try:
            # Signal avatar is thinking
            self._set_state("thinking")

            # Read latest.png as raw bytes (defensive: catch partial writes)
            image_bytes = LATEST_SCREENSHOT.read_bytes()
            if len(image_bytes) < 1024:
                print(f"[Analyst] Screenshot too small ({len(image_bytes)} bytes) — likely partial write, skipping.")
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
                print("[Analyst] Empty Gemini response — treating as IDLE.")
                self._set_state("idle")
                return

            print(f"[Analyst] Gemini response: {result[:80]!r}")

            # IDLE — stay silent (handle bare 'IDLE' or 'IDLE.' or 'IDLE\n' etc.)
            if result.upper().startswith("IDLE"):
                print("[Analyst] IDLE — no insight to share.")
                self._set_state("idle")
                return

            # Non-IDLE — speak the insight
            print(f"[Analyst] Insight detected — speaking ({len(result)} chars).")
            self.last_comment_time = datetime.now()
            self._trigger_mood("THINKING")
            self._speak(result)
            self._set_state("idle")

        except Exception as e:
            print(f"[Analyst] ERROR during analysis: {e}")
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
