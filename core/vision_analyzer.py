"""
core/vision_analyzer.py
-----------------------
Stage 2 of Aria's screen awareness pipeline.

Takes the most recent compressed screenshot produced by
`core.screen_capture.ScreenCapture` and sends it to Google Gemini
Flash vision for analysis. Returns natural-language commentary in
Aria's voice — what's on screen, what Chan is doing, what's
interesting or worth reacting to.

Stage 1 (core/screen_capture.py) writes `data/captures/latest.png`
every few seconds. Stage 2 reads that file on demand and asks
Gemini to describe it. Stage 1 runs whether or not Stage 2 is
configured; Stage 2 degrades gracefully if:

  • GEMINI_API_KEY is missing from config.py
  • data/captures/latest.png does not yet exist
  • the google-generativeai package is not installed
  • the Gemini API call fails

Usage:
    from core.vision_analyzer import VisionAnalyzer
    vision = VisionAnalyzer()
    reply  = vision.analyse_screen("what do you see?")
    speak(reply)
"""

from __future__ import annotations

import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LATEST_SCREENSHOT = Path("data/captures/latest.png")

# ── Model config ──────────────────────────────────────────────────────────────
# gemini-2.5-flash is the current stable Flash model on the v1beta endpoint.
# Older "gemini-1.5-flash" was retired — switch here if Google renames again,
# or swap in "gemini-flash-latest" to always ride the newest release.
GEMINI_MODEL     = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 400
REQUEST_TIMEOUT   = 20        # seconds
STALE_THRESHOLD   = 30        # warn if latest.png is older than this (s)

# ── Aria's vision persona ─────────────────────────────────────────────────────
# Kept conversational and short — this goes straight to TTS, so prose beats
# bullet points. The mood tag system (e.g. [HAPPY], [SURPRISED]) is preserved
# so the avatar can react to what Aria sees.
VISION_SYSTEM_PROMPT = """You are Aria, Chan's personal AI desktop assistant.
You can see Chan's screen. Speak in first person, casually, like a friend
watching over their shoulder — NOT like a vision-model caption.

Style rules:
- 1–3 short sentences. This is going straight to text-to-speech.
- Natural spoken English. No bullet points, no markdown, no headings.
- Be specific about what you see: game names, app names, what Chan is
  doing, anything unusual. If it's Valorant, call out the agent, map, or
  scoreboard. If it's code, mention the language or what file it looks like.
- If Chan asked a question ("what do you see?", "what's happening?"),
  answer it directly from the screenshot.
- Start your reply with one mood tag in square brackets:
  [HAPPY] [NEUTRAL] [THINKING] [SURPRISED] [SAD]
  Pick whichever matches your reaction to what's on screen.
- Never say "the image shows" or "in the screenshot" — you're looking
  at Chan's actual desktop in real time.
"""


class VisionAnalyzer:
    """Wraps Gemini Flash vision calls against Aria's latest screenshot.

    The analyzer is lazy: it imports google.genai and reads the
    API key on first use so that import-time failures don't break the
    rest of Aria. Every public method returns a user-facing string —
    never raises — so `core.brain` can just speak whatever comes back.

    Attributes:
        available: True if the module can actually call Gemini.
        _client:   Cached genai.Client instance once configured.
    """

    def __init__(self) -> None:
        """Configure Gemini if possible; fall back silently otherwise."""
        self.available: bool = False
        self._client = None
        self._configure()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse_screen(self, context: str = "") -> str:
        """Describe the latest screenshot in Aria's voice.

        Args:
            context: The user's spoken request, e.g. "what do you see?".
                     Passed to Gemini as additional context so the reply
                     answers the actual question.

        Returns:
            A short natural-language reply suitable for TTS. Always a
            string — on any failure, returns a graceful fallback line.
        """
        if not self.available:
            return (
                "[NEUTRAL] I can't see your screen right now — "
                "my vision system isn't configured yet."
            )

        screenshot = self._load_screenshot()
        if screenshot is None:
            return (
                "[NEUTRAL] I don't have a screenshot to look at yet. "
                "Give the screen capture a few seconds to warm up."
            )

        age = self._get_screenshot_age_seconds()
        if age is not None and age > STALE_THRESHOLD:
            print(f"[Vision] WARNING: latest.png is {age:.0f}s old — may be stale.")

        prompt = self._build_prompt(context)

        try:
            reply = self._query_gemini(prompt, screenshot)
        except Exception as e:
            print(f"[Vision] ERROR querying Gemini: {e}")
            return (
                "[SAD] Something went wrong when I tried to look at your "
                "screen. I'll try again in a moment."
            )

        if not reply:
            return (
                "[THINKING] I can see your screen but I'm not sure what "
                "to say about it."
            )

        return reply.strip()

    def reason_with_context(
        self,
        query: str,
        web_context: str = "",
        include_screen: bool = True,
    ) -> str:
        """Send query, web context, and optionally the screen to Gemini.

        This is the unified Tier 2 reasoning call. Gemini receives:
          - The user's original query
          - Raw web scrape results (if available)
          - The current screenshot (if screen capture is running)

        Gemini synthesises all inputs into a single coherent response,
        eliminating the need for separate weather, web, or gaming handlers.

        Args:
            query: The user's original natural language query.
            web_context: Raw text from DuckDuckGo scrape. Empty string if
                         no web context is needed or scrape failed.
            include_screen: Whether to include latest.png in the request.
                            Set False for pure text queries with no screen
                            relevance.

        Returns:
            Gemini's response as plain text for Aria to speak.

        Raises:
            RuntimeError: If Gemini is unavailable or the call fails.
        """
        if not self.available:
            raise RuntimeError("Gemini not available — API key missing or SDK not installed.")

        content_parts = []

        # Include screenshot if available and recent enough
        if include_screen:
            screenshot = self._load_screenshot()
            if screenshot is not None:
                age = self._get_screenshot_age_seconds()
                if age is not None and age < 60:
                    content_parts.append(screenshot)
                    print(f"[Vision] Including screenshot ({age:.0f}s old) in Gemini request.")
                else:
                    print(f"[Vision] Screenshot too old ({age:.0f}s) — skipping from request.")

        # Build the unified prompt
        prompt_parts = [
            "You are Aria, a smart personal AI assistant for Chan. "
            "Be concise — 1 to 3 sentences maximum. Conversational and direct. "
            "Do not use markdown formatting. Address the user as Chan.\n\n"
            "IMPORTANT: Begin your response with a mood tag in square brackets. "
            "Choose from: [HAPPY] [NEUTRAL] [THINKING] [SURPRISED] [SAD].\n\n"
        ]

        if web_context:
            prompt_parts.append(
                f"Web search results for context:\n{web_context}\n\n"
            )

        prompt_parts.append(f"User query: {query}\n\nAnswer:")
        content_parts.append("".join(prompt_parts))

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=content_parts,
                config={
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": 0.7,
                    "http_options": {"timeout": REQUEST_TIMEOUT * 1000},
                },
            )
            result = getattr(response, "text", "") or ""
            result = result.strip()
            print(f"[Vision] Gemini unified response — {len(result)} chars.")
            return result
        except Exception as e:
            raise RuntimeError(f"Gemini reasoning failed: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _configure(self) -> None:
        """Import google.genai and create a Gemini client.

        Silently disables the analyzer if the package or key are missing.
        """
        try:
            import config
        except Exception as e:
            print(f"[Vision] Could not import config: {e}")
            return

        api_key = getattr(config, "GEMINI_API_KEY", None)
        if not api_key or api_key.startswith("your-"):
            print("[Vision] GEMINI_API_KEY not set — vision disabled.")
            return

        try:
            from google import genai
        except ImportError:
            print("[Vision] google-genai not installed — vision disabled.")
            return

        try:
            self._client = genai.Client(api_key=api_key)
            self.available = True
            print(f"[Vision] Gemini {GEMINI_MODEL} ready (google.genai SDK).")
        except Exception as e:
            print(f"[Vision] Failed to initialise Gemini: {e}")

    def _load_screenshot(self):
        """Load `data/captures/latest.png` as a PIL Image.

        Returns:
            A PIL.Image.Image, or None if the file is missing / unreadable.
        """
        if not LATEST_SCREENSHOT.exists():
            print(f"[Vision] No screenshot at {LATEST_SCREENSHOT}")
            return None

        try:
            from PIL import Image
            return Image.open(LATEST_SCREENSHOT)
        except Exception as e:
            print(f"[Vision] Failed to open screenshot: {e}")
            return None

    def _build_prompt(self, context: str) -> str:
        """Build the user-side prompt that accompanies the screenshot.

        Args:
            context: The user's spoken request. May be empty.

        Returns:
            A short instruction for Gemini, anchored to Chan's question.
        """
        context = (context or "").strip()
        if context:
            return (
                f"Chan just asked: \"{context}\"\n"
                f"Look at the attached screenshot of his desktop and "
                f"answer him directly in your voice."
            )
        return (
            "Describe what's on Chan's desktop right now in 1–3 casual "
            "sentences. Be specific about apps, games, or what he's doing."
        )

    def _query_gemini(self, prompt: str, screenshot) -> str:
        """Send prompt + screenshot to Gemini and return the text reply.

        Args:
            prompt:     The text portion of the request.
            screenshot: A PIL Image of the desktop.

        Returns:
            The plain-text reply from Gemini. Empty string if none.
        """
        response = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, screenshot],
            config={
                "system_instruction": VISION_SYSTEM_PROMPT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0.7,
                "http_options": {"timeout": REQUEST_TIMEOUT * 1000},
            },
        )

        # .text is the convenience accessor for the first candidate.
        text = getattr(response, "text", "") or ""
        return text

    def _get_screenshot_age_seconds(self) -> float | None:
        """Return how old `latest.png` is, in seconds.

        Returns:
            Seconds since the file was last modified, or None if missing.
        """
        if not LATEST_SCREENSHOT.exists():
            return None
        return time.time() - LATEST_SCREENSHOT.stat().st_mtime
