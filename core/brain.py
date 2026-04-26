"""
Aria Brain Module
=================
Aria's reasoning orchestrator.

All queries are classified by core/router.py first.
brain.py dispatches to the correct handler based on tier.
No intent classification logic lives here.

Tier dispatch:
    Tier 1 — router.py local handlers (time, date, calendar)
    Tier 2 — Gemini Flash (web scrape + screenshot) as primary reasoner,
             Ollama/Mistral as offline fallback (USE_LOCAL_FALLBACK=True)
    Tier 3 — Claude API for complex open-ended reasoning only

Preserved across all tiers:
    - Episodic memory (store_episodic) — every exchange is recorded
    - Personality (record_interaction) — interaction count incremented
    - Memory context (build_memory_context) — injected into Claude prompts
    - Scheduler tool use — Claude can manage reminders via tool calls
"""

import json
import re
import httpx
import anthropic
from datetime import datetime
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_MODEL_LITE,
    COMPLEX_QUERY_KEYWORDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    MAX_CONVERSATION_TURNS,
)

# USE_LOCAL_FALLBACK may not exist in older config.py files.
# Default to False (Gemini primary) if the toggle is missing.
try:
    from config import USE_LOCAL_FALLBACK
except ImportError:
    USE_LOCAL_FALLBACK = False
from core.router import classify, handle_time, handle_date, handle_calendar
from core.personality import get_system_prompt, record_interaction
from core.memory import store_episodic, build_memory_context
from core.scheduler import add_reminder, add_reminder_minutes, list_reminders, cancel_reminder
from core.web_search import search_web
from core.vision_analyzer import VisionAnalyzer


# ── Lazy singleton for Gemini vision ──────────────────────────────────────────
# Constructed on first vision query so brain.py imports cheaply even when
# google-generativeai is missing or GEMINI_API_KEY is unset.
_vision_analyzer: VisionAnalyzer | None = None


def _get_vision_analyzer() -> VisionAnalyzer:
    """Return the shared VisionAnalyzer, constructing it on first use."""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer


# ── Ollama configuration ──────────────────────────────────────────────────────
# Ollama 0.6+ deprecated /api/generate in favour of /api/chat.
# The /api/chat endpoint uses a messages array instead of a prompt string.
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


# ── Claude tool definitions for scheduler ─────────────────────────────────────

TOOLS = [
    {
        "name": "add_reminder_at_time",
        "description": "Set a reminder for a specific date and time. Use when the user says something like 'remind me at 3pm to...' or 'set a reminder for tomorrow at 9am'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short name for the reminder (e.g. 'Team standup', 'Take a break').",
                },
                "year": {"type": "integer", "description": "Year (e.g. 2026)."},
                "month": {"type": "integer", "description": "Month (1-12)."},
                "day": {"type": "integer", "description": "Day of month (1-31)."},
                "hour": {"type": "integer", "description": "Hour in 24h format (0-23)."},
                "minute": {"type": "integer", "description": "Minute (0-59)."},
                "description": {
                    "type": "string",
                    "description": "Optional longer description.",
                    "default": "",
                },
            },
            "required": ["title", "year", "month", "day", "hour", "minute"],
        },
    },
    {
        "name": "add_reminder_in_minutes",
        "description": "Set a reminder N minutes from now. Use when the user says 'remind me in 30 minutes' or 'set a timer for 10 minutes'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short name for the reminder.",
                },
                "minutes": {
                    "type": "integer",
                    "description": "Number of minutes from now.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer description.",
                    "default": "",
                },
            },
            "required": ["title", "minutes"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List all upcoming reminders. Use when the user asks 'what reminders do I have?' or 'show my schedule'.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cancel_reminder",
        "description": "Cancel a reminder by title. Use when the user says 'cancel my reminder about...'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title (or part of it) of the reminder to cancel.",
                },
            },
            "required": ["title"],
        },
    },
]


# ── Model selection ───────────────────────────────────────────────────────────

def _select_model(user_input: str) -> str:
    """Return the appropriate Claude model based on query complexity.

    Uses ANTHROPIC_MODEL_LITE (Haiku) for simple voice interactions,
    escalates to ANTHROPIC_MODEL (Sonnet) for complex queries.

    Args:
        user_input: The user's message.

    Returns:
        The model string to use.
    """
    lowered = user_input.lower()
    word_count = len(lowered.split())
    if word_count > 25 or any(kw in lowered for kw in COMPLEX_QUERY_KEYWORDS):
        print(f"[Brain] Complex query detected — using {ANTHROPIC_MODEL}")
        return ANTHROPIC_MODEL
    return ANTHROPIC_MODEL_LITE


# ── Mood tag parsing ─────────────────────────────────────────────────────────

# Valid mood tags — any tag outside this set silently falls back to NEUTRAL.
# Keeps Ollama/Mistral's invented tags (SORRY, APOLOGETIC, etc.) from
# producing "Unknown mood" log noise downstream.
_VALID_MOODS = {"HAPPY", "NEUTRAL", "THINKING", "SURPRISED", "SAD"}


def _parse_mood_tag(text: str) -> tuple[str, str]:
    """Extract and validate a mood tag from an LLM response.

    Parses a `[MOOD]` prefix from the response and ALWAYS strips the
    bracketed prefix from the spoken text — even when the tag is
    unknown or stylised — so brackets never reach TTS.

    Tolerant of common Mistral/Claude formatting quirks:
        - Mixed or lowercase tags: `[Neutral]`, `[neutral]`
        - Markdown wrappers:       `**[NEUTRAL]**`
        - Leading whitespace:      `\\n[NEUTRAL] hi`
        - Plus signs / hyphens in tag names: `[VERY-HAPPY]`

    If no recognisable bracket pattern is found at the start of the
    text, the text is returned unchanged with NEUTRAL — the LLM simply
    forgot the tag, but there's nothing to strip.

    Args:
        text: Raw LLM response potentially starting with [MOOD_TAG].

    Returns:
        Tuple of (mood_tag, clean_text).
        mood_tag defaults to 'NEUTRAL' if missing or invalid.

    Examples:
        '[HAPPY] Hello Chan!'    -> ('HAPPY',   'Hello Chan!')
        '[Neutral] hello'        -> ('NEUTRAL', 'hello')
        '**[NEUTRAL]** hi'       -> ('NEUTRAL', 'hi')
        '[SORRY] My apologies'   -> ('NEUTRAL', 'My apologies')
        'No tag here'            -> ('NEUTRAL', 'No tag here')
    """
    # Allow leading whitespace, optional markdown bold/italic wrappers,
    # and ANY bracketed content — the _VALID_MOODS check decides whether
    # to fire a hotkey, but we always strip the bracket from spoken text
    # so TTS never reads "[NEUTRAL]" literally aloud.
    match = re.match(r'^\s*[*_]{0,3}\[([^\]\n]+)\][*_]{0,3}\s*', text)
    if match:
        # Extract a tag word for the whitelist check: take the last
        # alphabetic token inside the brackets (handles "Mood: Neutral").
        raw_tag = match.group(1)
        tokens  = re.findall(r'[A-Za-z]+', raw_tag)
        tag     = tokens[-1].upper() if tokens else ""
        clean   = text[match.end():]
        if tag in _VALID_MOODS:
            return tag, clean
        # Unknown / no recognisable tag — strip bracketed prefix anyway,
        # default to NEUTRAL silently so brackets never reach TTS.
        return "NEUTRAL", clean
    return "NEUTRAL", text


def _trigger_avatar_mood(mood: str) -> None:
    """Trigger a mood expression on the VTS avatar.

    Args:
        mood: Mood tag string (e.g. 'HAPPY', 'SAD').
    """
    try:
        from avatar.renderer import _controller
        if _controller:
            _controller.trigger_mood(mood)
    except Exception:
        pass  # Avatar not connected — continue silently


def _ensure_complete_sentence(text: str) -> str:
    """Append a period if a Tier 2 response ends mid-word.

    Gemini occasionally returns truncated responses (slow API, content
    filter mid-stream, partial stream). This guard catches the most
    obvious case — text ending with no sentence-ending punctuation —
    and appends a period so TTS doesn't speak a dangling fragment.

    Does not fix the upstream Gemini truncation, just sands off the
    rough edge so Aria sounds intentional rather than glitchy.

    Args:
        text: Cleaned response text (mood tag already stripped).

    Returns:
        Text guaranteed to end with sentence-closing punctuation.
    """
    if not text:
        return text
    text = text.rstrip()
    if text and text[-1] not in ".!?":
        print(f"[Brain] Response ends mid-word — appending '.' to ...{text[-20:]!r}")
        text += "."
    return text


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call from Claude and return the result.

    Args:
        tool_name: The name of the tool to execute.
        tool_input: The input parameters for the tool.

    Returns:
        The tool's result as a string.
    """
    if tool_name == "add_reminder_at_time":
        try:
            remind_at = datetime(
                year=tool_input["year"],
                month=tool_input["month"],
                day=tool_input["day"],
                hour=tool_input["hour"],
                minute=tool_input["minute"],
            )
            return add_reminder(
                title=tool_input["title"],
                remind_at=remind_at,
                description=tool_input.get("description", ""),
            )
        except (ValueError, KeyError) as e:
            return f"Invalid date/time: {e}"

    elif tool_name == "add_reminder_in_minutes":
        return add_reminder_minutes(
            title=tool_input["title"],
            minutes=tool_input["minutes"],
            description=tool_input.get("description", ""),
        )

    elif tool_name == "list_reminders":
        return list_reminders()

    elif tool_name == "cancel_reminder":
        return cancel_reminder(title=tool_input["title"])

    return f"Unknown tool: {tool_name}"


# ── Public entry point ────────────────────────────────────────────────────────

def think(user_input: str) -> str:
    """Generate Aria's response using the tiered routing system.

    Classifies the query via router.py, then dispatches to the
    appropriate handler. Claude is only called for Tier 3 queries.

    Every call stores the exchange in episodic memory and records
    the interaction for personality evolution.

    Args:
        user_input: The transcribed text from the user.

    Returns:
        Aria's text response.
    """
    # Store what the user said
    store_episodic("user", user_input)

    # Classify intent via router
    route = classify(user_input)
    tier = route["tier"]
    intent = route["intent"]

    # ── Tier 1: Local — zero network ─────────────────────────────────────
    if tier == 1:
        if intent == "time":
            response = handle_time()
        elif intent == "date":
            response = handle_date()
        elif intent == "calendar":
            response = handle_calendar()
        else:
            response = handle_time()  # fallback

        store_episodic("aria", response)
        record_interaction()
        return response

    # ── Tier 2: Gemini Flash (web + screen) or vision ──────────────────
    if tier == 2:
        if intent == "vision":
            response = _handle_vision(user_input)
        else:
            response = _handle_web_query(user_input, intent=intent)
        store_episodic("aria", response)
        record_interaction()
        return response

    # ── Tier 3: Claude API ───────────────────────────────────────────────
    response = _handle_claude(user_input)
    store_episodic("aria", response)
    record_interaction()
    return response


# ── Tier 2 handler ────────────────────────────────────────────────────────────

def _handle_web_query(text: str, intent: str = "") -> str:
    """Handle a Tier 2 query using Gemini Flash as primary reasoner.

    Pipeline:
        1. If USE_LOCAL_FALLBACK is True, skip Gemini entirely and
           route to Ollama via _handle_ollama_fallback().
        2. Scrape web context via DuckDuckGo.
        3. Send query + web context + screenshot to Gemini Flash
           via VisionAnalyzer.reason_with_context().
        4. If Gemini fails, fall back to Ollama.
        5. If Ollama also fails, escalate to Claude (Tier 3).

    Args:
        text: The user's original query.
        intent: The matched intent name from router.py (for logging).

    Returns:
        Aria's response based on web-retrieved content.
    """
    # ── Local fallback mode (offline / quota conservation) ───────────
    if USE_LOCAL_FALLBACK:
        print("[Brain] USE_LOCAL_FALLBACK is True — routing to Ollama.")
        return _handle_ollama_fallback(text)

    # ── Scrape web context ───────────────────────────────────────────
    web_context = ""
    try:
        web_context = search_web(text)
    except Exception as e:
        print(f"[Brain] Web scrape failed ({e}) — continuing without web context.")

    # ── Primary: Gemini Flash (web + screenshot) ─────────────────────
    try:
        analyzer = _get_vision_analyzer()
        if not analyzer.available:
            raise RuntimeError("Gemini not available.")

        raw = analyzer.reason_with_context(
            query=text,
            web_context=web_context,
            include_screen=True,
        )
        print(f"[Brain] Gemini Tier 2 response — {len(raw)} chars.")
        mood, clean_response = _parse_mood_tag(raw)
        clean_response = _ensure_complete_sentence(clean_response)
        _trigger_avatar_mood(mood)
        return clean_response

    except Exception as e:
        print(f"[Brain] Gemini failed ({e}) — falling back to Ollama.")

    # ── Fallback: Ollama local model ─────────────────────────────────
    return _handle_ollama_fallback(text, web_context=web_context)


def _handle_vision(text: str) -> str:
    """Handle a Tier 2 vision query via Gemini Flash.

    Delegates to VisionAnalyzer, which reads the most recent
    screenshot written by core.screen_capture and returns a short
    natural-language reply in Aria's voice (mood-tag prefixed).

    The analyzer never raises — on any failure it returns a
    graceful fallback string. We still parse the mood tag so the
    VTS avatar reacts appropriately.

    If USE_LOCAL_FALLBACK is True, vision queries return a polite
    decline since Ollama cannot process screenshots.

    Args:
        text: The user's original question (e.g. "what do you see?").

    Returns:
        Aria's response string, ready for TTS.
    """
    if USE_LOCAL_FALLBACK:
        print("[Brain] USE_LOCAL_FALLBACK is True — vision unavailable offline.")
        return (
            "I can't see your screen right now, Chan — "
            "I'm running in offline mode without Gemini."
        )

    print("[Brain] Tier 2 vision — querying Gemini Flash.")
    analyzer = _get_vision_analyzer()
    raw = analyzer.analyse_screen(context=text)
    mood, clean_response = _parse_mood_tag(raw)
    clean_response = _ensure_complete_sentence(clean_response)
    _trigger_avatar_mood(mood)
    return clean_response


def _query_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama model using the /api/chat endpoint.

    Uses the messages array format required by Ollama 0.6+.
    The /api/generate endpoint was deprecated and returns 404 on newer versions.

    Sets keep_alive: 0 so Mistral unloads from VRAM immediately after
    inference. Required on the RTX 4070 (12 GB) where Whisper large-v2
    leaves only ~6.5 GB for the Ollama runner — without this, Mistral
    stays resident, GPU OOMs on the next inference, and Ollama returns
    HTTP 500. Trade-off: each Ollama call now reloads the model from
    disk (~15-20 s on first inference after release), which is why the
    timeout is bumped from 30 s to 60 s.

    Args:
        prompt: Full prompt string including any web context.

    Returns:
        The model's response as a plain string.

    Raises:
        Exception: If Ollama is not running or returns an error.
    """
    response = httpx.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "keep_alive": 0,   # Unload model after inference — frees VRAM for Whisper
        },
        timeout=60.0,           # First call after VRAM release reloads from disk (~15-20s)
    )
    response.raise_for_status()
    data = response.json()
    result = data.get("message", {}).get("content", "").strip()
    if not result:
        raise ValueError("Ollama returned an empty response.")
    return result


def _handle_ollama_fallback(text: str, web_context: str = "") -> str:
    """Handle a Tier 2 query using Ollama as the offline fallback.

    Called when Gemini is unavailable or USE_LOCAL_FALLBACK is True.
    If web_context is empty, scrapes it fresh. If Ollama also fails,
    escalates to Claude (Tier 3).

    Args:
        text: The user's original query.
        web_context: Pre-scraped web context. Empty string triggers a fresh scrape.

    Returns:
        Aria's response string, ready for TTS.
    """
    # Scrape web context if not already provided
    if not web_context:
        try:
            web_context = search_web(text)
        except Exception as e:
            print(f"[Brain] Web scrape failed in Ollama fallback ({e}).")

    prompt = (
        "You are Aria, a helpful and concise personal AI assistant. "
        "Answer the following question using only the web information provided. "
        "Be brief — 1 to 3 sentences maximum. Address the user as Chan. "
        "If the web information does not contain a clear answer, say so.\n\n"
        "IMPORTANT: Begin every response with a mood tag in square brackets. "
        "Choose from: [HAPPY] [NEUTRAL] [THINKING] [SURPRISED] [SAD]. "
        "Example: '[HAPPY] Of course, Chan! Here's what I found.'\n\n"
        f"Web information:\n{web_context}\n\n"
        f"Question: {text}\n\n"
        "Answer:"
    )

    try:
        raw = _query_ollama(prompt)
        print(f"[Brain] Ollama fallback response — {len(raw)} chars.")
        mood, clean_response = _parse_mood_tag(raw)
        clean_response = _ensure_complete_sentence(clean_response)
        _trigger_avatar_mood(mood)
        return clean_response
    except Exception as e:
        print(f"[Brain] Ollama fallback failed ({e}) — escalating to Claude.")
        return _handle_claude(text)


# ── Tier 3 handler ────────────────────────────────────────────────────────────

def _handle_claude(user_input: str) -> str:
    """Handle a Tier 3 query using the Claude API with tool support.

    Builds the full system prompt with personality and memory context.
    If Claude decides to use a tool, executes it and sends the result
    back for a final natural language response.

    Args:
        user_input: The user's message.

    Returns:
        Claude's response text.
    """
    if not ANTHROPIC_API_KEY:
        return "I can't think right now — my API key isn't set. Add ANTHROPIC_API_KEY to your .env file, Chan."

    # Build the full prompt with memory context
    system_prompt = get_system_prompt()
    memory_context = build_memory_context(max_turns=MAX_CONVERSATION_TURNS)

    if memory_context:
        system_prompt += f"\n\n--- Memory ---\n{memory_context}"

    # Add current time so Aria knows when "now" is
    now = datetime.now()
    system_prompt += f"\n\nCurrent date and time: {now.strftime('%A %d %B %Y, %H:%M')}."

    # Request mood tag prefix for avatar expressions
    system_prompt += (
        "\n\nIMPORTANT: Begin every response with a mood tag in square brackets. "
        "Choose from: [HAPPY] [NEUTRAL] [THINKING] [SURPRISED] [SAD]. "
        "Example: '[HAPPY] Of course, Chan! Here's what I found.'"
    )

    try:
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        model = _select_model(user_input)
        messages = [{"role": "user", "content": user_input}]
        cached_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        # CLAUDE CALL — primary request (may include tool calls)
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=cached_system,
            tools=TOOLS,
            messages=messages,
        )

        # If Claude wants to use a tool, execute it and get a final response
        if response.stop_reason == "tool_use":
            tool_results = []
            assistant_content = response.content

            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"[Brain] Using tool: {block.name}")
                    result = _execute_tool(block.name, block.input)
                    print(f"[Brain] Tool result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Send tool results back for a natural language response
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            # CLAUDE CALL — follow-up after tool execution
            final_response = client.messages.create(
                model=model,
                max_tokens=250,
                system=cached_system,
                tools=TOOLS,
                tool_choice={"type": "none"},
                messages=messages,
            )

            reply = _extract_text(final_response)
            mood, clean_response = _parse_mood_tag(reply)
            _trigger_avatar_mood(mood)
            return clean_response

        reply = _extract_text(response)
        mood, clean_response = _parse_mood_tag(reply)
        _trigger_avatar_mood(mood)
        return clean_response

    except anthropic.AuthenticationError:
        return "My API key seems invalid, Chan. Double-check what's in your .env file."
    except anthropic.RateLimitError:
        return "I've been rate-limited by the API. Give me a moment and try again."
    except Exception as e:
        print(f"[Brain] Claude API error: {e}")
        return "Something went wrong reaching the API. Check the console for details."


def _extract_text(response) -> str:
    """Extract the text content from a Claude API response.

    Args:
        response: The Anthropic API response object.

    Returns:
        The text content, or a fallback message.
    """
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "I processed that, but I'm not sure what to say about it."
