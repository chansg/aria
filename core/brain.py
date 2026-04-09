"""
Aria Brain Module
=================
Aria's reasoning orchestrator.

All queries are classified by core/router.py first.
brain.py dispatches to the correct handler based on tier.
No intent classification logic lives here.

Tier dispatch:
    Tier 1 — router.py local handlers (time, date, calendar)
    Tier 2 — web_search.py + Ollama local model reasoning
    Tier 3 — Claude API for complex open-ended reasoning only

Preserved across all tiers:
    - Episodic memory (store_episodic) — every exchange is recorded
    - Personality (record_interaction) — interaction count incremented
    - Memory context (build_memory_context) — injected into Claude prompts
    - Scheduler tool use — Claude can manage reminders via tool calls
"""

import json
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
from core.router import classify, handle_time, handle_date, handle_calendar
from core.personality import get_system_prompt, record_interaction
from core.memory import store_episodic, build_memory_context
from core.scheduler import add_reminder, add_reminder_minutes, list_reminders, cancel_reminder
from core.web_search import search_web


# ── Ollama configuration ──────────────────────────────────────────────────────

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"


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

    # ── Tier 2: Web scrape + Ollama ──────────────────────────────────────
    if tier == 2:
        response = _handle_web_query(user_input)
        store_episodic("aria", response)
        record_interaction()
        return response

    # ── Tier 3: Claude API ───────────────────────────────────────────────
    response = _handle_claude(user_input)
    store_episodic("aria", response)
    record_interaction()
    return response


# ── Tier 2 handler ────────────────────────────────────────────────────────────

def _handle_web_query(text: str) -> str:
    """Handle a Tier 2 query: scrape web, reason with Ollama.

    Falls back to Claude if Ollama is unavailable.

    Args:
        text: The user's original query.

    Returns:
        Aria's response based on web-retrieved content.
    """
    try:
        web_context = search_web(text)

        prompt = (
            "You are Aria, a helpful and concise personal AI assistant. "
            "Answer the following question using only the web information provided. "
            "Be brief — 1 to 3 sentences maximum. Address the user as Chan. "
            "If the web information does not contain a clear answer, say so.\n\n"
            f"Web information:\n{web_context}\n\n"
            f"Question: {text}\n\n"
            "Answer:"
        )

        return _query_ollama(prompt)

    except Exception as e:
        print(f"[Brain] Tier 2 failed ({e}) — falling back to Claude.")
        return _handle_claude(text)


def _query_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama model and return the response.

    Args:
        prompt: Full prompt string including any web context.

    Returns:
        The model's response as a plain string.

    Raises:
        Exception: If Ollama is not running or returns an error.
    """
    response = httpx.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    result = response.json().get("response", "").strip()
    if not result:
        raise ValueError("Ollama returned an empty response.")
    return result


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

            return _extract_text(final_response)

        return _extract_text(response)

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
