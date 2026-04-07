"""
Aria Brain Module
=================
Handles AI reasoning via Anthropic Claude API or local Ollama model.
Toggle USE_LOCAL_MODEL in config.py to switch backends.

Uses Claude tool use to let Aria manage the schedule naturally
from voice commands.
"""

import json
import httpx
import anthropic
from datetime import datetime
from config import (
    USE_LOCAL_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_MODEL_LITE,
    COMPLEX_QUERY_KEYWORDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from core.personality import get_system_prompt, record_interaction
from core.memory import store_episodic, build_memory_context
from core.scheduler import add_reminder, add_reminder_minutes, list_reminders, cancel_reminder


# --- Tool definitions for Claude ---
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
        print(f"[Aria] Complex query detected — using {ANTHROPIC_MODEL}")
        return ANTHROPIC_MODEL
    return ANTHROPIC_MODEL_LITE


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


def think(user_input: str) -> str:
    """Process user input and return Aria's response.

    Routes to either the Claude API or a local Ollama model
    depending on the USE_LOCAL_MODEL config toggle.

    Args:
        user_input: The transcribed text from the user.

    Returns:
        Aria's text response.
    """
    # Store what the user said
    store_episodic("user", user_input)

    # Build the full prompt with memory context
    system_prompt = get_system_prompt()
    memory_context = build_memory_context()

    if memory_context:
        system_prompt += f"\n\n--- Memory ---\n{memory_context}"

    # Add current time so Aria knows when "now" is
    now = datetime.now()
    system_prompt += f"\n\nCurrent date and time: {now.strftime('%A %d %B %Y, %H:%M')}."

    # Route to the appropriate backend
    if USE_LOCAL_MODEL:
        response = _think_local(system_prompt, user_input)
    else:
        response = _think_claude(system_prompt, user_input)

    # Store Aria's response and record the interaction
    store_episodic("aria", response)
    record_interaction()

    return response


def _think_claude(system_prompt: str, user_input: str) -> str:
    """Send the prompt to Anthropic's Claude API with tool support.

    If Claude decides to use a tool, executes it and sends the result
    back for a final natural language response.

    Args:
        system_prompt: The full system prompt with personality and memory.
        user_input: The user's message.

    Returns:
        Claude's response text.
    """
    if not ANTHROPIC_API_KEY:
        return "I can't think right now — my API key isn't set. Add ANTHROPIC_API_KEY to your .env file, Chan."

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, default_headers={"anthropic-beta": "prompt-caching-2024-07-31"})
        model = _select_model(user_input)
        messages = [{"role": "user", "content": user_input}]
        cached_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        # First request — may include tool calls
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=cached_system,
            tools=TOOLS,
            messages=messages,
        )

        # If Claude wants to use a tool, execute it and get a final response
        if response.stop_reason == "tool_use":
            # Process all tool calls
            tool_results = []
            assistant_content = response.content

            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"[Aria] Using tool: {block.name}")
                    result = _execute_tool(block.name, block.input)
                    print(f"[Aria] Tool result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Send tool results back for a natural language response
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

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
        print(f"[Aria] Claude API error: {e}")
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


def _think_local(system_prompt: str, user_input: str) -> str:
    """Send the prompt to a local Ollama model.

    Note: Ollama doesn't support tool use the same way, so schedule
    commands are handled by keyword matching as a fallback.

    Args:
        system_prompt: The full system prompt with personality and memory.
        user_input: The user's message.

    Returns:
        The local model's response text.
    """
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    except httpx.ConnectError:
        return "I can't reach Ollama — is it running? Start it with 'ollama serve' in another terminal."
    except Exception as e:
        print(f"[Aria] Ollama error: {e}")
        return "Something went wrong with the local model. Check the console."
