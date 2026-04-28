"""
Aria Intent Router
==================
Single source of truth for intent classification in Aria.

Every user query is classified here before any handler is called.
Classification happens in tier order — cheapest first.

Tier system:
    TIER 1 (local)  — Python stdlib only. Zero network, zero cost.
    TIER 2 (web)    — DuckDuckGo scrape + Ollama local reasoning. Zero API cost.
    TIER 3 (claude) — Claude API. Used only when Tiers 1 and 2 cannot answer.

To add a new intent:
    1. Add an entry to INTENT_MAP with keywords and tier
    2. If Tier 1, add a handler function in this file
    3. If Tier 2, the query is passed to web_search + Ollama automatically
    4. If Tier 3, the query is passed to Claude automatically
"""

from __future__ import annotations
from datetime import datetime
from core.scheduler import list_reminders


# ── Intent map ───────────────────────────────────────────────────────────────
# Each entry defines: tier and keywords.
# Keywords are matched case-insensitively against the full query.
# Tier 1 is checked first, then Tier 2. First match wins.

INTENT_MAP: dict[str, dict] = {
    # Tier 1 — local, free
    "time": {
        "tier": 1,
        "keywords": [
            "what time", "current time", "what's the time",
            "tell me the time", "clock",
        ],
    },
    "date": {
        "tier": 1,
        "keywords": [
            "what date", "what day", "today's date", "what's today",
            "day is it", "date today",
        ],
    },
    "calendar": {
        "tier": 1,
        "keywords": [
            "what reminders", "show reminders", "list reminders",
            "my reminders", "my schedule", "show my schedule",
            "what's on my schedule", "what's on my agenda",
            "what's on today", "agenda",
        ],
    },
    "analysis_mode": {
        "tier": 1,
        "keywords": [
            "analysis mode on",  "enable analysis",  "start analysis",
            "analysis mode off", "disable analysis", "stop analysis",
            "analysis mode",     "toggle analysis",
        ],
    },

    # Tier 2 — web + Ollama, free
    "weather": {
        "tier": 2,
        "keywords": [
            "weather", "temperature", "forecast", "raining", "sunny",
            "cloudy", "cold outside", "warm outside", "hot outside",
            "umbrella", "degrees outside", "rain today", "snow",
        ],
    },
    # Tier 2 — Gemini vision (Stage 2 screen understanding)
    # Listed before web_search because vision's "what is on my screen"
    # phrase is more specific than web_search's "what is" prefix and
    # should win the tie-break. Weather stays ahead of vision since
    # its keywords ("weather", "temperature") never collide.
    "vision": {
        "tier": 2,
        "keywords": [
            # Direct vision requests
            "what do you see", "what can you see", "what do you notice",
            "what's on screen", "what's on my screen", "what is on my screen",
            "look at my screen", "look at the screen", "describe my screen",
            "analyse my screen", "analyze my screen",
            "what am i doing", "what game is this", "what game am i playing",
            "what app is this", "read my screen",
            # Natural language variants
            "can you see", "are you able to see", "do you see",
            "look at this", "what are you looking at",
            "describe what you see", "see my screen",
            "looking at my screen", "what do you observe",
            "tell me what you see",
            # Productivity and evaluation queries
            "is this good", "is this a good", "evaluate what you see",
            "what do you think of this", "review what you see", "review my screen",
            "what do you make of this", "give me feedback", "is this correct",
            "check this for me", "look at this and tell me",
            "what do you think about what you see",
            "does this look right", "any issues with this", "spot anything wrong",
        ],
    },

    "web_search": {
        "tier": 2,
        "keywords": [
            "search for", "look up", "find out", "who is", "what is",
            "tell me about", "latest news", "current news", "news about",
            "who won", "what happened", "how does", "when did",
            "where is", "price of", "how much is",
        ],
    },
}


# ── Classification ────────────────────────────────────────────────────────────

def classify(text: str) -> dict:
    """Classify user input and return the matched intent and tier.

    Checks intents in tier order (1 before 2) to ensure cheapest
    handler is always preferred. If no match is found, returns
    Tier 3 to escalate to Claude.

    Args:
        text: The transcribed user query. Aria's name may still
              be present — classification ignores it.

    Returns:
        dict with keys:
            intent (str): The matched intent name, or 'claude' if none.
            tier (int): 1, 2, or 3.
    """
    text_lower = text.lower().strip()

    # Check Tier 1 first
    for intent_name, config in INTENT_MAP.items():
        if config["tier"] == 1:
            if _matches(text_lower, config["keywords"]):
                print(f"[Router] Tier 1 matched: {intent_name} — handling locally.")
                return {"intent": intent_name, "tier": 1}

    # Then Tier 2
    for intent_name, config in INTENT_MAP.items():
        if config["tier"] == 2:
            if _matches(text_lower, config["keywords"]):
                backend = "Gemini Flash (vision)" if intent_name == "vision" else "Gemini Flash (web + screen)"
                print(f"[Router] Tier 2 matched: {intent_name} — {backend}.")
                return {"intent": intent_name, "tier": 2}

    # No match — escalate to Claude
    print("[Router] No match — Tier 3 escalation to Claude API.")
    return {"intent": "claude", "tier": 3}


def _matches(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the text.

    Args:
        text: Lowercased user query.
        keywords: List of keyword phrases to match against.

    Returns:
        True if any keyword is found in the text.
    """
    return any(kw in text for kw in keywords)


# ── Tier 1 local handlers ─────────────────────────────────────────────────────

def handle_time() -> str:
    """Return the current local time as a spoken string.

    Returns:
        Human-readable time string for Aria to speak.
    """
    now = datetime.now()
    return f"It's {now.strftime('%H:%M')}, Chan."


def handle_date() -> str:
    """Return the current date as a spoken string.

    Returns:
        Human-readable date string for Aria to speak.
    """
    now = datetime.now()
    return f"Today is {now.strftime('%A, %d %B %Y')}, Chan."


def handle_calendar() -> str:
    """Return upcoming scheduled reminders from the scheduler.

    Returns:
        Spoken summary of upcoming items, or a message if none exist.
    """
    try:
        return list_reminders()
    except Exception as e:
        print(f"[Router] Calendar lookup error: {e}")
        return "I couldn't check your schedule right now, Chan."
