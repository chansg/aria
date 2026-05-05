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
from core.logger import get_logger

log = get_logger(__name__)


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
    "market": {
        "tier": 1,
        "keywords": [
            # Direct asks
            "market update",   "market summary",   "market today",
            "stock update",    "stocks today",     "ticker update",
            # Conversational
            "how's the market",  "how is the market",  "how are the markets",
            "what's the market", "what are the markets", "give me the market",
            # Full-mode trigger (handler reads "full" / "details" from text)
            "full market update", "detailed market", "market details",
        ],
    },
    "stock_quote": {
        "tier": 1,
        "keywords": [
            "stock price", "share price", "ticker price", "latest price",
            "current price", "price for", "price of", "quote for",
            "quote on", "stock quote", "last close", "last closed",
            "closed at", "last end", "last ended",
        ],
    },
    "finance_followup": {
        "tier": 1,
        "keywords": [
            "is this recent", "is that recent", "how recent", "when was this",
            "when was that", "as of when", "is this live", "is that live",
            "what about", "how about", "last six months", "past six months",
            "six months", "last month", "past month", "last year",
            "performance", "performed",
        ],
    },
    "notifications": {
        "tier": 1,
        "keywords": [
            "queued insight", "queued insights", "notification", "notifications",
            "what did you notice", "what have you noticed",
            "anything worth flagging", "anything to flag",
            "show insights", "read insights", "clear insights",
            "clear notifications", "mark insights read",
        ],
    },
    "broker_account": {
        "tier": 1,
        "keywords": [
            "trading 212", "trading212", "paper account", "practice account",
            "demo account", "broker account", "broker summary",
            "account cash", "cash balance", "available cash",
            "open positions", "demo positions", "paper positions",
            "pending orders", "open orders", "paper portfolio",
            "demo portfolio",
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

EXPLICIT_VISION_REFERENCES = (
    "my screen",
    "the screen",
    "on screen",
    "on my monitor",
    "my monitor",
    "as you can see",
    "you can see from",
    "you can see on",
    "what i'm looking at",
    "what i am looking at",
)


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

    if _matches_explicit_vision_reference(text_lower):
        log.info("Tier 2 matched: vision — explicit screen reference.")
        return {"intent": "vision", "tier": 2}

    # Check Tier 1 first
    for intent_name, config in INTENT_MAP.items():
        if config["tier"] == 1:
            if intent_name == "stock_quote":
                if _matches_stock_quote(text, text_lower, config["keywords"]):
                    log.info("Tier 1 matched: %s — handling locally.", intent_name)
                    return {"intent": intent_name, "tier": 1}
                continue
            if intent_name == "finance_followup":
                if _matches_finance_followup(text, text_lower, config["keywords"]):
                    log.info("Tier 1 matched: %s — handling locally.", intent_name)
                    return {"intent": intent_name, "tier": 1}
                continue
            if _matches(text_lower, config["keywords"]):
                log.info("Tier 1 matched: %s — handling locally.", intent_name)
                return {"intent": intent_name, "tier": 1}

    # Then Tier 2
    for intent_name, config in INTENT_MAP.items():
        if config["tier"] == 2:
            if _matches(text_lower, config["keywords"]):
                backend = "Gemini Flash (vision)" if intent_name == "vision" else "Gemini Flash (web + screen)"
                log.info("Tier 2 matched: %s — %s.", intent_name, backend)
                return {"intent": intent_name, "tier": 2}

    # No match — escalate to Claude
    log.info("No match — Tier 3 escalation to Claude API.")
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


def _matches_explicit_vision_reference(text_lower: str) -> bool:
    """Route explicit screen/monitor references to vision before generic text."""
    return any(phrase in text_lower for phrase in EXPLICIT_VISION_REFERENCES)


def _matches_stock_quote(text: str, text_lower: str, keywords: list[str]) -> bool:
    """Match stock-quote wording without hijacking all price questions."""
    if not _matches(text_lower, keywords):
        return False

    explicit_stock_words = (
        "stock", "share", "ticker", "quote", "last close", "last closed",
        "last end", "last ended", "closed at",
    )
    if any(word in text_lower for word in explicit_stock_words):
        return True

    try:
        from core.market_analyst import extract_ticker_symbol
        return extract_ticker_symbol(text) is not None
    except Exception:
        return False


def _matches_finance_followup(text: str, text_lower: str, keywords: list[str]) -> bool:
    """Match follow-ups only when recent finance context makes them safe."""
    if not _matches(text_lower, keywords):
        return False

    try:
        from core.conversation_state import has_recent_finance_context
        if not has_recent_finance_context():
            return False
    except Exception:
        return False

    recency_phrases = (
        "is this recent", "is that recent", "how recent", "when was this",
        "when was that", "as of when", "is this live", "is that live",
    )
    if any(phrase in text_lower for phrase in recency_phrases):
        return True

    performance_phrases = (
        "last six months", "past six months", "six months", "last month",
        "past month", "last year", "performance", "performed",
    )
    if any(phrase in text_lower for phrase in performance_phrases):
        return True

    if "what about" in text_lower or "how about" in text_lower:
        try:
            from core.market_analyst import extract_ticker_symbol
            return extract_ticker_symbol(text) is not None
        except Exception:
            return False

    return False


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
        log.error("Calendar lookup error: %s", e)
        return "I couldn't check your schedule right now, Chan."


def handle_notifications(text: str = "") -> str:
    """Return or manage queued proactive analyst notifications."""
    try:
        from core.notifications import clear_notifications, mark_all_read, spoken_summary

        text_lower = text.lower()
        if "clear" in text_lower:
            removed = clear_notifications()
            return f"Cleared {removed} queued insight{'s' if removed != 1 else ''}, Chan."
        if "mark" in text_lower and "read" in text_lower:
            changed = mark_all_read()
            return f"Marked {changed} queued insight{'s' if changed != 1 else ''} as read, Chan."
        return spoken_summary()
    except Exception as e:
        log.error("Notification lookup error: %s", e, exc_info=True)
        return "I couldn't check queued insights right now, Chan."
