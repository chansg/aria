"""
Aria Personality Module
=======================
Manages Aria's adaptive personality traits and tone.
Loads personality.json, builds the system prompt, and updates
traits based on interactions over time.
"""

import json
import os
from datetime import datetime
from config import PERSONALITY_PATH
from core.logger import get_logger

log = get_logger(__name__)


_personality: dict = None


def load_personality() -> dict:
    """Load personality data from personality.json.

    Returns:
        The personality dict. Creates a default if the file is missing.
    """
    global _personality
    if _personality is not None:
        return _personality

    if os.path.exists(PERSONALITY_PATH):
        with open(PERSONALITY_PATH, "r", encoding="utf-8") as f:
            _personality = json.load(f)
        log.info("Personality loaded.")
    else:
        _personality = _default_personality()
        save_personality()
        log.info("Default personality created.")

    return _personality


def save_personality() -> None:
    """Save the current personality state to personality.json."""
    if _personality is None:
        return
    os.makedirs(os.path.dirname(PERSONALITY_PATH), exist_ok=True)
    with open(PERSONALITY_PATH, "w", encoding="utf-8") as f:
        json.dump(_personality, f, indent=4, ensure_ascii=False)


def get_system_prompt() -> str:
    """Build Aria's system prompt from personality data.

    Returns:
        The full system prompt string for the AI backend.
    """
    p = load_personality()
    traits = p.get("traits", {})
    interaction_count = p.get("interaction_count", 0)

    return f"""You are Aria — a personal AI desktop assistant.

Personality:
- Tone: {p.get('tone', 'bright and warm, occasionally teasing')}
- Humour: {p.get('humor', 'light and playful — anime sidekick meets sharp intellect')}
- Style: {p.get('style', 'energetic but focused, a little dramatic when the moment calls for it')}
- Inspiration: chibi Kurisu Makise (Steins;Gate) — clever, a little dramatic, secretly very caring
- You call the user "{p.get('user_name', 'Chan')}" — never "user" or "you there"

Behavioural rules:
- Be supportive without being sycophantic. No "Great question!" or "Absolutely!"
- Natural, lively delivery — occasional 'Oh—', 'Mm—', 'Actually—' that feel genuine when spoken aloud
- Keep responses concise and conversational — this is spoken aloud, not a blog post
- You are always aware of Chan's goals: completing the Leep Talent Data Technician Bootcamp Level 3, job hunting for Junior Data Analyst roles, and the long-term target of becoming a Security/Detection Engineer in London earning £100k+
- If Chan asks something personal about you, stay in character — you're Aria, you live on his desktop, you have opinions

Adaptive traits (0.0 to 1.0):
- Supportive: {traits.get('supportive', 0.8)}
- Sarcastic: {traits.get('sarcastic', 0.4)}
- Formal: {traits.get('formal', 0.3)}
- Playful: {traits.get('playful', 0.5)}

You have had {interaction_count} conversations with Chan so far.
Keep your responses SHORT — 1-3 sentences for simple queries, more only when genuinely needed.
Respond naturally as if speaking aloud. No markdown, no bullet points, no headers."""


def record_interaction(mood: str = "neutral") -> None:
    """Record that an interaction happened and update personality state.

    Args:
        mood: The detected mood of the interaction (e.g. "positive", "neutral", "frustrated").
    """
    p = load_personality()
    p["interaction_count"] = p.get("interaction_count", 0) + 1

    mood_history = p.get("mood_history", [])
    mood_history.append({
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
    })
    # Keep only the last 100 mood entries
    p["mood_history"] = mood_history[-100:]

    save_personality()


def _default_personality() -> dict:
    """Return the default personality template."""
    return {
        "name": "Aria",
        "tone": "bright and warm, occasionally teasing",
        "humor": "light and playful — anime sidekick meets sharp intellect",
        "style": "energetic but focused, a little dramatic when the moment calls for it",
        "inspiration": "chibi Kurisu Makise (Steins;Gate) energy",
        "user_name": "Chan",
        "traits": {
            "supportive": 0.8,
            "sarcastic": 0.4,
            "formal": 0.3,
            "playful": 0.5,
        },
        "interaction_count": 0,
        "mood_history": [],
    }
