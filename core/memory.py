"""
Aria Memory Module
==================
Three-layer memory system:
- Episodic: conversations and events (SQLite — memory.db)
- Semantic: facts about the user (SQLite — memory.db)
- Personality: adaptive tone (JSON — handled by personality.py)

All data stays local. No cloud storage.
"""

import sqlite3
import json
import os
from datetime import datetime
from config import MEMORY_DB_PATH


_conn: sqlite3.Connection = None


def init_memory() -> None:
    """Initialise the memory database and create tables if needed."""
    global _conn
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    _conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")

    # Episodic memory — conversations and events
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS episodic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT
        )
    """)

    # Semantic memory — facts about the user
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    _conn.commit()
    print("[Aria] Memory system initialised.")

    # Seed default semantic memories if table is empty
    cursor = _conn.execute("SELECT COUNT(*) FROM semantic")
    if cursor.fetchone()[0] == 0:
        _seed_semantic_memories()


def _seed_semantic_memories() -> None:
    """Insert the initial facts about Chan."""
    defaults = [
        ("identity", "name", "Chan"),
        ("education", "current_course", "Leep Talent Data Technician Bootcamp Level 3"),
        ("career", "job_hunting_for", "Junior Data Analyst roles"),
        ("career", "long_term_goal", "Security/Detection Engineer in London earning £100k+"),
    ]
    for category, key, value in defaults:
        store_semantic(category, key, value)
    print("[Aria] Seeded default memories about Chan.")


def store_episodic(role: str, content: str, summary: str = None) -> None:
    """Store a conversation turn in episodic memory.

    Args:
        role: Who said it — "user" or "aria".
        content: The full text of the message.
        summary: Optional one-line summary for context retrieval.
    """
    _ensure_connected()
    _conn.execute(
        "INSERT INTO episodic (timestamp, role, content, summary) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), role, content, summary),
    )
    _conn.commit()


def get_recent_episodic(limit: int = 8) -> list[dict]:
    """Retrieve the most recent conversation turns.

    Args:
        limit: Maximum number of turns to return.

    Returns:
        List of dicts with 'role', 'content', and 'timestamp' keys.
    """
    _ensure_connected()
    cursor = _conn.execute(
        "SELECT role, content, timestamp FROM episodic ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    rows.reverse()  # Chronological order
    return rows


def store_semantic(category: str, key: str, value: str) -> None:
    """Store or update a fact in semantic memory.

    Args:
        category: Fact category (e.g. "identity", "career", "preference").
        key: Unique fact key (e.g. "name", "favourite_language").
        value: The fact value.
    """
    _ensure_connected()
    _conn.execute(
        """INSERT INTO semantic (category, key, value, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               category = excluded.category,
               updated_at = excluded.updated_at""",
        (category, key, value, datetime.now().isoformat()),
    )
    _conn.commit()


def get_all_semantic() -> list[dict]:
    """Retrieve all semantic memories (facts about the user).

    Returns:
        List of dicts with 'category', 'key', and 'value'.
    """
    _ensure_connected()
    cursor = _conn.execute("SELECT category, key, value FROM semantic ORDER BY category, key")
    return [dict(row) for row in cursor.fetchall()]


def get_semantic_by_category(category: str) -> list[dict]:
    """Retrieve semantic memories filtered by category.

    Args:
        category: The category to filter by.

    Returns:
        List of dicts with 'key' and 'value'.
    """
    _ensure_connected()
    cursor = _conn.execute(
        "SELECT key, value FROM semantic WHERE category = ? ORDER BY key",
        (category,),
    )
    return [dict(row) for row in cursor.fetchall()]


def build_memory_context() -> str:
    """Build a context string from memory for the AI prompt.

    Combines semantic facts and recent conversation history into
    a string that can be injected into the system prompt.

    Returns:
        A formatted memory context string.
    """
    _ensure_connected()
    parts = []

    # Semantic memories
    facts = get_all_semantic()
    if facts:
        parts.append("What you know about Chan:")
        for fact in facts:
            parts.append(f"  - {fact['key']}: {fact['value']}")

    # Recent conversation (last 6 turns for context window efficiency)
    recent = get_recent_episodic(limit=6)
    if recent:
        parts.append("\nRecent conversation:")
        for turn in recent:
            speaker = "Chan" if turn["role"] == "user" else "Aria"
            parts.append(f"  {speaker}: {turn['content']}")

    return "\n".join(parts) if parts else ""


def _ensure_connected() -> None:
    """Ensure the database connection is open."""
    global _conn
    if _conn is None:
        init_memory()
