"""
Aria Scheduler Module
=====================
Manages reminders and calendar events via APScheduler.
Events are stored in calendar.db (SQLite) so they survive restarts.
When a reminder fires, Aria announces it aloud.
"""

import re
import sqlite3
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from config import CALENDAR_DB_PATH

# Module-level scheduler instance
_scheduler: BackgroundScheduler = None

# Callback function set by main.py so the scheduler can make Aria speak
_announce_callback = None


def init_scheduler(announce_fn=None) -> BackgroundScheduler:
    """Initialise the APScheduler with SQLite job storage.

    Args:
        announce_fn: A callback function(text: str) that makes Aria
                     speak and print the reminder. Set by main.py.

    Returns:
        The running BackgroundScheduler instance.
    """
    global _scheduler, _announce_callback
    _announce_callback = announce_fn

    os.makedirs(os.path.dirname(CALENDAR_DB_PATH), exist_ok=True)

    jobstores = {
        "default": SQLAlchemyJobStore(url=f"sqlite:///{CALENDAR_DB_PATH}")
    }

    _scheduler = BackgroundScheduler(jobstores=jobstores)
    _scheduler.start()

    _init_events_table()

    job_count = len(_scheduler.get_jobs())
    print(f"[Aria] Scheduler online — {job_count} pending reminder(s).")
    return _scheduler


def _init_events_table() -> None:
    """Create the events log table for human-readable event history."""
    conn = sqlite3.connect(CALENDAR_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            remind_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            fired INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def add_reminder(title: str, remind_at: datetime, description: str = "") -> str:
    """Schedule a one-time reminder.

    Args:
        title: Short name for the reminder (e.g. "Team standup").
        remind_at: When to fire the reminder (datetime object).
        description: Optional longer description.

    Returns:
        A confirmation message string.
    """
    if _scheduler is None:
        return "Scheduler not initialised."

    if remind_at <= datetime.now():
        return f"That time has already passed, Chan. Give me a future time."

    # Store in events table for history
    conn = sqlite3.connect(CALENDAR_DB_PATH)
    conn.execute(
        "INSERT INTO events (title, description, remind_at, created_at) VALUES (?, ?, ?, ?)",
        (title, description, remind_at.isoformat(), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    # Schedule the APScheduler job (sanitize title for safe job ID)
    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:50]
    job_id = f"reminder_{safe_title}_{int(remind_at.timestamp())}"
    _scheduler.add_job(
        _fire_reminder,
        "date",
        run_date=remind_at,
        args=[title, description],
        id=job_id,
        replace_existing=True,
    )

    time_str = remind_at.strftime("%H:%M on %A %d %B")
    return f"Reminder set: '{title}' at {time_str}."


def add_reminder_minutes(title: str, minutes: int, description: str = "") -> str:
    """Schedule a reminder N minutes from now.

    Args:
        title: Short name for the reminder.
        minutes: Minutes from now to fire.
        description: Optional longer description.

    Returns:
        A confirmation message string.
    """
    remind_at = datetime.now() + timedelta(minutes=minutes)
    return add_reminder(title, remind_at, description)


def list_reminders() -> str:
    """List all pending reminders.

    Returns:
        A formatted string of upcoming reminders, or a message if none.
    """
    if _scheduler is None:
        return "Scheduler not initialised."

    jobs = _scheduler.get_jobs()
    if not jobs:
        return "No pending reminders, Chan. Your schedule is clear."

    lines = ["Your upcoming reminders:"]
    for i, job in enumerate(jobs, 1):
        run_time = job.next_run_time.strftime("%H:%M on %A %d %B")
        # Extract title from job args
        title = job.args[0] if job.args else "Untitled"
        lines.append(f"  {i}. {title} — {run_time}")

    return "\n".join(lines)


def cancel_reminder(title: str) -> str:
    """Cancel a reminder by title (partial match).

    Args:
        title: The title (or part of it) to search for.

    Returns:
        Confirmation or not-found message.
    """
    if _scheduler is None:
        return "Scheduler not initialised."

    title_lower = title.lower()
    for job in _scheduler.get_jobs():
        job_title = job.args[0] if job.args else ""
        if title_lower in job_title.lower():
            job.remove()
            return f"Cancelled reminder: '{job_title}'."

    return f"Couldn't find a reminder matching '{title}', Chan."


def _fire_reminder(title: str, description: str = "") -> None:
    """Called by APScheduler when a reminder is due.

    Args:
        title: The reminder title.
        description: Optional description.
    """
    # Mark as fired in events table
    conn = sqlite3.connect(CALENDAR_DB_PATH)
    conn.execute(
        "UPDATE events SET fired = 1 WHERE title = ? AND fired = 0",
        (title,),
    )
    conn.commit()
    conn.close()

    message = f"Reminder, Chan: {title}."
    if description:
        message += f" {description}"

    print(f"\n[Aria] {message}")

    if _announce_callback:
        _announce_callback(message)


def shutdown_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        print("[Aria] Scheduler shut down.")
