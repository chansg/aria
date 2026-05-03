"""
core.notifications
------------------
Persistent local queue for Aria's Stage 3b notifications.

The queue is intentionally simple: JSON Lines on disk, append-only for new
items, rewrite only when marking/clearing. This keeps proactive insights
available for review without introducing a database dependency.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.logger import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()
DEFAULT_NOTIFICATIONS_PATH = Path("data") / "notifications.jsonl"


def _queue_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    try:
        import config

        configured = getattr(config, "NOTIFICATIONS_PATH", None)
        if configured:
            return Path(configured)
    except Exception:
        pass
    return DEFAULT_NOTIFICATIONS_PATH


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping invalid notification record at %s:%d: %s", path, line_number, exc)
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")


def enqueue_notification(
    text: str,
    *,
    source: str = "analyst",
    priority: str = "N05",
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a new unread notification to the queue."""
    clean_text = " ".join((text or "").split())
    if not clean_text:
        raise ValueError("notification text cannot be empty")

    record = {
        "id": f"n_{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "priority": priority,
        "status": "unread",
        "text": clean_text,
        "metadata": metadata or {},
    }

    queue_path = _queue_path(path)
    with _LOCK:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    log.info("Notification queued: id=%s priority=%s source=%s", record["id"], priority, source)
    return record


def list_notifications(
    *,
    status: str | None = None,
    limit: int | None = None,
    newest_first: bool = True,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return queued notifications, optionally filtered by status."""
    with _LOCK:
        records = _read_records(_queue_path(path))

    if status is not None:
        records = [record for record in records if record.get("status") == status]

    records.sort(key=lambda record: record.get("created_at", ""), reverse=newest_first)
    if limit is not None:
        records = records[: max(0, limit)]
    return records


def unread_count(*, path: str | Path | None = None) -> int:
    """Return the number of unread queued notifications."""
    return len(list_notifications(status="unread", path=path))


def latest_notification(*, path: str | Path | None = None) -> dict[str, Any] | None:
    """Return the newest notification, if any."""
    items = list_notifications(limit=1, path=path)
    return items[0] if items else None


def mark_all_read(*, path: str | Path | None = None) -> int:
    """Mark all unread notifications as read and return the number changed."""
    queue_path = _queue_path(path)
    with _LOCK:
        records = _read_records(queue_path)
        changed = 0
        read_at = datetime.now().isoformat(timespec="seconds")
        for record in records:
            if record.get("status") != "read":
                record["status"] = "read"
                record["read_at"] = read_at
                changed += 1
        _write_records(queue_path, records)
    log.info("Notifications marked read: %d", changed)
    return changed


def clear_notifications(*, path: str | Path | None = None) -> int:
    """Clear all notifications and return the number removed."""
    queue_path = _queue_path(path)
    with _LOCK:
        records = _read_records(queue_path)
        if queue_path.exists():
            queue_path.unlink()
    log.info("Notifications cleared: %d", len(records))
    return len(records)


def spoken_summary(*, limit: int = 3, path: str | Path | None = None) -> str:
    """Return a short spoken summary of unread notifications."""
    total = unread_count(path=path)
    if total == 0:
        return "No queued insights, Chan."

    items = list_notifications(status="unread", limit=limit, path=path)
    if total == 1:
        return f"I have one queued insight: {items[0]['text']}"

    latest = items[0]["text"]
    extra = total - 1
    return f"I have {total} queued insights, Chan. Latest: {latest} Plus {extra} more."
