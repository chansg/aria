"""
tests.test_notifications
------------------------
Network-free tests for the Stage 3b notification queue.
"""

from __future__ import annotations

from core.notifications import (
    clear_notifications,
    enqueue_notification,
    latest_notification,
    list_notifications,
    mark_all_read,
    spoken_summary,
    unread_count,
)


def test_enqueue_and_summarise_unread_notification(tmp_path) -> None:
    path = tmp_path / "notifications.jsonl"

    item = enqueue_notification("Check the failing test output.", path=path)

    assert item["status"] == "unread"
    assert unread_count(path=path) == 1
    assert latest_notification(path=path)["text"] == "Check the failing test output."
    assert "one queued insight" in spoken_summary(path=path)


def test_mark_all_read_updates_status(tmp_path) -> None:
    path = tmp_path / "notifications.jsonl"
    enqueue_notification("First.", path=path)
    enqueue_notification("Second.", path=path)

    assert mark_all_read(path=path) == 2
    assert unread_count(path=path) == 0
    assert all(item["status"] == "read" for item in list_notifications(path=path))


def test_clear_notifications_removes_queue(tmp_path) -> None:
    path = tmp_path / "notifications.jsonl"
    enqueue_notification("Temporary.", path=path)

    assert clear_notifications(path=path) == 1
    assert list_notifications(path=path) == []
    assert spoken_summary(path=path) == "No queued insights, Chan."
