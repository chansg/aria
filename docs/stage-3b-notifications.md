# Stage 3b Notifications

Stage 3b adds a local notification queue for proactive analyst insights.

## Goal

Aria should preserve useful observations without interrupting live voice
conversation. The proactive analyst can still monitor the desktop, but non-IDLE
insights are queued by default.

## Current Behavior

When analysis mode is enabled:

1. `core.proactive_analyst.ProactiveAnalyst` reviews the latest screenshot.
2. `IDLE` responses are ignored.
3. Non-IDLE insights are persisted through `core.notifications`.
4. The terminal dashboard shows unread count and latest unread insight.
5. Voice commands can read, mark, or clear the queue.

Default configuration:

```python
PROACTIVE_ANALYST_SPEAK_INSIGHTS = False
NOTIFICATIONS_PATH = "data/notifications.jsonl"
```

## Queue Format

Notifications are stored as JSON Lines at `data/notifications.jsonl`.

Each record includes:

- `id`
- `created_at`
- `source`
- `priority`
- `status`
- `text`
- `metadata`

Proactive analyst insights currently use priority `N05`.

## Voice Commands

Examples:

```text
Aria, what did you notice?
Aria, show queued insights
Aria, mark insights read
Aria, clear notifications
```

These route through the Tier 1 `notifications` intent and do not require a
network call.

## Design Constraints

- No new database dependency.
- No spoken interruption by default.
- Queue writes are local and append-only for new insights.
- Marking read and clearing rewrite/delete the queue file intentionally.
- Runtime queue data is not intended for Git.

## Next Work

- Add richer priority handling beyond `N05`.
- Add dashboard action hints.
- Add optional desktop notification integration.
- Add review/export command for a session summary.
