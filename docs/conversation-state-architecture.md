# Conversation State Architecture

## Purpose

Aria now has a small volatile session-state layer for live voice context.

This is separate from durable memory:

- `core.memory` remains the long-term SQLite-backed episodic and semantic store.
- `core.conversation_state` stores only the short-lived facts needed to answer
  immediate follow-ups without re-entering Claude or Gemini.

The goal is lower latency and better conversational continuity, not autonomous
self-modification.

## Current Scope

The first implementation focuses on finance conversation state.

After a successful quote request, Aria stores:

- symbol
- display name
- latest price
- previous close
- daily change percentage
- quote date
- data source
- original user text

This lets follow-ups such as "is this recent?" or "how has it performed over
six months?" stay in the local Tier 1 path.

## Runtime Flow

Before:

```text
voice -> transcription -> router -> Claude/Gemini if ambiguous -> TTS
```

After:

```text
voice -> transcription -> router -> fast session context -> local answer -> TTS
```

Longer-term analysis still belongs in a background consolidator, not the live
voice path.

## Why This Matters

The Apple quote test showed the local quote path was fast:

- reasoning: under one second
- TTS playback: materially longer than reasoning

That means memory architecture alone does not solve voice latency. The immediate
latency gains come from:

- fewer escalations to Claude for obvious follow-ups
- shorter local responses
- avoiding unnecessary long-form answers
- TTS text cleanup before synthesis

## Design Constraints

- Session state is in-memory only.
- It expires after a short TTL.
- Failed quote requests do not update state.
- Durable learning still requires explicit Stage 2/Stage 5 work.
- Financial answers remain analytical and informational, not trading execution.

## Follow-Up Work

Recommended next steps:

- Add a background memory consolidator for session reviews.
- Store structured unknowns and low-confidence outputs.
- Extend local finance intents for watchlists and portfolio summaries.
- Add validation checks that compare expected route behavior against `aria.log`.
