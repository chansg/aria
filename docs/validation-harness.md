# Validation Harness

Aria includes a deterministic validation command for pre-merge and post-runtime
health checks.

Run from the repository root:

```powershell
python tools\run_validation.py
```

The command writes a JSON report under:

```text
data/session_reviews/
```

That directory is ignored by Git because reports are runtime artifacts.

## Checks

The harness currently validates:

- `pytest` test suite
- local config shape without exposing secrets
- Kokoro-only TTS provider routing
- finance intent routing and ticker extraction
- Trading 212 demo-only/read-only safety rules

## Optional TTS Smoke Test

By default, the TTS check is config-only so validation remains fast and
non-interactive. To load Kokoro and synthesize a short sample:

```powershell
python tools\run_validation.py --tts-synthesize
```

This does not play audio. It only verifies synthesis succeeds.

## Fast Mode

When you only want configuration and routing checks:

```powershell
python tools\run_validation.py --skip-pytest
```

## Design Constraints

- No microphone selection.
- No live Trading 212 API calls.
- No order placement.
- No secrets written to reports.
- Reports are structured JSON so Codex, Claude, or Aria can review them later.
