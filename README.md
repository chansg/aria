# Project Aria

Aria is a Windows-first personal AI desktop assistant built to become a reliable, voice-led operating layer for daily work, screen context, memory, and finance research.

The project is intentionally personal and experimental, but the engineering goal is serious: Aria should feel natural to speak with while remaining observable, testable, modular, and governed by human approval at every meaningful decision point.

Aria is not being built as an autonomous trading bot, a black-box agent, or a decorative avatar project. The current priority is a stable assistant core: voice, reasoning, screen awareness, structured logs, memory, validation, and a safe paper-trading research environment.

## Intent

Aria is designed around one core principle:

> Aria proposes. Chan decides.

That principle applies across the whole project:

- Aria can listen, reason, search, inspect screen context, and summarize.
- Aria can store memory and produce structured feedback for later improvement.
- Aria can analyze financial data and eventually propose paper trades.
- Aria must not silently self-modify, merge code, execute live trades, or make irreversible decisions without approval.

The long-term direction is a personal AI analyst and desktop co-pilot that can move between conversation, screen understanding, financial research, and validated improvement loops.

## Current Capabilities

| Area | Current state |
| --- | --- |
| Voice input | Microphone selection, ambient calibration, silence detection, noise reduction |
| Speech-to-text | faster-whisper with CPU baseline and CUDA-ready configuration |
| Voice output | Kokoro ONNX as the primary TTS provider, tuned for low-latency conversation |
| Conversation mode | Optional always-listening conversation loop |
| Reasoning router | Three-tier routing for local tasks, web/screen context, and Claude reasoning |
| Screen awareness | Rolling screenshot capture with Gemini-based vision analysis |
| Proactive analysis | Opt-in background analyst loop with queued insights |
| Memory | SQLite-backed episodic and semantic memory foundation |
| Terminal dashboard | Rich live dashboard for state, latest response, logs, insights, and visual placeholder status |
| Finance MVP | Stock quote routing, daily market snapshots, compact spoken finance replies |
| Broker sandbox | Trading 212 demo adapter, currently read-only and demo-only |
| Validation | Pytest suite plus a validation harness that writes structured session reports |

## What Aria Is Not

Aria is deliberately not the following:

- Not a live-money autonomous trading system.
- Not a replacement for professional financial advice.
- Not a self-merging coding agent.
- Not dependent on an external visual avatar layer.
- Not allowed to hide failures behind silent fallbacks.

If a subsystem fails, the project should make that failure visible in `logs/aria.log` and in validation output.

## Architecture

```text
Microphone
  |
  v
voice/listener.py
  - PyAudio capture
  - silence detection
  - noise reduction
  |
  v
voice/transcriber.py
  - faster-whisper transcription
  |
  v
core/router.py
  - Tier 1: local deterministic handlers
  - Tier 2: web, screen, and Gemini context
  - Tier 3: Claude reasoning
  |
  v
core/brain.py
  - intent dispatch
  - memory writes
  - response shaping
  - mood/state cues
  |
  v
voice/speaker.py
  - Kokoro ONNX synthesis
  - sentence chunking
  - playback
  |
  v
core/terminal_ui.py
  - live dashboard
  - logs
  - insight queue
```

## Reasoning Tiers

| Tier | Purpose | Examples |
| --- | --- | --- |
| Tier 1 | Fast local actions | time, date, reminders, stock quotes, market snapshot, broker account summary |
| Tier 2 | External context | web search, finance news, current events, weather, screen analysis |
| Tier 3 | Complex reasoning | planning, explanation, personality-rich conversation, multi-step synthesis |

This structure exists to keep common voice interactions fast and cheap while preserving access to stronger reasoning when needed.

## Finance Direction

Aria's finance work is moving in stages.

The current finance layer is an assistant-friendly MVP:

- Fetch a small set of market prices and daily snapshots.
- Answer concise spoken stock quote questions.
- Preserve recent finance context for follow-up questions.
- Connect to Trading 212 demo in a read-only, audit-friendly way.

The intended direction is a safer personal quant research environment:

1. Build a dedicated market data spine.
2. Store normalized data locally, likely with DuckDB or Parquet.
3. Add feature generation and regime tagging.
4. Add backtesting before any execution logic.
5. Add a trade journal and outcome attribution.
6. Add risk controls and circuit breakers.
7. Allow paper-trade proposals in Trading 212 demo only.
8. Require explicit human approval before any paper order.
9. Treat live trading as out of scope until the paper system has a long validation record.

Trading 212 demo is a training and execution sandbox, not the source of truth for market history. Historical data, news, filings, and model features should come from a separate data pipeline.

## Roadmap

| Stage | Goal | Status |
| --- | --- | --- |
| 1 | Foundation: voice, reasoning, logs, dashboard, screen capture, market MVP | Current |
| 2 | Self-logging and feedback: structured session reviews and unknown/low-confidence capture | In progress |
| 3 | Improvement proposals: convert logs and validation output into reviewed GitHub issues | Planned |
| 4 | Autonomous test and validation: run tests, flag regressions, prepare reviewable PRs | Early foundation |
| 5 | Adaptive intelligence loop: improve memory, routing, and architecture from real usage | Deferred |

Stage 5 is intentionally deferred. It only becomes credible after Stage 2 and Stage 4 are boringly reliable.

## Safety Model

Aria's safety model is architectural, not cosmetic:

- API keys stay in local configuration or environment variables.
- `config.py`, runtime data, captures, logs, and private model assets are not intended for public commits.
- Trading 212 is demo-only in the current code path.
- Broker operations are read-only until a separate human-gated execution layer exists.
- Proactive analyst output is queued by default so it does not interrupt conversation.
- Test and validation output should exist before code is merged.
- Aria may propose changes, but the user reviews and approves them.

## Repository Layout

```text
aria/
├── main.py                         # desktop assistant entry point
├── config.example.py               # safe configuration template
├── requirements.txt
│
├── core/
│   ├── brain.py                    # reasoning dispatch and response shaping
│   ├── router.py                   # intent classification
│   ├── market_analyst.py           # quote and market snapshot logic
│   ├── brokers/
│   │   └── trading212.py           # Trading 212 demo adapter
│   ├── memory.py                   # SQLite memory layer
│   ├── notifications.py            # queued proactive insights
│   ├── proactive_analyst.py        # opt-in background analyst loop
│   ├── screen_capture.py           # rolling screenshot capture
│   ├── terminal_ui.py              # Rich terminal dashboard
│   ├── vision_analyzer.py          # Gemini screen reasoning
│   └── web_search.py               # web/weather context retrieval
│
├── voice/
│   ├── listener.py                 # microphone capture
│   ├── transcriber.py              # faster-whisper transcription
│   ├── speaker.py                  # TTS facade and playback
│   └── tts/                        # Kokoro and optional Piper providers
│
├── avatar/
│   └── renderer.py                 # lightweight visual placeholder facade
│
├── docs/                           # architecture notes and generated project docs
├── tests/                          # pytest coverage
├── tools/                          # validation and benchmarking tools
├── assets/                         # local model/audio assets
└── data/                           # local runtime state, ignored/private
```

## Setup

### Requirements

- Windows 11
- Python 3.13
- PowerShell 7 recommended
- NVIDIA GPU recommended for Kokoro ONNX CUDA
- Anthropic API key for Tier 3 reasoning
- Gemini API key for Tier 2 web/screen reasoning
- Trading 212 demo API key only if testing the broker sandbox

### Install

```powershell
git clone https://github.com/chansg/aria.git
cd aria

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
playwright install chromium
```

### Configure

```powershell
Copy-Item config.example.py config.py
```

Then edit `config.py` locally. Do not commit real API keys.

Important settings:

```python
ANTHROPIC_API_KEY = "..."
GEMINI_API_KEY = "..."

USE_LOCAL_FALLBACK = False
SCREEN_CAPTURE_ENABLED = True
CONVERSATION_MODE_DEFAULT = True

TTS_PROVIDER = "kokoro"
TTS_CONVERSATION_PROVIDER = "kokoro"
TTS_FALLBACK_PROVIDER = ""
TTS_FAIL_LOUD = True

TRADING212_ENV = "demo"
TRADING212_BASE_URL = "https://demo.trading212.com/api/v0"
```

### Run

```powershell
python main.py
```

On startup, Aria will ask for a microphone, calibrate ambient noise, initialize memory, start screen capture if enabled, and open the Rich terminal dashboard.

## Validation

Run the test suite:

```powershell
python -m pytest -q
```

Run the validation harness:

```powershell
python tools\run_validation.py
```

Validation reports are written to:

```text
data/session_reviews/
```

These reports are intended to become part of Aria's feedback loop: logs and validation data should drive issues, fixes, and later improvement proposals.

## Operating Principles

Development should follow these constraints:

- Prefer small reviewed changes over broad speculative refactors.
- Keep runtime failures visible in logs.
- Add tests for every bug discovered through manual voice sessions.
- Do not expand finance execution before backtesting, journaling, and risk controls exist.
- Do not let the assistant self-modify or merge unreviewed changes.
- Keep the voice experience fast enough to feel conversational.

## Current Focus

The near-term focus is:

1. Improve screen-aware routing so natural phrases like "on my screen" trigger vision.
2. Improve finance/current-event routing so market news goes through Tier 2 context.
3. Make `aria.log` and validation reports useful for debugging by humans and AI tools.
4. Keep Kokoro TTS fast and reliable without silent fallback.
5. Build the paper-trading research foundation without introducing live-trading risk.

## Project Status

Aria is active, experimental, and under rapid iteration. The codebase should be treated as a personal research system, not production software.

The target is not theatrical autonomy. The target is a dependable assistant that earns more responsibility through logs, tests, validation, and explicit human approval.
