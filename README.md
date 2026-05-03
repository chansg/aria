# Aria - Autonomous AI Desktop Assistant

Aria is a multi-modal AI desktop assistant built in Python for Windows 11. It listens for voice input, routes queries through a tiered reasoning system, analyses desktop screenshots when enabled, tracks local memory, and speaks responses through a configurable local TTS provider.

The visual model layer has been removed for now. `avatar/renderer.py` is a lightweight placeholder facade that records state and mood cues without depending on an external renderer. This keeps the core pipeline stable while finance, memory, logging, tests, and validation mature.

---

## Architecture

```text
Microphone input
      |
      v
voice/listener.py                    # PyAudio capture, silence detection, noise reduction
      |
      v
voice/transcriber.py                 # faster-whisper, CUDA, VAD filtering
      |
      v
core/router.py                       # Intent classification: Tier 1 / 2 / 3
      |
      +--> Tier 1                    # Local handlers: time, date, reminders, market update
      |
      +--> Tier 2                    # Web + screen context through Gemini, with Ollama fallback
      |
      +--> Tier 3                    # Claude API with personality, memory, and scheduler tools
      |
      v
core/brain.py                        # Dispatch, mood-tag parsing, memory persistence
      |
      v
voice/speaker.py                     # TTS facade, sentence splitting, markdown stripping
      |
      v
avatar/renderer.py                   # Local visual placeholder facade
```

---

## Tier System

| Tier | Engine | Query types | Cost |
|------|--------|-------------|------|
| 1 | Python stdlib / local modules | Time, date, reminders, market snapshots | Free |
| 2 | Gemini Flash + DuckDuckGo + screenshot context | Weather, web search, screen analysis | Gemini API |
| 2 fallback | Ollama / Mistral | Offline Tier 2 when `USE_LOCAL_FALLBACK = True` | Free |
| 3 | Anthropic Claude API | Personality, memory, complex reasoning | Claude API |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Speech-to-text | faster-whisper | CPU `small` default, CUDA-ready when NVIDIA runtime is healthy |
| Wake word | Whisper keyword spotting | Local, no extra wake-word API |
| Voice output | Kokoro ONNX | CUDA provider, fail-loud, no silent Piper fallback |
| Visual layer | Local placeholder facade | No external renderer dependency |
| Tier 2 reasoning | Google Gemini Flash | Web text + optional screenshot |
| Tier 2 fallback | Ollama / Mistral | Local offline reasoning |
| Tier 3 reasoning | Anthropic Claude API | Complex responses and tool use |
| Market analysis | yfinance + pandas | Daily OHLCV snapshots and anomaly summary |
| Memory | SQLite | Episodic and semantic memory |
| Personality | JSON | `data/personality.json` |
| Scheduler | APScheduler | Reminders and timed events |
| Web scraping | Playwright + BeautifulSoup | DuckDuckGo HTML endpoint |
| Screen capture | mss | Full desktop capture with rolling buffer |
| Terminal UI | rich | Live status and activity dashboard |

---

## Project Structure

```text
aria/
├── main.py
├── config.py                      # Local secrets - never committed
├── config.example.py
├── requirements.txt
│
├── core/
│   ├── brain.py                   # Tier dispatch, Gemini/Claude/Ollama handlers
│   ├── router.py                  # Intent classification
│   ├── market_analyst.py          # yfinance snapshots + spoken summary
│   ├── memory.py                  # SQLite episodic + semantic memory
│   ├── personality.py             # System prompt builder, interaction tracking
│   ├── proactive_analyst.py       # Optional Gemini screenshot loop
│   ├── scheduler.py               # APScheduler reminders
│   ├── screen_capture.py          # mss desktop capture
│   ├── terminal_ui.py             # rich dashboard
│   ├── vision_analyzer.py         # Gemini screenshot analysis
│   └── web_search.py              # Web search and weather context
│
├── voice/
│   ├── listener.py                # Microphone capture
│   ├── transcriber.py             # faster-whisper
│   ├── speaker.py                 # TTS facade and playback
│   ├── tts/                       # Kokoro ONNX and Piper providers
│   ├── trainer.py                 # Voice calibration
│   ├── wake.py                    # Wake-word spotting
│   └── chime.py                   # Wake chime
│
├── avatar/
│   ├── renderer.py                # Local placeholder facade
│   └── animations.py              # Shared state and mood constants
│
├── tests/
│   └── test_market_analyst.py
│
├── docs/
│   └── voice-runtime.md          # Kokoro CUDA baseline and troubleshooting
│
├── data/                          # Runtime data - never commit private files
├── assets/                        # Local audio/model/sprite assets
└── tools/
    ├── benchmark_tts.py           # TTS provider latency benchmark
    └── generate_sprites.py
```

---

## Setup

### Prerequisites

- Windows 11
- Python 3.13
- NVIDIA GPU recommended for Kokoro CUDA and optional Whisper CUDA
- Ollama with `mistral` pulled if you want the local Tier 2 fallback
- Kokoro ONNX full model and voices file for the primary voice
- Piper voice model is optional; it is not used as a silent fallback by default

### Installation

```powershell
git clone https://github.com/chansg/aria.git
cd aria

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
playwright install chromium
```

### Configuration

```powershell
Copy-Item config.example.py config.py
```

Required or commonly edited settings:

```python
ANTHROPIC_API_KEY = "..."
GEMINI_API_KEY = "..."

SCREEN_CAPTURE_ENABLED = True
SCREEN_CAPTURE_INTERVAL = 5.0
USE_LOCAL_FALLBACK = False
CONVERSATION_MODE_DEFAULT = True

TTS_PROVIDER = "kokoro"
TTS_CONVERSATION_PROVIDER = "kokoro"
TTS_FALLBACK_PROVIDER = ""
TTS_FAIL_LOUD = True

KOKORO_ONNX_MODEL_PATH = "assets/voices/kokoro/kokoro-v1.0.onnx"
KOKORO_ONNX_VOICES_PATH = "assets/voices/kokoro/voices-v1.0.bin"
KOKORO_ONNX_PROVIDER = "CUDAExecutionProvider"
KOKORO_DISABLE_PROVIDER_FALLBACK = True
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "en-us"
KOKORO_SPEED = 1.0

PROACTIVE_ANALYST_SPEAK_INSIGHTS = False

MARKET_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
MARKET_VOLUME_SPIKE_PCT = 30.0
MARKET_VOLATILITY_SIGMA = 2.0
```

### Run

```powershell
python main.py
```

On first run:

1. Select a microphone from the numbered list.
2. Stay quiet while ambient noise is calibrated.
3. Choose whether to run voice calibration.
4. Use the rich terminal dashboard to monitor state, logs, and responses.

---

## Voice Runtime Baseline

Aria's stable voice baseline is Kokoro ONNX on the NVIDIA CUDA execution provider.
The full `kokoro-v1.0.onnx` model is used for GPU inference. The int8 model is
kept only as a CPU-oriented asset because it is slower and less reliable on the
RTX 4070 CUDA path.

Conversation speech should log `provider=kokoro-onnx` and
`CUDAExecutionProvider`. Piper is retained as an optional provider but is not a
silent fallback in the default configuration. If Kokoro fails, Aria should fail
loudly in `logs/aria.log` so the runtime issue is visible.

Benchmark the current TTS path with:

```powershell
python tools\benchmark_tts.py --provider kokoro
```

Expected warm synthesis on the RTX 4070 is roughly half a second for short
conversation replies after the model has loaded. See
[`docs/voice-runtime.md`](docs/voice-runtime.md) for the full checklist and
troubleshooting notes.

---

## Market Analyst

The market analyst MVP adds a Tier 1 voice intent:

```text
Aria, market update
Aria, full market update
```

It fetches recent OHLCV data with yfinance, computes moving averages, daily change, 30-day volatility, volume deviation, and crossover signals, then writes a structured snapshot to:

```text
data/market/YYYY-MM-DD.json
```

The spoken summary is intentionally short by default. Full mode gives one sentence per configured ticker.

---

## Conversation Mode

| Mode | Behaviour | How to toggle |
|------|-----------|---------------|
| ON | Aria responds to all speech | Say "Aria, conversation mode off" |
| OFF | Aria only responds when addressed | Say "Aria, conversation mode on" |

---

## Phase Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Voice pipeline - Whisper, VAD, noisereduce | Complete |
| 2 | Claude brain + SQLite memory | Complete |
| 3 | TTS provider facade - Kokoro ONNX CUDA, fail-loud runtime | Complete |
| 4 | Web scraping - DuckDuckGo + Playwright | Complete |
| 5 | External visual model layer | Removed |
| 6 | Speech accuracy - Whisper prompts and calibration | Complete |
| 7 | Intent router - three-tier keyword classification | Complete |
| 8 | Screen capture - mss rolling buffer | Complete |
| 9 | Gemini vision - screen analysis on demand | Complete |
| 10 | Conversation mode toggle | Complete |
| 11 | Gemini unified Tier 2 - web + screen context | Complete |
| 12 | Stage 3a - Proactive Analyst Loop | Complete |
| 13 | Structured logging - `logs/aria.log` rotation | Complete |
| 14 | Rich terminal dashboard | Complete |
| 15 | Market analyst MVP - daily snapshots + spoken summary | Complete |
| 16 | Stage 3b - notification state + queued insights | Planned |
| 17 | Stage 3c - finance specialisation, sentiment, news, filings | Planned |
| 18 | Memory upgrade - semantic search | Planned |

---

## Placeholder Recommendation

The current placeholder is deliberately the simplest option: a terminal-visible state facade. It is cheap to maintain, testable, and keeps the application focused on finance and reliability.

For the next quick visual step, prefer a tiny local tray/window or rich-dashboard panel that shows:

- state: listening / thinking / speaking / idle
- last mood tag
- latest market snapshot status
- queued proactive insight count

That gives immediate feedback without reintroducing a heavy external rendering dependency.

---

## Known Issues / Deferred

| Issue | Reason | Resolution |
|-------|--------|------------|
| Rich UI unavailable | Optional dependency may be missing | App falls back to plain terminal output |
| Weather via search snippets | Search pages can return partial context | Prefer structured sources in future finance work |
| Advanced visual model | Not core to finance assistant reliability | Revisit after Stage 3b/3c validation |

---

## Author

Chanveer Grewal - [github.com/chansg](https://github.com/chansg)
