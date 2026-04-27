# Aria — Autonomous AI Desktop Assistant

A multi-modal AI desktop assistant built in Python on Windows 11. Aria listens for voice input, routes queries through a three-tier reasoning system, analyses the desktop via screen capture, and responds through a Live2D avatar with reactive facial expressions.

---

## Architecture

```
Microphone input
      │
      ▼
faster-whisper (large-v2, CUDA)     ← Speech-to-text, VAD filtering, noise reduction
      │
      ▼
core/router.py                      ← Intent classification — Tier 1 / 2 / 3
      │
      ├── Tier 1 (local)            ← time, date, calendar — zero network, zero cost
      │         │
      │         ▼
      │   Python stdlib
      │
      ├── Tier 2 (web + vision)     ← weather, web queries, screen analysis
      │         │
      │         ▼
      │   DuckDuckGo scrape (Playwright)
      │         +
      │   data/captures/latest.png  ← updated every 5s by screen capture module
      │         │
      │         ▼
      │   Gemini 1.5 Flash          ← unified multimodal reasoning (web + screen)
      │         │
      │   [fallback] Ollama/Mistral ← offline path (USE_LOCAL_FALLBACK = True)
      │
      └── Tier 3 (reasoning)        ← personality, memory, complex queries
                │
                ▼
          Anthropic Claude API
                +
          SQLite memory (episodic + semantic)
                +
          personality.json (evolving traits)
      │
      ▼
core/brain.py                       ← response + mood tag parser
      │
      ├── [HAPPY/SAD/THINKING/SURPRISED/NEUTRAL] tag → avatar/vts_controller.py
      │
      ▼
voice/speaker.py                    ← Piper TTS (hfc_female medium, 24050Hz)
      │
      ▼
VB-Audio Virtual Cable              ← routes TTS audio to VTube Studio for lip sync
      │
      ▼
VTube Studio (Hiyori_A model)       ← Live2D avatar, WebSocket API port 8001
```

---

## Tier System

| Tier | Engine | Query types | Cost |
|------|--------|-------------|------|
| 1 | Python stdlib | Time, date, reminders, calendar | Free |
| 2 | Gemini 1.5 Flash + DuckDuckGo | Weather, web search, screen analysis | Gemini API |
| 2 fallback | Ollama — Mistral 7B | Offline Tier 2 when `USE_LOCAL_FALLBACK = True` | Free |
| 3 | Anthropic Claude API | Personality, memory, reasoning | Claude API |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Speech-to-text | faster-whisper `large-v2` | CUDA on RTX 3080, VAD + noisereduce |
| Wake word | Picovoice Porcupine | Always-on, low power |
| Voice output | Piper TTS `hfc_female medium` | British female, 24050Hz ONNX |
| Lip sync | VB-Audio Virtual Cable | Routes Piper output to VTube Studio |
| Avatar | VTube Studio — Hiyori_A | Live2D, WebSocket API via `pyvts` |
| Tier 2 reasoning | Google Gemini 1.5 Flash | Multimodal — web text + screenshot |
| Tier 2 fallback | Ollama — Mistral 7B | Local offline reasoning |
| Tier 3 reasoning | Anthropic Claude API | claude-sonnet-4 |
| Memory | SQLite — episodic + semantic | `core/memory.py` |
| Personality | JSON state file | `data/personality.json` — evolves over time |
| Scheduler | APScheduler | Reminders and timed events |
| Web scraping | Playwright + BeautifulSoup | DuckDuckGo HTML endpoint |
| Screen capture | mss | Full desktop, every 5s, 10-frame rolling buffer |
| Vision analysis | Gemini 1.5 Flash | Reads `data/captures/latest.png` |
| Intent routing | `core/router.py` | Keyword-based, tier-ordered classification |

---

## Project Structure

```
aria/
├── main.py                        # Entry point — threading, pipeline, avatar init
├── config.py                      # Local secrets — never committed
├── config.example.py              # Safe template — copy to config.py
├── requirements.txt
│
├── core/
│   ├── brain.py                   # Tier dispatch, Gemini reasoning, Claude fallback
│   ├── router.py                  # Intent classification — single source of truth
│   ├── memory.py                  # SQLite episodic + semantic memory
│   ├── personality.py             # System prompt builder, mood tracking
│   ├── scheduler.py               # APScheduler reminders
│   ├── screen_capture.py          # mss desktop capture, rolling buffer
│   ├── vision_analyzer.py         # Gemini vision — analyse_screen(), reason_with_context()
│   ├── web_search.py              # DuckDuckGo scraper, query cleaning, cache
│   └── usage_tracker.py           # Claude API call logging
│
├── voice/
│   ├── listener.py                # Microphone capture, VAD, noise reduction
│   ├── transcriber.py             # faster-whisper large-v2, ARIA_VARIANTS matching
│   ├── speaker.py                 # Piper TTS, sentence splitting, markdown stripping
│   └── trainer.py                 # Voice calibration — WER measurement
│
├── avatar/
│   ├── vts_controller.py          # pyvts WebSocket client, hotkey triggering
│   ├── renderer.py                # Thin wrapper — create_avatar(), set_state()
│   └── animations.py              # State + mood tag constants
│
├── data/                          # Runtime data — never committed
│   ├── memory.db                  # SQLite conversations + facts
│   ├── calendar.db                # Scheduled reminders
│   ├── personality.json           # Aria's current personality state
│   ├── vts_token.json             # VTube Studio auth token
│   ├── web_cache.json             # DuckDuckGo scrape cache (15min TTL)
│   └── captures/                  # Screenshot buffer (latest.png + 10 frames)
│
├── assets/
│   └── voices/                    # Piper ONNX model files (gitignored)
│
└── tools/
    └── kokoro_voice_test.py       # TTS voice comparison utility
```

---

## VTube Studio — Hiyori_A Hotkey Mapping

| Mood tag | Hotkey | Expression |
|----------|--------|------------|
| `HAPPY` | `hiyori_m01` | Bright cheerful smile |
| `SURPRISED` | `hiyori_m02` | Wide eyes, open mouth |
| `THINKING` | `hiyori_m03` | Pensive, looking to side |
| `SAD` | `hiyori_m04` | Downcast expression |
| `NEUTRAL` | `None` | Base state — no hotkey fired |
| State: listening | `hiyori_m05` | Alert listening expression |

Mood tags are prefixed to every LLM response (`[HAPPY] Here's the weather...`), parsed in `brain.py`, and sent to `vts_controller.py` before the text reaches TTS.

---

## Setup

### Prerequisites
- Windows 11
- Python 3.13
- NVIDIA GPU (RTX recommended — used by Whisper large-v2)
- VTube Studio (Steam) — Hiyori_A model loaded, API enabled on port 8001
- VB-Audio Virtual Cable — routes Piper TTS output to VTube Studio
- Ollama — `ollama pull mistral` (offline fallback)
- espeak-ng 1.52.0 (system install — required for future TTS work)

### Installation

```bash
git clone https://github.com/chansg/aria.git
cd aria

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

playwright install chromium
```

### Configuration

```bash
cp config.example.py config.py
# Edit config.py — add API keys and configure settings
```

Required keys in `config.py`:

```python
ANTHROPIC_API_KEY         = "..."
GEMINI_API_KEY            = "..."
PICOVOICE_API_KEY         = "..."   # wake word detection
HF_TOKEN                  = "..."   # HuggingFace — speeds up Whisper downloads

SCREEN_CAPTURE_ENABLED    = True
SCREEN_CAPTURE_INTERVAL   = 5.0
USE_LOCAL_FALLBACK        = False   # True = route Tier 2 through Ollama
CONVERSATION_MODE_DEFAULT = True    # True = Aria responds without name prefix

# VTube Studio hotkeys — must match exact names in VTS
VTS_STATE_HOTKEYS = {
    "idle":      None,
    "listening": "hiyori_m05",
    "thinking":  "hiyori_m03",
    "dormant":   None,
}
VTS_MOOD_HOTKEYS = {
    "HAPPY":     "hiyori_m01",
    "NEUTRAL":   None,
    "THINKING":  "hiyori_m03",
    "SURPRISED": "hiyori_m02",
    "SAD":       "hiyori_m04",
}
```

### Run

```bash
python main.py
```

On first run:
1. Select microphone from the numbered list
2. Stay quiet for 2s while ambient noise is calibrated
3. VTube Studio will show a connection popup — click **Allow**
4. Auth token saved to `data/vts_token.json` for future sessions

---

## Conversation Mode

| Mode | Behaviour | How to toggle |
|------|-----------|---------------|
| ON (default) | Aria responds to all speech | Say "Aria, conversation mode off" |
| OFF | Aria only responds when name is spoken | Say "Aria, conversation mode on" |

---

## Phase Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Voice pipeline — Whisper large-v2, VAD, noisereduce | ✅ |
| 2 | Claude brain + SQLite memory | ✅ |
| 3 | Piper TTS — hfc_female medium, sentence splitting | ✅ |
| 4 | Web scraping — DuckDuckGo + Playwright | ✅ |
| 5 | Avatar — VTube Studio Hiyori_A via pyvts | ✅ |
| 6 | Speech accuracy — Whisper large-v2, VAD, initial prompt | ✅ |
| 7 | Intent router — three-tier keyword classification | ✅ |
| 8 | TTS fixes — markdown stripping, sentence chunking | ✅ |
| 9 | Screen capture — mss, 5s interval, 10-frame buffer | ✅ |
| 10 | Conversation mode toggle | ✅ |
| 11 | Gemini vision Stage 2 — screen analysis on demand | ✅ |
| 12 | VTube Studio hotkeys — Hiyori_A mood expressions | ✅ |
| 13 | Gemini unified Tier 2 — web + screen multimodal | ✅ |
| 14 | Kokoro-82M TTS | ⏸ Deferred — Python 3.13 incompatibility |
| 15 | Stage 3a — Proactive Analyst Loop | ✅ Complete |
| 16 | Stage 3b — Notification state + queued insights | 🔄 Planned |
| 17 | Stage 3c — Finance specialisation | 🔄 Planned |
| 18 | Memory upgrade — FAISS semantic search | 🔄 Planned |
| 19 | Voice recognition training | 🔄 Planned |

---

## Known Issues / Deferred

| Issue | Reason | Resolution |
|-------|--------|------------|
| Kokoro-82M TTS | `kokoro>=0.8` and `misaki>=0.7.5` dropped Python 3.13 support | Revisit when upstream restores 3.13 compatibility |
| Weather via DuckDuckGo | HTML endpoint returns limited weather data | Gemini reasons over partial results — accuracy varies |

---

## Author

Chanveer Grewal — [github.com/chansg](https://github.com/chansg)
