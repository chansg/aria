# Aria — Personal AI Desktop Assistant

An autonomous AI desktop assistant with a chibi anime avatar, natural voice,
persistent memory, and live web access. Built in Python on Windows 11.

## Features
- Wake word activation — "Hey Aria"
- Powered by Anthropic Claude API (with local Ollama fallback)
- Piper TTS — hfc_female medium voice (local ONNX inference)
- Chibi sprite avatar — desktop overlay with Win32 transparency
- Persistent memory — SQLite episodic + semantic
- Live web scraping — weather via Playwright/httpx
- Scheduling and reminders via APScheduler
- Voice recognition training with WER scoring

## Phase Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Voice pipeline (Whisper) | Complete |
| 2 | Claude brain + memory | Complete |
| 3 | Piper TTS voice output | Complete |
| 4 | Live web scraping — weather | Complete |
| 5 | Sprite avatar system | Complete |
| 6 | Voice recognition training | Planned |
| 7 | Intent router | Planned |
| 8 | Wake word polish + animations | Planned |

## Setup

### 1. Clone the repository
```
git clone https://github.com/chansg/aria.git
cd aria
```

### 2. Create a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
playwright install chromium
```

### 4. Download voice model
Download the Piper hfc_female medium ONNX model from HuggingFace:
```
mkdir assets\voices
curl -L -o assets/voices/en_US-hfc_female-medium.onnx ^
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx"
curl -L -o assets/voices/en_US-hfc_female-medium.onnx.json ^
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json"
```

### 5. Configure API keys
```
cp config.example.py config.py
# Edit config.py and add your Anthropic API key
```

### 6. Run
```
python main.py
```

## Project Structure
```
aria/
├── main.py                # Entry point — avatar + voice pipeline
├── config.py              # Not committed — contains API keys
├── config.example.py      # Safe template
├── core/
│   ├── brain.py           # Claude API + local intent routing
│   ├── memory.py          # SQLite persistent memory
│   ├── scheduler.py       # APScheduler reminders
│   ├── personality.py     # Aria's evolving personality
│   └── web_search.py      # Playwright web scraping (weather + more)
├── voice/
│   ├── listener.py        # Microphone capture + silence detection
│   ├── transcriber.py     # Whisper speech-to-text (CUDA)
│   ├── speaker.py         # Piper TTS voice output + lip-sync
│   ├── wake.py            # Wake word detection (local Whisper)
│   ├── trainer.py         # Voice recognition training + WER
│   └── chime.py           # Wake notification sound
├── avatar/
│   ├── renderer.py        # High-level avatar state API
│   ├── animations.py      # Tick-based animation system
│   └── window.py          # Win32 transparent desktop overlay
├── tools/
│   └── generate_sprites.py # Procedural sprite generation (Pillow)
├── assets/
│   ├── sprites/           # Aria expression sprites (6 states)
│   ├── voices/            # Piper ONNX voice models (gitignored)
│   └── sounds/            # Generated audio files (gitignored)
└── data/                  # Local data — not committed
```

## Tech Stack
| Component | Technology |
|-----------|-----------|
| AI Brain | Anthropic Claude API (Haiku + Sonnet) |
| Local Fallback | Ollama |
| Voice Input | faster-whisper (CUDA) |
| Voice Output | Piper TTS (hfc_female medium) |
| Wake Word | Whisper keyword spotting (no API keys) |
| Avatar | Pygame sprite system + Win32 overlay |
| Memory | SQLite (episodic + semantic) + JSON |
| Scheduler | APScheduler + SQLAlchemy |
| Web Scraping | Playwright + httpx |

## Author
Chanveer Grewal — github.com/chansg
