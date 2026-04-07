# Aria — Personal AI Desktop Assistant

An autonomous AI desktop assistant with a chibi anime avatar, British voice,
persistent memory, and live web access. Built in Python on Windows 11.

## Features
- 🎙️ Wake word activation — "Hey Aria"
- 🧠 Powered by Anthropic Claude API (with local model migration path)
- 🗣️ Coqui TTS — mature British female voice
- 🖥️ Chibi sprite avatar — desktop overlay, bottom right corner
- 💾 Persistent memory — SQLite + JSON
- 🌐 Live web scraping — weather and more
- 📅 Scheduling and reminders
- 📊 Claude API usage tracking

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
```

### 4. Configure API keys
```
cp config.example.py config.py
# Edit config.py and add your API keys
```

### 5. Run
```
python main.py
```

## Project Structure
```
aria/
├── main.py
├── config.py              # Not committed — contains API keys
├── config.example.py      # Safe template
├── core/
│   ├── brain.py           # Claude API + intent routing
│   ├── memory.py          # SQLite persistent memory
│   ├── scheduler.py       # APScheduler reminders
│   ├── personality.py     # Aria's evolving personality
│   ├── router.py          # Intent classification
│   └── usage_tracker.py   # Claude API usage logging
├── voice/
│   ├── listener.py        # Wake word + microphone
│   ├── transcriber.py     # Whisper speech-to-text
│   ├── speaker.py         # Coqui TTS voice output
│   └── trainer.py         # Voice recognition training
├── avatar/
│   ├── renderer.py        # Pygame sprite renderer
│   ├── animations.py      # Sleep, wake, talk, blink
│   └── window.py          # Transparent desktop overlay
├── assets/
│   └── sprites/           # Aria expression sprites
└── data/                  # Local data — not committed
```

## Tech Stack
| Component | Technology |
|-----------|-----------|
| AI Brain | Anthropic Claude API |
| Wake Word | Picovoice Porcupine |
| Voice Input | faster-whisper |
| Voice Output | Coqui TTS |
| Avatar | Pygame sprite system |
| Memory | SQLite + JSON |
| Scheduler | APScheduler |
| Web Access | Playwright |

## Author
Chanveer Grewal — github.com/chansg
