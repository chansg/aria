# config.example.py
# Copy this file to config.py and fill in your own values
# Never commit config.py to GitHub
#
# See config.py for the full list of settings. The constants below
# are the ones most likely to need editing on a new machine.

ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
PICOVOICE_API_KEY = "your-picovoice-api-key-here"
OPENWEATHER_API_KEY = ""  # Optional — currently using web scraping

# --- Ollama (local LLM for Tier 2 reasoning) ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# --- Conversation / Prompt Limits ---
MAX_CONVERSATION_TURNS = 10   # Recent exchanges sent as context to Claude
MAX_SPEAK_LENGTH = 500        # Max characters per TTS utterance
TRANSCRIPTION_TIMEOUT = 15    # Seconds before aborting a Whisper transcription

# --- VTube Studio — avatar expression hotkeys ---
# Set these to match the exact hotkey names configured in VTube Studio
# Leave as None if the hotkey is not yet configured
VTS_STATE_HOTKEYS = {
    "idle":      None,   # Default resting expression
    "listening": None,   # Alert/attentive expression
    "thinking":  None,   # Contemplative expression
    "dormant":   None,   # Sleeping expression
}

VTS_MOOD_HOTKEYS = {
    "HAPPY":     None,   # Happy/cheerful expression
    "NEUTRAL":   None,   # Default neutral expression
    "THINKING":  None,   # Thoughtful expression
    "SURPRISED": None,   # Surprised expression
    "SAD":       None,   # Sad/concerned expression
}

# VTube Studio API
VTS_PORT = 8001
VTS_ENABLED = True   # Set to False to run without VTube Studio

# ── Screen Capture ────────────────────────────────────────────────────────────
# Stage 1: Desktop screenshot capture for gameplay analysis pipeline
# Set SCREEN_CAPTURE_ENABLED = True to activate
# Gemini API key required for Stage 2 (vision analysis) — not needed today
#
# Where to add your Gemini API key when Stage 2 is ready:
#   GEMINI_API_KEY = "your-gemini-api-key-here"
#   Get your key at: https://aistudio.google.com/app/apikey
#
SCREEN_CAPTURE_ENABLED  = False   # Set to True to enable capture
SCREEN_CAPTURE_INTERVAL = 5.0     # Seconds between screenshots

# GEMINI_API_KEY = "your-gemini-api-key-here"  # Uncomment for Stage 2

# ── Conversation Mode ─────────────────────────────────────────────────────────
# True  = Aria responds to all speech without requiring her name (default)
# False = Aria only responds when her name is spoken first
CONVERSATION_MODE_DEFAULT = True

# --- TTS Voice Model ---
# Download from HuggingFace: rhasspy/piper-voices → en/en_US/hfc_female/medium/
# Save to: assets/voices/en_US-hfc_female-medium.onnx (and .onnx.json)
PIPER_MODEL_PATH = "assets/voices/en_US-hfc_female-medium.onnx"
