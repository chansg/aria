# config.example.py
# Copy this file to config.py and fill in your own values
# Never commit config.py to GitHub
#
# See config.py for the full list of settings. The constants below
# are the ones most likely to need editing on a new machine.

import os
from dotenv import load_dotenv

# Load environment variables from .env file (kept out of git)
load_dotenv()

# ── API Keys — load from .env, never hardcode ─────────────────────────────────
# Copy .env.example to .env and fill in your keys.
# NEVER commit .env to git.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")

# ── Reasoning engine ───────────────────────────────────────────────────────
# Tier 2 queries use Gemini Flash by default (web scrape + screenshot).
# USE_LOCAL_FALLBACK = True forces Tier 2 queries through Ollama/Mistral
# instead of Gemini. Use when offline or conserving Gemini API quota.
USE_LOCAL_FALLBACK = False

# --- Ollama (local LLM for Tier 2 fallback) ---
# Run `ollama list` to see which models are installed locally.
# The model name here must match exactly — a mismatch returns 404.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# --- Conversation / Prompt Limits ---
MAX_CONVERSATION_TURNS = 10   # Recent exchanges sent as context to Claude
# MAX_SPEAK_LENGTH is deprecated — speaker.py uses sentence-based chunking only.
# Kept here to avoid import errors in any legacy references.
MAX_SPEAK_LENGTH = 9999       # effectively disabled
TRANSCRIPTION_TIMEOUT = 15    # Seconds before aborting a Whisper transcription

# ── VTube Studio — Hiyori_A Model Configuration ───────────────────────────────
# Hotkey names must match exactly what is configured in VTube Studio.
# These have been manually created and confirmed in VTS for the Hiyori_A model.

VTS_STATE_HOTKEYS = {
    "idle":      None,           # Base state — no hotkey, Hiyori rests naturally
    "listening": "hiyori_m05",   # Active listening expression
    "thinking":  "hiyori_m03",   # Pensive, looking to the side
    "dormant":   None,           # Sleeping — no hotkey at this stage
}

VTS_MOOD_HOTKEYS = {
    "HAPPY":     "hiyori_m01",   # Bright cheerful smile
    "NEUTRAL":   None,           # Base state — no hotkey fired
    "THINKING":  "hiyori_m03",   # Pensive expression
    "SURPRISED": "hiyori_m02",   # Wide eyes, open mouth
    "SAD":       "hiyori_m04",   # Downcast, saddened
}

# VTube Studio API
VTS_PORT    = 8001
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
