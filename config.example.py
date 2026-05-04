# config.example.py
# Copy this file to config.py and fill in your own values
# Never commit config.py to GitHub
#
# See config.py for the full list of settings. The constants below
# are the ones most likely to need editing on a new machine.

ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
PICOVOICE_API_KEY = "your-picovoice-api-key-here"
OPENWEATHER_API_KEY = ""  # Optional — currently using web scraping

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
SPOKEN_RESPONSE_MAX_CHARS = 320  # Normal live replies are capped unless detail is requested
MAX_SPEAK_LENGTH = 9999      # Legacy compatibility; speaker.py chunks by sentence
TRANSCRIPTION_TIMEOUT = 45    # Seconds before giving up on a Whisper transcription
NOTIFICATIONS_PATH = "data/notifications.jsonl"

# --- Whisper Speech-to-Text ---
# Windows CPU fallback: use "small" for responsiveness or "medium" for accuracy.
# CUDA path: use WHISPER_DEVICE="cuda", WHISPER_COMPUTE_TYPE="float16", and a
# larger model only after the NVIDIA/cuBLAS runtime is confirmed healthy.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# ── Visual Placeholder ───────────────────────────────────────────────────────
# The visual layer is currently a local no-op facade. It keeps Aria's state
# and mood hooks intact while the project focuses on core reasoning, voice,
# market analysis, memory, and validation.
VISUAL_PLACEHOLDER_ENABLED = True

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

# --- TTS Voice Provider ---
# Kokoro is Aria's target voice. CUDA should use an NVIDIA GPU when
# onnxruntime-gpu[cuda,cudnn] is installed. Piper is optional, not a silent fallback.
TTS_PROVIDER = "kokoro"       # "kokoro" | "kokoro-onnx" | "piper"
TTS_CONVERSATION_PROVIDER = "kokoro"
TTS_FALLBACK_PROVIDER = ""
TTS_FAIL_LOUD = True

# Kokoro ONNX assets:
#   https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
# Use the full model for CUDA. The int8 model is better suited to CPU fallback.
KOKORO_ONNX_MODEL_PATH = "assets/voices/kokoro/kokoro-v1.0.onnx"
KOKORO_ONNX_VOICES_PATH = "assets/voices/kokoro/voices-v1.0.bin"
KOKORO_ONNX_PROVIDER = "CUDAExecutionProvider"
KOKORO_DISABLE_PROVIDER_FALLBACK = True
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "en-us"
KOKORO_SPEED = 1.0

# Stage 3b should surface proactive insights as notifications. Keep spontaneous
# speech off for now so analyst output cannot interrupt live conversation.
PROACTIVE_ANALYST_SPEAK_INSIGHTS = False

# Piper fallback voice:
# Download from HuggingFace: rhasspy/piper-voices -> en/en_US/hfc_female/medium/
# Save to: assets/voices/en_US-hfc_female-medium.onnx (and .onnx.json)
PIPER_MODEL_PATH = "assets/voices/en_US-hfc_female-medium.onnx"

# ── Market Analyst (Phase 18) ───────────────────────────────────────────────
# Daily stock snapshot triggered by "Aria, market update".
# Tickers fetched via yfinance — no API key required.
# Add or remove tickers freely; the spoken summary scales to whatever is here.
MARKET_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
# Volume must exceed 30-day average by this percentage to be flagged as a spike.
MARKET_VOLUME_SPIKE_PCT = 30.0
# Today's daily change in absolute % must exceed this many σ of the
# 30-day return distribution to be flagged as a "large move".
MARKET_VOLATILITY_SIGMA = 2.0
# Hard timeout for yfinance calls so voice turns do not hang silently.
MARKET_DATA_TIMEOUT_SECONDS = 8.0
# Tighter timeout for single-stock voice quotes.
MARKET_QUOTE_TIMEOUT_SECONDS = 3.0

# ── Trading 212 Demo Broker Adapter ─────────────────────────────────────────
# Phase 1 is read-only and demo-only. Do not point this at live.trading212.com.
TRADING212_ENV = "demo"
TRADING212_BASE_URL = "https://demo.trading212.com/api/v0"
TRADING212_API_KEY = ""
TRADING212_API_SECRET = ""
TRADING212_TIMEOUT_SECONDS = 10.0
TRADING212_AUDIT_LOG_PATH = "data/broker/trading212_audit.jsonl"
TRADING212_TRAINING_LOG_PATH = "data/broker/trading212_account_state.jsonl"
