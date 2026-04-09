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

# --- TTS Voice Model ---
# Download from HuggingFace: rhasspy/piper-voices → en/en_US/hfc_female/medium/
# Save to: assets/voices/en_US-hfc_female-medium.onnx (and .onnx.json)
PIPER_MODEL_PATH = "assets/voices/en_US-hfc_female-medium.onnx"
