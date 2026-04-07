# config.example.py
# Copy this file to config.py and fill in your own values
# Never commit config.py to GitHub
#
# See config.py for the full list of settings. The constants below
# are the ones most likely to need editing on a new machine.

ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
PICOVOICE_API_KEY = "your-picovoice-api-key-here"
OPENWEATHER_API_KEY = ""  # Optional — currently using web scraping

# --- Conversation / Prompt Limits ---
MAX_CONVERSATION_TURNS = 10   # Recent exchanges sent as context to Claude
MAX_SPEAK_LENGTH = 500        # Max characters per TTS utterance
TRANSCRIPTION_TIMEOUT = 15    # Seconds before aborting a Whisper transcription
