"""
Aria Voice Recognition Trainer
===============================
Calibration and passive learning system for improving speech recognition.

- Calibration mode: 10 phrases read aloud, scored by Word Error Rate (WER)
- Passive learning: tracks corrections to build a personal vocabulary
- Voice profile: stores calibration results and learned corrections in
  data/voice_profile.json

All data stays local. No cloud storage.
"""

import json
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from config import DATA_DIR
from voice.listener import record_audio, calibrate_silence
from voice.transcriber import load_model, transcribe_audio

VOICE_PROFILE_PATH = os.path.join(DATA_DIR, "voice_profile.json")

# 10 calibration phrases — mix of conversational, technical, and Aria-specific
CALIBRATION_PHRASES = [
    "Hey Aria, what's the weather like today?",
    "Set a reminder for three o'clock tomorrow afternoon.",
    "Explain the difference between a list and a tuple in Python.",
    "Cancel my meeting with the data engineering team.",
    "What time is it in London right now?",
    "Aria, help me debug this SQL query please.",
    "Remind me in thirty minutes to check the oven.",
    "How do I write a for loop in JavaScript?",
    "Show me all my reminders for this week.",
    "Aria, tell me a fun fact about cybersecurity.",
]


def _load_profile() -> dict:
    """Load or create the voice profile.

    Returns:
        The voice profile dict.
    """
    if os.path.exists(VOICE_PROFILE_PATH):
        with open(VOICE_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return _default_profile()


def _save_profile(profile: dict) -> None:
    """Save the voice profile to disk.

    Args:
        profile: The voice profile dict to save.
    """
    os.makedirs(os.path.dirname(VOICE_PROFILE_PATH), exist_ok=True)
    with open(VOICE_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)


def _default_profile() -> dict:
    """Return the default voice profile template.

    Returns:
        A dict with empty calibration data and correction history.
    """
    return {
        "calibrated": False,
        "calibration_date": None,
        "overall_wer": None,
        "phrase_results": [],
        "corrections": {},
        "correction_count": 0,
        "whisper_hints": [],
    }


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis text.

    Uses a simple edit-distance-based approach at the word level.
    WER = (substitutions + insertions + deletions) / reference_length

    Args:
        reference: The expected (ground truth) text.
        hypothesis: The transcribed (predicted) text.

    Returns:
        WER as a float (0.0 = perfect, 1.0+ = very poor).
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming for word-level edit distance
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[n][m] / n


def run_calibration(device_index: int = None) -> dict:
    """Run the full 10-phrase calibration session.

    Prompts the user to read each phrase, transcribes it, and scores
    the result by WER. Saves results to the voice profile.

    Args:
        device_index: Audio input device index (None for default).

    Returns:
        The updated voice profile dict with calibration results.
    """
    print()
    print("=" * 55)
    print("  ARIA VOICE CALIBRATION")
    print("  Read each phrase clearly when prompted.")
    print("=" * 55)
    print()

    # Ensure Whisper model is loaded
    load_model()

    profile = _load_profile()
    results = []
    total_wer = 0.0

    for i, phrase in enumerate(CALIBRATION_PHRASES, 1):
        print(f"\n--- Phrase {i}/10 ---")
        print(f"  Read aloud: \"{phrase}\"")
        print()
        input("  Press Enter when ready, then speak...")

        audio_data = record_audio(device_index=device_index)

        if not audio_data:
            print("  [!] No audio captured. Skipping this phrase.")
            results.append({
                "phrase": phrase,
                "transcription": "",
                "wer": 1.0,
                "skipped": True,
            })
            total_wer += 1.0
            continue

        transcription = transcribe_audio(audio_data)
        wer = compute_wer(phrase, transcription)
        total_wer += wer

        # Quality label
        if wer == 0.0:
            quality = "PERFECT"
        elif wer < 0.15:
            quality = "GREAT"
        elif wer < 0.30:
            quality = "GOOD"
        elif wer < 0.50:
            quality = "FAIR"
        else:
            quality = "POOR"

        print(f"  Expected:     \"{phrase}\"")
        print(f"  Transcribed:  \"{transcription}\"")
        print(f"  WER: {wer:.1%} — {quality}")

        results.append({
            "phrase": phrase,
            "transcription": transcription,
            "wer": round(wer, 4),
            "skipped": False,
        })

    # Calculate overall WER
    overall_wer = total_wer / len(CALIBRATION_PHRASES)

    # Overall quality assessment
    print("\n" + "=" * 55)
    print(f"  CALIBRATION COMPLETE")
    print(f"  Overall WER: {overall_wer:.1%}")

    if overall_wer < 0.10:
        print("  Rating: EXCELLENT — Whisper hears you clearly!")
    elif overall_wer < 0.20:
        print("  Rating: GOOD — Most commands will be understood.")
    elif overall_wer < 0.35:
        print("  Rating: FAIR — Some commands may need repeating.")
    else:
        print("  Rating: NEEDS IMPROVEMENT — Try speaking more clearly,")
        print("  or move to a quieter environment.")
    print("=" * 55)
    print()

    # Save to profile
    profile["calibrated"] = True
    profile["calibration_date"] = datetime.now().isoformat()
    profile["overall_wer"] = round(overall_wer, 4)
    profile["phrase_results"] = results

    _save_profile(profile)
    print("[Aria] Voice profile saved.")

    return profile


def record_correction(misheard: str, correct: str) -> None:
    """Record a voice recognition correction for passive learning.

    Tracks what Whisper got wrong and what it should have been,
    building a personal vocabulary of common misrecognitions.

    Args:
        misheard: What Whisper transcribed (incorrect).
        correct: What the user actually said (correct).
    """
    profile = _load_profile()

    misheard_lower = misheard.lower().strip()
    correct_lower = correct.lower().strip()

    if misheard_lower == correct_lower:
        return

    corrections = profile.get("corrections", {})

    if misheard_lower in corrections:
        entry = corrections[misheard_lower]
        entry["correct"] = correct_lower
        entry["count"] = entry.get("count", 0) + 1
        entry["last_seen"] = datetime.now().isoformat()
    else:
        corrections[misheard_lower] = {
            "correct": correct_lower,
            "count": 1,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        }

    profile["corrections"] = corrections
    profile["correction_count"] = sum(e["count"] for e in corrections.values())

    # Update whisper_hints — top corrections that appear 3+ times
    frequent = [
        entry["correct"]
        for entry in corrections.values()
        if entry["count"] >= 3
    ]
    profile["whisper_hints"] = list(set(frequent))

    _save_profile(profile)
    print(f"[Aria] Correction recorded: \"{misheard}\" → \"{correct}\"")


def apply_corrections(text: str) -> str:
    """Apply known corrections to transcribed text.

    Replaces frequently misheard words/phrases with the correct version
    based on the accumulated correction history.

    Args:
        text: The raw transcribed text from Whisper.

    Returns:
        The corrected text.
    """
    profile = _load_profile()
    corrections = profile.get("corrections", {})

    if not corrections:
        return text

    corrected = text.lower().strip()
    for misheard, entry in corrections.items():
        if entry.get("count", 0) >= 2 and misheard in corrected:
            corrected = corrected.replace(misheard, entry["correct"])

    # Preserve original casing for the first character
    if text and corrected:
        corrected = text[0] + corrected[1:] if len(corrected) > 1 else text[0]

    return corrected


def get_profile_summary() -> str:
    """Get a human-readable summary of the voice profile.

    Returns:
        A formatted string summarising calibration status and corrections.
    """
    profile = _load_profile()

    if not profile.get("calibrated"):
        return "Voice profile: Not calibrated yet. Run calibration to improve recognition."

    wer = profile.get("overall_wer", 0)
    date = profile.get("calibration_date", "unknown")
    correction_count = profile.get("correction_count", 0)
    hint_count = len(profile.get("whisper_hints", []))

    return (
        f"Voice profile: Calibrated on {date[:10]}, "
        f"WER {wer:.1%}, "
        f"{correction_count} corrections recorded, "
        f"{hint_count} learned hints."
    )


if __name__ == "__main__":
    print("=== Aria Voice Trainer ===")
    print()
    print("Options:")
    print("  [1] Run full calibration (10 phrases)")
    print("  [2] View profile summary")
    print()

    choice = input("Select: ").strip()

    if choice == "1":
        calibrate_silence()
        run_calibration()
    elif choice == "2":
        print(get_profile_summary())
    else:
        print("Invalid choice.")
