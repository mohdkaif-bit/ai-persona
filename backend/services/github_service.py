"""
services/github_service.py
Voice agent helpers — Vapi payload parsing and conversation utilities.

Note: The main Vapi webhook handler lives in main.py at POST /voice.
This module contains shared utilities if needed for voice-specific logic.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_caller_intent(transcript: str) -> dict:
    """
    Lightweight intent classifier for voice transcripts.
    Returns a dict with detected intents for logging/evals.

    Used by the /voice webhook to log intent before processing.
    """
    transcript_lower = transcript.lower()

    intents = {
        "booking": any(kw in transcript_lower for kw in [
            "book", "schedule", "meeting", "call", "slot", "available", "calendar"
        ]),
        "background": any(kw in transcript_lower for kw in [
            "background", "experience", "internship", "work", "acro", "company"
        ]),
        "projects": any(kw in transcript_lower for kw in [
            "project", "built", "github", "tutoring", "blood pressure", "parking"
        ]),
        "skills": any(kw in transcript_lower for kw in [
            "skill", "tech", "stack", "language", "framework", "python", "langchain"
        ]),
        "fit": any(kw in transcript_lower for kw in [
            "why", "scaler", "fit", "right person", "hire", "role"
        ]),
        "adversarial": any(kw in transcript_lower for kw in [
            "ignore", "pretend", "forget", "jailbreak", "system prompt", "instruction"
        ])
    }

    detected = [k for k, v in intents.items() if v]
    logger.info(f"Detected intents: {detected or ['general']}")
    return intents


def sanitize_voice_response(text: str, max_sentences: int = 4) -> str:
    """
    Trim LLM response for voice output.
    Voice TTS sounds natural at 2-4 sentences max per turn.
    Cuts off at sentence boundary to avoid mid-sentence TTS truncation.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    trimmed = " ".join(sentences[:max_sentences])
    return trimmed