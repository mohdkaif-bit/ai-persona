"""
models/schemas.py
Pydantic schemas for all API request/response types.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_history: Optional[List[Message]] = Field(default_factory=list)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    retrieved_sections: Optional[List[str]] = None  # For debugging/evals


# ─────────────────────────────────────────
# VOICE (Vapi webhook)
# ─────────────────────────────────────────

class VapiMessage(BaseModel):
    role: str
    content: str


class VapiCall(BaseModel):
    id: str
    status: Optional[str] = None


class VapiRequest(BaseModel):
    """
    Vapi sends a webhook payload with the conversation so far.
    We respond with the assistant's next message.
    """
    message: dict  # Full Vapi message object
    call: Optional[VapiCall] = None


class VapiResponse(BaseModel):
    """
    Vapi expects: { "response": { "content": "..." } }
    """
    content: str


# ─────────────────────────────────────────
# CALENDAR
# ─────────────────────────────────────────

class SlotRequest(BaseModel):
    days_ahead: Optional[int] = 7  # Look ahead N days


class Slot(BaseModel):
    start: str      # ISO 8601
    end: str        # ISO 8601
    display: str    # Human-readable: "Monday, June 9 at 3:00 PM IST"


class SlotsResponse(BaseModel):
    slots: List[Slot]
    booking_url: str


class BookingRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    slot_start: str   # ISO 8601 datetime string
    slot_end: str     # ISO 8601 datetime string
    notes: Optional[str] = ""


class BookingResponse(BaseModel):
    success: bool
    booking_id: Optional[str] = None
    confirmation_message: str
    meeting_link: Optional[str] = None