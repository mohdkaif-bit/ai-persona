"""
main.py
FastAPI entry point for Mohd Kaif AI Persona backend.

Routes:
    POST /chat           — Chat with the persona (frontend)
    POST /chat/book      — Book a meeting from chat
    POST /voice          — Vapi webhook (voice agent)
    GET  /calendar/slots — Get available slots
    POST /calendar/book  — Book a confirmed meeting
    GET  /health         — Health check
"""

import logging
import re
from fastapi.responses import StreamingResponse
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.models.schemas import (
    ChatRequest, ChatResponse,
    BookingRequest, BookingResponse,
    SlotsResponse
)

load_dotenv()

from backend.models.schemas import (
    ChatRequest, ChatResponse,
    BookingRequest, BookingResponse,
    SlotsResponse
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models on startup so first request isn't slow."""
    logger.info("Warming up embedding model and reranker...")
    from backend.rag.embeddings import get_embedding_model
    from backend.rag.retriever import get_reranker, build_bm25_index
    get_embedding_model()
    get_reranker()
    build_bm25_index()
    logger.info("Warmup complete. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Mohd Kaif AI Persona",
    description="AI persona for Scaler screening — RAG-grounded, voice + chat + calendar booking.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# BOOKING INTENT DETECTION
# ─────────────────────────────────────────

BOOKING_KEYWORDS = [
    "book", "schedule", "meeting", "call", "slot", "available",
    "availability", "calendar", "interview", "talk", "connect", "when can"
]


def is_booking_intent(message: str) -> bool:
    return any(kw in message.lower() for kw in BOOKING_KEYWORDS)


# ─────────────────────────────────────────
# CHAT ROUTES
# ─────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint used by frontend.
    Handles RAG retrieval, booking intent detection, and Groq response.
    """
    from backend.services.llm_service import get_persona_response
    from backend.services.calendar_service import get_available_slots

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Booking intent — inject real slot data so LLM can propose times
    augmented_message = user_message
    if is_booking_intent(user_message):
        logger.info("Booking intent detected — fetching slots.")
        slots = await get_available_slots(days_ahead=7)
        if slots:
            slot_lines = "\n".join(
                f"- {s.display} (start: {s.start}, end: {s.end})"
                for s in slots[:5]
            )
            augmented_message = (
                f"{user_message}\n\n"
                f"[SYSTEM: Real available slots from Kaif's calendar:]\n"
                f"{slot_lines}\n"
                f"[Propose 2-3 of these. Once user confirms, ask for name and email to book.]"
            )
        else:
            augmented_message = (
                f"{user_message}\n\n"
                f"[SYSTEM: No slots available in next 7 days. "
                f"Direct user to app.cal.com/mohd-kaif-ryjbbg]"
            )

    reply, retrieved_sections = get_persona_response(
        user_message=augmented_message,
        history=req.conversation_history,
        top_k=5
    )

    return ChatResponse(
        reply=reply,
        session_id=req.session_id,
        retrieved_sections=retrieved_sections
    )


@app.post("/chat/book", response_model=BookingResponse)
async def chat_book(req: BookingRequest):
    """
    Called from frontend when user confirms a slot.
    Body: { name, email, slot_start, slot_end, notes }
    """
    from backend.services.calendar_service import book_meeting

    result = await book_meeting(
        name=req.name,
        email=req.email,
        slot_start=req.slot_start,
        slot_end=req.slot_end,
        notes=req.notes or ""
    )
    return result


# ─────────────────────────────────────────
# VOICE ROUTE (Vapi webhook)
# ─────────────────────────────────────────

@app.post("/voice")
async def vapi_webhook(payload: dict):
    """
    Vapi webhook — responds to assistant-request events.
    Returns: { "response": { "content": "..." } }
    """
    from backend.models.schemas import Message
    from backend.services.llm_service import get_persona_response
    from backend.services.calendar_service import get_available_slots

    try:
        message_type = payload.get("message", {}).get("type", "")
        logger.info(f"Vapi event: {message_type}")

        # Non-conversational events — just ack
        if message_type in ("end-of-call-report", "status-update", "hang"):
            return {"status": "ok"}

        # Extract latest user message + history
        messages = payload.get("message", {}).get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]

        if not user_messages:
            return {"response": {"content": (
                "Hi! I'm Kaif's AI representative. I can tell you about his background, "
                "projects, and fit for the AI Engineer role at Scaler — and I can check "
                "his calendar and book a call. What would you like to know?"
            )}}

        user_message = user_messages[-1].get("content", "")

        if not user_message.strip():
            return {"status": "ok"}

        # Build history

        # Build history
        history = [
            Message(role=m["role"], content=m["content"])
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        logger.info(f"Voice query: '{user_message[:80]}'")

        # Booking intent
        augmented_message = user_message
        if is_booking_intent(user_message):
            slots = await get_available_slots(days_ahead=7)
            if slots:
                slot_lines = ", ".join(s.display for s in slots[:3])
                augmented_message = (
                    f"{user_message}\n\n"
                    f"[SYSTEM: Available slots: {slot_lines}. "
                    f"Propose these naturally. Keep it brief — this is a voice call.]"
                )

        reply, _ = get_persona_response(
            user_message=augmented_message,
            history=history,
            top_k=4,
            use_reranker=False
        )

        return {"response": {"content": reply}}

    except Exception as e:
        logger.error(f"Vapi webhook error: {e}", exc_info=True)
        return {
            "response": {
                "content": "I ran into a technical issue. Please try again or reach Kaif at m.kaif.sondat@gmail.com."
            }
        }

from fastapi.responses import StreamingResponse
import json

@app.post("/voice/chat/completions")
async def vapi_custom_llm(payload: dict):
    from backend.services.llm_service import get_persona_response
    from backend.services.calendar_service import get_available_slots, book_meeting
    from backend.models.schemas import Message

    try:
        messages = payload.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]

        if not user_messages:
            reply = "Hi! I'm Kaif's AI representative. What would you like to know?"
        else:
            user_message = user_messages[-1].get("content", "")

            if not user_message.strip():
                reply = "I didn't catch that. Could you repeat?"
            else:
                history = [
                    Message(role=m["role"], content=m["content"])
                    for m in messages[:-1]
                    if m.get("role") in ("user", "assistant") and m.get("content")
                ]

                # Extract full conversation text for context
                full_history_text = " ".join(
                    m["content"] for m in messages
                    if m.get("content")
                )

                augmented_message = user_message

                # Check if we already have slot + name + email in conversation
                # and user is confirming booking
                has_slot = any(
                    kw in full_history_text.lower()
                    for kw in ["9 am", "10 am", "11 am", "monday", "tuesday",
                               "wednesday", "thursday", "friday", "june", "july"]
                )
                has_name = any(
                    m.get("role") == "assistant" and "name" in m.get("content", "").lower()
                    for m in messages
                )
                has_email = any(
                    "@" in m.get("content", "")
                    for m in messages
                    if m.get("role") == "user"
                )

                # Extract email from conversation
                user_email = None
                user_name = None
                for m in messages:
                    if m.get("role") == "user":
                        match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', m.get("content", ""))
                        if match:
                            user_email = match.group(0)
                    # Name is usually the message after assistant asks for name
                for i, m in enumerate(messages):
                    if (m.get("role") == "assistant"
                            and "name" in m.get("content", "").lower()
                            and i + 1 < len(messages)
                            and messages[i+1].get("role") == "user"
                            and "@" not in messages[i+1].get("content", "")):
                        user_name = messages[i+1]["content"].strip()

                # If we have all 3 — trigger actual booking
                booking_already_confirmed = any(
                    "confirmed your booking" in m.get("content", "").lower()
                    for m in messages
                    if m.get("role") == "assistant"
                )
                user_confirming = any(
                    kw in user_message.lower()
                    for kw in ["yes", "confirm", "book it", "go ahead", "that works", "sounds good"]
                )
                if has_slot and user_email and user_name and not booking_already_confirmed and user_confirming:
                    # Find the confirmed slot from history
                    slots = await get_available_slots(days_ahead=7)
                    confirmed_slot = slots[0] if slots else None

                    # Try to match slot from conversation
                    for m in messages:
                        content = m.get("content", "").lower()
                        for slot in slots:
                            display_lower = slot.display.lower()
                            words = display_lower.split()
                            if any(w in content for w in words if len(w) > 3):
                                confirmed_slot = slot
                                break

                    if confirmed_slot:
                        try:
                            result = await book_meeting(
                                name=user_name,
                                email=user_email,
                                slot_start=confirmed_slot.start,
                                slot_end=confirmed_slot.end,
                                notes="Booked via voice agent"
                            )
                            reply = (
                                f"Perfect, I've confirmed your booking for "
                                f"{confirmed_slot.display}. "
                                f"A confirmation has been sent to {user_email}. "
                                f"Looking forward to speaking with you!"
                            )
                        except Exception as e:
                            logger.error(f"Booking failed: {e}")
                            reply = (
                                f"I had trouble confirming the booking. "
                                f"Please book directly at app.cal.com/mohd-kaif-ryjbbg"
                            )
                    else:
                        reply = "I couldn't match a slot. Please book directly at app.cal.com/mohd-kaif-ryjbbg"

                elif is_booking_intent(user_message):
                    # Fetch and propose slots
                    slots = await get_available_slots(days_ahead=7)
                    if slots:
                        slot_lines = ", ".join(s.display for s in slots[:3])
                        augmented_message = (
                            f"{user_message}\n\n"
                            f"[SYSTEM: Available slots: {slot_lines}. "
                            f"Propose these naturally. Keep it brief — voice call.]"
                        )
                    reply, _ = get_persona_response(
                        user_message=augmented_message,
                        history=history,
                        top_k=3,
                        use_reranker=False
                    )
                else:
                    reply, _ = get_persona_response(
                        user_message=augmented_message,
                        history=history,
                        top_k=3,
                        use_reranker=False
                    )

    except Exception as e:
        logger.error(f"Custom LLM error: {e}", exc_info=True)
        reply = "I ran into a technical issue. Please try again."

    async def stream():
        chunk = {
            "id": "chatcmpl-kaif",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": reply},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": "chatcmpl-kaif",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
# ─────────────────────────────────────────
# CALENDAR ROUTES
# ─────────────────────────────────────────

@app.get("/calendar/slots", response_model=SlotsResponse)
async def get_slots(days_ahead: int = Query(default=7, ge=1, le=30)):
    """Returns available slots for the next N days."""
    from backend.services.calendar_service import get_available_slots

    slots = await get_available_slots(days_ahead=days_ahead)
    return SlotsResponse(
        slots=slots,
        booking_url="https://app.cal.com/mohd-kaif-ryjbbg"
    )


@app.post("/calendar/book", response_model=BookingResponse)
async def book_meeting_route(req: BookingRequest):
    """Books a confirmed meeting slot."""
    from backend.services.calendar_service import book_meeting

    result = await book_meeting(
        name=req.name,
        email=req.email,
        slot_start=req.slot_start,
        slot_end=req.slot_end,
        notes=req.notes or ""
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.confirmation_message)

    return result


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    from backend.rag.vector_store import get_collection_stats
    stats = get_collection_stats()
    return {"status": "ok", "collection": stats}