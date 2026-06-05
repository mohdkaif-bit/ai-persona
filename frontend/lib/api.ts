const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  session_id: string;
  retrieved_sections: string[];
}

export interface Slot {
  start: string;
  end: string;
  display: string;
}

export interface SlotsResponse {
  slots: Slot[];
  booking_url: string;
}

export interface BookingRequest {
  name: string;
  email: string;
  slot_start: string;
  slot_end: string;
  notes: string;
}

export interface BookingResponse {
  success: boolean;
  booking_id: string;
  confirmation_message: string;
  meeting_link: string;
}

// ── API Functions ──────────────────────────────────────────────────────────

export async function sendMessage(
  message: string,
  history: Message[],
  sessionId: string
): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_history: history,
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Chat API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<ChatResponse>;
}

export async function getSlots(daysAhead = 7): Promise<SlotsResponse> {
  const res = await fetch(`${BASE_URL}/calendar/slots?days_ahead=${daysAhead}`);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Slots API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<SlotsResponse>;
}

export async function bookSlot(
  name: string,
  email: string,
  slotStart: string,
  slotEnd: string,
  notes: string
): Promise<BookingResponse> {
  const body: BookingRequest = {
    name,
    email,
    slot_start: slotStart,
    slot_end: slotEnd,
    notes,
  };

  const res = await fetch(`${BASE_URL}/calendar/book`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Booking API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<BookingResponse>;
}