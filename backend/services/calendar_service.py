"""
services/calendar_service.py
Cal.com API integration for fetching availability and booking meetings.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List
import httpx

from backend.models.schemas import Slot, BookingResponse

logger = logging.getLogger(__name__)

CAL_API_KEY = os.getenv("CAL_API_KEY")
CAL_USERNAME = "mohd-kaif-ryjbbg"
CAL_BASE_URL = "https://api.cal.com/v2"
CAL_EVENT_TYPE_ID = os.getenv("CAL_EVENT_TYPE_ID", "")


def get_booking_headers() -> dict:
    """Headers for bookings endpoint."""
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-08-13"
    }


def get_slots_headers() -> dict:
    """Headers for slots endpoint — requires different api-version."""
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-09-04"
    }


def format_slot_display(start_iso: str) -> str:
    """
    Convert ISO 8601 to human-readable IST display string.
    Windows-compatible — no %-d or %-I.
    """
    try:
        dt_utc = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        dt_ist = dt_utc.astimezone(ist_offset)

        day = str(dt_ist.day)
        hour = dt_ist.strftime("%I").lstrip("0") or "12"
        minute = dt_ist.strftime("%M")
        ampm = dt_ist.strftime("%p")
        weekday = dt_ist.strftime("%A")
        month = dt_ist.strftime("%B")

        if minute == "00":
            return f"{weekday}, {month} {day} at {hour} {ampm} IST"
        return f"{weekday}, {month} {day} at {hour}:{minute} {ampm} IST"

    except Exception as e:
        logger.error(f"Date format error: {e}")
        return start_iso


async def get_available_slots(days_ahead: int = 7) -> List[Slot]:
    """
    Fetch available slots from Cal.com for the next N days.
    Cal.com v2 slots API (2024-09-04):
      - params: start, end (date strings YYYY-MM-DD), eventTypeId
      - returns: { "data": { "2026-06-10": [{"start": "..."}], ... } }
    """
    now = datetime.now(timezone.utc)
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    url = f"{CAL_BASE_URL}/slots"
    params = {
        "start": start_date,
        "end": end_date,
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
    }

    logger.info(f"Fetching slots: {start_date} → {end_date}, eventTypeId: {CAL_EVENT_TYPE_ID}")
    logger.info(f"Params: {params}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=get_slots_headers(), params=params)

        logger.info(f"Cal.com slots status: {resp.status_code}")
        logger.info(f"Raw response: {resp.text[:600]}")

        resp.raise_for_status()
        data = resp.json()

        # v2 with 2024-09-04 returns flat dict keyed by date
        # { "data": { "2026-06-10": [{"start": "2026-06-10T03:30:00.000Z"}], ... } }
        slots_by_date = data.get("data", {})

        # Handle both response shapes just in case
        if isinstance(slots_by_date, dict) and "slots" in slots_by_date:
            slots_by_date = slots_by_date["slots"]

        logger.info(f"Date keys returned: {list(slots_by_date.keys())[:5]}")

        slots: List[Slot] = []
        for date_key in sorted(slots_by_date.keys()):
            day_slots = slots_by_date[date_key]
            for slot in day_slots:
                # v2 uses "start" key (not "time")
                start = slot.get("start") or slot.get("time", "")
                if not start:
                    continue
                try:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_dt = start_dt + timedelta(minutes=30)
                    end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    end = start

                slots.append(Slot(
                    start=start,
                    end=end,
                    display=format_slot_display(start)
                ))

        logger.info(f"Parsed {len(slots)} slots total.")
        return slots[:10]

    except httpx.HTTPStatusError as e:
        logger.error(f"Cal.com slots error {e.response.status_code}: {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Slot fetch error: {e}", exc_info=True)
        return []


async def book_meeting(
    name: str,
    email: str,
    slot_start: str,
    slot_end: str,
    notes: str = ""
) -> BookingResponse:
    """
    Book a meeting on Cal.com v2.
    Uses 2024-08-13 api-version (bookings endpoint).
    Does NOT send 'end' or 'responses' — Cal.com v2 rejects those.
    """
    url = f"{CAL_BASE_URL}/bookings"

    payload = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID) if CAL_EVENT_TYPE_ID else None,
        "start": slot_start,
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": "Asia/Kolkata",
            "language": "en"
        },
        "metadata": {}
    }

    payload = {k: v for k, v in payload.items() if v is not None}
    logger.info(f"Booking payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=get_booking_headers(), json=payload)

        logger.info(f"Booking response {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()

        booking_data = data.get("data", {})
        booking_id = str(booking_data.get("id", ""))
        meeting_link = (
            booking_data.get("meetingUrl")
            or booking_data.get("videoCallUrl")
            or ""
        )

        display_time = format_slot_display(slot_start)
        logger.info(f"Booking confirmed: {booking_id} for {email}")

        return BookingResponse(
            success=True,
            booking_id=booking_id,
            confirmation_message=(
                f"Done! Your call with Kaif is confirmed for {display_time}. "
                f"A confirmation has been sent to {email}."
            ),
            meeting_link=meeting_link
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Booking error {e.response.status_code}: {e.response.text}")
        return BookingResponse(
            success=False,
            confirmation_message=(
                f"Booking failed. Please try directly at cal.com/{CAL_USERNAME}."
            )
        )
    except Exception as e:
        logger.error(f"Booking error: {e}", exc_info=True)
        return BookingResponse(
            success=False,
            confirmation_message=f"Something went wrong. Please book at cal.com/{CAL_USERNAME}."
        )