"""
emergency_call.py — Twilio-powered emergency calling for MediBot.

Features:
  - Outbound voice call to a pre-configured emergency contact
  - Reads an AI-generated emergency summary aloud via TwiML <Say>
  - Simultaneous SMS alert with the same summary
  - /api/emergency/twiml  — serves TwiML when Twilio fetches it (webhook mode)
  - /api/emergency/status — receives call status callbacks from Twilio
  - /api/emergency/call-status/<sid> — poll live call status from frontend

Required env vars:
  TWILIO_ACCOUNT_SID    from console.twilio.com
  TWILIO_AUTH_TOKEN     from console.twilio.com
  TWILIO_FROM_NUMBER    your Twilio phone number  e.g. +14155551234
  EMERGENCY_TO_NUMBER   recipient number          e.g. +919876543210

Optional:
  PUBLIC_BASE_URL       public HTTPS URL of your server, e.g. https://app.onrender.com
                        Required for status callbacks. If absent, inline TwiML is used.
"""

from __future__ import annotations
import os
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ── env ───────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER  = os.environ.get("TWILIO_FROM_NUMBER", "")
EMERGENCY_TO_NUMBER = os.environ.get("EMERGENCY_TO_NUMBER", "")
PUBLIC_BASE_URL     = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# ── lazy Twilio client ─────────────────────────────────────────────────────────
_twilio_client = None

def _get_client():
    global _twilio_client
    if _twilio_client:
        return _twilio_client
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError(
            "Twilio credentials missing.\n"
            "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file."
        )
    try:
        from twilio.rest import Client
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialised.")
        return _twilio_client
    except ImportError:
        raise RuntimeError("twilio package not installed. Run: pip install twilio")


def _missing_vars() -> list[str]:
    missing = []
    if not TWILIO_ACCOUNT_SID:  missing.append("TWILIO_ACCOUNT_SID")
    if not TWILIO_AUTH_TOKEN:   missing.append("TWILIO_AUTH_TOKEN")
    if not TWILIO_FROM_NUMBER:  missing.append("TWILIO_FROM_NUMBER")
    if not EMERGENCY_TO_NUMBER: missing.append("EMERGENCY_TO_NUMBER")
    return missing


# ── TwiML builder ─────────────────────────────────────────────────────────────

def build_twiml(summary: str, patient_name: str = "the patient") -> str:
    """Return a TwiML XML string Twilio reads aloud when the recipient answers."""
    safe = (
        summary
        .replace("&", "and")
        .replace("<", "")
        .replace(">", "")
        .replace('"', "'")
    )[:450]
    name = patient_name.replace("&", "and").replace("<", "").replace(">", "")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Amy" language="en-GB">
    This is an automated emergency alert from MediBot Medical Assistant.
    {name} may require immediate medical attention.
    {safe}
    Please call back or contact emergency services immediately.
  </Say>
  <Pause length="1"/>
  <Say voice="Polly.Amy" language="en-GB">
    Repeating: {name} needs immediate help. {safe} Please respond now.
  </Say>
</Response>"""


def build_sms(summary: str, patient_name: str = "the patient") -> str:
    return (
        f"\U0001f6a8 MEDIBOT EMERGENCY ALERT\n"
        f"Patient: {patient_name}\n"
        f"{summary[:300]}\n"
        f"Please respond immediately or call emergency services."
    )


# ── public API ────────────────────────────────────────────────────────────────

def trigger_emergency(
    summary: str,
    patient_name: str = "the patient",
    to_number: str | None = None,
) -> dict:
    """
    Place an emergency voice call + send SMS via Twilio.

    Returns a result dict:
      success        bool
      call_sid       Twilio call SID (if call placed)
      call_status    initial status string
      sms_sid        Twilio message SID (if SMS sent)
      sms_status     initial SMS status
      to             recipient number
      errors         list of error strings
    """
    missing = _missing_vars()
    if missing:
        return {"success": False, "errors": [f"Missing env vars: {', '.join(missing)}"]}

    recipient = to_number or EMERGENCY_TO_NUMBER
    if not recipient:
        return {"success": False, "errors": ["No recipient phone number configured."]}

    client  = _get_client()
    result  = {"success": False, "errors": [], "to": recipient}

    # ── voice call ────────────────────────────────────────────────────────────
    try:
        twiml_xml = build_twiml(summary, patient_name)

        call_kwargs: dict = {
            "to":      recipient,
            "from_":   TWILIO_FROM_NUMBER,
            "timeout": 30,
        }

        if PUBLIC_BASE_URL:
            enc_summary = quote(summary[:300], safe="")
            enc_name    = quote(patient_name, safe="")
            call_kwargs["url"] = (
                f"{PUBLIC_BASE_URL}/api/emergency/twiml"
                f"?summary={enc_summary}&name={enc_name}"
            )
            call_kwargs["status_callback"]        = f"{PUBLIC_BASE_URL}/api/emergency/status"
            call_kwargs["status_callback_method"] = "POST"
            call_kwargs["status_callback_event"]  = ["initiated", "ringing", "answered", "completed"]
        else:
            # inline TwiML — works without a public URL (local dev)
            call_kwargs["twiml"] = twiml_xml

        call = client.calls.create(**call_kwargs)
        result["call_sid"]    = call.sid
        result["call_status"] = call.status
        logger.info(f"Emergency call initiated: {call.sid} → {recipient} [{call.status}]")

    except Exception as e:
        logger.error(f"Emergency call failed: {e}")
        result["errors"].append(f"Call error: {e}")

    # ── SMS ───────────────────────────────────────────────────────────────────
    try:
        msg = client.messages.create(
            to=recipient,
            from_=TWILIO_FROM_NUMBER,
            body=build_sms(summary, patient_name),
        )
        result["sms_sid"]    = msg.sid
        result["sms_status"] = msg.status
        logger.info(f"Emergency SMS sent: {msg.sid} [{msg.status}]")

    except Exception as e:
        logger.error(f"Emergency SMS failed: {e}")
        result["errors"].append(f"SMS error: {e}")

    result["success"] = "call_sid" in result or "sms_sid" in result
    return result


def get_call_status(call_sid: str) -> dict:
    """Poll the live status of a call by SID."""
    try:
        call = _get_client().calls(call_sid).fetch()
        return {
            "sid":      call.sid,
            "status":   call.status,   # queued|ringing|in-progress|completed|busy|failed|no-answer
            "duration": call.duration,
            "to":       call.to,
            "from_":    call.from_,
            "direction": call.direction,
        }
    except Exception as e:
        return {"error": str(e)}


def config_status() -> dict:
    """Return which Twilio env vars are configured (values masked)."""
    return {
        "TWILIO_ACCOUNT_SID":  "set" if TWILIO_ACCOUNT_SID else "MISSING",
        "TWILIO_AUTH_TOKEN":   "set" if TWILIO_AUTH_TOKEN  else "MISSING",
        "TWILIO_FROM_NUMBER":  TWILIO_FROM_NUMBER  or "MISSING",
        "EMERGENCY_TO_NUMBER": EMERGENCY_TO_NUMBER or "MISSING",
        "PUBLIC_BASE_URL":     PUBLIC_BASE_URL     or "not set (inline TwiML mode)",
        "ready":               not bool(_missing_vars()),
    }
