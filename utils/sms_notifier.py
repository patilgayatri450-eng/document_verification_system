"""
sms_notifier.py
----------------
Sends SMS reminders via Twilio. Requires a Twilio account (free trial
works for testing, but trial accounts can only text verified numbers —
see README for setup instructions and paid-account notes for India).

Reads credentials from environment variables so they're never hardcoded:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER   (the Twilio number you're sending from, E.164 format)

If these aren't set, send_sms() logs a warning and returns False instead
of crashing — so the reminder script degrades gracefully rather than
failing outright if SMS isn't configured yet.
"""

import os
import re

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")


def send_sms(to_number: str, message: str) -> tuple[bool, str]:
    """
    Sends an SMS via Twilio. Returns (success, reason) — reason is a
    short human-readable string either way (e.g. "sent", "Twilio
    credentials not configured", "Twilio API error: ..."), so callers
    can log exactly why a send succeeded or failed without needing to
    inspect exceptions themselves.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        reason = "Twilio credentials not configured (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER)"
        print(f"[sms_notifier] {reason} — skipping SMS. See README.")
        return False, reason

    if not to_number:
        reason = "No phone number on file for this student"
        print(f"[sms_notifier] {reason} — skipping SMS.")
        return False, reason

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=to_number)
        print(f"[sms_notifier] Sent SMS to {to_number}")
        return True, "sent"
    except Exception as e:
        # Twilio's exception messages can include ANSI color codes and
        # run long (multi-paragraph HTTP error explanations) — strip
        # escape codes and cap the length so this always fits safely
        # in a database column, regardless of what Twilio sends back.
        raw = str(e)
        clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        reason = f"Twilio API error: {clean[:400]}"
        print(f"[sms_notifier] Failed to send SMS to {to_number}: {raw}")
        return False, reason


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sms_notifier.py <phone_number> <message>")
    else:
        success, reason = send_sms(sys.argv[1], sys.argv[2])
        print(f"success={success} reason={reason}")
