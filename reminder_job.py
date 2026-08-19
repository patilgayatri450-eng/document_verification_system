"""
reminder_job.py
-----------------
The actual "find soon-expiring documents and text students" logic,
factored out so it can run two ways:
  1. Manually / on a schedule outside the app: send_expiry_reminders.py
  2. Fully automatically inside the running app: app.py starts a
     background scheduler (APScheduler) that calls run_reminder_check()
     on its own, on an interval — no manual script execution needed.
"""

from datetime import date, timedelta


def run_reminder_check(app, days: int = 7, dry_run: bool = False) -> dict:
    """
    Finds documents expiring within `days` days that haven't already
    had a reminder sent, and texts each student. Must be called with
    an active Flask app (for the app context / database access).

    Returns a summary dict: {"found": N, "sent": N} — useful for
    logging from whichever caller invoked it.
    """
    # Imported inside the function to avoid a circular import with
    # app.py, which imports run_scheduler_startup from this module.
    from models_db import db, Document, Notification
    from utils.sms_notifier import send_sms

    cutoff = date.today() + timedelta(days=days)

    with app.app_context():
        expiring_docs = Document.query.filter(
            Document.expiry_date != None,
            Document.expiry_date <= cutoff,
            Document.expiry_date >= date.today(),
            Document.expiry_reminder_sent == False,
        ).all()

        if not expiring_docs:
            return {"found": 0, "sent": 0}

        sent_count = 0
        for doc in expiring_docs:
            student = doc.student
            days_left = (doc.expiry_date - date.today()).days
            message = (
                f"Hi {student.full_name}, your document '{doc.original_filename}' "
                f"({doc.doc_type or 'document'}) expires on {doc.expiry_date.strftime('%d %b %Y')} "
                f"({days_left} day{'s' if days_left != 1 else ''} left). Please resubmit an updated "
                f"copy through the Legal Document Analysis System."
            )

            print(f"[reminder_job] {student.full_name} ({student.phone_number or 'no phone on file'}): "
                  f"'{doc.original_filename}' expires {doc.expiry_date}")

            if dry_run:
                continue

            # Each document is its own try/commit: one bad record (a
            # DB error, a Twilio hiccup, whatever) rolls back and gets
            # skipped without losing the results already saved for
            # every other document in this batch.
            try:
                sent, reason = send_sms(student.phone_number, message)

                db.session.add(Notification(
                    student_id=student.id,
                    document_id=doc.id,
                    message=message,
                    status="sent" if sent else "failed",
                    failure_reason=None if sent else reason,
                ))

                if sent:
                    doc.expiry_reminder_sent = True
                    sent_count += 1

                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"[reminder_job] Skipped '{doc.original_filename}' after an unexpected error: {e}")

        return {"found": len(expiring_docs), "sent": sent_count}


def start_background_scheduler(app, days: int = 7, interval_hours: int = 24):
    """
    Starts an APScheduler background job that calls run_reminder_check()
    automatically on a fixed interval for as long as the app process is
    running — this is what makes reminders "fully automatic" with no
    manual script execution or OS-level task scheduler required.

    Runs once immediately on startup, then every `interval_hours` after
    that (default: once a day).
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(daemon=True)

    def job():
        # This must never raise — it runs both at startup (synchronously,
        # in the main process) and on every scheduled interval. A crash
        # here previously took down the entire app at launch (see the
        # MySQL "data too long" incident) — reminders are a secondary
        # feature and must never be able to prevent the app from serving
        # requests.
        try:
            result = run_reminder_check(app, days=days)
            if result["found"]:
                print(f"[reminder_job] Checked expiring documents: found {result['found']}, "
                      f"sent {result['sent']}.")
        except Exception as e:
            print(f"[reminder_job] Reminder check failed, will retry on the next scheduled run: {e}")

    scheduler.add_job(job, "interval", hours=interval_hours)
    scheduler.start()

    # Run once immediately too, rather than waiting a full interval for
    # the first check.
    job()

    return scheduler
