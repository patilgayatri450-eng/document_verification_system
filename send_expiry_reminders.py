"""
send_expiry_reminders.py
--------------------------
Manual/scheduled way to run the expiry-reminder check outside the app
(e.g. from Windows Task Scheduler or cron). NOTE: as of this version,
the running app ALSO does this automatically on its own (see
reminder_job.start_background_scheduler(), started in app.py) — so you
don't need this script or an external scheduler for reminders to work.
This is still useful for testing, forcing an immediate check, or
running reminders on a machine where the app itself isn't running
continuously.

Run manually:
    python send_expiry_reminders.py
    python send_expiry_reminders.py --days 14
    python send_expiry_reminders.py --dry-run
"""

import argparse
import os

os.environ["DISABLE_AUTO_SCHEDULER"] = "1"  # this script does its own one-off check below

from app import app
from reminder_job import run_reminder_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,
                         help="Remind for documents expiring within this many days (default 7)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would be sent without actually sending or marking as sent")
    args = parser.parse_args()

    result = run_reminder_check(app, days=args.days, dry_run=args.dry_run)

    if result["found"] == 0:
        print(f"No documents expiring within {args.days} days that need a reminder.")
    elif args.dry_run:
        print(f"Found {result['found']} document(s) expiring within {args.days} days (dry run — nothing sent).")
    else:
        print(f"Found {result['found']} document(s) expiring within {args.days} days — "
              f"{result['sent']} reminder(s) actually sent (the rest were skipped, usually missing "
              f"Twilio config or phone number — check the notification table or console output above).")


if __name__ == "__main__":
    main()
