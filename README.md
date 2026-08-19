# Legal Document Analysis System

Pipeline: **barcode/QR extraction → OCR fallback → institutional dataset lookup → document-type recognition → CNN / MobileNetV2 visual authenticity check → fused verdict → manual clerk approval**

An MCA-level project: two role-based dashboards (student/clerk), a
5-table relational schema (users, document types, documents, an
append-only audit log, and SMS notification history), automatic
document-type recognition, SMS expiry reminders, and both SQLite
(zero-setup) and MySQL (Workbench-manageable) backends.


## Folder structure

```
doc_verification_system/
├── app.py                     # Flask web app (upload -> result)
├── requirements.txt
├── data/
│   └── valid_barcodes.csv     # sample dataset — REPLACE with your real records
├── models/
│   ├── train_cnn.py           # trains a from-scratch CNN
│   ├── train_mobilenetv2.py   # transfer learning with MobileNetV2
│   ├── model_utils.py         # loads a trained model + runs inference
│   ├── cnn_model.h5           # created after you run train_cnn.py
│   └── mobilenetv2_model.h5   # created after you run train_mobilenetv2.py
├── utils/
│   ├── barcode_extractor.py   # OpenCV + pyzbar decoding
│   ├── dataset_lookup.py      # checks decoded barcode against CSV
│   └── fusion.py              # combines both checks into final verdict
├── templates/                 # index.html, result.html
├── static/style.css
└── uploads/                   # uploaded files land here
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **pyzbar note:** on Linux you may need the system zbar library first:
> `sudo apt-get install libzbar0` (macOS: `brew install zbar`). If pyzbar
> still fails to import, `barcode_extractor.py` automatically falls back
> to OpenCV's built-in QR-only decoder, so the app won't crash — you'll
> just lose non-QR barcode support until zbar is installed.

## 2. Dataset — no real college data needed

You don't have to collect anything from your college to get this
working. The project ships with `data/generate_synthetic_dataset.py`,
which builds a **complete synthetic dataset from scratch**:

```bash
python data/generate_synthetic_dataset.py --count 300
```

This single command:
- Generates fake "Greenfield College" ID cards / marksheets /
  certificates / govt IDs with random names and real, scannable
  Code128 barcodes printed on them
- Writes the matching records into `data/valid_barcodes.csv` (so those
  barcodes actually pass the dataset lookup step)
- Creates a tampered "forged" counterpart of every genuine image
  (edited ID number, blurred barcode, swapped name, patch overlay —
  a mix of tamper types)
- Splits everything 80/20 into `data/train/{genuine,forged}` and
  `data/val/{genuine,forged}`, ready for the training scripts below

A 300-image dataset (600 images total across both classes) is already
generated and included in this zip, so you can skip straight to step 4
(training) if you just want to see it work. Run the generator again
with a different `--count` any time you want a bigger/fresh batch —
it overwrites the previous one.

**When you're ready for real-world accuracy**, swap in actual document
scans later: replace images in `data/train/genuine` and
`data/val/genuine` with real ones, create matching forged edits, and
update `data/valid_barcodes.csv` with your institution's real
barcode-to-record mapping (same column format the generator uses:
`barcode_value,document_type,record_id,holder_name,issue_year,status`).
Public datasets like MIDV-500/MIDV-2019/FMIDV are also usable for
further pretraining if you want more variety later.

## 3. Image data for the CNN / MobileNetV2 models

These two models judge whether the document **image itself** looks
tampered — separate from the barcode lookup. After step 2 you'll
already have:

```
data/train/genuine/*.jpg
data/train/forged/*.jpg
data/val/genuine/*.jpg
data/val/forged/*.jpg
```

populated and ready — no extra work needed here unless you want to add
your own real samples on top.

## 4. Train the models

```bash
python models/train_cnn.py
python models/train_mobilenetv2.py
```

Each saves a `.h5` file into `models/`. You can train just one if you
only want to compare a single model first — `app.py` defaults to
MobileNetV2 (set `VISION_MODEL=cnn_model.h5` env var to switch).

## 5. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000`, upload a document image, and it will:
1. Decode the barcode/QR
2. Look it up in `data/valid_barcodes.csv`
3. Run the vision model for a tamper check (only if the barcode matched)
4. Show VALID / NOT VALID with the reasons

## Automatic document-type recognition

Every uploaded document now gets its type recognized independently of
whether its barcode matched anything — `utils/doc_type_classifier.py`
checks the filename and OCR'd page text against keyword rules
(marksheet, HSC/SSC certificate, income certificate, domicile/nationality
certificate, caste certificate, student ID, government ID, degree
certificate). This is why a real document's type now shows correctly
in the clerk's per-student list even when its barcode isn't in your
sample dataset — type recognition and barcode validity are separate
checks. Recognized types are backed by a `document_type` lookup table
(seeded automatically on first run) with a default validity period per
type.

## Barcode auto-fill fix

Previously, when a barcode couldn't be decoded and OCR found a
possible number, you had to manually re-type it from the result
message into the manual-barcode field. Now that OCR suggestion is
passed straight back and pre-fills the barcode field for you — just
double-check it against the document and resubmit.

## Five-table relational schema

Beyond `user` and `document`, the schema now includes:
- **`document_type`** — lookup table of recognized document types
- **`audit_log`** — append-only trail of every significant action
  (login, register, upload, approve, reject, delete, set/clear expiry)
  — viewable at `/clerk/audit-log`
- **`notification`** — full history of every SMS reminder attempt
  (sent or failed, with the reason) — viewable at `/clerk/notifications`

`database/schema.sql` has been updated to match all five tables.

## Manual clerk approval

Every document now has two independent results:

- **Automatic check** (barcode + vision model, as before) — shown as
  Valid/Not Valid, always visible.
- **Clerk review** — starts as **Pending** on every upload. A clerk
  opens the document's detail page (`/clerk/document/<id>`) and clicks
  **Approve** or **Reject**, optionally with a note. This status
  (Pending/Approved/Rejected) shows on both the clerk's per-student
  document list and the student's own dashboard, so students can see
  whether a clerk has actually signed off — the automatic check alone
  doesn't grant final approval, it's just a first-pass signal for the
  clerk to review against.

**Note:** this adds new columns to the `Document` table
(`clerk_status`, `clerk_reviewed_at`, `clerk_notes`). If you already
have an `app.db` from testing an earlier version, delete it before
running — `db.create_all()` only creates missing tables, it doesn't
alter existing ones, so an old `app.db` won't have these columns.

## Public home page

`/` is always the public landing page now (no login required, and it
stays the landing page even when logged in — with a "Go to my
Dashboard" button in that case instead of auto-redirecting). It has a
"For Students" section (login/register) and a "For Clerks" section
(login), plus a quick visual walkthrough of the verification flow.

## Clerk can also register students and add documents

Beyond viewing/approving/deleting, a clerk can now:

- **Register a student account** — `/clerk/register-student`, linked
  from the "+ Register Student" button on the clerk dashboard. Useful
  for a walk-in student who hands over physical documents and doesn't
  have an account yet.
- **Submit a document on a student's behalf** — an upload form now
  sits at the top of each student's document page
  (`/clerk/student/<id>`), identical to the student's own upload form
  (same file types, same optional manual-barcode field), just usable
  by the clerk directly.

## Two dashboards: student and clerk

The app now has accounts and two role-based views, backed by a SQLite
database (`app.db`, created automatically on first run):

- **Student**: registers/logs in at `/register` and `/login`, uploads
  documents at their own dashboard (`/student`), and sees their own
  submission history with each verification result.
- **Clerk**: logs in and sees *every* document submitted by *every*
  student at `/clerk`, searchable by student name, with **View** (full
  detail + image preview) and **Delete** actions on each document.

A default clerk account is created the first time you run `python
app.py` — check the terminal output for the generated username/password
(`clerk` / `clerk123` by default). **Change this password** or create
your own clerk accounts instead:

```bash
python create_clerk.py <username> "<full name>" <password>
```

Clerk signup is deliberately not exposed as a public web form — only
students can self-register — so run `create_clerk.py` from the
terminal for each real clerk who needs access.

Document ownership: every uploaded document is tied to the student who
submitted it (`student_id` in the `Document` table) and to their name,
so the clerk dashboard's "search by student name" and per-document
detail view both work directly off that relationship.

## PDF uploads

The app also accepts `.pdf` files, not just images. When a PDF is
uploaded, `utils/pdf_utils.py` renders its **first page** to a PNG
(via PyMuPDF — no external Poppler install needed) and runs that
image through the same barcode → lookup → vision-model pipeline. If
your documents are typically multi-page PDFs with the barcode/photo
on a later page, adjust `pdf_to_image()`'s `page.load_page(0)` call in
`utils/pdf_utils.py` to the correct page index.

## When barcode detection fails entirely

Some real-world barcodes — especially from low-cost CSC/VLE-printed
government certificates — are genuinely too low-quality to decode as a
barcode at all, even with heavy image preprocessing. For these, the
app falls back automatically to OCR: it reads the human-readable
digits normally printed just below the barcode (standard practice for
most institutional barcodes) and checks that against the dataset.

OCR isn't perfect — it can occasionally misread an edge digit — so
this fallback only auto-accepts a result if it actually matches a real
record in `data/valid_barcodes.csv`. If it doesn't find a confident
match, the result screen shows the OCR's best-guess number so the
student can visually compare it to their document, and the upload form
has an optional **"Barcode number"** field to type it in manually and
resubmit — this skips detection entirely and uses exactly what's typed.

`pytesseract` requires the Tesseract OCR engine installed separately
from the pip package:
- **Windows**: install from https://github.com/UB-Mannheim/tesseract/wiki,
  then add the install folder to your PATH (or set
  `pytesseract.pytesseract.tesseract_cmd` in `utils/ocr_fallback.py`
  to the full path of `tesseract.exe`)
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## SMS expiry reminders

Students can get texted automatically when a document is coming up on
its expiry date. Two pieces make this work:

**1. A clerk sets the expiry date.** On any document's detail page
(`/clerk/document/<id>`), there's a "Set / update expiry date" field.
There's no automatic expiry extraction from the document image itself
(that would mean reliably OCR-reading a "valid until" line, which
varies a lot by document type) — the clerk enters it, typically off
what's printed on the document.

**2. `send_expiry_reminders.py` texts students whose documents are
expiring soon.** Run it manually:

```bash
python send_expiry_reminders.py            # reminds for anything expiring within 7 days
python send_expiry_reminders.py --days 14  # change the window
python send_expiry_reminders.py --dry-run  # preview without sending or marking anything
```

Or schedule it to run daily so it's fully automatic:
- **Windows**: Task Scheduler → create a daily task running
  `C:\doc_verification_system\venv\Scripts\python.exe C:\doc_verification_system\send_expiry_reminders.py`
- **Mac/Linux**: cron — `0 9 * * * /path/to/venv/bin/python /path/to/send_expiry_reminders.py`

Each document only gets reminded once (`expiry_reminder_sent` flips to
true after a successful send) — updating the expiry date resets that,
so it can remind again for the new date.

## SMS expiry reminders — now fully automatic

Two things happen without any manual steps:

1. **Instant check on set** — the moment a clerk sets or updates a
   document's expiry date, the app immediately checks whether it falls
   within the reminder window and texts the student right then if so
   — no waiting, no running a script by hand.
2. **Background daily check** — the app also starts a background job
   (via APScheduler) the moment `python app.py` runs, checking *all*
   documents on an interval (default: once every 24 hours) for the
   rest of the time the app is running — so a document that becomes
   due later (e.g. tomorrow it's suddenly within 7 days) still gets
   caught automatically, not just at the moment its expiry was set.

Configure the window and interval via environment variables (both optional):
```powershell
setx REMINDER_DAYS "7"              # remind when a document is within this many days of expiring
setx REMINDER_INTERVAL_HOURS "24"   # how often the background check runs
```

`send_expiry_reminders.py` still exists for manually forcing a check
(useful for testing, or running reminders on a machine where the app
itself isn't running continuously) — but it's no longer required for
reminders to work day-to-day.

### Setting up Twilio (the actual SMS sender)

SMS sending goes through [Twilio](https://www.twilio.com). You need an
account and three values set as environment variables:

```
TWILIO_ACCOUNT_SID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
```

1. Sign up at twilio.com, verify your email/phone.
2. From the Twilio Console dashboard, copy your **Account SID** and
   **Auth Token**.
3. Get a Twilio phone number (Console → Phone Numbers → Buy a number,
   or trial accounts get one free number) — that's `TWILIO_FROM_NUMBER`.
4. **Trial accounts can only text phone numbers you've manually
   verified** in the Twilio Console (Verified Caller IDs) — fine for
   testing with your own number, but you'll need to upgrade to a paid
   account before it can text arbitrary students. For India specifically,
   also check Twilio's current regulatory requirements for sending SMS
   to Indian numbers (a registered sender ID/DLT registration is
   typically required for production use — trial/test sends work
   without it).

Setting environment variables on Windows (PowerShell), permanently:
```powershell
setx TWILIO_ACCOUNT_SID "your_sid_here"
setx TWILIO_AUTH_TOKEN "your_token_here"
setx TWILIO_FROM_NUMBER "+1xxxxxxxxxx"
```
(Close and reopen your terminal after `setx` — same PATH-refresh
requirement as before.)

If these aren't set, `send_expiry_reminders.py` still runs fine and
prints which documents *would* be reminded — it just skips the actual
SMS send and logs why, rather than crashing.

## Using MySQL instead of SQLite (MySQL Workbench)

The app uses SQLite by default (`app.db`, zero setup). To use MySQL
instead — so you can browse/manage the data in MySQL Workbench — the
data models don't change, only where they're stored.

**1. Create the database.** Open MySQL Workbench, connect to your
MySQL server, and either:
- Run `database/schema.sql` (included in this project) as a SQL script
  — this creates the `doc_verification` database and both tables
  explicitly, so you can see the exact schema in Workbench right away, or
- Just create an empty database (`CREATE DATABASE doc_verification;`)
  and let the app create the tables itself on first run (step 3 below
  does this automatically via `db.create_all()`) — either approach
  ends up with the same schema.

**2. Install the MySQL driver** (already in `requirements.txt`):
```bash
pip install -r requirements.txt
```

**3. Point the app at MySQL** by setting the `DATABASE_URL` environment
variable before running it:
```powershell
setx DATABASE_URL "mysql+pymysql://root:your_password@localhost/doc_verification"
```
(Replace `root`/`your_password` with your actual MySQL Workbench
credentials, and reopen your terminal after `setx`, same as above.)

Run `python app.py` again — it now reads/writes MySQL instead of the
`app.db` SQLite file, and everything (student/clerk accounts, uploaded
documents, expiry dates) is visible and editable directly in MySQL
Workbench under the `doc_verification` schema.

**Note:** I wasn't able to spin up a live MySQL server in the sandbox
I built this in (package installs hit broken mirror links there), so
`database/schema.sql` was hand-written to match `models_db.py` exactly
rather than tested against a real MySQL instance. It's the same
standard syntax used elsewhere, but if you hit an error running it in
Workbench, send me the exact error and I'll fix it.

## Tuning the decision logic

Edit `utils/fusion.py` — e.g. `FORGERY_THRESHOLD` (default 0.60)
controls how confident the vision model must be before a "forged"
verdict overrides an otherwise-valid barcode match.
