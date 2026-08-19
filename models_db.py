"""
models_db.py
------------
Database models:
  User          - either role="student" or role="clerk"
  DocumentType  - lookup table of recognized document types, with a
                   default validity period used to suggest an expiry
                   date when a clerk reviews a document of that type.
  Document      - a submitted document, always owned by a student
                   (uploader), carrying its barcode-check + vision-model
                   results, and manageable (view/delete) by any clerk.
  AuditLog      - append-only record of significant actions taken in
                   the system (uploads, approvals, rejections, deletes,
                   logins) — who did what, to what, and when.
  Notification  - record of every SMS reminder attempt (sent or
                   skipped), separate from the boolean flag on Document
                   so there's a full history rather than just the
                   latest state.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "student" or "clerk"
    phone_number = db.Column(db.String(20))  # for SMS expiry reminders — E.164 format, e.g. +919876543210

    documents = db.relationship("Document", backref="student", lazy=True,
                                 cascade="all, delete-orphan", foreign_keys="Document.student_id")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_clerk(self):
        return self.role == "clerk"


class DocumentType(db.Model):
    """Lookup table of recognized document types. Seeded automatically
    on first run (see seed_document_types() in app.py) from the same
    rules utils/doc_type_classifier.py uses to recognize documents."""
    id = db.Column(db.Integer, primary_key=True)
    type_key = db.Column(db.String(50), unique=True, nullable=False)  # e.g. "marksheet"
    display_name = db.Column(db.String(120), nullable=False)          # e.g. "Marksheet / Statement of Marks"
    default_validity_days = db.Column(db.Integer)  # used to suggest an expiry date; NULL = no default


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doc_type_id = db.Column(db.Integer, db.ForeignKey("document_type.id"))  # nullable — may be unrecognized

    original_filename = db.Column(db.String(255), nullable=False)
    stored_path = db.Column(db.String(500), nullable=False)  # image used for verification

    barcode_value = db.Column(db.String(120))
    doc_type = db.Column(db.String(50))  # display name snapshot — kept even if DocumentType row later changes

    barcode_found = db.Column(db.Boolean, default=False)
    record_status = db.Column(db.String(50))  # active / revoked / None

    vision_label = db.Column(db.String(20))
    vision_confidence = db.Column(db.Float)

    is_valid = db.Column(db.Boolean, default=False)
    reasons = db.Column(db.Text)  # newline-joined reasons from fusion.py

    # Manual clerk review — separate from the automatic barcode/vision
    # verdict above. A clerk can override or confirm the automatic result.
    clerk_status = db.Column(db.String(20), default="pending")  # pending / approved / rejected
    clerk_reviewed_at = db.Column(db.DateTime)
    clerk_notes = db.Column(db.Text)

    # Expiry tracking — set by the clerk (e.g. from a "valid until" date
    # printed on the document itself). send_expiry_reminders.py checks
    # this and SMS's the student when it's coming up.
    expiry_date = db.Column(db.Date)
    expiry_reminder_sent = db.Column(db.Boolean, default=False)

    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    document_type_ref = db.relationship("DocumentType", foreign_keys=[doc_type_id])


class AuditLog(db.Model):
    """Append-only activity trail: who did what, to what, and when.
    Never updated or deleted after creation — only ever appended to."""
    id = db.Column(db.Integer, primary_key=True)
    actor_user_id = db.Column(db.Integer, db.ForeignKey("user.id"))  # nullable — e.g. a failed login attempt
    actor_name = db.Column(db.String(120))  # snapshot, so it reads fine even if the user is later deleted
    action = db.Column(db.String(50), nullable=False)  # e.g. "upload", "approve", "reject", "delete", "login"
    target_type = db.Column(db.String(50))  # e.g. "document", "user"
    target_id = db.Column(db.Integer)
    details = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    actor = db.relationship("User", foreign_keys=[actor_user_id])


class Notification(db.Model):
    """History of every SMS reminder attempt — separate from the
    Document.expiry_reminder_sent flag, which only tracks latest state."""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(20), nullable=False)  # "sent" or "failed"
    failure_reason = db.Column(db.Text)  # Text, not String — Twilio error messages can run long
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship("User", foreign_keys=[student_id])
    document = db.relationship("Document", foreign_keys=[document_id])
