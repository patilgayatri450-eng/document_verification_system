-- ============================================================
-- Legal Document Analysis System — MySQL schema
-- ============================================================
-- Matches models_db.py exactly (User -> `user`, DocumentType ->
-- `document_type`, Document -> `document`, AuditLog -> `audit_log`,
-- Notification -> `notification`; Flask-SQLAlchemy's default table
-- naming). Import this directly in MySQL Workbench, or just point
-- the app at an empty MySQL database and let SQLAlchemy's
-- db.create_all() build the same tables automatically on first run.
--
-- Usage in MySQL Workbench:
--   1. Open a new SQL tab connected to your MySQL server.
--   2. Run this whole script (it creates the database and all tables).
-- Or from the command line:
--   mysql -u root -p < schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS doc_verification
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE doc_verification;

-- ------------------------------------------------------------
-- user  (students and clerks)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(80)  NOT NULL UNIQUE,
    full_name       VARCHAR(120) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20)  NOT NULL,          -- 'student' or 'clerk'
    phone_number    VARCHAR(20)                     -- E.164 format, e.g. +919876543210
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- document_type  (lookup table of recognized document types)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_type (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    type_key                VARCHAR(50)  NOT NULL UNIQUE,   -- e.g. 'marksheet'
    display_name            VARCHAR(120) NOT NULL,          -- e.g. 'Marksheet / Statement of Marks'
    default_validity_days   INT                             -- NULL = no default suggestion
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- document  (one row per submitted document)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    student_id              INT NOT NULL,
    doc_type_id             INT,                    -- nullable — may be unrecognized

    original_filename       VARCHAR(255) NOT NULL,
    stored_path              VARCHAR(500) NOT NULL,

    barcode_value            VARCHAR(120),
    doc_type                 VARCHAR(50),             -- display-name snapshot

    barcode_found             TINYINT(1) DEFAULT 0,
    record_status             VARCHAR(50),             -- 'active' / 'revoked' / NULL

    vision_label               VARCHAR(20),
    vision_confidence          FLOAT,

    is_valid                   TINYINT(1) DEFAULT 0,
    reasons                     TEXT,

    -- Manual clerk review, independent of the automatic check above
    clerk_status                VARCHAR(20) DEFAULT 'pending',   -- pending / approved / rejected
    clerk_reviewed_at           DATETIME,
    clerk_notes                  TEXT,

    -- Expiry tracking for SMS reminders
    expiry_date                  DATE,
    expiry_reminder_sent         TINYINT(1) DEFAULT 0,

    uploaded_at                   DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_document_student
        FOREIGN KEY (student_id) REFERENCES user(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_document_type
        FOREIGN KEY (doc_type_id) REFERENCES document_type(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- audit_log  (append-only activity trail)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    actor_user_id   INT,                            -- nullable, e.g. failed/anonymous actions
    actor_name      VARCHAR(120),                   -- snapshot — survives user deletion
    action          VARCHAR(50) NOT NULL,           -- 'login', 'upload', 'approve', 'reject', 'delete', ...
    target_type     VARCHAR(50),                    -- 'document', 'user'
    target_id       INT,
    details         VARCHAR(500),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_audit_actor
        FOREIGN KEY (actor_user_id) REFERENCES user(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- notification  (SMS reminder history)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notification (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT NOT NULL,
    document_id     INT NOT NULL,
    message         VARCHAR(500) NOT NULL,
    status          VARCHAR(20) NOT NULL,           -- 'sent' or 'failed'
    failure_reason  TEXT,
    sent_at         DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_notification_student
        FOREIGN KEY (student_id) REFERENCES user(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_notification_document
        FOREIGN KEY (document_id) REFERENCES document(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Helpful indexes for the queries the app runs most often
-- ------------------------------------------------------------
CREATE INDEX idx_document_student_id ON document(student_id);
CREATE INDEX idx_document_expiry ON document(expiry_date, expiry_reminder_sent);
CREATE INDEX idx_audit_created_at ON audit_log(created_at);
CREATE INDEX idx_notification_document ON notification(document_id);
