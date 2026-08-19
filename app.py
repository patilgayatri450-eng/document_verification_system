"""
Student Document Verification System
-------------------------------------

Flask application for:

STUDENT
- Register
- Login
- Upload document
- Automatic barcode detection
- OCR fallback
- Institutional barcode verification
- AI authenticity verification
- View verification history

CLERK
- Login
- View students
- View student documents
- Upload documents for students
- Approve / reject documents
- Set expiry dates
- Delete documents
- Audit log
- Notifications
"""

import os
from pathlib import Path
from functools import wraps
from datetime import datetime

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    abort,
    send_from_directory
)

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.utils import secure_filename

from models_db import (
    db,
    User,
    Document,
    DocumentType,
    AuditLog,
    Notification
)

from utils.barcode_extractor import get_primary_code
from utils.dataset_lookup import lookup_barcode
from utils.fusion import final_verdict
from utils.pdf_utils import pdf_to_image
from utils.ocr_fallback import ocr_extract_digit_candidates
from utils.doc_type_classifier import (
    classify_document_type,
    RULES as DOC_TYPE_RULES
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {
    "png",
    "jpg",
    "jpeg",
    "pdf"
}

# IMPORTANT:
# Your actual model is:
# models/mobilenetv2_model.keras

VISION_MODEL_NAME = os.environ.get(
    "VISION_MODEL",
    "mobilenetv2_model.keras"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-change-me"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'app.db'}"
)

db.init_app(app)


# ============================================================
# LOGIN
# ============================================================

login_manager = LoginManager(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# ============================================================
# ROLE CHECK
# ============================================================

def role_required(role):

    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):

            if not current_user.is_authenticated:
                return redirect(
                    url_for("login")
                )

            if current_user.role != role:
                abort(403)

            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# FILE CHECK
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXT
    )


# ============================================================
# AUDIT LOG
# ============================================================

def log_action(
    action,
    target_type=None,
    target_id=None,
    details=None
):

    try:

        actor = (
            current_user
            if current_user.is_authenticated
            else None
        )

        entry = AuditLog(
            actor_user_id=(
                actor.id
                if actor
                else None
            ),

            actor_name=(
                actor.full_name
                if actor
                else "anonymous"
            ),

            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details
        )

        db.session.add(entry)
        db.session.commit()

    except Exception as e:

        print(
            f"[audit_log] Failed: {e}"
        )


# ============================================================
# DOCUMENT VERIFICATION PIPELINE
# ============================================================

def run_verification_pipeline(
    save_path: Path,
    filename: str,
    manual_barcode: str | None = None
):

    print("\n")
    print("=" * 60)
    print("STARTING DOCUMENT VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1 - Convert PDF to image
    # --------------------------------------------------------

    if filename.lower().endswith(".pdf"):

        image_path = (
            UPLOAD_DIR /
            (
                Path(filename).stem +
                "_page1.png"
            )
        )

        pdf_to_image(
            str(save_path),
            str(image_path)
        )

    else:

        image_path = save_path

    print(
        f"[1] Image: {image_path}"
    )

    # --------------------------------------------------------
    # STEP 2 - Barcode detection
    # --------------------------------------------------------

    barcode_value = None

    ocr_candidates = []

    if manual_barcode:

        barcode_value = (
            manual_barcode
            .strip()
        )

        print(
            f"[2] Manual barcode: "
            f"{barcode_value}"
        )

    else:

        try:

            barcode_value = get_primary_code(
                str(image_path)
            )

        except Exception as e:

            print(
                f"[barcode] Error: {e}"
            )

            barcode_value = None

        print(
            f"[2] Automatic barcode: "
            f"{barcode_value}"
        )

    # --------------------------------------------------------
    # STEP 3 - OCR fallback
    # --------------------------------------------------------

    if not barcode_value:

        print(
            "[3] Barcode not decoded."
        )

        try:

            ocr_candidates = (
                ocr_extract_digit_candidates(
                    str(image_path)
                )
            )

        except Exception as e:

            print(
                f"[OCR] Error: {e}"
            )

            ocr_candidates = []

        print(
            f"[3] OCR candidates: "
            f"{ocr_candidates}"
        )

        # IMPORTANT:
        # If OCR finds a number that exists
        # in institutional records, use it
        # immediately.

        for candidate in ocr_candidates:

            candidate = (
                str(candidate)
                .strip()
            )

            check = lookup_barcode(
                candidate
            )

            if check["found"]:

                barcode_value = candidate

                print(
                    "[3] OCR barcode matched "
                    "institutional record:"
                )

                print(
                    f"    {barcode_value}"
                )

                break

    # --------------------------------------------------------
    # STEP 4 - Database lookup
    # --------------------------------------------------------

    if barcode_value:

        barcode_result = lookup_barcode(
            barcode_value
        )

    else:

        barcode_result = {
            "found": False,
            "status": None,
            "record": None
        }

    print(
        "[4] DATABASE RESULT:"
    )

    print(
        barcode_result
    )

    # --------------------------------------------------------
    # STEP 5 - AI verification
    # --------------------------------------------------------

    vision_result = {
        "label": "n/a",
        "confidence": 0.0
    }

    # Run AI if barcode exists in records
    if barcode_result["found"]:

        try:

            from models.model_utils import (
                predict_authenticity
            )

            print(
                "[5] Loading AI model:"
            )

            print(
                VISION_MODEL_NAME
            )

            vision_result = (
                predict_authenticity(
                    str(image_path),
                    VISION_MODEL_NAME
                )
            )

            print(
                "[5] AI RESULT:"
            )

            print(
                vision_result
            )

        except Exception as e:

            print(
                f"[AI ERROR] {e}"
            )

            # Don't automatically make it forged.
            # Keep result visible.

            vision_result = {
                "label": "n/a",
                "confidence": 0.0
            }

    else:

        print(
            "[5] AI verification skipped "
            "because barcode was not found."
        )

    # --------------------------------------------------------
    # STEP 6 - FINAL VERDICT
    # --------------------------------------------------------

    verdict = final_verdict(
        barcode_value,
        barcode_result,
        vision_result
    )

    print(
        "[6] FINAL VERDICT:"
    )

    print(
        verdict
    )

    # --------------------------------------------------------
    # STEP 7 - DOCUMENT TYPE
    # --------------------------------------------------------

    doc_type_id = None

    if barcode_result["found"]:

        record = (
            barcode_result["record"]
        )

        doc_type_display = (
            record.get(
                "document_type",
                "Unknown"
            )
        )

    else:

        try:

            (
                type_key,
                doc_type_display
            ) = classify_document_type(
                str(image_path),
                filename
            )

            if type_key:

                dt_row = (
                    DocumentType.query
                    .filter_by(
                        type_key=type_key
                    )
                    .first()
                )

                if dt_row:

                    doc_type_id = dt_row.id

        except Exception as e:

            print(
                f"[DOC TYPE] Error: {e}"
            )

            doc_type_display = (
                "Unknown"
            )

    # --------------------------------------------------------
    # STEP 8 - REASONS
    # --------------------------------------------------------

    reasons = verdict["reasons"]

    # IMPORTANT:
    # Do NOT tell the student to resubmit
    # if OCR successfully matched the barcode.

    if (
        not barcode_value
        and
        ocr_candidates
    ):

        reasons = reasons + [
            "OCR detected possible barcode "
            f"numbers: {', '.join(ocr_candidates)}."
        ]

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "barcode_value": barcode_value,

        "doc_type": doc_type_display,

        "doc_type_id": doc_type_id,

        "barcode_found": (
            barcode_result["found"]
        ),

        "record_status": (
            barcode_result["status"]
        ),

        "vision_label": (
            vision_result["label"]
        ),

        "vision_confidence": (
            vision_result["confidence"]
        ),

        "is_valid": (
            verdict["valid"]
        ),

        "reasons": "\n".join(
            reasons
        )
    }

    print(
        "[7] SAVING RESULT:"
    )

    print(
        result
    )

    print(
        "=" * 60
    )
    print(
        "VERIFICATION COMPLETED"
    )
    print(
        "=" * 60
    )

    return result, None


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "home.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = (
            request.form["username"]
            .strip()
        )

        password = (
            request.form["password"]
        )

        user = (
            User.query
            .filter_by(
                username=username
            )
            .first()
        )

        if (
            user
            and
            user.check_password(
                password
            )
        ):

            login_user(user)

            log_action(
                "login",
                target_type="user",
                target_id=user.id
            )

            if user.is_clerk():

                return redirect(
                    url_for(
                        "clerk_dashboard"
                    )
                )

            return redirect(
                url_for(
                    "student_dashboard"
                )
            )

        flash(
            "Invalid username or password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = (
            request.form["username"]
            .strip()
        )

        full_name = (
            request.form["full_name"]
            .strip()
        )

        password = (
            request.form["password"]
        )

        phone_number = (
            request.form
            .get(
                "phone_number",
                ""
            )
            .strip()
            or None
        )

        if (
            User.query
            .filter_by(
                username=username
            )
            .first()
        ):

            flash(
                "That username is already taken."
            )

            return render_template(
                "register.html"
            )

        user = User(
            username=username,
            full_name=full_name,
            role="student",
            phone_number=phone_number
        )

        user.set_password(
            password
        )

        db.session.add(user)
        db.session.commit()

        login_user(user)

        log_action(
            "register",
            target_type="user",
            target_id=user.id,
            details="student registration"
        )

        return redirect(
            url_for(
                "student_dashboard"
            )
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("login")
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

@app.route(
    "/student",
    methods=["GET"]
)
@role_required("student")
def student_dashboard():

    docs = (
        Document.query
        .filter_by(
            student_id=current_user.id
        )
        .order_by(
            Document.uploaded_at.desc()
        )
        .all()
    )

    suggested_barcode = (
        request.args.get(
            "suggested_barcode",
            ""
        )
    )

    return render_template(
        "student_dashboard.html",
        docs=docs,
        suggested_barcode=suggested_barcode
    )


# ============================================================
# STUDENT UPLOAD
# ============================================================

@app.route(
    "/student/upload",
    methods=["POST"]
)
@role_required("student")
def student_upload():

    file = request.files.get(
        "document"
    )

    if (
        not file
        or
        file.filename == ""
        or
        not allowed_file(
            file.filename
        )
    ):

        flash(
            "Please upload a .png/.jpg/.jpeg/.pdf file."
        )

        return redirect(
            url_for(
                "student_dashboard"
            )
        )

    filename = secure_filename(
        file.filename
    )

    save_path = (
        UPLOAD_DIR /
        f"{current_user.id}_{filename}"
    )

    file.save(save_path)

    manual_barcode = (
        request.form
        .get(
            "manual_barcode",
            ""
        )
        .strip()
        or None
    )

    try:

        result, suggested = (
            run_verification_pipeline(
                save_path,
                filename,
                manual_barcode
            )
        )

    except Exception as e:

        print(
            f"[UPLOAD ERROR] {e}"
        )

        flash(
            f"Could not process document: {e}"
        )

        return redirect(
            url_for(
                "student_dashboard"
            )
        )

    doc = Document(
        student_id=current_user.id,
        original_filename=filename,
        stored_path=str(save_path),
        **result
    )

    db.session.add(doc)

    db.session.commit()

    log_action(
        "upload",
        target_type="document",
        target_id=doc.id,
        details=filename
    )

    # --------------------------------------------------------
    # Show result immediately
    # --------------------------------------------------------

    if result["is_valid"]:

        flash(
            f"Document verification completed. "
            f"Barcode: {result['barcode_value']} — VALID."
        )

    else:

        flash(
            f"Document verification completed. "
            f"Barcode: {result['barcode_value'] or 'Not detected'} "
            f"— NOT VALID."
        )

    return redirect(
        url_for(
            "student_dashboard"
        )
    )


# ============================================================
# CLERK DASHBOARD
# ============================================================

@app.route(
    "/clerk",
    methods=["GET"]
)
@role_required("clerk")
def clerk_dashboard():

    name_query = (
        request.args
        .get(
            "student_name",
            ""
        )
        .strip()
    )

    query = User.query.filter_by(
        role="student"
    )

    if name_query:

        query = query.filter(
            User.full_name.ilike(
                f"%{name_query}%"
            )
        )

    students = (
        query
        .order_by(
            User.full_name
        )
        .all()
    )

    students_with_counts = [

        (
            student,
            Document.query
            .filter_by(
                student_id=student.id
            )
            .count()
        )

        for student in students
    ]

    return render_template(
        "clerk_dashboard.html",
        students=students_with_counts,
        name_query=name_query
    )


# ============================================================
# CLERK STUDENT DOCUMENTS
# ============================================================

@app.route(
    "/clerk/student/<int:student_id>",
    methods=["GET"]
)
@role_required("clerk")
def clerk_student_documents(
    student_id
):

    student = db.session.get(
        User,
        student_id
    )

    if (
        not student
        or
        student.role != "student"
    ):

        abort(404)

    docs = (
        Document.query
        .filter_by(
            student_id=student.id
        )
        .order_by(
            Document.uploaded_at.desc()
        )
        .all()
    )

    suggested_barcode = (
        request.args
        .get(
            "suggested_barcode",
            ""
        )
    )

    return render_template(
        "clerk_student_documents.html",
        student=student,
        docs=docs,
        suggested_barcode=suggested_barcode
    )


# ============================================================
# CLERK REGISTER STUDENT
# ============================================================

@app.route(
    "/clerk/register-student",
    methods=["GET", "POST"]
)
@role_required("clerk")
def clerk_register_student():

    if request.method == "POST":

        username = (
            request.form["username"]
            .strip()
        )

        full_name = (
            request.form["full_name"]
            .strip()
        )

        password = (
            request.form["password"]
        )

        phone_number = (
            request.form
            .get(
                "phone_number",
                ""
            )
            .strip()
            or None
        )

        if (
            User.query
            .filter_by(
                username=username
            )
            .first()
        ):

            flash(
                "That username is already taken."
            )

            return render_template(
                "clerk_register_student.html"
            )

        student = User(
            username=username,
            full_name=full_name,
            role="student",
            phone_number=phone_number
        )

        student.set_password(
            password
        )

        db.session.add(student)

        db.session.commit()

        log_action(
            "register",
            target_type="user",
            target_id=student.id,
            details="registered by clerk"
        )

        flash(
            f"Student account created for "
            f"{full_name}."
        )

        return redirect(
            url_for(
                "clerk_student_documents",
                student_id=student.id
            )
        )

    return render_template(
        "clerk_register_student.html"
    )


# ============================================================
# CLERK UPLOAD FOR STUDENT
# ============================================================

@app.route(
    "/clerk/student/<int:student_id>/upload",
    methods=["POST"]
)
@role_required("clerk")
def clerk_upload_for_student(
    student_id
):

    student = db.session.get(
        User,
        student_id
    )

    if (
        not student
        or
        student.role != "student"
    ):

        abort(404)

    file = request.files.get(
        "document"
    )

    if (
        not file
        or
        file.filename == ""
        or
        not allowed_file(
            file.filename
        )
    ):

        flash(
            "Please upload a .png/.jpg/.jpeg/.pdf file."
        )

        return redirect(
            url_for(
                "clerk_student_documents",
                student_id=student_id
            )
        )

    filename = secure_filename(
        file.filename
    )

    save_path = (
        UPLOAD_DIR /
        f"{student.id}_{filename}"
    )

    file.save(save_path)

    manual_barcode = (
        request.form
        .get(
            "manual_barcode",
            ""
        )
        .strip()
        or None
    )

    try:

        result, suggested = (
            run_verification_pipeline(
                save_path,
                filename,
                manual_barcode
            )
        )

    except Exception as e:

        flash(
            f"Could not process document: {e}"
        )

        return redirect(
            url_for(
                "clerk_student_documents",
                student_id=student_id
            )
        )

    doc = Document(
        student_id=student.id,
        original_filename=filename,
        stored_path=str(save_path),
        **result
    )

    db.session.add(doc)

    db.session.commit()

    log_action(
        "upload",
        target_type="document",
        target_id=doc.id,
        details=(
            f"{filename} "
            f"(on behalf of {student.full_name})"
        )
    )

    if result["is_valid"]:

        flash(
            f"Document verified successfully — "
            f"VALID. Barcode: "
            f"{result['barcode_value']}"
        )

    else:

        flash(
            f"Document verification completed — "
            f"NOT VALID. Barcode: "
            f"{result['barcode_value'] or 'Not detected'}"
        )

    return redirect(
        url_for(
            "clerk_student_documents",
            student_id=student_id
        )
    )


# ============================================================
# UPLOADED FILES
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
@login_required
def uploaded_file(filename):

    if (
        not current_user.is_clerk()
        and
        not filename.startswith(
            f"{current_user.id}_"
        )
    ):

        abort(403)

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ============================================================
# CLERK DOCUMENT VIEW
# ============================================================

@app.route(
    "/clerk/document/<int:doc_id>",
    methods=["GET"]
)
@role_required("clerk")
def clerk_view_document(doc_id):

    doc = (
        db.session.get(
            Document,
            doc_id
        )
        or abort(404)
    )

    return render_template(
        "document_detail.html",
        doc=doc
    )


# ============================================================
# APPROVE
# ============================================================

@app.route(
    "/clerk/document/<int:doc_id>/approve",
    methods=["POST"]
)
@role_required("clerk")
def clerk_approve_document(
    doc_id
):

    doc = (
        db.session.get(
            Document,
            doc_id
        )
        or abort(404)
    )

    doc.clerk_status = "approved"

    doc.clerk_reviewed_at = (
        datetime.utcnow()
    )

    doc.clerk_notes = (
        request.form
        .get(
            "notes",
            ""
        )
        .strip()
        or None
    )

    db.session.commit()

    log_action(
        "approve",
        target_type="document",
        target_id=doc.id,
        details=doc.clerk_notes
    )

    flash(
        "Document approved."
    )

    return redirect(
        url_for(
            "clerk_view_document",
            doc_id=doc.id
        )
    )


# ============================================================
# REJECT
# ============================================================

@app.route(
    "/clerk/document/<int:doc_id>/reject",
    methods=["POST"]
)
@role_required("clerk")
def clerk_reject_document(
    doc_id
):

    doc = (
        db.session.get(
            Document,
            doc_id
        )
        or abort(404)
    )

    doc.clerk_status = "rejected"

    doc.clerk_reviewed_at = (
        datetime.utcnow()
    )

    doc.clerk_notes = (
        request.form
        .get(
            "notes",
            ""
        )
        .strip()
        or None
    )

    db.session.commit()

    log_action(
        "reject",
        target_type="document",
        target_id=doc.id,
        details=doc.clerk_notes
    )

    flash(
        "Document rejected."
    )

    return redirect(
        url_for(
            "clerk_view_document",
            doc_id=doc.id
        )
    )


# ============================================================
# SET EXPIRY
# ============================================================

@app.route(
    "/clerk/document/<int:doc_id>/set-expiry",
    methods=["POST"]
)
@role_required("clerk")
def clerk_set_expiry(doc_id):

    doc = (
        db.session.get(
            Document,
            doc_id
        )
        or abort(404)
    )

    expiry_str = (
        request.form
        .get(
            "expiry_date",
            ""
        )
        .strip()
    )

    if expiry_str:

        try:

            doc.expiry_date = (
                datetime.strptime(
                    expiry_str,
                    "%Y-%m-%d"
                ).date()
            )

        except ValueError:

            flash(
                "Invalid date format."
            )

            return redirect(
                url_for(
                    "clerk_view_document",
                    doc_id=doc.id
                )
            )

        doc.expiry_reminder_sent = False

        db.session.commit()

        log_action(
            "set_expiry",
            target_type="document",
            target_id=doc.id,
            details=expiry_str
        )

        flash(
            f"Expiry date set to "
            f"{doc.expiry_date.strftime('%d %b %Y')}."
        )

    else:

        doc.expiry_date = None

        db.session.commit()

        log_action(
            "clear_expiry",
            target_type="document",
            target_id=doc.id
        )

        flash(
            "Expiry date cleared."
        )

    return redirect(
        url_for(
            "clerk_view_document",
            doc_id=doc.id
        )
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.route(
    "/clerk/document/<int:doc_id>/delete",
    methods=["POST"]
)
@role_required("clerk")
def clerk_delete_document(
    doc_id
):

    doc = (
        db.session.get(
            Document,
            doc_id
        )
        or abort(404)
    )

    student_id = doc.student_id

    filename = (
        doc.original_filename
    )

    try:

        Path(
            doc.stored_path
        ).unlink(
            missing_ok=True
        )

    except Exception:
        pass

    db.session.delete(doc)

    db.session.commit()

    log_action(
        "delete",
        target_type="document",
        target_id=doc_id,
        details=filename
    )

    flash(
        "Document deleted."
    )

    return redirect(
        url_for(
            "clerk_student_documents",
            student_id=student_id
        )
    )


# ============================================================
# AUDIT LOG
# ============================================================

@app.route(
    "/clerk/audit-log",
    methods=["GET"]
)
@role_required("clerk")
def clerk_audit_log():

    entries = (
        AuditLog.query
        .order_by(
            AuditLog.created_at.desc()
        )
        .limit(200)
        .all()
    )

    return render_template(
        "audit_log.html",
        entries=entries
    )


# ============================================================
# NOTIFICATIONS
# ============================================================

@app.route(
    "/clerk/notifications",
    methods=["GET"]
)
@role_required("clerk")
def clerk_notifications():

    entries = (
        Notification.query
        .order_by(
            Notification.sent_at.desc()
        )
        .limit(200)
        .all()
    )

    return render_template(
        "notifications.html",
        entries=entries
    )


# ============================================================
# ERROR
# ============================================================

@app.errorhandler(403)
def forbidden(e):

    return (
        render_template(
            "error.html",
            message="You don't have access to that page."
        ),
        403
    )


# ============================================================
# DEFAULT CLERK
# ============================================================

def seed_default_clerk():

    if not User.query.filter_by(
        role="clerk"
    ).first():

        clerk = User(
            username="clerk",
            full_name="Front Desk Clerk",
            role="clerk"
        )

        clerk.set_password(
            "clerk123"
        )

        db.session.add(clerk)

        db.session.commit()

        print(
            "\n[first run] "
            "Clerk login:"
        )

        print(
            "username: clerk"
        )

        print(
            "password: clerk123\n"
        )


# ============================================================
# DOCUMENT TYPES
# ============================================================

def seed_document_types():

    if DocumentType.query.first():

        return

    default_validity = {

        "marksheet": None,

        "hsc_ssc_certificate": None,

        "income_certificate": 365,

        "nationality_domicile": None,

        "caste_certificate": 365,

        "student_id": 365,

        "govt_id": None,

        "degree_certificate": None
    }

    for (
        type_key,
        display_name,
        _patterns
    ) in DOC_TYPE_RULES:

        db.session.add(
            DocumentType(
                type_key=type_key,
                display_name=display_name,
                default_validity_days=(
                    default_validity
                    .get(type_key)
                )
            )
        )

    db.session.commit()

    print(
        "[first run] "
        "Document types seeded."
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

with app.app_context():

    db.create_all()

    seed_default_clerk()

    seed_document_types()


# ============================================================
# BACKGROUND REMINDER
# ============================================================

if not os.environ.get(
    "DISABLE_AUTO_SCHEDULER"
):

    try:

        from reminder_job import (
            start_background_scheduler
        )

        start_background_scheduler(
            app,
            days=int(
                os.environ.get(
                    "REMINDER_DAYS",
                    7
                )
            ),
            interval_hours=int(
                os.environ.get(
                    "REMINDER_INTERVAL_HOURS",
                    24
                )
            )
        )

    except Exception as e:

        print(
            f"[scheduler] "
            f"Could not start: {e}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "STUDENT DOCUMENT VERIFICATION SYSTEM"
    )
    print("=" * 60)
    print(
        "AI Model:",
        VISION_MODEL_NAME
    )
    print(
        "Upload directory:",
        UPLOAD_DIR
    )
    print(
        "URL: http://127.0.0.1:5000"
    )
    print("=" * 60)
    print()

    app.run(
        debug=True,
        use_reloader=False
    )