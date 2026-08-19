"""
fusion.py
---------

Final document verification logic.

PRIMARY VERIFICATION:
    Barcode / QR code institutional record

SECONDARY VERIFICATION:
    AI visual authenticity model

Decision:
    - Barcode missing              -> NOT VALID
    - Barcode not found            -> NOT VALID
    - Barcode inactive/revoked     -> NOT VALID
    - Barcode active               -> VALID

AI prediction is displayed separately
and does not override an active institutional
barcode record.
"""

FORGERY_THRESHOLD = 0.60


def final_verdict(
    barcode_value,
    barcode_result,
    vision_result
):

    # ======================================================
    # 1. BARCODE NOT DETECTED
    # ======================================================

    if not barcode_value:

        return {
            "valid": False,

            "reasons": [
                "No barcode/QR code could be detected "
                "on the document."
            ],

            "verification_method": "barcode",
            "ai_warning": None
        }

    # ======================================================
    # 2. BARCODE NOT FOUND
    # ======================================================

    if not barcode_result.get("found", False):

        return {
            "valid": False,

            "reasons": [
                f"Barcode '{barcode_value}' was not found "
                "in institutional records."
            ],

            "verification_method": "barcode",
            "ai_warning": None
        }

    # ======================================================
    # 3. CHECK STATUS
    # ======================================================

    status = str(
        barcode_result.get("status", "")
    ).strip().lower()

    # ======================================================
    # 4. BARCODE NOT ACTIVE
    # ======================================================

    if status != "active":

        return {
            "valid": False,

            "reasons": [
                f"Record found but status is '{status}'."
            ],

            "verification_method": "barcode",
            "ai_warning": None
        }

    # ======================================================
    # 5. BARCODE IS ACTIVE
    # ======================================================

    reasons = [

        "Barcode matched an active institutional record."
    ]

    # ======================================================
    # 6. AI RESULT
    # ======================================================

    ai_warning = None

    if vision_result:

        label = str(
            vision_result.get("label", "")
        ).lower().strip()

        confidence = float(
            vision_result.get("confidence", 0)
        )

        # --------------------------------------------------
        # AI says forged
        # --------------------------------------------------

        if (
            label == "forged"
            and confidence >= FORGERY_THRESHOLD
        ):

            ai_warning = (
                f"AI visual model flagged the document "
                f"as potentially forged "
                f"({confidence * 100:.2f}% confidence)."
            )

            reasons.append(
                "Institutional barcode verification passed."
            )

        # --------------------------------------------------
        # AI says genuine
        # --------------------------------------------------

        else:

            reasons.append(
                f"AI visual authenticity check: "
                f"genuine ({confidence * 100:.2f}% confidence)."
            )

    # ======================================================
    # 7. FINAL RESULT
    # ======================================================

    return {

        "valid": True,

        "reasons": reasons,

        "verification_method": "active_institutional_barcode",

        "ai_warning": ai_warning
    }