"""
generate_synthetic_dataset.py
------------------------------
Generates a complete synthetic training dataset from scratch — no real
college records or scanned documents required. Useful when you can't
access real student data (privacy, no access, etc.).

What it does:
  1. Creates N fake "ID card" style document images, each with a random
     name/ID/year and a real scannable barcode printed on it.
  2. Writes matching rows into data/valid_barcodes.csv, so the barcodes
     these images carry actually pass the dataset lookup step.
  3. For each genuine image, creates a tampered "forged" counterpart
     (edited text, blurred barcode, or overlaid patch) — covering the
     kinds of tampering the vision model should learn to catch.
  4. Splits everything into data/train/{genuine,forged} and
     data/val/{genuine,forged} (80/20).

This gives your CNN/MobileNetV2 models something concrete to train on
immediately. It's a good starting point — swap in real scans later
(see README) for better real-world accuracy, but this alone is enough
to get the pipeline fully working end to end.

Run:
    python data/generate_synthetic_dataset.py --count 300
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import barcode
from barcode.writer import ImageWriter

BASE_DIR = Path(__file__).resolve().parent
GENUINE_DIR = BASE_DIR / "_generated_genuine"
FORGED_DIR = BASE_DIR / "_generated_forged"
CSV_PATH = BASE_DIR / "valid_barcodes.csv"

FIRST_NAMES = ["Aditi", "Rohan", "Sneha", "Vikram", "Priya", "Arjun", "Kabir",
               "Meera", "Sanjay", "Divya", "Karan", "Neha", "Rahul", "Isha",
               "Aman", "Pooja", "Varun", "Anjali", "Yash", "Riya"]
LAST_NAMES = ["Sharma", "Mehta", "Patil", "Singh", "Nair", "Rao", "Khan",
              "Gupta", "Reddy", "Kulkarni", "Iyer", "Chauhan", "Verma", "Joshi"]
DOC_TYPES = ["student_id", "marksheet", "certificate", "govt_id"]

W, H = 640, 400


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_barcode_image(value: str, out_path: Path):
    code128 = barcode.get("code128", value, writer=ImageWriter())
    code128.save(str(out_path.with_suffix("")), options={"write_text": False, "module_height": 8})


def make_document(record: dict, out_path: Path):
    img = Image.new("RGB", (W, H), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 60], fill=(30, 64, 175))
    draw.text((20, 18), "GREENFIELD COLLEGE", font=_font(22), fill="white")

    draw.text((20, 90), record["document_type"].replace("_", " ").upper(), font=_font(18), fill=(30, 30, 30))
    draw.text((20, 130), f"Name: {record['holder_name']}", font=_font(16), fill=(20, 20, 20))
    draw.text((20, 160), f"Record ID: {record['record_id']}", font=_font(16), fill=(20, 20, 20))
    draw.text((20, 190), f"Issue Year: {record['issue_year']}", font=_font(16), fill=(20, 20, 20))

    # NOTE: paste the barcode at its native rendered size — resizing a
    # barcode image (especially shrinking) can distort bar widths enough
    # that decoders like pyzbar fail to read it back. Keep it 1:1.
    bc_tmp = out_path.parent / f"_bc_{record['barcode_value']}"
    make_barcode_image(record["barcode_value"], bc_tmp)
    bc_img = Image.open(str(bc_tmp) + ".png")
    # Cap width so it never overflows the 640px canvas; only shrink if needed,
    # using a generous minimum that still reads reliably.
    if bc_img.width > 560:
        ratio = 560 / bc_img.width
        bc_img = bc_img.resize((560, int(bc_img.height * ratio)), Image.LANCZOS)
    img.paste(bc_img, (20, 280))
    Path(str(bc_tmp) + ".png").unlink(missing_ok=True)

    img.save(out_path)


def make_forged_variant(genuine_path: Path, record: dict, out_path: Path):
    """
    Applies one of several tamper types to a copy of the genuine image.
    Effects are made deliberately strong/visible — a model training from
    scratch on a few hundred images needs a clear signal to key in on;
    subtle single-pixel edits get lost in the noise at this data scale.
    """
    img = Image.open(genuine_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    tamper_type = random.choice(["text_edit", "barcode_blur", "patch_overlay", "name_swap", "noise_block"])

    if tamper_type == "text_edit":
        draw.rectangle([15, 150, 340, 180], fill=(255, 230, 180))  # visible highlight, not background color
        fake_id = record["record_id"][:-2] + str(random.randint(10, 99))
        draw.text((20, 158), f"Record ID: {fake_id}", font=_font(18), fill=(180, 0, 0))

    elif tamper_type == "barcode_blur":
        bc_region = img.crop((15, 275, 590, 365))
        bc_region = bc_region.filter(ImageFilter.GaussianBlur(9))  # much heavier blur
        img.paste(bc_region, (15, 275))

    elif tamper_type == "patch_overlay":
        draw.rectangle([15, 120, 350, 150], fill=(255, 245, 150))
        fake_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        draw.text((20, 128), f"Name: {fake_name}", font=_font(18), fill=(20, 20, 20))
        draw.rectangle([15, 120, 350, 150], outline=(200, 0, 0), width=2)  # visible tamper border

    elif tamper_type == "noise_block":
        # Simulates a scanned-over-scan / print-and-recapture artifact
        import numpy as np
        arr = np.array(img).astype(np.int16)
        y0, y1, x0, x1 = 100, 220, 15, 400
        noise = np.random.randint(-45, 45, arr[y0:y1, x0:x1].shape)
        arr[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1] + noise, 0, 255)
        img = Image.fromarray(arr.astype("uint8"))

    else:  # name_swap with a visible pasted-patch artifact
        fake_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        draw.rectangle([15, 120, 350, 150], fill=(250, 250, 235))
        draw.text((20, 128), f"Name: {fake_name}", font=_font(18), fill=(20, 20, 20))
        draw.rectangle([13, 118, 352, 152], outline=(200, 0, 0), width=3)

    img.save(out_path)


def generate_records(count: int) -> list[dict]:
    records = []
    for i in range(count):
        doc_type = random.choice(DOC_TYPES)
        prefix = {"student_id": "ID", "marksheet": "MSC", "certificate": "CERT", "govt_id": "GOVID"}[doc_type]
        year = random.choice([2021, 2022, 2023, 2024])
        record_num = f"{prefix}{10000 + i}"
        barcode_value = f"COL{year}{record_num}"
        records.append({
            "barcode_value": barcode_value,
            "document_type": doc_type,
            "record_id": record_num,
            "holder_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "issue_year": year,
            "status": "active",
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300, help="number of genuine documents to generate")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    for d in [GENUINE_DIR, FORGED_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    records = generate_records(args.count)

    print(f"Generating {args.count} genuine documents + matching forged versions...")
    for i, record in enumerate(records):
        genuine_path = GENUINE_DIR / f"genuine_{i:04d}.jpg"
        forged_path = FORGED_DIR / f"forged_{i:04d}.jpg"
        make_document(record, genuine_path)
        make_forged_variant(genuine_path, record, forged_path)

    # Write dataset CSV (this is what dataset_lookup.py checks against)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["barcode_value", "document_type", "record_id",
                                                 "holder_name", "issue_year", "status"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {CSV_PATH}")

    # Split into train/val for both classes
    for label, src_dir in [("genuine", GENUINE_DIR), ("forged", FORGED_DIR)]:
        files = sorted(src_dir.glob("*.jpg"))
        random.shuffle(files)
        n_val = int(len(files) * args.val_split)
        val_files, train_files = files[:n_val], files[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            out_dir = BASE_DIR / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, out_dir / f.name)

    shutil.rmtree(GENUINE_DIR)
    shutil.rmtree(FORGED_DIR)

    print("Done. Dataset ready at:")
    print(f"  {BASE_DIR}/train/genuine, {BASE_DIR}/train/forged")
    print(f"  {BASE_DIR}/val/genuine, {BASE_DIR}/val/forged")
    print(f"  {CSV_PATH} (barcode lookup dataset)")
    print("\nNext: python models/train_cnn.py  and  python models/train_mobilenetv2.py")


if __name__ == "__main__":
    main()
