"""
dataset_lookup.py
-----------------
Looks up a barcode number in the institutional
valid_barcodes.csv dataset.
"""

import pandas as pd
from pathlib import Path


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "valid_barcodes.csv"
)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Barcode dataset not found: {path}"
        )

    df = pd.read_csv(path, dtype=str)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    if "barcode_value" not in df.columns:
        raise ValueError(
            "CSV must contain a 'barcode_value' column."
        )

    # Clean barcode values
    df["barcode_value"] = (
        df["barcode_value"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Clean status
    if "status" in df.columns:
        df["status"] = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def lookup_barcode(
    barcode_value: str,
    df: pd.DataFrame | None = None
) -> dict:

    if df is None:
        df = load_dataset()

    if not barcode_value:
        return {
            "found": False,
            "status": None,
            "record": None
        }

    # Clean incoming barcode
    barcode = str(barcode_value).strip()

    # Exact match
    match = df[
        df["barcode_value"].astype(str).str.strip() == barcode
    ]

    if match.empty:
        return {
            "found": False,
            "status": None,
            "record": None
        }

    record = match.iloc[0].to_dict()

    return {
        "found": True,
        "status": str(record.get("status", "unknown")).strip().lower(),
        "record": record
    }


if __name__ == "__main__":
    import sys

    code = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "0202223200161"
    )

    print(lookup_barcode(code))