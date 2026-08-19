"""
create_clerk.py
----------------
Creates an additional clerk account. Run this from the terminal —
clerk signup is intentionally not exposed as a web form, so random
visitors can't self-register as a clerk.

Usage:
    python create_clerk.py <username> <full_name> <password>

Example:
    python create_clerk.py priya "Priya Nair" S3cureP@ss
"""

import sys
from app import app
from models_db import db, User


def main():
    if len(sys.argv) != 4:
        print("Usage: python create_clerk.py <username> <full_name> <password>")
        sys.exit(1)

    username, full_name, password = sys.argv[1], sys.argv[2], sys.argv[3]

    with app.app_context():
        if User.query.filter_by(username=username).first():
            print(f"Username '{username}' already exists.")
            sys.exit(1)

        clerk = User(username=username, full_name=full_name, role="clerk")
        clerk.set_password(password)
        db.session.add(clerk)
        db.session.commit()
        print(f"Created clerk account: {username}")


if __name__ == "__main__":
    main()
