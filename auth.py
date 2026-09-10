"""
auth.py
-------
Simple, explainable authentication:

  * Passwords are hashed with bcrypt (never stored in plain text).
  * "Being logged in" means the user's id is stored in a signed session
    cookie (Starlette's SessionMiddleware, added in app.py). No JWTs,
    no refresh tokens, no separate sessions table - just a cookie the
    browser sends back on every request, and MySQL as the source of
    truth for who that user actually is.

Every page that requires login calls `get_logged_in_user(request, db)`
and redirects to /login if it returns None.
"""

import bcrypt
from fastapi import Request
from sqlalchemy.orm import Session

from models import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def log_in_user(request: Request, user: User) -> None:
    """Store the user's id in the signed session cookie."""
    request.session["user_id"] = user.id


def log_out_user(request: Request) -> None:
    request.session.clear()


def get_logged_in_user(request: Request, db: Session) -> User | None:
    """Return the currently logged-in User, or None if nobody is logged in."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()
