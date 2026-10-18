"""CSRF token issue + check, shared by every state-changing POST handler."""
from __future__ import annotations

import hashlib
import hmac
import os
from flask import jsonify, request, session

from .blueprint import bp


# ---------- CSRF protection ----------------------------------------------

def csrf_token() -> str:
    """Return (and cache in Flask session) a per-session CSRF token."""
    if "csrf_token" not in session:
        session["csrf_token"] = hashlib.sha256(os.urandom(32)).hexdigest()
    return session["csrf_token"]


def csrf_protect() -> "tuple | None":
    """Raise 403 if the submitted CSRF token doesn't match the session token.

    Called at the top of every state-changing POST handler. Every such route is
    a form submission carrying the per-session token from csrf_token(); returns
    a (Response, 403) tuple on mismatch, or None when the token is valid so the
    caller can proceed.
    """
    token = request.form.get("csrf_token") or request.args.get("csrf_token")
    expected = session.get("csrf_token", "")
    if not token or not hmac.compare_digest(token, expected):
        return jsonify(error="CSRF token missing or invalid"), 403


# Inject the token generator into every template so forms can use it.
@bp.app_context_processor
def _inject_csrf():
    return {"csrf_token": csrf_token}
