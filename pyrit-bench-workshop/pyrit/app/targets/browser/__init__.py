"""Browser-driven targets (Playwright/Patchright).

  engine.py   engine import, availability checks, pre-action parsing
  target.py   BrowserUITarget -- types into a chat UI and reads the reply
  capture.py  interactive helpers: record selectors, record a login session
  scripts.py  the JavaScript injected by capture.py
  vision.py   ask a vision LLM what a timed-out page shows (diagnostic only)
  sessions.py find a saved login / local page after the project moves machines
"""
from __future__ import annotations

from .capture import capture_login_session, capture_ui_selectors
from .engine import engine_available, install_hint, parse_pre_actions
from .sessions import (LocalPageMissing, SavedSessionMissing,
                       resolve_local_url, resolve_session_path)
from .target import BrowserUITarget
from .vision import diagnose_ui_timeout

__all__ = [
    "BrowserUITarget",
    "capture_login_session", "capture_ui_selectors",
    "diagnose_ui_timeout", "engine_available", "install_hint", "parse_pre_actions",
    "resolve_local_url", "resolve_session_path",
    "LocalPageMissing", "SavedSessionMissing",
]
