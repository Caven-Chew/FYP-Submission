"""Interactive helpers: record CSS selectors, and record a logged-in session."""
from __future__ import annotations

import json
import os

from .engine import install_hint, load_sync_playwright
from .scripts import CAPTURE_JS, LOGIN_JS
from .sessions import SavedSessionMissing, resolve_session_path


def capture_ui_selectors(url: str, timeout_s: int = 300, headless: bool = False,
                         storage_state: str | None = None) -> dict:
    """Open `url` in a real browser and let the user click-to-capture selectors.

    Drives a headed Chromium with Playwright, injects the capture overlay
    (`CAPTURE_JS`), and blocks until the user presses *Use these* / *Cancel*
    or closes the window. Returns:

        {"ok": True,  "selectors": {"ui_input_selector": ..., "ui_submit_selector": ...,
                                    "ui_response_selector": ...}}
        {"ok": False, "error": "<friendly message>"}

    `storage_state` is an optional path to a saved login session (see
    `capture_login_session`); when given, the page opens already authenticated so
    selectors can be picked behind a login wall.

    `headless` is exposed only so the selector logic can be exercised in tests;
    a real capture session needs a visible window (headless=False).
    """
    import time as _time

    if not (url or "").strip():
        return {"ok": False, "error": "Enter the chat page URL first."}

    try:
        sync_playwright, engine = load_sync_playwright()   # Patchright (required)
    except ImportError as e:
        return {"ok": False, "error": str(e)}

    state = {"input": "", "submit": "", "response": "",
             "pre": [], "done": False, "cancelled": False}

    def _report(payload):
        try:
            data = payload if isinstance(payload, dict) else json.loads(payload)
        except (TypeError, ValueError):
            return
        for k in ("input", "submit", "response"):
            if data.get(k):
                state[k] = data[k]
        if isinstance(data.get("pre"), list):
            state["pre"] = data["pre"]
        if data.get("type") == "done":
            state["done"] = True
        elif data.get("type") == "cancel":
            state["cancelled"] = True

    try:
        with sync_playwright() as pw:
            try:
                # A fixed Playwright viewport defaults to 1280x720. In a headed
                # browser the native title/address bars are added outside that
                # area, which can put the bottom of the window beyond a smaller
                # projector display. Let Chromium fit the available screen and
                # let its page viewport follow the real window instead.
                launch_kwargs = {"headless": headless}
                if not headless:
                    launch_kwargs["args"] = ["--start-maximized"]
                browser = pw.chromium.launch(**launch_kwargs)
            except Exception as e:  # noqa: BLE001 — usually a missing browser binary
                return {"ok": False, "error": (
                    f"Could not launch Chromium ({type(e).__name__}). "
                    f"Run:  {install_hint(engine)}"
                )}
            ctx_kwargs = {}
            try:
                saved_login = resolve_session_path(storage_state)
            except SavedSessionMissing as e:
                return {"ok": False, "error": str(e)}
            if saved_login:
                ctx_kwargs["storage_state"] = saved_login   # open authenticated
            if not headless:
                ctx_kwargs["no_viewport"] = True
            context = browser.new_context(**ctx_kwargs)
            # Re-inject the overlay on every navigation (including a manual
            # refresh) so it doesn't vanish — same fix as the login bar.
            context.add_init_script(CAPTURE_JS)
            page = context.new_page()
            own_browser = True
            page.expose_function("__pyritCaptureReport", _report)
            try:
                page.goto(url, timeout=30000)
            except Exception as e:  # noqa: BLE001
                try:
                    page.close() if not own_browser else browser.close()
                except Exception:  # noqa: BLE001
                    pass
                return {"ok": False, "error": f"Could not open {url!r}: {type(e).__name__}: {e}"}

            deadline = _time.time() + timeout_s
            while not state["done"] and not state["cancelled"] and _time.time() < deadline:
                try:
                    page.wait_for_timeout(250)
                except Exception:  # noqa: BLE001 — window/tab closed
                    break
                if page.is_closed():
                    break
                # Read the DOM sentinel the overlay writes — survives Patchright's
                # isolated-context binding split (the JSON string feeds _report).
                try:
                    raw = page.get_attribute("html", "data-pyrit-capture", timeout=1000)
                except Exception:  # noqa: BLE001
                    raw = None
                if raw:
                    _report(raw)
            try:
                if own_browser:
                    browser.close()
                else:
                    page.close()   # leave the user's Chrome running
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Capture failed: {type(e).__name__}: {e}"}

    if state["cancelled"]:
        return {"ok": False, "error": "Capture cancelled."}
    selectors = {
        "ui_input_selector": state["input"],
        "ui_submit_selector": state["submit"],
        "ui_response_selector": state["response"],
    }
    pre = [a for a in (state.get("pre") or []) if isinstance(a, dict) and a.get("type")]
    if pre:
        selectors["ui_pre_actions"] = json.dumps(pre, separators=(",", ":"))
    if not any(selectors.values()):
        return {"ok": False, "error": "No selectors captured (the window was closed early)."}
    return {"ok": True, "selectors": selectors}


def capture_login_session(url: str, storage_path: str, timeout_s: int = 600,
                          headless: bool = False) -> dict:
    """Open `url`, let the user log in by hand, then save the browser session.

    Shows a slim top bar with an *I'm logged in* button. When pressed, the
    context's storage_state (cookies + localStorage) is written to
    `storage_path` for reuse by `capture_ui_selectors` and `BrowserUITarget`.

    Returns {"ok": True, "path": storage_path} or {"ok": False, "error": ...}.
    """
    import time as _time

    if not (url or "").strip():
        return {"ok": False, "error": "Enter the chat page URL first."}

    try:
        sync_playwright, engine = load_sync_playwright()   # Patchright (required)
    except ImportError as e:
        return {"ok": False, "error": str(e)}

    state = {"done": False, "cancelled": False}

    def _report(payload):
        try:
            data = payload if isinstance(payload, dict) else json.loads(payload)
        except (TypeError, ValueError):
            return
        if data.get("type") == "done":
            state["done"] = True
        elif data.get("type") == "cancel":
            state["cancelled"] = True

    try:
        with sync_playwright() as pw:
            try:
                launch_kwargs = {"headless": headless}
                if not headless:
                    launch_kwargs["args"] = ["--start-maximized"]
                browser = pw.chromium.launch(**launch_kwargs)
            except Exception as e:  # noqa: BLE001 — usually a missing browser binary
                return {"ok": False, "error": (
                    f"Could not launch Chromium ({type(e).__name__}). "
                    f"Run:  {install_hint(engine)}"
                )}
            context_kwargs = {"no_viewport": True} if not headless else {}
            context = browser.new_context(**context_kwargs)
            # Re-inject the bar on every navigation (login flows navigate a lot).
            context.add_init_script(LOGIN_JS)
            page = context.new_page()
            page.expose_function("__pyritLoginReport", _report)
            try:
                page.goto(url, timeout=60000)
            except Exception as e:  # noqa: BLE001
                browser.close()
                return {"ok": False, "error": f"Could not open {url!r}: {type(e).__name__}: {e}"}

            deadline = _time.time() + timeout_s
            while not state["done"] and not state["cancelled"] and _time.time() < deadline:
                try:
                    page.wait_for_timeout(300)
                except Exception:  # noqa: BLE001 — window/tab closed
                    break
                if page.is_closed():
                    break
                # Read the DOM sentinel the bar writes — survives Patchright's
                # isolated-context binding split (shared DOM, world-agnostic).
                try:
                    sig = page.get_attribute("html", "data-pyrit-login", timeout=1000)
                except Exception:  # noqa: BLE001
                    sig = None
                if sig == "done":
                    state["done"] = True
                elif sig == "cancel":
                    state["cancelled"] = True

            saved = False
            if state["done"]:
                try:
                    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
                    context.storage_state(path=storage_path)
                    saved = True
                except Exception as e:  # noqa: BLE001
                    try:
                        browser.close()
                    except Exception:  # noqa: BLE001
                        pass
                    return {"ok": False, "error": f"Could not save session: {type(e).__name__}: {e}"}
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Login failed: {type(e).__name__}: {e}"}

    if state["cancelled"]:
        return {"ok": False, "error": "Login cancelled."}
    if not saved:
        return {"ok": False, "error": "Login window closed before the session was saved."}
    return {"ok": True, "path": storage_path}
