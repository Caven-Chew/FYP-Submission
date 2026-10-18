"""Browser engine plumbing: Patchright import, availability checks, pre-action parsing."""
from __future__ import annotations

import json


# ---------- browser engine selection (stealth) ---------------------------
# Browser UI targets REQUIRE Patchright — a drop-in Playwright replacement that
# strips the automation tells bot-detection services (Cloudflare, DataDome,
# Google's "this browser may not be secure") look for: the CDP Runtime.enable
# leak, navigator.webdriver, and the --enable-automation flag. We deliberately
# do NOT fall back to stock Playwright, so a UI target never silently runs the
# weaker (easily-detected) engine against a protected site. Install with:
#   pip install patchright && patchright install chromium

ENGINE_MISSING_MSG = (
    "Patchright is required for Browser UI targets — it evades the bot detection "
    "(Cloudflare / DataDome / Google) that stock automation can't. Install it with:\n"
    "  pip install patchright\n"
    "  patchright install chromium"
)


def load_sync_playwright():
    """Return (sync_playwright factory, "patchright"). Raises ImportError if absent."""
    try:
        from patchright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(ENGINE_MISSING_MSG) from e
    return sync_playwright, "patchright"


def load_async_playwright():
    """Return (async_playwright factory, "patchright"). Raises ImportError if absent."""
    try:
        from patchright.async_api import async_playwright
    except ImportError as e:
        raise ImportError(ENGINE_MISSING_MSG) from e
    return async_playwright, "patchright"


def engine_available() -> bool:
    """True if Patchright (the required engine) can be imported."""
    try:
        load_async_playwright()
        return True
    except ImportError:
        return False


def install_hint(engine: str) -> str:
    """Browser-install command for the engine."""
    return f"{engine} install chromium"


# ---------- the UI target (Playwright) -----------------------------------

def parse_pre_actions(raw) -> list:
    """Normalise the ``ui_pre_actions`` config into a list of action dicts.

    Pre-actions are run once after the page loads, before each prompt is typed,
    to clear cookie banners / welcome modals that would otherwise sit on top of
    the chat input. Accepts an already-parsed list or a JSON string; anything
    unparseable yields an empty list (pre-actions are always optional).

    Each action is one of:
      {"type": "click", "selector": "<css>"}   click an element (e.g. Accept)
      {"type": "key",   "key": "Enter"}        press a key (Enter, Escape, …)
      {"type": "wait",  "ms": 1000}            pause before the next step
    """
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            return []
    return [a for a in items if isinstance(a, dict) and a.get("type")] \
        if isinstance(items, list) else []
