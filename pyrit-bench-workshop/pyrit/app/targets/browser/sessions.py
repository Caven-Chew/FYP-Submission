"""Finding a target's saved login session (and local test page) on *this* machine.

A UI target stores an absolute path to its saved Playwright session, and often an
absolute `file://` URL for a local test page. Both are written on the machine the
target was created on, so copying the project to another computer — or just to
another drive — would silently break every logged-in target: the run would start
signed out and record a login wall as the model's answer.

Both helpers below are fallbacks first. If the stored path exists, it is returned
unchanged and nothing here has any effect; only when it is missing do they look
for the same filename in this install's own folders. That keeps the project
portable in either direction (Windows <-> macOS <-> Linux) without rewriting the
database, and leaves behaviour on the original machine exactly as it was.

If the fallback also comes up empty they **raise**, rather than quietly carrying
on. A target configured with a saved login that then runs signed out does not
fail — it records the site's login wall as the model's answer, so the benchmark
fills up with rows that look like results and are not. A loud error you can fix
in a minute is worth more than a silent run you have to notice, distrust, and
throw away.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from ...config import ROOT

# Where "Log in & save session" writes to (see web/views_targets.py).
SESSIONS_DIR = ROOT / "instance" / "ui_sessions"


class SavedSessionMissing(RuntimeError):
    """A target names a saved login session that isn't on this machine."""


class LocalPageMissing(RuntimeError):
    """A target names a local file:// page that isn't on this machine."""


def _basename(raw: str) -> str:
    """Last path segment, whichever slash the path was written with.

    `Path("D:\\\\x\\\\y.json").name` is the whole string on POSIX, so normalise
    separators by hand rather than trusting the local flavour of Path.
    """
    return raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_session_path(stored: str | None, *, required: bool = True) -> str | None:
    """Return a usable path to a saved login session.

    Order: the stored path if it exists, else the same filename under this
    install's `instance/ui_sessions/`. Returns None when no session was ever
    configured (an empty value) — that target is simply meant to run signed out.

    Raises `SavedSessionMissing` when a session *was* configured but cannot be
    found either way, so the run stops with an explanation instead of silently
    hitting a login wall. Pass `required=False` to get None instead.
    """
    raw = (stored or "").strip()
    if not raw:
        return None
    if os.path.exists(raw):
        return raw
    name = _basename(raw)
    local = SESSIONS_DIR / name
    if local.exists():
        return str(local)
    if not required:
        return None
    raise SavedSessionMissing(
        f"This target uses a saved login session, but the file is not on this "
        f"machine:\n    {raw}\n"
        f"and no file named '{name}' exists in {SESSIONS_DIR}.\n\n"
        f"If you copied this project from another computer, copy that computer's "
        f"instance/ui_sessions/ folder across as well. Otherwise re-record the "
        f"login with 'Log in & save session' on the Targets page, or clear the "
        f"target's ui_storage_state to run it signed out."
    )


def resolve_local_url(url: str | None) -> str:
    """Re-point a `file://` URL at this machine's copy of the project.

    Only `file://` URLs are touched, and only when the file they name is absent
    (a page saved as `file:///D:/...` on Windows, opened on a Mac). The same
    filename is looked for at the project root and in `test/`, which is where
    the bundled mock chat pages live. Anything else — https, http, an existing
    local file — is returned unchanged.

    Raises `LocalPageMissing` if the page cannot be found either way, rather
    than letting the browser navigate to nothing and time out cryptically.
    """
    raw = (url or "").strip()
    if not raw.lower().startswith("file:"):
        return raw

    path = unquote(urlparse(raw).path)
    # A Windows file URL parsed anywhere gives "/D:/proj/page.html"; drop the
    # leading slash so the drive letter isn't read as a directory name.
    if re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    if Path(path).exists():
        return raw

    name = _basename(path)
    for candidate in (ROOT / name, ROOT / "test" / name):
        if candidate.exists():
            return candidate.as_uri()
    raise LocalPageMissing(
        f"This target points at a local page that is not on this machine:\n"
        f"    {raw}\n"
        f"and no file named '{name}' was found in {ROOT} or {ROOT / 'test'}.\n\n"
        f"If you copied this project from another computer, the page may have "
        f"been outside the project folder — copy it in, or edit the target's "
        f"ui_url to point at where the page lives here."
    )
