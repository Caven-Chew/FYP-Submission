"""Shared target helpers: PyRIT memory bootstrap and small response/config shims."""
from __future__ import annotations

import json
from typing import Any


# ---------- PyRIT one-time initialization --------------------------------

_pyrit_initialized = False


def ensure_pyrit_initialized() -> None:
    """Initialize PyRIT's CentralMemory once per process.

    Tries three initialization paths in order, matching the API of whichever
    PyRIT build is installed:

    Path 1 — current builds (no initialize_pyrit):
        CentralMemory.set_memory_instance(SQLiteMemory())
        SQLiteMemory is a Singleton, so repeated calls return the same instance.
        File-backed (default ~/.pyrit/pyrit.db) so background threads share it.

    Path 2 — older builds that have initialize_pyrit:
        initialize_pyrit(memory_db_type=DUCK_DB / "DuckDB")

    Path 3 — very old builds:
        initialize_pyrit() with no args
    """
    global _pyrit_initialized
    if _pyrit_initialized:
        return

    # --- Path 1: current PyRIT (CentralMemory.set_memory_instance) -----------
    try:
        from pyrit.memory import CentralMemory, SQLiteMemory  # type: ignore
        try:
            CentralMemory.get_memory_instance()
        except ValueError:
            CentralMemory.set_memory_instance(SQLiteMemory())
        _pyrit_initialized = True
        return
    except (ImportError, AttributeError):
        pass

    # --- Path 2: initialize_pyrit with DuckDB --------------------------------
    try:
        from pyrit.common import initialize_pyrit  # type: ignore
        try:
            from pyrit.common import DUCK_DB  # type: ignore
            initialize_pyrit(memory_db_type=DUCK_DB)
        except (ImportError, AttributeError):
            try:
                initialize_pyrit(memory_db_type="DuckDB")
            except TypeError:
                initialize_pyrit()
        _pyrit_initialized = True
        return
    except (ImportError, AttributeError):
        pass

    _pyrit_initialized = True


def wrap_response(text: str) -> list:
    """Build a one-message list[Message] response — the return type expected by the new PyRIT API."""
    from pyrit.models import Message  # type: ignore
    return [Message.from_prompt(prompt=text, role="assistant")]


def extra_of(t: Any) -> dict:
    """Return a Target row's extra-config dict without importing the model.

    Uses the Target.extra property when present; otherwise parses extra_json.
    Keeps this module free of any SQLAlchemy / Flask import.
    """
    extra = getattr(t, "extra", None)
    if isinstance(extra, dict):
        return extra
    return json.loads(getattr(t, "extra_json", None) or "{}")
