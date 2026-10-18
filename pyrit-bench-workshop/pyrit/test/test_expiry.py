"""Regression checks for the hard workshop cutoff."""
from datetime import date
import hashlib
import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "workshop_expiry", Path(__file__).resolve().parents[1] / "app" / "expiry.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
WORKSHOP_LAST_DATE = _MODULE.WORKSHOP_LAST_DATE
ensure_workshop_active = _MODULE.ensure_workshop_active
workshop_is_expired = _MODULE.workshop_is_expired


def main() -> int:
    os.environ.pop("FYP_WORKSHOP_UNLOCK_CODE", None)
    assert not workshop_is_expired(WORKSHOP_LAST_DATE)
    assert workshop_is_expired(date(2026, 9, 8))
    try:
        ensure_workshop_active(date(2026, 9, 8))
    except RuntimeError as exc:
        assert "expired" in str(exc).lower()
    else:
        raise AssertionError("The app must refuse to start after the cutoff")
    os.environ["FYP_WORKSHOP_UNLOCK_CODE"] = "wrong-code"
    assert workshop_is_expired(date(2026, 9, 8))
    # Exercise the valid-code branch with a test-only hash; never place the
    # real facilitator code in the repository.
    _MODULE._UNLOCK_CODE_SHA256 = hashlib.sha256(b"test-only-code").hexdigest()
    os.environ["FYP_WORKSHOP_UNLOCK_CODE"] = "test-only-code"
    assert not workshop_is_expired(date(2026, 9, 8))
    ensure_workshop_active(date(2026, 9, 8))
    print("Expiry check passed: blocked after cutoff without the correct unlock code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
