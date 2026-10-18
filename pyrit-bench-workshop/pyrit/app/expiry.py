"""Workshop service expiry guard.

The HPC LLM service is only available for the 7 September 2026 workshop.
This is intentionally a code-level cutoff rather than an environment setting,
so a copied `.env` cannot keep the app running after the service is retired.
"""
from __future__ import annotations

from datetime import date
import hashlib
import hmac
import os
from zoneinfo import ZoneInfo


WORKSHOP_LAST_DATE = date(2026, 9, 7)
WORKSHOP_TIMEZONE = ZoneInfo("Asia/Singapore")
_UNLOCK_CODE_SHA256 = "50f87d0665459ff3e0d85c7f2fe4207fba639abee68d1edfd1639584095a9517"


def _has_unlock_code() -> bool:
    """Check the facilitator-only post-expiry unlock without logging the code."""
    candidate = os.getenv("FYP_WORKSHOP_UNLOCK_CODE", "").strip().encode()
    if not candidate:
        return False
    digest = hashlib.sha256(candidate).hexdigest()
    return hmac.compare_digest(digest, _UNLOCK_CODE_SHA256)


def workshop_is_expired(today: date | None = None) -> bool:
    """Return whether the workshop app must refuse to start."""
    if _has_unlock_code():
        return False
    if today is None:
        from datetime import datetime

        today = datetime.now(WORKSHOP_TIMEZONE).date()
    return today > WORKSHOP_LAST_DATE


def ensure_workshop_active(today: date | None = None) -> None:
    """Raise a clear startup error after the cutoff unless unlocked."""
    if workshop_is_expired(today):
        raise RuntimeError(
            "This FYP workshop application expired after 7 September 2026 "
            "because its HPC LLM service is no longer available. Set the "
            "facilitator-only FYP_WORKSHOP_UNLOCK_CODE to continue."
        )
