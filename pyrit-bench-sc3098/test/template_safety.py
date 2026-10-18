"""Check public template text against the application's canonical key patterns.

No environment files are opened and no configured credential values are read.
"""
from app.web.redact import KEY_PATTERNS, UNQUOTED_ASSIGN


def key_pattern_labels(text: str) -> list[str]:
    """Return matched pattern labels without returning any matched values."""
    return [label for label, pattern in [*KEY_PATTERNS, UNQUOTED_ASSIGN]
            if pattern.search(text)]
