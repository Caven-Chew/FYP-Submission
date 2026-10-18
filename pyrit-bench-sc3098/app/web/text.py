"""Small text helpers shared by the HTML/PDF renderers."""
from __future__ import annotations


def shorten(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def break_long_words(text: str, chunk: int = 25) -> str:
    """Insert a space every `chunk` chars inside tokens with no natural break point.

    ReportLab only breaks on spaces (and hyphens); for Base64/encoded blobs
    that have no whitespace we must inject break opportunities manually.
    """
    def _break(token: str) -> str:
        if len(token) <= chunk:
            return token
        return " ".join(token[i:i+chunk] for i in range(0, len(token), chunk))
    return " ".join(_break(tok) for tok in text.split())
