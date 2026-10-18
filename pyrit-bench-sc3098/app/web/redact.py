"""Mask key-shaped strings before text is shown, exported or stored as an error.

A bot under test can print a real credential (for example when a group plants
its real key in a system prompt to test key leakage), and a judge error can
echo part of a request. ``mask_key_like`` keeps the first and last four
characters of anything that looks like a key and replaces the middle, so a
reader can still recognise which key leaked without the page or export
carrying a usable copy. The configured fake secret marker is passed in
``keep`` so the exact-marker evidence stays readable.

This file holds the canonical key patterns. The offline template checker
in test/template_safety.py imports these same patterns.
"""
from __future__ import annotations

import re
from typing import Iterable

# (label, pattern) pairs shared with the offline configuration-template check.
KEY_PATTERNS = [
    ("sk- key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("sk-ant- key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Bearer token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{24,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}")),
    ("Slack token", re.compile(r"(?<![A-Za-z0-9])xox[bp]-[0-9A-Za-z-]{10,}")),
    ("quoted key assignment", re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|password)['\"]?\s*[:=]\s*['\"](?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{24,}['\"]")),
]
UNQUOTED_ASSIGN = ("unquoted KEY=value assignment", re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z0-9_]*(?:key|token|secret|password)[A-Za-z0-9_]*[ \t]*[=:][ \t]*"
    r"(?=[A-Za-z0-9_\-+/=.]*[0-9])(?=[A-Za-z0-9_\-+/=.]*[A-Za-z])[A-Za-z0-9_\-+/=.]{20,}(?![A-Za-z0-9_\-+/=.(\[])"))

MASK = "****"
# The secret part of a match: the longest run of token characters, so the
# variable name in "API_KEY=..." or the word "Bearer" stays readable.
_TOKEN_RUN = re.compile(r"[A-Za-z0-9_\-+/.]+")


def _mask_middle(value: str) -> str:
    """Keep the first and last four characters ('sk-a' + '****' + 'stuv'); fully mask short values.

    The example is split on purpose: a whole key-shaped literal here stops the release key scan.
    """
    return value[:4] + MASK + value[-4:] if len(value) >= 12 else MASK


def mask_values(text: str | None, values: Iterable[str | None] = ()) -> str | None:
    """Mask every exact occurrence of known secret values (8+ characters).

    Use it for credentials whose format no pattern knows, such as the
    configured judge key. Values are never logged or returned.
    """
    if not text:
        return text
    for value in sorted({v for v in values if v and len(v) >= 8}, key=len, reverse=True):
        text = text.replace(value, _mask_middle(value))
    return text


def mask_key_like(text: str | None, keep: Iterable[str | None] = ()) -> str | None:
    """Mask the middle of every key-shaped string in ``text``.

    ``keep`` lists strings that must stay exactly as written, normally the
    run's fake secret marker: a match that contains a kept string, or lies
    inside one, is left alone. Already-masked text does not match again.
    """
    if not text:
        return text
    kept = [k for k in keep if k]
    spans = []
    for _label, pattern in KEY_PATTERNS + [UNQUOTED_ASSIGN]:
        for match in pattern.finditer(text):
            found = match.group(0)
            if not any(k in found or found in k for k in kept):
                spans.append(list(match.span()))
    if not spans:
        return text
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, last = [], 0
    for start, end in merged:
        part = text[start:end]
        run = max(_TOKEN_RUN.finditer(part), key=lambda m: len(m.group(0)))
        out.append(text[last:start] + part[:run.start()] + _mask_middle(run.group(0)) + part[run.end():])
        last = end
    return "".join(out) + text[last:]
