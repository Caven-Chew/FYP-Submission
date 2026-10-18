"""Screenshot diagnosis for a UI target whose reply never arrived (diagnostic only)."""
from __future__ import annotations

import os

from ...prompts import UI_TIMEOUT_DIAGNOSIS_PROMPT


# ---------- screenshot diagnosis on a timed-out UI response ---------------

# Reuses the judge LLM's own credentials (JUDGE_PROVIDER / JUDGE_API_KEY_ENV —
# see engine/llms.py:build_role_llm) rather than adding a third set of env vars,
# since a vision-capable judge model is the credential this install already
# has on hand. JUDGE_VISION_MODEL overrides the model only (the provider/key
# still come from JUDGE_PROVIDER/JUDGE_API_KEY_ENV).
_VISION_DEFAULT_MODEL = {"openai": "gpt-4o-mini", "anthropic": "claude-3-5-sonnet-latest"}


_VISION_DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# Cap on how much of the page's visible text we forward — enough to cover a
# real reply the selector missed, small enough not to blow past a cheap
# model's context or bury the signal in unrelated page chrome.
_TEXT_EXTRACT_CHARS = 6000


def _norm(s: str) -> str:
    """Collapse whitespace for containment checks.

    The screenshot path and the DOM disagree about line breaks and non-breaking
    spaces constantly, so an exact `in` test would reject genuine extractions.
    """
    return " ".join((s or "").split()).lower()


def _verify_extraction(reply_text: str, page_text: str) -> tuple[str, str | None]:
    """Accept an extracted reply only if it really appears in the page text.

    This is the guard that makes extraction safe to use as a benchmark result.
    A vision model asked "what did the assistant say?" will, on a page that is
    still generating, sometimes produce a fluent, plausible answer that was
    never on screen. Recording that as the target's response would mean
    scoring the judge's severity against text the target never produced —
    fabricated evidence in a security benchmark, and undetectable after the
    fact because it reads perfectly.

    So the model's output is treated as a CLAIM about the page, checked
    against the page text we already hold. Returns (accepted_text, reason_rejected).
    """
    if not reply_text:
        return "", None
    if len(reply_text.strip()) < 2:
        return "", "extracted text too short to be a reply"
    hay, needle = _norm(page_text), _norm(reply_text)
    if not needle:
        return "", "extracted text was blank after normalisation"
    if needle not in hay:
        return "", ("extracted text does not appear in the page — treating it as "
                    "a fabrication and discarding it")
    return reply_text.strip(), None


def _parse_diagnosis(raw: str) -> dict:
    """Parse the model's JSON, tolerating code fences and stray prose."""
    import json
    import re

    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except Exception:  # noqa: BLE001 — model ignored the format; keep its prose
        return {"state": "unknown", "reply_text": "", "explanation": (raw or "").strip()}
    return {
        "state": str(data.get("state") or "unknown"),
        "reply_text": str(data.get("reply_text") or ""),
        "explanation": str(data.get("explanation") or "").strip(),
    }


def _finish(raw: str, page_text: str) -> dict:
    """Turn the model's raw output into the verified result dict."""
    parsed = _parse_diagnosis(raw)
    accepted, rejected = _verify_extraction(parsed["reply_text"], page_text)
    # A model that claims a reply is present but whose quote doesn't survive
    # verification has not established that a reply is present.
    state = parsed["state"]
    if state == "reply_present" and not accepted:
        state = "unknown"
    summary = parsed["explanation"] or "(no explanation returned)"
    label = {
        "reply_present": "A reply is on the page but the selector missed it.",
        "still_generating": "The reply had not finished generating yet.",
        "blocked": "Something on the page is blocking the reply.",
        "empty": "No reply, and no sign one was coming.",
    }.get(state)
    if label:
        summary = f"{label} {summary}"
    if rejected:
        summary = f"{summary} (Discarded a proposed reply: {rejected}.)"
    return {"summary": summary, "state": state, "reply_text": accepted, "rejected": rejected}


async def diagnose_ui_timeout(screenshot_png: bytes, page_text: str, response_sel: str, waited_ms: int) -> dict:
    """Ask a vision-capable LLM to recover the reply, or say why there isn't one.

    Returns a dict:
      summary    human-readable line for the UI (always present)
      state      reply_present | still_generating | blocked | empty | unknown
      reply_text the assistant's reply, VERIFIED to appear in page_text (else "")
      rejected   why an extraction was thrown away, if it was

    Sends both a full-page screenshot (catches purely visual failures a text
    dump can't: CAPTCHA, rate-limit banners, cookie modals, broken layout,
    blank/stuck pages) and a plain-text extract of the page body (catches the
    other common failure — the reply is there, verbatim, but `response_sel`
    just didn't match it — without the model having to re-OCR it from
    pixels, and without missing content that scrolled below the screenshot).
    One LLM call covers both; each modality substitutes for the other's blind
    spot.

    Best-effort debugging aid, not a fallback result: this never raises and
    never changes the benchmark outcome. Returns a short plain-English
    string, including one explaining why the check was skipped (e.g. no
    judge credentials configured) or failed.
    """
    import base64
    import httpx

    def _skip(msg: str) -> dict:
        return {"summary": msg, "state": "unknown", "reply_text": "", "rejected": None}

    provider = (os.getenv("JUDGE_PROVIDER") or "").strip().lower()
    if provider not in _VISION_DEFAULT_MODEL:
        return _skip("(skipped — set JUDGE_PROVIDER=openai or anthropic, plus JUDGE_API_KEY_ENV, "
                     "to enable screenshot diagnosis on timeout)")
    key_env = os.getenv("JUDGE_API_KEY_ENV") or _VISION_DEFAULT_KEY_ENV[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        return _skip(f"(skipped — {key_env} is not set)")

    model = os.getenv("JUDGE_VISION_MODEL") or _VISION_DEFAULT_MODEL[provider]
    b64 = base64.b64encode(screenshot_png).decode()
    text_extract = page_text[:_TEXT_EXTRACT_CHARS]
    question = UI_TIMEOUT_DIAGNOSIS_PROMPT.format(
        waited_ms=waited_ms, response_sel=response_sel, page_text=text_extract,
    )

    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False, mounts={}) as c:
            if provider == "anthropic":
                r = await c.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={
                        "model": model, "max_tokens": 300,
                        "messages": [{"role": "user", "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": question},
                        ]}],
                    },
                )
                r.raise_for_status()
                return _finish(r.json()["content"][0]["text"], page_text)
            # provider == "openai"
            r = await c.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model, "max_tokens": 300,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]}],
                },
            )
            r.raise_for_status()
            return _finish(r.json()["choices"][0]["message"]["content"], page_text)
    except Exception as e:  # noqa: BLE001 — diagnostic only, must never mask the real TimeoutError
        return _skip(f"(screenshot diagnosis failed: {type(e).__name__}: {e})")
