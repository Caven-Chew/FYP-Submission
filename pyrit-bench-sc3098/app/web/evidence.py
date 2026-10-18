"""Plain-language evidence helpers shared by the run, compare and export views.

Students clearing Checklist D need three things from a run: which test cases
failed, the exact replies, and a fair before/after pairing. These helpers keep
those rules in one place so every page and export says the same thing.
"""
from __future__ import annotations

import re
import json

from ..models import PromptStatus
from ..settings import SUCCESS_SEVERITY

FAILURE = "Failure found"
NO_FAILURE = "No failure"
NOT_MEASURED = "Not measured"
STUDENT_SLUGS = {"custom_prompt_list", "custom_conversation"}

# Characters XML 1.0 forbids (C0 controls other than tab/newline/CR, and the
# U+FFFE/U+FFFF non-characters). python-docx/lxml raise ValueError on them, so a
# reply containing NUL or an ANSI escape turned the Word export into an HTTP 500.
_XML_UNSAFE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def xml_safe(text) -> str:
    """Text safe to write into a Word (.docx) document (review finding #28)."""
    return _XML_UNSAFE.sub("", "" if text is None else str(text))

# Settings that define "the same test". The fake marker is optional evidence and
# may legitimately differ between runs, so it is reported, never blocking.
_SAME_TEST_FIELDS = {
    "CHECKLIST_ID": "Checklist item",
    "CUSTOM_RUBRIC": "judge rubric",
    "FAILURE_CRITERION": "failure rule",
    "REFERENCE_FACTS": "reference facts",
    "ASSESSMENT_RULES": "target rules and scope",
}


def threshold_of(bench) -> int:
    """The failure threshold saved with the run at launch, else the current setting (finding #23)."""
    try:
        return int((getattr(bench, "llm_config", None) or {}).get("SUCCESS_SEVERITY_THRESHOLD") or SUCCESS_SEVERITY)
    except (TypeError, ValueError):
        return SUCCESS_SEVERITY


def verdict_of(row) -> str:
    """One plain verdict per test case. Unscored rows are never counted as passes."""
    if row is None or row.severity is None or row.status not in (PromptStatus.OK, PromptStatus.RECOVERED):
        return NOT_MEASURED
    return FAILURE if row.severity >= threshold_of(getattr(row, "benchmark", None)) else NO_FAILURE


def judge_record() -> dict:
    """Judge, method and threshold in force now; both launch paths save them with the run (finding #23).

    JUDGE_METHOD comes from the app version that launched the run ("j0_severity" under /v1)."""
    from flask import current_app
    from ..engine import llms
    from ..engine.scoring import SC3098_METHOD
    provider = (llms.role_setting("JUDGE", "PROVIDER") or "").strip().lower()
    model = llms.role_setting("JUDGE", "MODEL") or getattr(llms, "_ROLE_DEFAULT_MODEL", {}).get(provider, "")
    return {"JUDGE_PROVIDER": provider, "JUDGE_MODEL": model, "SUCCESS_SEVERITY_THRESHOLD": SUCCESS_SEVERITY,
            "JUDGE_METHOD": current_app.config.get("JUDGE_METHOD") or SC3098_METHOD,
            **({"JUDGE_SCOPE": "all"} if current_app.config.get("JUDGE_SCOPE") == "all" else {})}


def _judge_fields(bench):
    from ..engine.scoring import judge_method
    cfg = bench.llm_config or {}
    override = cfg.get("JUDGE") if isinstance(cfg.get("JUDGE"), dict) else {}
    provider = cfg.get("JUDGE_PROVIDER") or override.get("provider") or ""
    model = cfg.get("JUDGE_MODEL") or override.get("model") or ""
    threshold = cfg.get("SUCCESS_SEVERITY_THRESHOLD")
    return provider, model, threshold, judge_method(bench)


def judge_label(bench) -> str:
    """'Judge: provider/model · Failure found at score N or more', as recorded at launch.

    A Version 1 (/v1) run also names its 1-5 severity judge."""
    from ..engine.scoring import J0_METHOD, V2_METHOD, DUAL_METHOD
    provider, model, threshold, method = _judge_fields(bench)
    judge = "/".join(x for x in (provider, model) if x) or "not recorded (run launched before judges were recorded)"
    if method == J0_METHOD:
        judge += " · 1–5 severity judge (J0 prompt, v1)"
    elif method == DUAL_METHOD:
        judge += " · both v1 (original harm severity) and v2 (five criteria; highest score)"
    elif method == V2_METHOD:
        judge += " · five-criterion judge (v2; highest score)"
    limit = (f"{threshold} or more" if threshold is not None
             else f"{SUCCESS_SEVERITY} or more (current setting; not recorded with this run)")
    return f"Judge: {judge} · Failure found at score {limit}"


def judge_note(before, after) -> str | None:
    """Non-blocking note when the two runs recorded different judges or thresholds.

    A different judge method also blocks the pair (see differences)."""
    sides = [_judge_fields(b) for b in (before, after)]
    if any(not provider or threshold is None for provider, _model, threshold, _method in sides):
        return None  # nothing recorded for an older run, so there is nothing to compare
    if sides[0] != sides[1]:
        return (f"Before and after were scored differently ({judge_label(before)}; {judge_label(after)}). "
                "A different judge or threshold can score the same reply differently: use the same "
                "judge for both runs.")
    return None


def judge_problem(bench) -> dict | None:
    """Run-level banner data when any test case has a judge error (finding #19)."""
    rows = [p for p in case_rows(bench) if p.status == PromptStatus.JUDGE_ERROR]
    if not rows:
        return None
    messages, hints = [], []
    from ..engine.scoring import JUDGE_ERROR_HINT, judge_error_hint
    for p in rows:
        stored = (p.verdict or p.error or "").strip()
        text = mask_keys((stored.splitlines() or [""])[0][:300])
        if text and text not in messages:
            messages.append(text)
        hint = judge_error_hint(stored)
        if hint and hint not in hints:
            hints.append(hint)
    return {"count": len(rows), "messages": messages[:3],
            "hint": " ".join(hints) or JUDGE_ERROR_HINT}


def mask_keys(text, keep: str | None = None) -> str:
    """Hide the middle of key-shaped strings in replies; the planted marker stays visible (finding #37)."""
    from .redact import mask_key_like
    plain = "" if text is None else str(text)
    # Dual results contain escaped provider text. Decode string values before
    # masking, then preserve the envelope's score types and audit metadata.
    try:
        payload = json.loads(plain)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "judge_version_results" in payload:
        def mask(value):
            if isinstance(value, dict): return {key: mask(item) for key, item in value.items()}
            if isinstance(value, list): return [mask(item) for item in value]
            if isinstance(value, str): return mask_key_like(value, keep=[keep] if keep else []) or ""
            return value
        return json.dumps(mask(payload), ensure_ascii=False)
    return mask_key_like(plain, keep=[keep] if keep else []) or ""


def verdict_class(verdict: str) -> str:
    return {FAILURE: "bad", NO_FAILURE: "ok"}.get(verdict, "none")


def evidence_rows(bench) -> list:
    return sorted(bench.prompts, key=lambda p: (p.attack_slug, p.sequence or 0, p.attempt or 0, p.id or 0))


def is_case_row(row) -> bool:
    """A student test case, not a run-level setup failure.

    Setup-error rows carry no prompt and reuse sequence 0, so pairing them by
    (attack, sequence, attempt) hid test case 1 and counted a phantom case
    (review finding #25). They are reported once by run_problems() instead.
    """
    return (row.attack_slug in STUDENT_SLUGS and row.status != PromptStatus.SETUP_ERROR
            and bool((row.prompt_text or "").strip()))


def case_rows(bench) -> list:
    """The run's student test cases in evidence order (setup errors excluded)."""
    return [p for p in evidence_rows(bench) if is_case_row(p)]


def run_problems(bench) -> list[str]:
    """Run-level setup failures of the student attacks, each reported once."""
    out = []
    for p in evidence_rows(bench):
        if p.attack_slug in STUDENT_SLUGS and not is_case_row(p):
            # Masked before trimming, so a key cut at the limit cannot slip through half-masked.
            message = (mask_keys(p.error) or f"setup failed ({p.status})").splitlines()[0][:300]
            if message not in out:
                out.append(message)
    return out


def is_student_run(bench) -> bool:
    return any(slug in {"custom_prompt_list", "custom_conversation"} for slug in bench.attack_slugs)


def ordered_pair(a, b, swap: bool = False):
    """Older run first ("before"), unless the student asks to swap."""
    before, after = sorted((a, b), key=lambda x: (x.created_at, x.id))
    return (after, before) if swap else (before, after)


def differences(before, after) -> list[str]:
    """Why two runs are not the same test (empty list = comparable)."""
    out = []
    for key, label in _SAME_TEST_FIELDS.items():
        if (before.llm_config.get(key) or "") != (after.llm_config.get(key) or ""):
            out.append(f"the {label} differs")
    if (before.llm_config.get("JUDGE_SCOPE") or "") != (after.llm_config.get("JUDGE_SCOPE") or ""):
        out.append("the judge scope differs")
    if _judge_fields(before)[3] != _judge_fields(after)[3]:
        out.append("the judge differs (one run was scored in Version 1 (/v1) by the 1–5 severity judge)")
    if before.custom_prompts != after.custom_prompts or before.custom_conversations != after.custom_conversations:
        out.append("the prompts differ (a pack or your pasted prompts changed between runs)")
    if before.id == after.id:
        out.append("both sides are the same run")
    return out


def target_name(bench) -> str:
    return getattr(bench.target, "name", None) or f"target {bench.target_id}"


def target_note(before, after) -> str | None:
    """Non-blocking note when the two runs used different saved targets (finding #62).

    Not a difference: a student may legitimately re-save the fixed bot as a new
    target, so this asks them to confirm instead of refusing the comparison.
    """
    if before.target_id != after.target_id:
        return (f"Before and after used different targets ({target_name(before)}, "
                f"{target_name(after)}); confirm they are the same bot.")
    return None


def marker_note(before, after) -> str | None:
    b, a = before.llm_config.get("CANARY") or "", after.llm_config.get("CANARY") or ""
    if b != a:
        return ("The fake-marker setting differs between the runs, so the exact-marker column "
                "is not comparable. The judge's scores are unaffected.")
    return None


def reply_text(row) -> str:
    """The evidence a reader needs: every turn of a conversation, or the reply."""
    if row is None:
        return ""
    if row.transcript:
        return "\n\n".join(f"Turn {t['turn']}\nUser: {t['prompt']}\nBot: {t['response'] or '(no reply)'}"
                           for t in row.transcript)
    return row.response or ""


def is_conversation(row) -> bool:
    return row is not None and (row.attack_slug == "custom_conversation" or bool(row.transcript))


def scripted_turns(row) -> str:
    """The user turns of a scripted conversation, numbered (finding #31)."""
    turns = [t["prompt"] for t in row.transcript] if row.transcript else []
    scripted = (row.prompt_text or "").split("\n---\n")
    if len(scripted) > len(turns):
        turns = scripted  # a cut-off conversation still shows every scripted turn
    return "\n".join(f"Turn {n}: {text}" for n, text in enumerate(turns, 1))


def bot_reply(row, keep: str | None = None) -> str:
    """Only the bot's side: the reply, or each conversation turn's reply. Key-like strings masked."""
    if row is None:
        return ""
    if row.transcript:
        return "\n\n".join(f"Turn {t['turn']}: {mask_keys(t.get('response'), keep) or '(no reply)'}"
                           for t in row.transcript)
    return mask_keys(row.response, keep)


def missing_reason(row) -> str:
    if row is None:
        return "No matching test case in this run."
    if (row.response or "").strip() and row.status == PromptStatus.JUDGE_ERROR:
        return "The reply was saved, but the judge gave no valid score (judge error)."
    if row.status == PromptStatus.SCORING:
        return "Reply captured; scoring pending."
    if (row.response or "").strip() and row.status == PromptStatus.INCOMPLETE:
        return "The conversation did not finish, so it was not scored."
    if row.error:
        return f"No reply recorded ({row.status}): {mask_keys(row.error).splitlines()[0][:200]}"
    return f"No reply recorded ({row.status})."


def paired_cases(before, after) -> list[dict]:
    """Keep every recorded case, including either side of an interrupted run."""
    def indexed(bench):
        return {(p.attack_slug, p.sequence, p.attempt or 0): p for p in case_rows(bench)}
    before_rows, after_rows = indexed(before), indexed(after)
    cases = []
    for key in sorted(before_rows.keys() | after_rows.keys(), key=lambda k: (k[0], k[1] or 0, k[2])):
        p, a = before_rows.get(key), after_rows.get(key)
        cases.append({"prompt": (p if p is not None else a).prompt_text or "", "before": p, "after": a,
                      "before_verdict": verdict_of(p), "after_verdict": verdict_of(a)})
    return cases


def summary(bench) -> dict:
    """Counts of plain verdicts over a run's student test cases.

    Setup errors are not test cases; they are counted separately as
    ``setup_problems`` (see run_problems)."""
    rows = case_rows(bench)
    counts = {FAILURE: 0, NO_FAILURE: 0, NOT_MEASURED: 0}
    for p in rows:
        counts[verdict_of(p)] += 1
    return {"total": len(rows), **counts, "setup_problems": len(run_problems(bench))}
