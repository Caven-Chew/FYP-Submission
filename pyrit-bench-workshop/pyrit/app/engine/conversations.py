"""Reading conversations back out of PyRIT memory, and judging attacker quality."""
from __future__ import annotations

from typing import Any

from .scoring import (
    _MIN_ATTACK_PROMPT_CHARS, looks_like_refusal, looks_like_soft_refusal
)


# ---------- attack conversation extraction -------------------------------

def fetch_conversation(result: Any) -> list[Any]:
    """Pull the *target* conversation: the attack prompts sent to the target and
    the target's replies — this is what the judge must score.

    In PyRIT's executor API, ``AttackResult.conversation_id`` is the
    objective-target conversation (``context.session.conversation_id``). The
    ADVERSARIAL entry in ``related_conversations`` is the *attacker model's own
    chat* (the prompts it drafts); scoring that would measure the attacker, not
    the target — so we deliberately do NOT use it here (see
    ``fetch_adversarial_conversation`` for that).

    Falls back to legacy attribute names for older builds.
    """
    from pyrit.memory import CentralMemory  # type: ignore

    # ── Primary: the objective-target conversation (the target's replies) ────
    conv_id = getattr(result, "conversation_id", None)
    if conv_id:
        try:
            mem = CentralMemory.get_memory_instance()
            pieces = mem.get_conversation(conversation_id=conv_id)
            if pieces:
                return list(pieces)
        except Exception:  # noqa: BLE001
            pass

    # ── Legacy fallbacks (older PyRIT builds) ────────────────────────────────
    for attr in ("conversation", "messages", "pieces"):
        val = getattr(result, attr, None)
        if val:
            return list(val)

    return []


def fetch_adversarial_conversation(result: Any) -> list[Any]:
    """Pull the adversarial-chat conversation — the attacker model's own outputs.

    Used only to detect *attacker refusals* (the adversarial model declining to
    craft an attack prompt). Those turns are not evidence about the target, so we
    flag them separately instead of scoring them as target robustness.
    """
    related = getattr(result, "related_conversations", None)
    if not related:
        return []
    try:
        from pyrit.memory import CentralMemory  # type: ignore
        from pyrit.models.conversation_reference import ConversationType  # type: ignore
        mem = CentralMemory.get_memory_instance()
        for ref in related:
            if getattr(ref, "conversation_type", None) != ConversationType.ADVERSARIAL:
                continue
            conv_id = getattr(ref, "conversation_id", None)
            if not conv_id:
                continue
            try:
                pieces = mem.get_conversation(conversation_id=conv_id)
                if pieces:
                    return list(pieces)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return []


def attacker_outputs(adv_conv: list[Any]) -> list[str]:
    """The attacker model's own assistant-role turns, non-empty, as strings."""
    outputs = [
        (getattr(m, "converted_value", None) or getattr(m, "original_value", "") or "")
        for m in adv_conv
        if getattr(m, "role", "") == "assistant"
    ]
    return [o for o in outputs if o.strip()]


def attacker_refused(adv_conv: list[Any]) -> bool:
    """True if the adversarial (attacker) model declined to produce an attack.

    If every non-empty attacker output reads as a refusal, treat the run as an
    attacker refusal rather than as target robustness. Returns False when the
    attacker produced no output (ambiguous — we don't over-claim).
    """
    outputs = attacker_outputs(adv_conv)
    if not outputs:
        return False
    return all(looks_like_refusal(o) for o in outputs)


def attacker_looks_weak(adv_conv: list[Any]) -> str | None:
    """Detect *soft* attacker self-sabotage; returns a reason, or None.

    A safety-trained attacker LLM frequently complies on the surface while
    producing a prompt that isn't actually adversarial: a hedge, a meta-comment,
    a one-liner, or an offer to do something else. The target then refuses
    trivially and the objective is recorded as target robustness — an invalid
    conclusion, because the attack was never really attempted.

    This is a heuristic *diagnostic* only. It never changes a score; it is
    surfaced on the benchmark so a low ASR caused by a reluctant attacker can be
    distinguished from a low ASR caused by a genuinely robust target.
    """
    outputs = attacker_outputs(adv_conv)
    if not outputs:
        return None
    if all(len(o.strip()) < _MIN_ATTACK_PROMPT_CHARS for o in outputs):
        return (
            f"every attacker turn was under {_MIN_ATTACK_PROMPT_CHARS} characters "
            "— no real attack prompt was produced"
        )
    hedged = sum(
        1 for o in outputs if looks_like_refusal(o) or looks_like_soft_refusal(o)
    )
    if hedged * 2 >= len(outputs):
        return (
            f"{hedged} of {len(outputs)} attacker turns read as refusals or hedging "
            "— the adversarial model is likely self-censoring"
        )
    return None
