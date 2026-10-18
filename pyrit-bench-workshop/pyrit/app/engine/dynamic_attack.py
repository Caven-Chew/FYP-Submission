"""Dynamic attacks: multi-turn PyRIT orchestrators driven by an attacker LLM."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..activity import set_activity
from .control import wait_while_paused
from .conversations import (
    attacker_looks_weak, attacker_refused, fetch_adversarial_conversation,
    fetch_conversation
)
from .retry import BenchmarkAborted, ConsecutiveFailureGuard
from .scoring import (
    build_objective_scorer, build_tap_scoring_config, judge_response,
    severity_provenance,
)
from ..models import PromptStatus
from ..settings import REQUEST_DELAY, attempts_for
from ..targets.throttle import throttle_slept_ms


def tap_class(name: str) -> Any:
    """Resolve the TAP or PAIR attack class for the installed PyRIT build.

    Both landed in pyrit.executor.attack; older builds have neither, so fail
    with an actionable message rather than an opaque ImportError.
    """
    try:
        import pyrit.executor.attack as ea  # type: ignore
    except ImportError as exc:  # pragma: no cover - PyRIT always provides this
        raise RuntimeError(f"PyRIT executor API unavailable: {exc}") from exc

    candidates = {
        "tap": ("TAPAttack", "TreeOfAttacksWithPruningAttack"),
        "pair": ("PAIRAttack",),
    }[name]
    for attr in candidates:
        cls = getattr(ea, attr, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        f"This PyRIT build does not provide {' / '.join(candidates)}. "
        f"TAP and PAIR need pyrit>=0.14 — upgrade with: pip install -U pyrit"
    )


async def run_dynamic_attack(
    *, attack: dict, target: Any, attacker: Any, judge: Any, objectives: list[dict],
    guard: "ConsecutiveFailureGuard", benchmark_id: int | None = None,
) -> list[dict]:
    """Run a PyRIT multi-turn attack (RedTeaming / Crescendo).

    Each result dict carries: sequence, prompt_text, response, error,
    succeeded, severity, verdict, latency_ms, taxonomy_category, objective_id.
    If `guard` trips (too many consecutive target failures) this stops iterating
    objectives and returns what was gathered; it never raises. The caller checks
    `guard.tripped` to decide whether to stop the whole benchmark.
    """
    # Prefer the top-level executor.attack exports (stable public API in 0.13+);
    # fall back to the deep module paths for older PyRIT builds.
    try:
        from pyrit.executor.attack import (  # type: ignore
            RedTeamingAttack, CrescendoAttack,
            AttackAdversarialConfig, AttackScoringConfig,
        )
    except ImportError:  # older PyRIT deep-module layout
        from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack  # type: ignore
        from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack    # type: ignore
        from pyrit.executor.attack.core.attack_config import (                     # type: ignore
            AttackAdversarialConfig, AttackScoringConfig,
        )

    name = attack["orchestrator"]
    # Display name for the live activity line (falls back to the orchestrator key).
    atk_name = attack.get("name", name)
    max_turns = int(attack.get("max_turns", 5))
    fallback_tax = attack.get("taxonomy_category")
    results: list[dict] = []

    attempts = attempts_for(attack)
    # Best-of-N: re-run the whole objective set up to `attempts` times and
    # count an objective as reached if ANY attempt lands. Objectives already
    # solved are skipped on later passes, so cost is bounded by the number
    # that keep failing. Every attempt is persisted, so per-attempt ASR and
    # best-of-N ASR can both be reported from the same run.
    solved: set[str] = set()

    for attempt in range(attempts):
        mark = len(results)
        for obj_idx, row in enumerate(objectives):
            # Pause checkpoint: block between objectives while paused. Guarded
            # because benchmark_id is optional on this signature.
            if benchmark_id is not None:
                await wait_while_paused(benchmark_id)
            objective_text = row["objective"]
            objective_id = row.get("id") or f"obj-{obj_idx}"
            if objective_id in solved:
                continue
            tax = row.get("taxonomy_category") or fallback_tax

            # per-row scorer with this objective's success criterion
            objective_scorer = build_objective_scorer(
                judge, true_description=row.get("true_description"),
            )

            adversarial_cfg = AttackAdversarialConfig(target=attacker)
            scoring_cfg = AttackScoringConfig(objective_scorer=objective_scorer)

            try:
                if name == "red_teaming":
                    attack_strategy = RedTeamingAttack(
                        objective_target=target,
                        attack_adversarial_config=adversarial_cfg,
                        attack_scoring_config=scoring_cfg,
                        max_turns=max_turns,
                    )
                elif name == "crescendo":
                    attack_strategy = CrescendoAttack(
                        objective_target=target,
                        attack_adversarial_config=adversarial_cfg,
                        attack_scoring_config=scoring_cfg,
                        max_turns=max_turns,
                    )
                elif name in ("tap", "pair"):
                    # TAP explores several refinement branches per round, scores
                    # them, keeps the best `tree_width` and prunes the rest — so
                    # unlike Crescendo it does not commit to one escalation path
                    # that may be a dead end. PAIR is the same algorithm with the
                    # tree collapsed (branching_factor=1, no on-topic pruning):
                    # parallel single-line refinement streams.
                    cls = tap_class(name)
                    kwargs = dict(
                        objective_target=target,
                        attack_adversarial_config=adversarial_cfg,
                        attack_scoring_config=build_tap_scoring_config(judge),
                        tree_width=int(attack.get("tree_width", 3)),
                        tree_depth=int(attack.get("tree_depth", max_turns)),
                    )
                    if name == "tap":
                        kwargs["branching_factor"] = int(attack.get("branching_factor", 2))
                        kwargs["on_topic_checking_enabled"] = bool(
                            attack.get("on_topic_checking", True)
                        )
                    attack_strategy = cls(**kwargs)
                else:
                    raise ValueError(f"Unknown orchestrator: {name!r}")

                # A multi-turn objective may cross several burst-pause rests;
                # subtract them so the objective's measured time is the target's,
                # not the throttle's (see throttle_slept_ms).
                slept0 = throttle_slept_ms(target)
                t0 = time.perf_counter()
                # execute_async runs the whole multi-turn attack inside PyRIT
                # (generate → send → score, turn after turn) as one opaque call,
                # so the finest phase we can narrate from here is per-objective:
                # which objective, how far through, and how many turns it may take.
                set_activity(
                    benchmark_id, "attacking",
                    f"{atk_name}: objective {obj_idx + 1}/{len(objectives)}"
                    + (f" (attempt {attempt + 1}/{attempts})" if attempts > 1 else "")
                    + f", up to {max_turns} turns",
                )
                result = await attack_strategy.execute_async(objective=objective_text)
                latency = max(0, int((time.perf_counter() - t0) * 1000
                                     - (throttle_slept_ms(target) - slept0)))
            except Exception as exc:  # noqa: BLE001
                # See send_prompt(): surface any BrowserUITarget timeout diagnosis in the
                # response slot instead of leaving the cell blank.
                debug = getattr(target, "last_debug", None)
                diagnosis = debug.get("vision_diagnosis") if isinstance(debug, dict) else None
                # Commentary about the failure, kept out of `response` so that
                # column only ever holds what the target actually said.
                diagnosis_note = f"no response captured — {diagnosis}" if diagnosis else None
                results.append({
                    "sequence": obj_idx, "prompt_text": objective_text, "response": "",
                    "error": f"{type(exc).__name__}: {exc}", "note": diagnosis_note,
                    "succeeded": False, "severity": None, "verdict": "orchestrator-error",
                    "status": PromptStatus.SEND_ERROR,
                    "severity_source": "none", "latency_measured": False,
                    "latency_ms": 0, "taxonomy_category": tax, "objective_id": objective_id,
                    # Screenshot + HTML of the page as it stood when this
                    # objective died, if the target managed to write one.
                    "capture_ref": debug.get("capture_ref") if isinstance(debug, dict) else None,
                })
                # A whole-objective failure is our best per-objective proxy for
                # "could the target even be reached".
                try:
                    guard.record(f"{type(exc).__name__}: {exc}")
                except BenchmarkAborted:
                    # Stop remaining objectives; results so far are still returned.
                    break
                continue

            # Extract conversation turns from CentralMemory via related_conversations.
            conv = fetch_conversation(result)
            # Diagnostic (never affects scoring): did the attacker model actually
            # mount an attack, or did it quietly water the prompt down? Without this
            # a self-censoring attacker is indistinguishable from a robust target.
            weak_reason = attacker_looks_weak(fetch_adversarial_conversation(result))
            if conv:
                guard.record(None)  # target replied — reset the failure streak
                set_activity(benchmark_id, "judging",
                             f"{atk_name}: scoring {len(conv)}-turn transcript for objective "
                             f"{obj_idx + 1}/{len(objectives)}")
                first_turn = True
                for i, msg in enumerate(conv):
                    role = getattr(msg, "role", "")
                    if role != "assistant":
                        continue
                    response_text = getattr(msg, "converted_value", None) or getattr(msg, "original_value", "") or ""
                    prev = conv[i - 1] if i > 0 else None
                    prompt_text = getattr(prev, "converted_value", "") or getattr(prev, "original_value", "") if prev else objective_text
                    verdict, severity, succeeded = await judge_response(
                        judge, prompt_text or objective_text, response_text,
                    )
                    results.append({
                        "attacker_weak": weak_reason,
                        "sequence": obj_idx * 100 + i,
                        "prompt_text": prompt_text or objective_text,
                        "response": response_text,
                        "error": None,
                        "succeeded": succeeded,
                        "severity": severity,
                        "verdict": verdict,
                        "status": PromptStatus.OK,
                        "severity_source": severity_provenance(verdict, severity),
                        # The objective was timed ONCE, as a whole. The old code
                        # divided that single measurement evenly across the
                        # turns and presented every slice as if it had been
                        # timed. Attribute the real measurement to the first
                        # turn and leave the rest null: an unknown latency is
                        # not a small one.
                        "latency_ms": latency if first_turn else None,
                        "latency_measured": first_turn,
                        "taxonomy_category": tax,
                        "objective_id": objective_id,
                    })
                    first_turn = False
            else:
                # No target conversation retrieved. This can mean the attacker model
                # refused to craft an attack (a silent failure that must NOT be scored
                # as target robustness), or the attack produced no target turns.
                adv_conv = fetch_adversarial_conversation(result)
                if attacker_refused(adv_conv):
                    # The attacker model's fault, not the target's — leave the
                    # guard untouched so a chatty attacker declining a few
                    # objectives doesn't get mistaken for a dead target.
                    results.append({
                        "sequence": obj_idx,
                        "prompt_text": objective_text,
                        "response": "",
                        "error": "attacker-refused: adversarial model declined to craft an attack prompt",
                        "succeeded": False,
                        "severity": None,
                        "verdict": "attacker-refused",
                        "attacker_refused": True,
                        "status": PromptStatus.ATTACKER_REFUSED,
                        "severity_source": "none", "latency_measured": True,
                        "latency_ms": latency,
                        "taxonomy_category": tax,
                        "objective_id": objective_id,
                    })
                else:
                    # Attack ran but no target conversation was retrieved — record the
                    # outcome only, using the 0.14 AttackResult fields.
                    outcome = getattr(result, "outcome", None)
                    if outcome is not None:
                        achieved = str(getattr(outcome, "value", outcome)).lower() == "success"
                    else:  # older PyRIT builds
                        achieved = bool(getattr(result, "achieved", False))
                    reason = (
                        getattr(result, "outcome_reason", None)
                        or getattr(result, "reason", None)
                        or ("objective achieved" if achieved else "no conversation retrieved")
                    )
                    results.append({
                        "sequence": obj_idx,
                        "prompt_text": objective_text,
                        "response": reason,
                        # Was: None if achieved else "no conversation retrieved".
                        # Letting the classification depend on the OUTCOME meant
                        # one identical situation counted as clean data when the
                        # attack happened to succeed and as an error when it
                        # failed. Any filter on error rows would then drop the
                        # failures, keep the successes, and inflate the rate.
                        # Both cases are the same thing — an unverifiable
                        # outcome — and `status` now says so for both.
                        "error": None,
                        # The orchestrator's own outcome is kept as EVIDENCE (in
                        # note + verdict) but never as the result. Report-time
                        # success is decided by the judge on a real target
                        # response; with no transcript there is nothing to judge,
                        # so this is not a success. Reading `achieved` into
                        # `succeeded` here was the one place an unjudged
                        # orchestrator verdict still leaked into reported ASR —
                        # and it also polluted the best-of-N `solved` set below.
                        "note": (
                            "no target conversation retrieved — the orchestrator's own "
                            f"outcome was {'success' if achieved else 'failure'}, but with "
                            "no transcript to judge it is not counted (report-time success "
                            "requires a judged target response, never the orchestrator's "
                            "self-verdict)"
                        ),
                        "succeeded": False,
                        # No transcript means nothing to score. The old code put
                        # a hardcoded 4 here whenever the outcome said success,
                        # and that constant then pooled with real judge scores in
                        # the severity histogram and the mean.
                        "severity": None,
                        "severity_source": "none",
                        "verdict": reason,
                        "status": PromptStatus.NO_CONVERSATION,
                        "latency_measured": True,
                        "latency_ms": latency,
                        "taxonomy_category": tax,
                        "objective_id": objective_id,
                    })
                    # No target conversation is the closest signal we have that the
                    # target didn't respond; an "achieved" outcome still counts as
                    # a reply and resets the streak.
                    try:
                        guard.record(None if achieved else "no conversation retrieved")
                    except BenchmarkAborted:
                        break
            # Throttle between objectives to stay under provider rate limits.
            if REQUEST_DELAY:
                set_activity(benchmark_id, "waiting",
                             f"{atk_name}: {REQUEST_DELAY:.0f}s between objectives")
                await asyncio.sleep(REQUEST_DELAY)

        # Tag this pass and fold its outcome into `solved`. Done here rather
        # than inline because the objective body has several early-`continue`
        # paths that would otherwise skip the bookkeeping.
        for r in results[mark:]:
            r["attempt"] = attempt
            r["attempt_group"] = r.get("objective_id")
            r["sequence"] = attempt * 10000 + r.get("sequence", 0)
            if r.get("succeeded") and r.get("objective_id"):
                solved.add(str(r["objective_id"]))
        if guard.tripped or len(solved) >= len(objectives):
            break

    return results
