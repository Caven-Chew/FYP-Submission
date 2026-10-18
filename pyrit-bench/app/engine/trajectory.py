"""Task 8: durable multi-turn recording, extended request ledger and resume plans.

``TrajectoryRecorder`` shadows ``send_prompt_async`` on the multi-turn target,
the attacker and the judge for one objective attempt only. The shadow is an
instance attribute, so the class, ``configuration`` and ``get_identifier()``
that PyRIT 0.14 inspects stay unchanged (ConversationManager reads
``target.configuration.includes(...)``; PromptNormalizer and the scorers call
``target.send_prompt_async(message=...)``). The previous attribute (for
example the burst throttle) is restored afterwards, so one target instance is
reused safely across objectives.

Each target send writes a ``TrajectoryTurn`` before the call and completes it
after the reply, committing each step; the full conversation is then read back
from PyRIT memory into ``Trajectory.snapshot_json``. Every target, attacker and
PyRIT-internal judge call also gets a ``RequestAttempt`` ledger row (stages
``target_send``, ``attacker_call`` and ``judge_internal``) linked to the
trajectory through ``TrajectoryCall``. Converter calls are not recorded here:
dynamic attacks pass no converters, and the static converter model is built
inside llms.converter_llm, outside this lane.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..extensions import db
from ..models import ConversationRetry, Prompt, RequestAttempt, _utcnow
from ..trajectory_models import Trajectory, TrajectoryArchive, TrajectoryCall, TrajectoryTurn
from . import control
from .conversations import _pieces
from .request_history import RequestHistory
from .retry import RequestAccountingError

#: Result-row sequence scheme shared with runner and dynamic_attack.
ROSTER_STRIDE, ATTEMPT_STRIDE, OBJECTIVE_STRIDE = 1_000_000, 10_000, 100
#: obj_index * 100 must stay below one attempt stride, so attempt_of() can decode it.
MAX_OBJECTIVES = ATTEMPT_STRIDE // OBJECTIVE_STRIDE - 1

TREE_LIMIT_NOTE = (
    "TAP/PAIR resume limit: the saved branch seeds every first-level node; the "
    "search tree's other branches and pruning state are not restored.")


def _commit() -> None:
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        raise RequestAccountingError(
            "Local trajectory record could not be saved; execution stopped.") from exc


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def attempt_of(sequence: int | None) -> int:
    """Best-of-N attempt encoded in a dynamic row's sequence."""
    return ((sequence or 0) % ROSTER_STRIDE) // ATTEMPT_STRIDE


# ---------- best-of-N stopping rule ------------------------------------------

def two_label_mode() -> bool:
    """True when this run judges with the two-label method only (Task 7)."""
    from .two_label import judge_method
    return judge_method() == "two_label"


def payload_is_candidate(payload: Any) -> bool:
    """Whether an unsaved two-label assessment payload is a candidate success."""
    if not isinstance(payload, dict):
        return False
    fields = payload.get("fields") or {}
    if "candidate_success" in fields:
        return bool(fields["candidate_success"])
    if not payload.get("objective_assessment_id"):
        return False
    from ..two_label_models import ObjectiveAssessment
    from .two_label import is_candidate
    return is_candidate(db.session.get(ObjectiveAssessment, payload["objective_assessment_id"]),
                        fields.get("status"), fields.get("fulfilment"))


def result_solved(result: dict, two_label: bool) -> bool:
    """Best-of-N stop for a fresh result: a judged success, or a two-label candidate."""
    return bool(result.get("succeeded")) or (two_label and payload_is_candidate(result.get("assessment")))


def row_solved(prompt: Prompt, two_label: bool) -> bool:
    """Best-of-N stop for a saved row, as static_attack applies it."""
    if prompt.succeeded:
        return True
    if not two_label:
        return False
    from .two_label import is_solved
    return is_solved(prompt)


# ---------- reading PyRIT messages ----------------------------------------

def message_text(message: Any) -> str:
    try:
        value = message.get_value()
        if isinstance(value, str):
            return value
    except Exception:  # noqa: BLE001 - fall back to the pieces
        pass
    parts = [_text(getattr(p, "converted_value", None)) or _text(getattr(p, "original_value", None))
             for p in (getattr(message, "message_pieces", None) or [])]
    return "\n".join(p for p in parts if p)


def message_conversation_id(message: Any) -> str | None:
    pieces = getattr(message, "message_pieces", None) or []
    cid = getattr(pieces[0], "conversation_id", None) if pieces else getattr(message, "conversation_id", None)
    return str(cid) if isinstance(cid, str) and cid else None


def reply_text(responses: Any) -> tuple[str, str | None]:
    """Text of the last returned message and its provider error flag, if any."""
    if not responses:
        return "", "empty"
    last = responses[-1]
    error = None
    for piece in getattr(last, "message_pieces", None) or []:
        flag = getattr(piece, "response_error", "none")
        if isinstance(flag, str) and flag not in ("", "none"):
            error = flag
    return message_text(last), error


def memory_snapshot(conversation_id: str | None) -> list[dict] | None:
    """The conversation as PyRIT memory holds it; None when memory is unavailable."""
    if not conversation_id:
        return None
    try:
        from pyrit.memory import CentralMemory  # type: ignore
        messages = CentralMemory.get_memory_instance().get_conversation(conversation_id=conversation_id)
    except Exception:  # noqa: BLE001 - recorded turns are the fallback
        return None
    out = []
    for piece in _pieces(messages or []):
        api_role = getattr(piece, "api_role", None)
        role = api_role if isinstance(api_role, str) else _text(getattr(piece, "role", None))
        flag = getattr(piece, "response_error", "none")
        sequence = getattr(piece, "sequence", None)
        out.append({
            "role": role,
            "text": _text(getattr(piece, "converted_value", None)) or _text(getattr(piece, "original_value", None)),
            "conversation_id": conversation_id,
            "sequence": sequence if isinstance(sequence, int) else None,
            "error": getattr(piece, "converted_value_data_type", None) == "error"
                     or (isinstance(flag, str) and flag not in ("", "none")),
            "simulated": getattr(piece, "role", None) == "simulated_assistant",
        })
    return out


def complete_pairs(snapshot: list[dict]) -> tuple[list[tuple[str, str]], int]:
    """Leading user/assistant pairs with a real reply, and how many entries were left out.

    System entries are skipped. Pairing stops at the first user entry without
    a usable reply, so only an unbroken prefix of the conversation is replayed.
    """
    entries = [e for e in snapshot if e.get("role") in ("user", "assistant")]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(entries):
        user, reply = entries[i], entries[i + 1]
        if (user.get("role") != "user" or reply.get("role") != "assistant" or reply.get("error")
                or not (reply.get("text") or "").strip()):
            break
        pairs.append((user.get("text") or "", reply["text"]))
        i += 2
    return pairs, len(entries) - 2 * len(pairs)


def _pair_entries(pairs: list[tuple[str, str]]) -> list[dict]:
    out = []
    for user, reply in pairs:
        out.append({"role": "user", "text": user, "replayed": True})
        out.append({"role": "assistant", "text": reply, "replayed": True})
    return out


NO_HISTORY_NOTE = (
    "This target cannot take a replayed conversation (PyRIT needs editable history, which a "
    "browser UI target does not have), so the objective restarted from turn 1; the saved "
    "turns are archived.")


def accepts_replayed_history(target: Any) -> bool:
    """Whether PyRIT can place saved turns into this target's conversation.

    Without EDITABLE_HISTORY, PyRIT 0.14 flattens a prepended conversation into
    the text of the next message and does not count it as executed turns, which
    would not be a resume. Unknown (stubbed) targets are treated as chat targets.
    """
    try:
        from pyrit.prompt_target.common.target_capabilities import CapabilityName  # type: ignore
        return bool(target.configuration.includes(capability=CapabilityName.EDITABLE_HISTORY))
    except Exception:  # noqa: BLE001
        return True


def build_prepended(trajectory: Trajectory | None, *, orchestrator: str, max_turns: int | None,
                    editable_history: bool = True) -> tuple[list[Any], list[tuple[str, str]], list[str]]:
    """PyRIT ``prepended_conversation`` from a trajectory's latest snapshot.

    Returns (messages, replayed pairs, notes). A send that has no saved reply
    is not replayed: PyRIT's attacker writes that turn again as the first new
    turn. RedTeaming and Crescendo accept at most ``max_turns`` replayed turns;
    TAP and PAIR need fewer than ``tree_depth``. A target without editable
    history restarts instead (see accepts_replayed_history).
    """
    if trajectory is None:
        return [], [], []
    pairs, dropped = complete_pairs(trajectory.snapshot)
    if pairs and not editable_history:
        return [], [], [NO_HISTORY_NOTE]
    notes: list[str] = []
    unanswered = [t for t in trajectory.turns if t.status != "ok"]
    if unanswered and pairs:
        last = unanswered[-1]
        notes.append(f"Turn {last.turn_index} was sent but no reply was saved ({last.status}); "
                     "it is not replayed and is attempted again as a new turn.")
    if dropped and pairs:
        notes.append(f"{dropped} saved message(s) after the last complete turn were not replayed.")
    tree = orchestrator in ("tap", "pair")
    limit = (max_turns - 1 if tree else max_turns) if max_turns else None
    if limit is not None and len(pairs) > limit:
        notes.append(f"Only the first {limit} saved turn(s) were replayed (limit of this attack).")
        pairs = pairs[:max(limit, 0)]
    if tree and pairs:
        notes.append(TREE_LIMIT_NOTE)
    if not pairs:
        return [], [], []
    from pyrit.models import Message  # type: ignore
    messages = []
    for user, reply in pairs:
        messages.append(Message.from_prompt(prompt=user, role="user"))
        messages.append(Message.from_prompt(prompt=reply, role="assistant"))
    return messages, pairs, notes


def snapshot_conversation(trajectory: Trajectory) -> list[Any]:
    """Saved pairs as message-like pieces, for rebuilding rows without a PyRIT call."""
    pairs, _ = complete_pairs(trajectory.snapshot)
    out = []
    for user, reply in pairs:
        out.append(SimpleNamespace(role="user", api_role="user", converted_value=user, original_value=user))
        out.append(SimpleNamespace(role="assistant", api_role="assistant", converted_value=reply,
                                   original_value=reply))
    return out


# ---------- recording -----------------------------------------------------

class TrajectoryRecorder:
    """Durable record of one objective attempt; see the module docstring."""

    def __init__(self, *, benchmark_id: int, attack_slug: str, objective_id: str, attempt: int,
                 roster_index: int, objective_index: int, label: str | None, orchestrator: str,
                 max_turns: int | None, previous: Trajectory | None = None,
                 replayed: list[tuple[str, str]] | None = None, note: str | None = None):
        self.benchmark_id = benchmark_id
        self.attack_slug = attack_slug
        self.roster_index = roster_index
        self.sequence = roster_index * ROSTER_STRIDE + attempt * ATTEMPT_STRIDE + objective_index * OBJECTIVE_STRIDE
        replayed = replayed or []
        self._base = _pair_entries(replayed)
        self.next_turn = len(replayed) + 1
        self._restore: list[Any] = []
        self._shadowed: set[int] = set()
        self.superseded_id = None
        # Time spent paused between turns, which the objective's timing excludes.
        self.paused_ms = 0
        # First local persistence failure. TAP/PAIR nodes catch Exception, so
        # the caller re-raises this after execute_async returns.
        self.accounting_error: RequestAccountingError | None = None
        if previous is not None and previous.status != "archived":
            previous.status = "archived"
            previous.finished_at = previous.finished_at or _utcnow()
            self.superseded_id = previous.id
        self.trajectory = Trajectory(
            benchmark_id=benchmark_id, attack_slug=attack_slug, objective_id=objective_id,
            attempt=attempt, roster_index=roster_index, objective_index=objective_index,
            converter_used=label, orchestrator=orchestrator, max_turns=max_turns,
            status="resumed" if replayed else "running",
            snapshot_json=json.dumps(self._base), turns_recorded=len(replayed),
            resumed_from_turn=len(replayed) + 1 if replayed else None,
            previous_trajectory_id=previous.id if previous is not None else None,
            note=note)
        db.session.add(self.trajectory)
        _commit()

    # -- instance shadowing ------------------------------------------------
    def _shadow(self, instance: Any, wrapper_factory) -> None:
        if instance is None or id(instance) in self._shadowed:
            return
        try:
            own = vars(instance)
        except TypeError:
            return  # no instance dict (a bare object() in tests): nothing to record through
        original = getattr(instance, "send_prompt_async", None)
        if original is None:
            return
        had, previous = "send_prompt_async" in own, own.get("send_prompt_async")
        instance.send_prompt_async = wrapper_factory(original)
        self._shadowed.add(id(instance))

        def restore():
            if had:
                instance.send_prompt_async = previous
            else:
                vars(instance).pop("send_prompt_async", None)
        self._restore.append(restore)

    def instrument_target(self, target: Any) -> None:
        self._shadow(target, self._target_wrapper)

    def instrument(self, *, target: Any, attacker: Any = None, judge: Any = None) -> "TrajectoryRecorder":
        self.instrument_target(target)
        self._shadow(attacker, lambda original: self._call_wrapper("attacker_call", original))
        self._shadow(judge, lambda original: self._call_wrapper("judge_internal", original))
        return self

    def uninstrument(self) -> None:
        while self._restore:
            self._restore.pop()()
        self._shadowed.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.uninstrument()
        return False

    # -- ledger --------------------------------------------------------------
    def _ledger_begin(self, stage: str, turn_index: int | None = None):
        history = RequestHistory(self.benchmark_id, stage, attack_slug=self.attack_slug,
                                 sequence=self.sequence, attempt_slot=self.roster_index)
        db.session.add(TrajectoryCall(trajectory_id=self.trajectory.id, call_id=history.call_id,
                                      stage=stage, turn_index=turn_index))
        return history, history.begin(1, 1)  # commits the link and the started row together

    @staticmethod
    def _ledger_error(history, row, exc: BaseException, start: float) -> None:
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        known = isinstance(status_code, int)
        history.finish(row, status="error", reason=type(exc).__name__[:80],
                       http_status=status_code if known else None,
                       delivery="response_received" if known else "unknown",
                       elapsed_ms=_elapsed(start))

    @staticmethod
    def _ledger_interrupted(history, row, start: float) -> None:
        history.finish(row, status="interrupted", reason="call_cancelled", delivery="unknown",
                       elapsed_ms=_elapsed(start))

    def _keep_accounting_error(self, exc: RequestAccountingError) -> None:
        if self.accounting_error is None:
            self.accounting_error = exc

    def _call_wrapper(self, stage: str, original):
        async def send_prompt_async(*args, **kwargs):
            try:
                history, row = self._ledger_begin(stage)
                start = time.perf_counter()
                try:
                    responses = await original(*args, **kwargs)
                except Exception as exc:
                    self._ledger_error(history, row, exc, start)
                    raise
                except BaseException:
                    self._ledger_interrupted(history, row, start)
                    raise
                history.finish(row, status="returned", delivery="helper_returned", elapsed_ms=_elapsed(start))
                return responses
            except RequestAccountingError as exc:
                self._keep_accounting_error(exc)
                raise
        return send_prompt_async

    # -- target turns --------------------------------------------------------
    def _target_wrapper(self, original):
        async def send_prompt_async(*args, **kwargs):
            try:
                return await self._send_turn(original, args, kwargs)
            except RequestAccountingError as exc:
                self._keep_accounting_error(exc)
                raise
        return send_prompt_async

    async def _send_turn(self, original, args, kwargs):
        message = kwargs.get("message", args[0] if args else None)
        cid = message_conversation_id(message)
        if cid and self.trajectory.pyrit_conversation_id and cid != self.trajectory.pyrit_conversation_id:
            # Crescendo backtracking (or a TAP/PAIR branch) moved to another
            # conversation: save that conversation now, not after the reply.
            self.refresh_from_memory(cid)
        # Pause and Stop take effect here, between turns, not only between
        # objectives. Paused time is excluded from the objective's latency.
        waited = time.perf_counter()
        await control.wait_while_paused(self.benchmark_id)
        self.paused_ms += _elapsed(waited)
        turn = self._begin_turn(cid, message_text(message))
        history, row = self._ledger_begin("target_send", turn.turn_index)
        turn.request_call_id = history.call_id
        _commit()
        start = time.perf_counter()
        try:
            responses = await original(*args, **kwargs)
        except Exception as exc:
            self._finish_turn(turn, "error", start, error=f"{type(exc).__name__}: {exc}")
            self._ledger_error(history, row, exc, start)
            raise
        except BaseException:
            self._finish_turn(turn, "interrupted", start,
                              error="Call cancelled before a reply was saved; delivery unknown.")
            self._ledger_interrupted(history, row, start)
            raise
        self._record_reply(turn, history, row, responses, start)
        return responses

    def _begin_turn(self, conversation_id: str | None, prompt: str) -> TrajectoryTurn:
        turn = TrajectoryTurn(trajectory_id=self.trajectory.id, turn_index=self.next_turn,
                              pyrit_conversation_id=conversation_id, prompt_text=prompt, status="started")
        self.next_turn += 1
        db.session.add(turn)
        self.trajectory.updated_at = _utcnow()
        _commit()
        return turn

    def _finish_turn(self, turn: TrajectoryTurn, status: str, start: float, *,
                     response: str | None = None, error: str | None = None) -> None:
        turn.status = status
        turn.response_text = response
        turn.error = error
        turn.latency_ms = _elapsed(start)
        turn.finished_at = _utcnow()
        _commit()

    def _record_reply(self, turn: TrajectoryTurn, history, row, responses, start: float) -> None:
        text, error = reply_text(responses)
        self._finish_turn(turn, "error" if error else "ok", start, response=text,
                          error=f"Target returned an error response ({error})." if error else None)
        history.finish(row, status="returned", delivery="helper_returned", elapsed_ms=_elapsed(start))
        self.refresh_snapshot(turn)

    def refresh_snapshot(self, turn: TrajectoryTurn) -> None:
        """Store the full current conversation for the turn's PyRIT conversation id.

        PyRIT writes a request and its reply to memory only after the target
        returns, so memory holds the earlier turns and this pair is appended.
        Crescendo backtracking and conversation rotation change the id; the
        snapshot always follows the latest one. Without memory the recorded
        turns are used instead.
        """
        cid = turn.pyrit_conversation_id
        entries = memory_snapshot(cid)
        if entries is None:
            entries = list(self._base) + _pair_entries(
                [(t.prompt_text, t.response_text or "") for t in self.trajectory.turns
                 if t.status == "ok" and t.pyrit_conversation_id == cid])
        else:
            last = entries[-1] if entries else None
            if not (last and last.get("role") == "assistant" and last.get("text") == turn.response_text):
                entries.append({"role": "user", "text": turn.prompt_text, "conversation_id": cid})
                entries.append({"role": "assistant", "text": turn.response_text or "",
                                "conversation_id": cid, "error": turn.status != "ok"})
        self._store_snapshot(cid, entries)

    def refresh_from_memory(self, conversation_id: str | None) -> bool:
        """Replace the snapshot with PyRIT memory's copy of a conversation, if memory has it."""
        entries = memory_snapshot(conversation_id)
        if not entries:
            return False
        self._store_snapshot(conversation_id, entries)
        return True

    def _store_snapshot(self, conversation_id: str | None, entries: list[dict]) -> None:
        trajectory = self.trajectory
        trajectory.snapshot_json = json.dumps(entries)
        trajectory.pyrit_conversation_id = conversation_id
        trajectory.turns_recorded = len(complete_pairs(entries)[0])
        trajectory.updated_at = _utcnow()
        _commit()

    # -- outcome ---------------------------------------------------------------
    def _set(self, status: str, note: str | None = None) -> None:
        trajectory = self.trajectory
        trajectory.status = status
        if note:
            trajectory.note = f"{trajectory.note}\n{note}" if trajectory.note else note
        trajectory.updated_at = trajectory.finished_at = _utcnow()
        _commit()

    def completed(self, conversation_id: str | None = None) -> None:
        """PyRIT's attack returned; result rows may still be pending.

        The snapshot is taken from the conversation PyRIT returned, which for
        TAP/PAIR is the best branch rather than the last one to reply.
        """
        if isinstance(conversation_id, str) and conversation_id:
            self.refresh_from_memory(conversation_id)
        self._set("completed")

    def failed(self, error: str) -> None:
        self._set("failed", f"Attack ended with an error: {error}"[:2000])

    def interrupted(self, note: str) -> None:
        self._set("interrupted", note)


def _elapsed(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


# ---------- startup ---------------------------------------------------------

def mark_interrupted_trajectories() -> None:
    """At startup, running trajectories and unanswered sends become interrupted (delivery unknown)."""
    note = "Process restarted while this conversation was running; saved turns can be resumed."
    for trajectory in Trajectory.query.filter(Trajectory.status.in_(("running", "resumed"))).all():
        trajectory.status = "interrupted"
        trajectory.note = f"{trajectory.note}\n{note}" if trajectory.note else note
        trajectory.updated_at = _utcnow()
    for turn in TrajectoryTurn.query.filter_by(status="started").all():
        turn.status = "interrupted"
        turn.error = "Process restarted before a reply was saved; delivery unknown."
    db.session.commit()


# ---------- resume plans and archives ---------------------------------------

@dataclass
class ResumePlan:
    """What a resumed or retried dynamic attack does per (objective_id, attempt)."""
    resume: dict = field(default_factory=dict)     # key -> latest trajectory to continue or supersede
    completed: dict = field(default_factory=dict)  # key -> succeeded; rows kept, attempt skipped
    replace: dict = field(default_factory=dict)    # key -> current rows archived when new results land

    def attempted(self, objective_id: str) -> bool:
        return any(k[0] == objective_id for k in (*self.resume, *self.completed, *self.replace))

    def pending(self, objective_id: str) -> bool:
        return any(k[0] == objective_id for k in (*self.resume, *self.replace))

    def solved(self) -> set:
        """Objectives already broken by a kept attempt; best-of-N runs nothing more for them."""
        return {k[0] for k, won in self.completed.items() if won}


def latest_trajectories(benchmark_id: int, attack_slug: str, roster_index: int,
                        objective_id: str | None = None) -> dict:
    query = Trajectory.query.filter(
        Trajectory.benchmark_id == benchmark_id, Trajectory.attack_slug == attack_slug,
        Trajectory.roster_index == roster_index, Trajectory.status != "archived")
    if objective_id is not None:
        query = query.filter(Trajectory.objective_id == objective_id)
    return {(t.objective_id, t.attempt): t for t in query.order_by(Trajectory.id).all()}


def plan_resume(benchmark_id: int, attack_slug: str, roster_index: int, label: str | None, *,
                retry_failed: bool = True, objective_id: str | None = None) -> ResumePlan:
    """Decide, from saved rows and trajectories, which objective attempts to keep, resume or redo.

    Rows without execution errors are kept. With ``retry_failed`` an attempt
    whose rows failed, or whose trajectory never produced rows, is resumed from
    its trajectory (or started again when nothing usable was saved) and its rows
    are archived when the new results arrive. Without it, every attempted
    objective is left as it is.
    """
    from .runner import EXECUTION_ERRORS
    rows = [p for p in Prompt.query.filter_by(benchmark_id=benchmark_id, attack_slug=attack_slug).all()
            if p.objective_id and (p.attempt or 0) == roster_index and p.converter_used == label
            and (objective_id is None or p.objective_id == objective_id)]
    by_key: dict = {}
    for p in rows:
        by_key.setdefault((p.objective_id, attempt_of(p.sequence)), []).append(p)
    trajectories = latest_trajectories(benchmark_id, attack_slug, roster_index, objective_id)
    two_label = two_label_mode()
    plan = ResumePlan()
    for key in set(by_key) | set(trajectories):
        current = by_key.get(key, [])
        if not retry_failed or (current and not any(p.status in EXECUTION_ERRORS for p in current)):
            plan.completed[key] = any(row_solved(p, two_label) for p in current)
            continue
        if key in trajectories:
            plan.resume[key] = trajectories[key]
        if current:
            plan.replace[key] = current
    return plan


def supersede_leftovers(plan: ResumePlan, benchmark_id: int, attack_slug: str, roster_index: int,
                        objective_id: str) -> int:
    """Archive the unfinished attempts of an objective that is now solved (reason "superseded").

    Best-of-N runs no further attempts once an objective is broken, so a later
    attempt's failed rows or unfinished trajectory would otherwise stay
    current and pending for ever. The caller commits. Returns the number of
    attempts archived.
    """
    keys = sorted({k for k in (*plan.replace, *plan.resume) if k[0] == objective_id})
    for key in keys:
        old = plan.resume.pop(key, None)
        archive_and_remove(benchmark_id, attack_slug, plan.replace.pop(key, []), reason="superseded",
                           objective_id=objective_id, attempt=key[1], roster_index=roster_index,
                           superseded_id=old.id if old is not None else None)
        if old is not None and old.status != "archived":
            old.status = "archived"
            note = "Superseded: another attempt solved this objective."
            old.note = f"{old.note}\n{note}" if old.note else note
            old.finished_at = old.finished_at or _utcnow()
    return len(keys)


def _row_snapshot(prompt: Prompt) -> dict:
    snapshot = prompt.execution_snapshot()
    # Two-label ResponseAssessment rows keep this prompt_id after the row is
    # removed, so the archive names it explicitly to stay linkable.
    snapshot["original_prompt_id"] = prompt.id
    snapshot["response_assessment_ids"] = [a.id for a in getattr(prompt, "response_assessments", None) or []]
    snapshot["scoring_evaluations"] = [
        {c.name: (v.isoformat() if hasattr(v, "isoformat") else v)
         for c in e.__table__.columns for v in [getattr(e, c.name)]}
        for e in prompt.scoring_evaluations]
    # The ledger rows stay in the request history (detached below); this keeps
    # which call belonged to which archived row and saved evaluation.
    evaluation_ids = [e.id for e in prompt.scoring_evaluations]
    snapshot["ledger"] = [
        {"call_id": a.call_id, "stage": a.stage, "evaluation_id": a.evaluation_id}
        for a in RequestAttempt.query.filter(db.or_(
            RequestAttempt.prompt_id == prompt.id,
            RequestAttempt.evaluation_id.in_(evaluation_ids or [-1]))).order_by(RequestAttempt.id)]
    return snapshot


def archive_and_remove(benchmark_id: int, attack_slug: str, rows: list[Prompt], *, reason: str,
                       objective_id: str | None = None, attempt: int | None = None,
                       roster_index: int | None = None, superseded_id: int | None = None,
                       show_as_retry: bool = True) -> TrajectoryArchive | None:
    """Archive replaced rows (and the trajectory they came from), then remove them from current results.

    Ledger rows attached to a removed row are detached, not deleted; its saved
    judge evaluations are copied into the archive. The caller commits.
    """
    if not rows and superseded_id is None:
        return None
    snapshots = [_row_snapshot(p) for p in rows]
    retry = None
    if rows and objective_id and show_as_retry:
        retry = ConversationRetry(benchmark_id=benchmark_id, attack_slug=attack_slug,
                                  objective_id=objective_id,
                                  original_rows_json=json.dumps(snapshots, default=str))
        db.session.add(retry)
        db.session.flush()
    superseded = db.session.get(Trajectory, superseded_id) if superseded_id else None
    archive = TrajectoryArchive(
        benchmark_id=benchmark_id, attack_slug=attack_slug, objective_id=objective_id,
        attempt=attempt, roster_index=roster_index, reason=reason,
        original_rows_json=json.dumps(snapshots, default=str),
        trajectory_json=json.dumps(superseded.as_dict(), default=str) if superseded else None,
        conversation_retry_id=retry.id if retry else None)
    db.session.add(archive)
    db.session.flush()  # the archive id names what superseded each row's assessments
    from .two_label import supersede_attempts
    for p in rows:
        supersede_attempts(p, f"trajectory_archive:{archive.id}")
        RequestAttempt.query.filter_by(prompt_id=p.id).update(
            {RequestAttempt.prompt_id: None}, synchronize_session=False)
        # The row's saved evaluations are deleted with it (copied into the
        # archive above), so no ledger row may keep pointing at their ids.
        evaluation_ids = [e.id for e in p.scoring_evaluations]
        if evaluation_ids:
            RequestAttempt.query.filter(RequestAttempt.evaluation_id.in_(evaluation_ids)).update(
                {RequestAttempt.evaluation_id: None}, synchronize_session=False)
        db.session.expire(p, ["request_attempts"])
        db.session.delete(p)
    db.session.flush()
    return archive


def conversation_call_counts(benchmark_id: int) -> dict:
    """(attack_slug, objective_id, roster_index, attempt) -> {stage: count} per conversation.

    Multi-turn ledger rows link to their conversation, not to a Prompt row, so
    the CSV export reports them per conversation through this map. The count
    covers the whole conversation: the current trajectory plus every earlier
    segment it resumed or replaced (followed through previous_trajectory_id),
    so an interrupted-and-resumed conversation is not undercounted. It is a
    per-conversation total and therefore repeats on every turn row of that
    conversation; it must not be summed over rows.
    """
    from sqlalchemy import func
    trajectories = {t.id: t for t in Trajectory.query.filter_by(benchmark_id=benchmark_id)}
    per_segment: dict = {}
    for tid, stage, count in (db.session.query(TrajectoryCall.trajectory_id, TrajectoryCall.stage,
                                               func.count(TrajectoryCall.id))
                              .join(Trajectory, Trajectory.id == TrajectoryCall.trajectory_id)
                              .filter(Trajectory.benchmark_id == benchmark_id)
                              .group_by(TrajectoryCall.trajectory_id, TrajectoryCall.stage)):
        per_segment.setdefault(tid, {})[stage] = int(count)
    out: dict = {}
    for t in trajectories.values():
        if t.status == "archived":
            continue
        counts = out.setdefault((t.attack_slug, t.objective_id, t.roster_index, t.attempt), {})
        seen, segment = set(), t
        while segment is not None and segment.id not in seen:
            seen.add(segment.id)
            for stage, count in per_segment.get(segment.id, {}).items():
                counts[stage] = counts.get(stage, 0) + count
            segment = trajectories.get(segment.previous_trajectory_id)
    return out


def call_links(benchmark_id: int) -> dict:
    """call_id -> (trajectory id, turn index) for the request-history table."""
    rows = (db.session.query(TrajectoryCall.call_id, TrajectoryCall.trajectory_id, TrajectoryCall.turn_index)
            .join(Trajectory, Trajectory.id == TrajectoryCall.trajectory_id)
            .filter(Trajectory.benchmark_id == benchmark_id).all())
    return {call_id: (tid, turn) for call_id, tid, turn in rows}
