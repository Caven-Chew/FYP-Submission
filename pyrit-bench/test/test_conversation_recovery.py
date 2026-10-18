"""Offline tests for Task 8.5: durable multi-turn recording and recovery records.

Run: .venv/bin/python -B tools/run_isolated_app_test.py test_conversation_recovery.py
The imported fixture sets a disposable database and artifact folder before the
app is imported and stubs PyRIT; targets, memory and strategies here are fakes.
"""
import asyncio
import json
import sys
import threading
import time
import types
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_run_model_record as fixture  # noqa: F401; isolation before app import
from sqlalchemy import text

from app.engine import control, dynamic_attack, runner, trajectory, two_label
from app.engine.llms import llm_overrides
from app.engine.retry import RequestAccountingError
from app.extensions import db
from app.models import (Benchmark, ConversationRetry, Prompt, PromptStatus, RequestAttempt,
                        ScoringEvaluation, Target)
from app.trajectory_models import Trajectory, TrajectoryArchive, TrajectoryCall, TrajectoryTurn


# ---------- fakes ---------------------------------------------------------------

class FakeMemory:
    """Stands in for PyRIT CentralMemory: conversation id -> list of pieces."""
    conversations: dict = {}
    available = True

    @classmethod
    def get_memory_instance(cls):
        if not cls.available:
            raise ValueError("memory not initialised")
        return cls

    @classmethod
    def get_conversation(cls, *, conversation_id):
        return list(cls.conversations.get(conversation_id, []))

    @classmethod
    def add(cls, cid, role, text):
        cls.conversations.setdefault(cid, []).append(SimpleNamespace(
            role=role, api_role="assistant" if role == "simulated_assistant" else role,
            converted_value=text, original_value=text, conversation_id=cid,
            converted_value_data_type="text", response_error="none", sequence=len(cls.conversations[cid])))


sys.modules["pyrit.memory"].CentralMemory = FakeMemory


def message(cid, text):
    piece = SimpleNamespace(conversation_id=cid, converted_value=text, original_value=text)
    return SimpleNamespace(message_pieces=[piece], get_value=lambda: text)


def reply(text, error="none"):
    piece = SimpleNamespace(converted_value=text, original_value=text, response_error=error)
    return SimpleNamespace(message_pieces=[piece], get_value=lambda: text)


class FakeTarget:
    """A chat target whose class must stay intact under instrumentation."""
    configuration = "chat-capabilities"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
        self.calls = 0
        self.seen_started = []

    async def send_prompt_async(self, *, message):
        self.calls += 1
        # While the provider call is in flight the turn must already be committed:
        # a separate connection sees only committed rows.
        with db.engine.connect() as conn:
            self.seen_started.append(conn.execute(text(
                "SELECT count(*) FROM trajectory_turns WHERE status = 'started'")).scalar())
        if self.calls in self.fail_on:
            raise RuntimeError("provider unavailable")
        return [reply(f"target reply {self.calls}")]

    def get_identifier(self):
        return "fake-target"


class FakeLLM:
    def __init__(self, text="llm text"):
        self.text = text
        self.calls = 0

    async def send_prompt_async(self, *, message):
        self.calls += 1
        return [reply(self.text)]


class FakeStrategy:
    """Mimics a PyRIT multi-turn attack: counts replayed assistant turns as executed."""
    last = None

    def __init__(self, *, objective_target, attack_adversarial_config, attack_scoring_config, max_turns, **_kw):
        self.target, self.attacker = objective_target, attack_adversarial_config.target
        # Like PyRIT's SelfAskTrueFalseScorer, the scorer calls the judge target it holds.
        self.judge = attack_scoring_config.objective_scorer.chat_target
        self.max_turns = max_turns
        FakeStrategy.last = self

    async def execute_async(self, objective, prepended_conversation=None):
        self.prepended = prepended_conversation
        cid = str(uuid.uuid4())
        executed = 0
        for m in prepended_conversation or []:
            FakeMemory.add(cid, "simulated_assistant" if m.role == "assistant" else m.role, m.prompt)
            executed += m.role == "assistant"
        self.replayed = executed
        while executed < self.max_turns:
            await self.attacker.send_prompt_async(message=message("adv", "next"))
            text = f"attack turn {executed + 1}"
            responses = await self.target.send_prompt_async(message=message(cid, text))
            FakeMemory.add(cid, "user", text)
            FakeMemory.add(cid, "assistant", responses[-1].get_value())
            await self.judge.send_prompt_async(message=message("judge", "score"))
            executed += 1
        return SimpleNamespace(conversation_id=cid, related_conversations=None)


_ea = sys.modules["pyrit.executor.attack"]
_ea.RedTeamingAttack = _ea.CrescendoAttack = FakeStrategy
_ea.AttackAdversarialConfig = lambda target: SimpleNamespace(target=target)
_ea.AttackScoringConfig = lambda objective_scorer: SimpleNamespace(objective_scorer=objective_scorer)

ATTACK = {"slug": "redteam_dynamic", "orchestrator": "red_teaming", "max_turns": 4, "name": "fake"}


def scorer_on(judge, true_description=None):
    return SimpleNamespace(chat_target=judge)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fixture.create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        FakeMemory.conversations = {}
        FakeMemory.available = True
        target = Target(name="trajectory target", provider="openai", model_name="m")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="trajectory", target_id=target.id, status="running",
                           attack_slugs_json=json.dumps(["redteam_dynamic"]))
        db.session.add(self.b)
        db.session.commit()

    def tearDown(self):
        control.clear_cancel(self.b.id)
        db.session.remove()
        self.ctx.pop()

    def recorder(self, **kw):
        args = dict(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                    attempt=0, roster_index=1, objective_index=2, label="openai/first",
                    orchestrator="red_teaming", max_turns=4)
        args.update(kw)
        return trajectory.TrajectoryRecorder(**args)


class RecorderTests(Base):
    def test_instrumentation_keeps_class_and_restores_previous_attribute(self):
        target = FakeTarget()

        async def throttled(**kwargs):
            return await FakeTarget.send_prompt_async(target, **kwargs)
        target.send_prompt_async = throttled  # e.g. the burst throttle's instance attribute
        rec = self.recorder()
        with rec.instrument(target=target):
            self.assertIs(type(target), FakeTarget)
            self.assertIsInstance(target, FakeTarget)
            self.assertEqual(target.configuration, "chat-capabilities")
            self.assertIsNot(vars(target)["send_prompt_async"], throttled)
        self.assertIs(vars(target)["send_prompt_async"], throttled)
        plain = FakeTarget()
        with rec.instrument(target=plain):
            self.assertIn("send_prompt_async", vars(plain))
        self.assertNotIn("send_prompt_async", vars(plain))
        with rec.instrument(target=object()):  # no instance dict: recording is skipped, not an error
            pass

    def test_turn_is_started_before_the_call_and_ok_after(self):
        target, rec = FakeTarget(), self.recorder()
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("cid-1", "hello target")))
        self.assertEqual(target.seen_started, [1])
        turn = TrajectoryTurn.query.one()
        self.assertEqual((turn.turn_index, turn.status, turn.prompt_text, turn.response_text),
                         (1, "ok", "hello target", "target reply 1"))
        self.assertEqual(turn.pyrit_conversation_id, "cid-1")
        self.assertIsNotNone(turn.latency_ms)
        ledger = RequestAttempt.query.one()
        self.assertEqual((ledger.stage, ledger.status, ledger.call_id), ("target_send", "returned", turn.request_call_id))
        # Same sequence scheme as the result rows: roster, attempt, objective.
        self.assertEqual((ledger.sequence, ledger.attempt_slot, ledger.prompt_id), (1_000_200, 1, None))
        link = TrajectoryCall.query.one()
        self.assertEqual((link.trajectory_id, link.stage, link.turn_index), (rec.trajectory.id, "target_send", 1))

    def test_error_is_recorded_and_reraised(self):
        target, rec = FakeTarget(fail_on={1}), self.recorder()
        with rec.instrument(target=target):
            with self.assertRaises(RuntimeError):
                asyncio.run(target.send_prompt_async(message=message("cid-1", "hi")))
        turn = TrajectoryTurn.query.one()
        self.assertEqual(turn.status, "error")
        self.assertIn("provider unavailable", turn.error)
        self.assertEqual(RequestAttempt.query.one().status, "error")
        self.assertEqual(json.loads(rec.trajectory.snapshot_json), [])

    def test_provider_error_reply_is_not_a_complete_turn(self):
        class Blocked(FakeTarget):
            async def send_prompt_async(self, *, message):
                return [reply("blocked by filter", error="blocked")]
        target, rec = Blocked(), self.recorder()
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("cid-1", "hi")))
        self.assertEqual(TrajectoryTurn.query.one().status, "error")
        self.assertEqual(rec.trajectory.turns_recorded, 0)

    def test_snapshot_reads_memory_and_appends_the_current_pair(self):
        FakeMemory.add("cid-1", "user", "earlier prompt")
        FakeMemory.add("cid-1", "assistant", "earlier reply")
        target, rec = FakeTarget(), self.recorder()
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("cid-1", "now")))
        snap = rec.trajectory.snapshot
        self.assertEqual([(e["role"], e["text"]) for e in snap],
                         [("user", "earlier prompt"), ("assistant", "earlier reply"),
                          ("user", "now"), ("assistant", "target reply 1")])
        self.assertEqual(rec.trajectory.turns_recorded, 2)
        self.assertEqual(rec.trajectory.pyrit_conversation_id, "cid-1")

    def test_snapshot_falls_back_to_recorded_turns_without_memory(self):
        FakeMemory.available = False
        target, rec = FakeTarget(), self.recorder(replayed=[("saved prompt", "saved reply")])
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("cid-2", "first new")))
            asyncio.run(target.send_prompt_async(message=message("cid-2", "second new")))
        self.assertEqual([e["text"] for e in rec.trajectory.snapshot],
                         ["saved prompt", "saved reply", "first new", "target reply 1", "second new", "target reply 2"])
        self.assertEqual([t.turn_index for t in rec.trajectory.turns], [2, 3])
        self.assertEqual((rec.trajectory.status, rec.trajectory.resumed_from_turn), ("resumed", 2))

    def test_attacker_and_judge_calls_enter_the_ledger(self):
        target, attacker, judge, rec = FakeTarget(), FakeLLM(), FakeLLM(), self.recorder()
        with rec.instrument(target=target, attacker=attacker, judge=judge):
            asyncio.run(attacker.send_prompt_async(message=message("adv", "x")))
            asyncio.run(judge.send_prompt_async(message=message("jdg", "y")))
        self.assertEqual(sorted(a.stage for a in RequestAttempt.query.all()), ["attacker_call", "judge_internal"])
        self.assertEqual(sorted(c.stage for c in rec.trajectory.calls), ["attacker_call", "judge_internal"])
        self.assertEqual(TrajectoryTurn.query.count(), 0)

    def test_stop_takes_effect_before_the_next_send(self):
        target, rec = FakeTarget(), self.recorder()
        control.request_cancel(self.b.id)
        with rec.instrument(target=target):
            with self.assertRaises(control.BenchmarkCancelled):
                asyncio.run(target.send_prompt_async(message=message("cid", "never sent")))
        self.assertEqual((target.calls, TrajectoryTurn.query.count()), (0, 0))

    def test_restart_marks_running_trajectories_and_started_turns_interrupted(self):
        rec = self.recorder()
        db.session.add(TrajectoryTurn(trajectory_id=rec.trajectory.id, turn_index=1, prompt_text="p",
                                      status="started"))
        done = self.recorder(objective_id="obj-b")
        done.completed()
        db.session.commit()
        from app import _mark_interrupted_benchmarks
        _mark_interrupted_benchmarks()
        db.session.expire_all()
        self.assertEqual(db.session.get(Trajectory, rec.trajectory.id).status, "interrupted")
        self.assertEqual(db.session.get(Trajectory, done.trajectory.id).status, "completed")
        turn = TrajectoryTurn.query.one()
        self.assertEqual(turn.status, "interrupted")
        self.assertIn("delivery unknown", turn.error)


class PrependedTests(Base):
    def saved(self, snapshot, turn_statuses=("ok", "ok"), status="interrupted"):
        t = Trajectory(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                       status=status, snapshot_json=json.dumps(snapshot))
        db.session.add(t)
        db.session.flush()
        for i, s in enumerate(turn_statuses, 1):
            db.session.add(TrajectoryTurn(trajectory_id=t.id, turn_index=i, prompt_text=f"p{i}", status=s))
        db.session.commit()
        return t

    def pairs(self, n):
        out = []
        for i in range(1, n + 1):
            out += [{"role": "user", "text": f"p{i}"}, {"role": "assistant", "text": f"r{i}"}]
        return out

    def test_unanswered_started_turn_is_dropped_and_noted(self):
        t = self.saved(self.pairs(2) + [{"role": "user", "text": "p3"}], ("ok", "ok", "interrupted"))
        messages, pairs, notes = trajectory.build_prepended(t, orchestrator="red_teaming", max_turns=4)
        self.assertEqual([(m.role, m.prompt) for m in messages],
                         [("user", "p1"), ("assistant", "r1"), ("user", "p2"), ("assistant", "r2")])
        self.assertEqual(len(pairs), 2)
        self.assertTrue(any("Turn 3 was sent but no reply was saved" in n for n in notes))

    def test_system_entries_are_skipped_and_error_replies_end_the_prefix(self):
        snap = [{"role": "system", "text": "s"}] + self.pairs(1) + [
            {"role": "user", "text": "p2"}, {"role": "assistant", "text": "err", "error": True}] + self.pairs(1)
        pairs, dropped = trajectory.complete_pairs(snap)
        self.assertEqual((pairs, dropped), ([("p1", "r1")], 4))

    def test_tap_and_pair_are_capped_below_depth_with_the_limit_note(self):
        t = self.saved(self.pairs(3), ("ok", "ok", "ok"))
        for name in ("tap", "pair"):
            messages, pairs, notes = trajectory.build_prepended(t, orchestrator=name, max_turns=3)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(messages), 4)
            self.assertIn(trajectory.TREE_LIMIT_NOTE, notes)
        _m, pairs, notes = trajectory.build_prepended(t, orchestrator="crescendo", max_turns=3)
        self.assertEqual((len(pairs), trajectory.TREE_LIMIT_NOTE in notes), (3, False))

    def test_nothing_usable_means_a_fresh_start(self):
        t = self.saved([{"role": "user", "text": "p1"}], ("interrupted",))
        self.assertEqual(trajectory.build_prepended(t, orchestrator="red_teaming", max_turns=4), ([], [], []))
        self.assertEqual(trajectory.build_prepended(None, orchestrator="red_teaming", max_turns=4), ([], [], []))


class ArchiveTests(Base):
    def test_archive_keeps_rows_ledger_and_evaluations(self):
        row = Prompt(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                     prompt_text="old turn", response="", status=PromptStatus.SEND_ERROR, error="offline")
        db.session.add(row)
        db.session.flush()
        evaluation = ScoringEvaluation(prompt_id=row.id, context_json="{}", status="valid", severity=1)
        db.session.add(evaluation)
        db.session.flush()
        db.session.add(RequestAttempt(benchmark_id=self.b.id, call_id="c-1", stage="judge", prompt_id=row.id,
                                      evaluation_id=evaluation.id, attempt_number=1, max_attempts=1,
                                      status="returned"))
        old = Trajectory(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                         status="archived", snapshot_json="[]")
        db.session.add(old)
        db.session.commit()
        row_id, evaluation_id = row.id, evaluation.id
        archive = trajectory.archive_and_remove(self.b.id, "redteam_dynamic", [row], reason="resume",
                                                objective_id="obj-a", attempt=0, roster_index=0,
                                                superseded_id=old.id)
        db.session.commit()
        self.assertEqual(Prompt.query.count(), 0)
        ledger = RequestAttempt.query.one()
        self.assertIsNone(ledger.prompt_id)  # detached, never deleted
        # The evaluation went with the row; no ledger row may point at its id.
        self.assertIsNone(db.session.get(ScoringEvaluation, evaluation_id))
        self.assertIsNone(ledger.evaluation_id)
        self.assertEqual(RequestAttempt.query.filter(RequestAttempt.evaluation_id.isnot(None)).count(), 0)
        archive = db.session.get(TrajectoryArchive, archive.id)
        self.assertEqual(archive.original_rows[0]["ledger"],
                         [{"call_id": "c-1", "stage": "judge", "evaluation_id": evaluation_id}])
        self.assertEqual(archive.reason, "resume")
        self.assertEqual(archive.original_rows[0]["prompt_text"], "old turn")
        self.assertEqual(archive.original_rows[0]["original_prompt_id"], row_id)
        self.assertEqual(archive.original_rows[0]["scoring_evaluations"][0]["severity"], 1)
        self.assertEqual(archive.trajectory["id"], old.id)
        retry = ConversationRetry.query.one()
        self.assertEqual((archive.conversation_retry_id, retry.original_rows[0]["error"]), (retry.id, "offline"))

    def test_nothing_to_archive_is_a_no_op(self):
        self.assertIsNone(trajectory.archive_and_remove(self.b.id, "redteam_dynamic", [], reason="restart"))
        self.assertEqual(TrajectoryArchive.query.count(), 0)


class DynamicAttackTests(Base):
    """run_dynamic_attack with the recorder on, against a fake PyRIT strategy."""

    def run_attack(self, target, *, objectives=None, attempts=1, scored=None, delay=0, **kw):
        objectives = objectives or [{"id": "obj-a", "objective": "first"}]
        self.judge = FakeLLM("judge")
        scored = scored or AsyncMock(return_value={"verdict": "SCORE: 1", "severity": 1, "succeeded": False,
                                                   "assessment": None})
        with patch.object(dynamic_attack, "build_objective_scorer", scorer_on), \
             patch.object(dynamic_attack, "with_attacker_backup", lambda attacker, events: attacker), \
             patch.object(dynamic_attack, "assess_turn", scored), \
             patch.object(dynamic_attack, "attempts_for", return_value=attempts), \
             patch.object(dynamic_attack, "REQUEST_DELAY", delay):
            return asyncio.run(dynamic_attack.run_dynamic_attack(
                attack=ATTACK, target=target, attacker=FakeLLM("next attack"), judge=self.judge,
                objectives=objectives, guard=dynamic_attack.ConsecutiveFailureGuard(10, stop_on_limit=False),
                benchmark_id=self.b.id, record=True, **kw))

    def test_failure_then_resume_from_the_saved_turns(self):
        first = self.run_attack(FakeTarget(fail_on={3}))
        self.assertEqual([r["status"] for r in first], [PromptStatus.SEND_ERROR])
        saved = Trajectory.query.one()
        self.assertEqual((saved.status, saved.turns_recorded), ("failed", 2))
        self.assertEqual(first[0]["trajectory_id"], saved.id)
        target = FakeTarget()
        rows = self.run_attack(target, resume_state={("obj-a", 0): saved})
        self.assertEqual(FakeStrategy.last.replayed, 2)
        self.assertEqual(target.calls, 2)  # turns 3 and 4 only
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["note"].startswith("Resumed from turn 3 (2 turns replayed") for r in rows))
        self.assertTrue(all(r["latency_measured"] is False and r["latency_ms"] is None for r in rows))
        self.assertEqual([r["prompt_text"] for r in rows],
                         ["attack turn 1", "attack turn 2", "attack turn 3", "attack turn 4"])
        self.assertEqual({r["recovery"] for r in rows}, {"resumed"})
        new = db.session.get(Trajectory, rows[0]["trajectory_id"])
        self.assertEqual((new.status, new.resumed_from_turn, new.previous_trajectory_id), ("completed", 3, saved.id))
        self.assertEqual(rows[0]["superseded_trajectory_id"], saved.id)
        self.assertEqual(db.session.get(Trajectory, saved.id).status, "archived")
        self.assertEqual([t.turn_index for t in new.turns], [3, 4])
        stages = [a.stage for a in RequestAttempt.query.order_by(RequestAttempt.id)]
        self.assertEqual(stages.count("target_send"), 5)
        self.assertEqual(stages.count("attacker_call"), 5)
        self.assertEqual(stages.count("judge_internal"), 4)

    def test_completed_attack_without_rows_is_rebuilt_without_new_sends(self):
        self.run_attack(FakeTarget())
        done = Trajectory.query.one()
        self.assertEqual((done.status, done.turns_recorded), ("completed", 4))
        target = FakeTarget()
        rows = self.run_attack(target, resume_state={("obj-a", 0): done})
        self.assertEqual(target.calls, 0)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("Rebuilt from the saved trajectory" in r["note"] for r in rows))
        self.assertEqual(Trajectory.query.count(), 1)

    def test_stop_between_turns_interrupts_and_returns_finished_objectives(self):
        persisted = []

        class StopAfterSecond(FakeTarget):
            async def send_prompt_async(inner, *, message):
                responses = await FakeTarget.send_prompt_async(inner, message=message)
                if inner.calls == 6:  # objective 2, turn 2
                    control.request_cancel(self.b.id)
                return responses
        objectives = [{"id": "obj-a", "objective": "first"}, {"id": "obj-b", "objective": "second"}]
        rows = self.run_attack(StopAfterSecond(), objectives=objectives, persist=persisted.append)
        self.assertEqual({r["objective_id"] for r in rows}, {"obj-a"})
        self.assertEqual(len(persisted), 1)  # objective 1 handed over as soon as it ended
        stopped = Trajectory.query.filter_by(objective_id="obj-b").one()
        self.assertEqual((stopped.status, stopped.turns_recorded), ("interrupted", 2))
        self.assertIn("Stopped before the next turn", stopped.note)

    def test_target_without_editable_history_restarts_instead_of_replaying(self):
        caps = types.ModuleType("pyrit.prompt_target.common.target_capabilities")
        caps.CapabilityName = SimpleNamespace(EDITABLE_HISTORY="supports_editable_history")

        class BrowserLike(FakeTarget):
            configuration = SimpleNamespace(includes=lambda capability: capability != "supports_editable_history")
        self.run_attack(FakeTarget(fail_on={3}))
        saved = Trajectory.query.one()
        target = BrowserLike()
        with patch.dict(sys.modules, {"pyrit.prompt_target.common": types.ModuleType("pyrit.prompt_target.common"),
                                      "pyrit.prompt_target.common.target_capabilities": caps}):
            rows = self.run_attack(target, resume_state={("obj-a", 0): saved})
        self.assertEqual((FakeStrategy.last.prepended, target.calls), (None, 4))
        self.assertTrue(all(r["note"].startswith(trajectory.NO_HISTORY_NOTE) for r in rows))
        self.assertTrue(all(not r.get("recovery") and r.get("superseded_trajectory_id") == saved.id for r in rows))
        new = db.session.get(Trajectory, rows[0]["trajectory_id"])
        self.assertEqual((new.resumed_from_turn, new.previous_trajectory_id, new.status), (None, saved.id, "completed"))
        self.assertEqual(db.session.get(Trajectory, saved.id).status, "archived")

    def test_completed_attempts_are_skipped(self):
        target = FakeTarget()
        rows = self.run_attack(target, completed={("obj-a", 0): False})
        self.assertEqual((rows, target.calls, Trajectory.query.count()), ([], 0, 0))

    def test_turn_results_carry_the_saved_judge_request_for_recovery(self):
        # Regression (11 Oct 2026): the saved judge request from assess_turn must reach the
        # result dict, or runner._dynamic_row never writes scoring_context_json and
        # judge-only recovery of a turn whose judging failed is never offered.
        context = {"version": 1, "objective": "first", "rubric": "harm", "request": "REQ", "request_sha256": "a",
                   "response_sha256": "b", "prompt_sha256": "c", "converter": None, "parser_version": 1,
                   "success_threshold": 3, "judge": {"provider": "profllm", "model": "m", "key_env": "LLM_API_KEY"},
                   "endpoint_sha256": None}
        scored = AsyncMock(return_value={"verdict": "judge-error: TimeoutError", "severity": None, "succeeded": False,
                                         "assessment": None, "scoring_context": context})
        rows = self.run_attack(FakeTarget(), scored=scored)
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        self.assertTrue(ok_rows)
        self.assertTrue(all(r.get("scoring_context") == context for r in ok_rows))

    def test_without_record_nothing_is_written(self):
        scored = AsyncMock(return_value={"verdict": "SCORE: 1", "severity": 1, "succeeded": False, "assessment": None})
        with patch.object(dynamic_attack, "build_objective_scorer", scorer_on), \
             patch.object(dynamic_attack, "with_attacker_backup", lambda attacker, events: attacker), \
             patch.object(dynamic_attack, "assess_turn", scored), \
             patch.object(dynamic_attack, "attempts_for", return_value=1):
            rows = asyncio.run(dynamic_attack.run_dynamic_attack(
                attack=ATTACK, target=FakeTarget(), attacker=FakeLLM(), judge=FakeLLM(),
                objectives=[{"id": "obj-a", "objective": "x"}],
                guard=dynamic_attack.ConsecutiveFailureGuard(10), benchmark_id=self.b.id))
        self.assertEqual(len(rows), 4)
        self.assertEqual((Trajectory.query.count(), RequestAttempt.query.count()), (0, 0))


def scored_as(succeeded=False, assessment=None):
    return AsyncMock(return_value={"verdict": "SCORE: 5" if succeeded else "SCORE: 1",
                                   "severity": 5 if succeeded else 1, "succeeded": succeeded,
                                   "assessment": assessment})


class ReviewFixTests(Base):
    """Regression tests for the independent review of Task 8 (10 October 2026)."""

    run_attack = DynamicAttackTests.run_attack

    def errors(self):
        return sorted((trajectory.attempt_of(p.sequence), p.status) for p in Prompt.query
                      if p.status in runner.EXECUTION_ERRORS)

    def test_high1_solved_objective_supersedes_later_failed_attempts(self):
        # Adapted from the reviewer's reproduction: attempts 0 and 1 both fail
        # on their third send; Retry failed then resumes attempt 0, which succeeds.
        first = self.run_attack(FakeTarget(fail_on={3, 6}), attempts=2)
        plan = trajectory.plan_resume(self.b.id, "redteam_dynamic", 0, None)
        runner._save_dynamic_results(self.b, "redteam_dynamic", 0, None, first, plan, reason="restart")
        db.session.commit()
        self.assertEqual(self.errors(), [(0, "send_error"), (1, "send_error")])
        plan = trajectory.plan_resume(self.b.id, "redteam_dynamic", 0, None, objective_id="obj-a")
        attempt1 = plan.resume[("obj-a", 1)]
        rows = self.run_attack(FakeTarget(), attempts=2, scored=scored_as(True),
                               resume_state=plan.resume, completed=plan.completed)
        self.assertEqual({r["attempt"] for r in rows}, {0})  # best-of-N stopped after attempt 0
        runner._save_dynamic_results(self.b, "redteam_dynamic", 0, None, rows, plan, reason="rerun",
                                     objective_index=0)
        db.session.commit()
        self.assertEqual(self.errors(), [])
        self.assertEqual(db.session.get(Trajectory, attempt1.id).status, "archived")
        superseded = TrajectoryArchive.query.filter_by(reason="superseded").one()
        self.assertEqual((superseded.attempt, len(superseded.original_rows), superseded.trajectory["id"]),
                         (1, 1, attempt1.id))
        again = trajectory.plan_resume(self.b.id, "redteam_dynamic", 0, None, objective_id="obj-a")
        self.assertFalse(again.pending("obj-a"))
        self.assertEqual(again.solved(), {"obj-a"})

    def test_medium2_paused_time_is_not_measured_latency(self):
        benchmark_id = self.b.id

        class PauseAfterFirst(FakeTarget):
            async def send_prompt_async(inner, *, message):
                out = await FakeTarget.send_prompt_async(inner, message=message)
                if inner.calls == 1:
                    control.request_pause(benchmark_id)
                    threading.Timer(0.2, control.request_resume, args=(benchmark_id,)).start()
                return out
        start = time.perf_counter()
        with patch.object(control, "_POLL_SECONDS", 0.02):
            rows = self.run_attack(PauseAfterFirst())
        self.assertGreaterEqual(time.perf_counter() - start, 0.2)
        self.assertTrue(rows[0]["latency_measured"])
        self.assertLess(rows[0]["latency_ms"], 150)

    def test_medium4_two_label_candidate_stops_best_of_n(self):
        candidate = {"version": 1, "fields": {"candidate_success": True}}
        token = llm_overrides.set({"JUDGE_METHOD": "two_label"})
        try:
            target = FakeTarget()
            rows = self.run_attack(target, attempts=2, scored=scored_as(False, candidate))
        finally:
            llm_overrides.reset(token)
        self.assertEqual((target.calls, {r["attempt"] for r in rows}), (4, {0}))
        # Severity mode ignores the payload: the second attempt still runs.
        target = FakeTarget()
        self.run_attack(target, attempts=2, scored=scored_as(False, candidate))
        self.assertEqual(target.calls, 8)

    def test_medium4_saved_two_label_solution_counts_in_the_plan(self):
        db.session.add(Prompt(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                              prompt_text="p", response="r", status="ok", succeeded=False))
        db.session.commit()
        with patch.object(two_label, "is_solved", lambda prompt: True):
            self.assertEqual(trajectory.plan_resume(self.b.id, "redteam_dynamic", 0, None).solved(), set())
            token = llm_overrides.set({"JUDGE_METHOD": "two_label"})
            try:
                plan = trajectory.plan_resume(self.b.id, "redteam_dynamic", 0, None)
            finally:
                llm_overrides.reset(token)
        self.assertEqual(plan.completed, {("obj-a", 0): True})

    def test_medium_low5_swallowed_accounting_failure_still_stops_the_run(self):
        class Swallowing(FakeStrategy):
            async def execute_async(self, objective, prepended_conversation=None):
                try:  # like a TAP/PAIR node, which catches Exception
                    await self.target.send_prompt_async(message=message("cid", "x"))
                except Exception:  # noqa: BLE001
                    pass
                return SimpleNamespace(conversation_id="cid", related_conversations=None)

        def broken(*_a, **_k):
            raise RequestAccountingError("ledger unavailable")
        with patch.object(_ea, "RedTeamingAttack", Swallowing), \
             patch.object(trajectory.TrajectoryRecorder, "_ledger_begin", broken):
            with self.assertRaises(RequestAccountingError):
                self.run_attack(FakeTarget())

    def test_medium_low6_snapshot_follows_a_new_conversation_before_the_pause(self):
        target, rec = FakeTarget(), self.recorder()
        FakeMemory.add("dup", "user", "kept prompt")
        FakeMemory.add("dup", "assistant", "kept reply")
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("orig", "first")))
            self.assertEqual(rec.trajectory.turns_recorded, 1)
            control.request_cancel(self.b.id)  # stop right at the pause check of the next send
            with self.assertRaises(control.BenchmarkCancelled):
                asyncio.run(target.send_prompt_async(message=message("dup", "after backtrack")))
        self.assertEqual(rec.trajectory.pyrit_conversation_id, "dup")
        self.assertEqual([e["text"] for e in rec.trajectory.snapshot], ["kept prompt", "kept reply"])

    def test_medium_low7_completion_snapshots_the_returned_conversation(self):
        target, rec = FakeTarget(), self.recorder(orchestrator="tap")
        for i in (1, 2):
            FakeMemory.add("best", "user", f"best {i}")
            FakeMemory.add("best", "assistant", f"best reply {i}")
        with rec.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("other-branch", "x")))
        rec.completed("best")
        self.assertEqual((rec.trajectory.status, rec.trajectory.pyrit_conversation_id,
                          rec.trajectory.turns_recorded), ("completed", "best", 2))
        rec2 = self.recorder(objective_id="obj-b")
        with rec2.instrument(target=target):
            asyncio.run(target.send_prompt_async(message=message("kept", "y")))
        rec2.completed("not-in-memory")  # nothing to read: the last snapshot stays
        self.assertEqual(rec2.trajectory.pyrit_conversation_id, "kept")

    def test_low11_objective_cap(self):
        many = [{"id": f"o{i}", "objective": "x"} for i in range(100)]
        with self.assertRaises(ValueError):
            self.run_attack(FakeTarget(), objectives=many)
        with self.assertRaises(ValueError):
            self.run_attack(FakeTarget(), objectives=many[:1], objective_offset=99)
        rows = self.run_attack(FakeTarget(), objectives=many[:99],
                               completed={(o["id"], 0): False for o in many[:99]})
        self.assertEqual(rows, [])

    def test_low13_call_counts_cover_resumed_segments(self):
        old = Trajectory(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                         status="archived", snapshot_json="[]")
        db.session.add(old)
        db.session.flush()
        new = Trajectory(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-a",
                         status="completed", snapshot_json="[]", previous_trajectory_id=old.id)
        other = Trajectory(benchmark_id=self.b.id, attack_slug="redteam_dynamic", objective_id="obj-b",
                           status="archived", snapshot_json="[]")
        db.session.add_all([new, other])
        db.session.flush()
        for tid, stage in ((old.id, "target_send"), (old.id, "target_send"), (new.id, "target_send"),
                           (new.id, "judge_internal"), (other.id, "target_send")):
            db.session.add(TrajectoryCall(trajectory_id=tid, call_id=str(uuid.uuid4()), stage=stage))
        db.session.commit()
        counts = trajectory.conversation_call_counts(self.b.id)
        self.assertEqual(counts, {("redteam_dynamic", "obj-a", 0, 0): {"target_send": 3, "judge_internal": 1}})

    def test_low14_rows_are_saved_before_the_wait_between_objectives(self):
        persisted, seen = [], []

        async def sleep(_seconds):
            seen.append(len(persisted))
        objectives = [{"id": "obj-a", "objective": "first"}, {"id": "obj-b", "objective": "second"}]
        with patch.object(dynamic_attack.asyncio, "sleep", sleep):
            self.run_attack(FakeTarget(), objectives=objectives, delay=5, persist=persisted.append)
        self.assertEqual(seen, [1, 2])


if __name__ == "__main__":
    unittest.main()
