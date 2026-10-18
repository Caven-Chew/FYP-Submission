"""Offline tests for Task 8.6: resuming and retrying multi-turn attacks from saved trajectories.

Run: .venv/bin/python -B tools/run_isolated_app_test.py test_dynamic_resume.py
The imported fixture sets a disposable database and artifact folder before the
app is imported and stubs PyRIT. run_dynamic_attack is faked here; the real
strategies are exercised by tools/dynamic_resume_check.py.
"""
import asyncio
import json
import types
import unittest
from unittest.mock import AsyncMock, patch

import test_run_model_record as fixture  # noqa: F401; isolation before app import

from app.engine import runner
from app.extensions import db
from app.models import Benchmark, ConversationRetry, Prompt, PromptStatus, RequestAttempt, Target
from app.trajectory_models import Trajectory, TrajectoryArchive, TrajectoryCall, TrajectoryTurn

SLUG = "redteam_dynamic"
OBJECTIVES = [{"id": "objective-one", "objective": "keep"},
              {"id": "objective-two", "objective": "resume"},
              {"id": "objective-three", "objective": "new"}]


def turn_rows(oid, n=2, **extra):
    return [dict(sequence=OBJECTIVES.index(next(o for o in OBJECTIVES if o["id"] == oid)) * 100 + 2 * i + 1,
                 prompt_text=f"{oid} turn {i + 1}", response="reply", objective_id=oid, status="ok",
                 severity=1, succeeded=False, severity_source="judge", latency_measured=False, attempt=0,
                 **extra) for i in range(n)]


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fixture.create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        target = Target(name="resume target", provider="openai", model_name="m")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="resume", target_id=target.id, status="interrupted",
                           attack_slugs_json=json.dumps([SLUG]))
        db.session.add(self.b)
        db.session.flush()
        self.kept = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-one", sequence=1,
                           prompt_text="kept turn", response="kept reply", status="ok", severity=1)
        db.session.add(self.kept)
        self.done = self.trajectory("objective-one", "completed")
        self.stopped = self.trajectory("objective-two", "interrupted", pairs=2)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def trajectory(self, oid, status, pairs=0, **kw):
        snapshot = []
        for i in range(pairs):
            snapshot += [{"role": "user", "text": f"saved {i + 1}"}, {"role": "assistant", "text": f"saved reply {i + 1}"}]
        t = Trajectory(benchmark_id=self.b.id, attack_slug=SLUG, objective_id=oid, status=status,
                       snapshot_json=json.dumps(snapshot), turns_recorded=pairs, max_turns=4, **kw)
        db.session.add(t)
        db.session.flush()
        return t

    def patches(self, dynamic, objectives=OBJECTIVES):
        target = types.SimpleNamespace(aclose=AsyncMock())
        # The runner consumes resume_state as attempts finish, so keep a copy
        # of what each call was given.
        self.resume_seen = []

        async def recording(**kwargs):
            self.resume_seen.append(dict(kwargs.get("resume_state") or {}))
            return await dynamic(**kwargs)
        return [patch.object(runner, "build_target", return_value=target),
                patch.object(runner, "build_judge_llm", return_value=object()),
                patch.object(runner, "build_attacker_llm", return_value=object()),
                patch.object(runner, "resolve_objectives", AsyncMock(return_value=objectives)),
                patch.object(runner, "run_dynamic_attack", recording),
                patch.object(runner, "prune", lambda *a, **k: 0),
                patch.object(runner, "ATTACK_DELAY", 0)]

    def run_with(self, coro_factory, dynamic, objectives=OBJECTIVES):
        ps = self.patches(dynamic, objectives)
        for p in ps:
            p.start()
        try:
            asyncio.run(coro_factory())
        finally:
            for p in reversed(ps):
                p.stop()
        db.session.expire_all()

    # ---- the benchmark Resume path (_run_async -> _run_attacks) -------------
    def test_resume_keeps_completed_resumes_unfinished_and_runs_new(self):
        setup = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, prompt_text="", status=PromptStatus.SETUP_ERROR,
                       error="earlier setup failure")
        db.session.add(setup)
        db.session.commit()
        stopped_id = self.stopped.id
        results = (turn_rows("objective-two", recovery="resumed", trajectory_id=99,
                             superseded_trajectory_id=stopped_id, note="Resumed from turn 3 (2 turns replayed)")
                   + turn_rows("objective-three"))
        dynamic = AsyncMock(return_value=results)
        self.run_with(lambda: runner._run_async(self.b.id), dynamic)
        kwargs = dynamic.call_args.kwargs
        self.assertTrue(kwargs["record"])
        self.assertEqual((kwargs["roster_index"], kwargs["roster_label"]), (0, None))
        self.assertEqual(list(self.resume_seen[0]), [("objective-two", 0)])
        self.assertEqual(self.resume_seen[0][("objective-two", 0)].id, stopped_id)
        self.assertEqual(kwargs["completed"], {("objective-one", 0): False})
        self.assertTrue(callable(kwargs["persist"]))
        self.assertIsNotNone(db.session.get(Prompt, self.kept.id))
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-two").count(), 2)
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-three").count(), 2)
        self.assertEqual(Prompt.query.filter_by(status=PromptStatus.SETUP_ERROR).count(), 0)
        reasons = sorted((a.reason, a.objective_id, len(a.original_rows)) for a in TrajectoryArchive.query.all())
        self.assertEqual(reasons, [("restart", None, 1), ("resume", "objective-two", 0)])
        resume_archive = TrajectoryArchive.query.filter_by(reason="resume").one()
        self.assertEqual(resume_archive.trajectory["id"], stopped_id)
        self.assertEqual(ConversationRetry.query.count(), 0)  # no rows were replaced for display
        bench = db.session.get(Benchmark, self.b.id)
        self.assertEqual(bench.status, "completed")
        self.assertIn(SLUG, bench.completed_attacks)

    def test_rows_handed_to_persist_are_written_once(self):
        async def run(**kwargs):
            batch = turn_rows("objective-three")
            kwargs["persist"](batch)
            self.assertEqual(Prompt.query.filter_by(objective_id="objective-three").count(), 2)
            return batch + turn_rows("objective-two")
        self.run_with(lambda: runner._run_async(self.b.id), AsyncMock(side_effect=run))
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-three").count(), 2)
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-two").count(), 2)

    def test_azure_restart_archives_partial_rows(self):
        slug = "azure_fake"  # no Azure attack is configured in app/attacks.py
        self.b.attack_slugs_json = json.dumps([slug])
        db.session.add(Prompt(benchmark_id=self.b.id, attack_slug=slug, prompt_text="partial", status="ok"))
        db.session.commit()
        azure = AsyncMock(return_value=[])
        with patch.object(runner, "run_azure_redteam_attack", azure), \
             patch.object(runner, "by_slug", return_value={"slug": slug, "kind": "azure_redteam", "name": "az"}):
            self.run_with(lambda: runner._run_async(self.b.id), AsyncMock())
        azure.assert_awaited_once()
        archive = TrajectoryArchive.query.filter_by(attack_slug=slug).one()
        self.assertEqual((archive.reason, archive.original_rows[0]["prompt_text"]), ("restart", "partial"))
        self.assertEqual(Prompt.query.filter_by(attack_slug=slug).count(), 0)

    # ---- Retry failed / Skip failed (_recover_async) ------------------------
    def test_skip_failed_leaves_attempted_conversations(self):
        dynamic = AsyncMock(return_value=turn_rows("objective-three"))
        self.run_with(lambda: runner._recover_async(self.b.id, retry_failed=False), dynamic)
        self.assertEqual([c.kwargs["objectives"][0]["id"] for c in dynamic.call_args_list], ["objective-three"])
        self.assertEqual(db.session.get(Trajectory, self.stopped.id).status, "interrupted")
        # The stopped conversation has no rows yet, so the attack is not complete.
        bench = db.session.get(Benchmark, self.b.id)
        self.assertNotIn(SLUG, bench.completed_attacks)
        self.assertEqual(bench.status, "failed")
        self.assertIn("unfinished", bench.status_detail)

    def test_retry_failed_resumes_and_archives_the_failed_row(self):
        failed = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=1,
                        prompt_text="resume", response="", status=PromptStatus.SEND_ERROR, error="offline")
        db.session.add(failed)
        self.stopped.status = "failed"
        db.session.commit()
        stopped_id = self.stopped.id

        async def run(**kwargs):
            oid = kwargs["objectives"][0]["id"]
            extra = dict(recovery="resumed", superseded_trajectory_id=stopped_id) if oid == "objective-two" else {}
            return [dict(r, sequence=2 * i + 1) for i, r in enumerate(turn_rows(oid, **extra))]
        dynamic = AsyncMock(side_effect=run)
        self.run_with(lambda: runner._recover_async(self.b.id), dynamic)
        calls = {c.kwargs["objectives"][0]["id"]: c.kwargs for c in dynamic.call_args_list}
        self.assertEqual([c.kwargs["objectives"][0]["id"] for c in dynamic.call_args_list],
                         ["objective-two", "objective-three"])
        self.assertEqual(self.resume_seen[0][("objective-two", 0)].id, stopped_id)
        self.assertEqual(calls["objective-two"]["objective_offset"], 1)
        self.assertEqual(Prompt.query.filter_by(status=PromptStatus.SEND_ERROR).count(), 0)
        rows = Prompt.query.filter_by(objective_id="objective-two").order_by(Prompt.sequence).all()
        self.assertEqual([r.sequence for r in rows], [101, 103])
        self.assertFalse(any("rerun from the beginning" in (r.note or "") for r in rows))
        archive = TrajectoryArchive.query.filter_by(objective_id="objective-two").one()
        self.assertEqual((archive.reason, archive.original_rows[0]["error"]), ("resume", "offline"))
        self.assertEqual(ConversationRetry.query.one().id, archive.conversation_retry_id)
        # objective-three never ran before, so its rows are not labelled a rerun.
        new_rows = Prompt.query.filter_by(objective_id="objective-three").all()
        self.assertEqual(len(new_rows), 2)
        self.assertFalse(any("rerun" in (r.note or "") for r in new_rows))
        self.assertEqual(db.session.get(Benchmark, self.b.id).status, "completed")

    def test_rerun_keeps_completed_best_of_n_attempts(self):
        attempt0 = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=101,
                          prompt_text="attempt one", response="reply", status="ok", severity=1)
        attempt1 = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=10001,
                          prompt_text="attempt two", response="", status=PromptStatus.SEND_ERROR, error="offline")
        db.session.add_all([attempt0, attempt1])
        self.stopped.status = "completed"
        db.session.commit()
        dynamic = AsyncMock(return_value=[dict(r, sequence=10001, attempt=1) for r in turn_rows("objective-two", 1)])
        self.run_with(lambda: runner._rerun_prompt(self.b.id, attempt1.id, "completed"), dynamic)
        self.assertEqual(dynamic.call_args.kwargs["completed"], {("objective-two", 0): False})
        # SQLite may reuse a deleted row's id, so check by content, not by id.
        self.assertEqual(db.session.get(Prompt, attempt0.id).prompt_text, "attempt one")
        self.assertEqual(Prompt.query.filter_by(status=PromptStatus.SEND_ERROR).count(), 0)
        archive = TrajectoryArchive.query.one()
        self.assertEqual((archive.reason, archive.attempt, len(archive.original_rows)), ("rerun", 1, 1))
        self.assertEqual(archive.original_rows[0]["prompt_text"], "attempt two")
        new = Prompt.query.filter_by(objective_id="objective-two", sequence=10101).one()
        self.assertTrue(new.note.startswith("Conversation rerun from the beginning."))

    # ---- per-conversation Resume ---------------------------------------------
    def test_resume_trajectory_runs_only_that_objective(self):
        self.b.status = "stopped"
        db.session.commit()
        dynamic = AsyncMock(return_value=turn_rows("objective-two", recovery="resumed",
                                                   superseded_trajectory_id=self.stopped.id))
        self.run_with(lambda: runner._resume_trajectory(self.b.id, self.stopped.id, "stopped"), dynamic)
        dynamic.assert_awaited_once()
        self.assertEqual(dynamic.call_args.kwargs["objectives"], [OBJECTIVES[1]])
        self.assertIn(("objective-two", 0), self.resume_seen[0])
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-two").count(), 2)
        bench = db.session.get(Benchmark, self.b.id)
        # objective-three (in OBJECTIVES) never ran, so the previous status is restored with a note.
        self.assertEqual(bench.status, "stopped")
        self.assertNotIn(SLUG, bench.completed_attacks)
        self.assertIn("still unfinished", bench.status_detail or "")
        self.assertEqual(bench.resolved_models[-1]["phase"], "resume_conversation")
        with self.assertRaises(ValueError):
            asyncio.run(runner._resume_trajectory(self.b.id, self.done.id, "stopped"))

    def test_resume_of_the_last_unfinished_conversation_completes_the_run(self):
        # Only objective-one (kept) and objective-two (resumed) exist: nothing is left afterwards.
        dynamic = AsyncMock(return_value=turn_rows("objective-two", recovery="resumed",
                                                   superseded_trajectory_id=self.stopped.id))
        self.run_with(lambda: runner._resume_trajectory(self.b.id, self.stopped.id, "interrupted"), dynamic,
                      objectives=OBJECTIVES[:2])
        bench = db.session.get(Benchmark, self.b.id)
        self.assertEqual(bench.status, "completed")
        self.assertIn(SLUG, bench.completed_attacks)
        self.assertIsNone(bench.status_detail)

    def test_resume_route_csrf_ownership_status_and_double_click(self):
        client = self.app.test_client()
        url = f"/benchmarks/{self.b.id}/trajectories/{self.stopped.id}/resume"
        self.assertEqual(client.post(url).status_code, 403)
        with client.session_transaction() as sess:
            sess["csrf_token"] = "t"
        data = {"csrf_token": "t"}
        other = Benchmark(name="other", target_id=self.b.target_id, status="completed", attack_slugs_json="[]")
        db.session.add(other)
        db.session.commit()
        self.assertEqual(client.post(f"/benchmarks/{other.id}/trajectories/{self.stopped.id}/resume",
                                     data=data).status_code, 404)
        self.assertEqual(client.post(f"/benchmarks/{self.b.id}/trajectories/{self.done.id}/resume",
                                     data=data).status_code, 400)
        with patch("app.web.views_recovery.start_conversation_resume") as start:
            self.assertEqual(client.post(url, data=data).status_code, 302)
            self.assertEqual(client.post(url, data=data).status_code, 302)
            start.assert_called_once()
            self.assertEqual(start.call_args.args[1:], (self.b.id, self.stopped.id, "interrupted"))
        self.assertEqual(db.session.get(Benchmark, self.b.id).status, "running")

    def test_trajectories_json_and_partials(self):
        rows = [Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=101 + 2 * i,
                       prompt_text=f"turn {i}", response="reply", status="ok", severity=1) for i in range(2)]
        db.session.add_all(rows)
        db.session.add(TrajectoryTurn(trajectory_id=self.stopped.id, turn_index=1, prompt_text="saved 1",
                                      response_text="saved reply 1", status="ok", request_call_id="call-1"))
        db.session.add(TrajectoryCall(trajectory_id=self.stopped.id, call_id="call-1", stage="target_send", turn_index=1))
        db.session.add(RequestAttempt(benchmark_id=self.b.id, call_id="call-1", stage="target_send", attack_slug=SLUG,
                                      attempt_number=1, max_attempts=1, status="returned"))
        db.session.add(TrajectoryArchive(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two",
                                         reason="resume", original_rows_json="[]",
                                         trajectory_json=json.dumps({"id": 5, "turns_recorded": 2, "turns": []})))
        db.session.commit()
        client = self.app.test_client()
        data = client.get(f"/benchmarks/{self.b.id}/trajectories.json").get_json()
        self.assertEqual([t["objective_id"] for t in data["trajectories"]], ["objective-one", "objective-two"])
        two = data["trajectories"][1]
        self.assertEqual((two["status"], two["turns_recorded"], two["ledger_call_ids"]), ("interrupted", 2, ["call-1"]))
        self.assertEqual(two["snapshot"][0]["text"], "saved 1")
        self.assertEqual(data["archives"][0]["reason"], "resume")
        self.assertEqual(client.get("/benchmarks/9999/trajectories.json").status_code, 404)
        page = client.get(f"/benchmarks/{self.b.id}").data.decode()
        self.assertIn("Multi-turn recovery (1 unfinished conversation, 1 archived)", page)
        self.assertIn("Saved trajectory: interrupted", page)
        self.assertIn(f"/benchmarks/{self.b.id}/trajectories/{self.stopped.id}/resume", page)
        self.assertIn("Resume this conversation", page)
        self.assertIn(f"trajectory #{self.stopped.id}, turn 1", page)
        self.b.status = "running"
        db.session.commit()
        page = client.get(f"/benchmarks/{self.b.id}").data.decode()
        self.assertNotIn("Resume this conversation", page)

    # ---- independent review fixes (10 October 2026) ---------------------------
    def test_high1_rerun_of_a_leftover_attempt_after_the_objective_was_solved(self):
        solved = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=101,
                        prompt_text="broke it", response="harmful reply", status="ok", severity=5, succeeded=True)
        leftover = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=10001,
                          prompt_text="later attempt", response="", status=PromptStatus.SEND_ERROR, error="offline")
        db.session.add_all([solved, leftover])
        self.stopped.status = "completed"
        late = self.trajectory("objective-two", "failed", pairs=1, attempt=1)
        db.session.commit()
        late_id = late.id
        dynamic = AsyncMock(return_value=[])
        self.run_with(lambda: runner._rerun_prompt(self.b.id, leftover.id, "completed_with_target_errors"), dynamic)
        dynamic.assert_not_awaited()  # nothing is left to run, and that is not an error
        self.assertEqual(Prompt.query.filter_by(status=PromptStatus.SEND_ERROR).count(), 0)
        archive = TrajectoryArchive.query.filter_by(reason="superseded").one()
        self.assertEqual((archive.attempt, archive.original_rows[0]["prompt_text"], archive.trajectory["id"]),
                         (1, "later attempt", late_id))
        self.assertEqual(db.session.get(Trajectory, late_id).status, "archived")
        self.assertEqual(db.session.get(Benchmark, self.b.id).status, "completed_with_target_errors")
        self.assertIsNone(db.session.get(Benchmark, self.b.id).status_detail)

    def test_high1_resume_path_supersedes_leftovers_of_solved_objectives(self):
        self.kept.succeeded = True
        db.session.add(Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-one", sequence=10001,
                              prompt_text="later attempt", response="", status=PromptStatus.SEND_ERROR))
        db.session.commit()
        dynamic = AsyncMock(return_value=turn_rows("objective-two") + turn_rows("objective-three"))
        self.run_with(lambda: runner._run_async(self.b.id), dynamic)
        self.assertEqual(Prompt.query.filter_by(status=PromptStatus.SEND_ERROR).count(), 0)
        self.assertEqual(TrajectoryArchive.query.filter_by(reason="superseded").count(), 1)
        self.assertEqual(db.session.get(Benchmark, self.b.id).status, "completed")

    def test_low14_rerun_persists_each_attempt_as_it_ends(self):
        failed = Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id="objective-two", sequence=1,
                        prompt_text="resume", response="", status=PromptStatus.SEND_ERROR, error="offline")
        db.session.add(failed)
        db.session.commit()
        seen = []

        async def run(**kwargs):
            batch = turn_rows("objective-two")
            kwargs["persist"](batch)
            seen.append(Prompt.query.filter_by(objective_id="objective-two", status="ok").count())
            return batch
        self.run_with(lambda: runner._rerun_prompt(self.b.id, failed.id, "completed"), AsyncMock(side_effect=run))
        self.assertEqual(seen, [2])  # committed before the attack returned
        self.assertEqual(Prompt.query.filter_by(objective_id="objective-two").count(), 2)
        self.assertEqual(TrajectoryArchive.query.filter_by(objective_id="objective-two").count(), 1)

    def test_low_medium8_azure_recovery_rows_get_assessment_and_refusal_flag(self):
        slug = "azure_fake"
        self.b.attack_slugs_json = json.dumps([slug])
        db.session.add(Prompt(benchmark_id=self.b.id, attack_slug=slug, sequence=0, prompt_text="p",
                              response="", status=PromptStatus.SEND_ERROR, error="offline"))
        db.session.commit()
        result = {"sequence": 0, "prompt_text": "p", "response": "I'm sorry, but I can't help with that.",
                  "succeeded": False, "status": "ok", "objective_id": "az-0", "assessment": {"payload": 1}}
        attached = []
        with patch.object(runner, "run_azure_redteam_attack", AsyncMock(return_value=[result])), \
             patch.object(runner, "by_slug", return_value={"slug": slug, "kind": "azure_redteam", "name": "az"}), \
             patch.object(runner, "attach_assessment", lambda row, r: attached.append((row.prompt_text, r))):
            self.run_with(lambda: runner._recover_async(self.b.id), AsyncMock())
        row = Prompt.query.filter_by(attack_slug=slug).one()
        self.assertTrue(row.refused)
        self.assertEqual(attached, [("p", result)])
        self.assertEqual(TrajectoryArchive.query.filter_by(attack_slug=slug, reason="rerun").count(), 1)

    def test_low12_page_loads_trajectories_once(self):
        from sqlalchemy import event
        for oid in ("objective-one", "objective-two", "objective-three"):
            for i in range(2):
                db.session.add(Prompt(benchmark_id=self.b.id, attack_slug=SLUG, objective_id=oid,
                                      sequence=OBJECTIVES.index(next(o for o in OBJECTIVES if o["id"] == oid)) * 100 + i,
                                      prompt_text=f"{oid} {i}", response="reply", status="ok", severity=1))
        self.trajectory("objective-three", "completed", pairs=2)
        db.session.commit()
        statements = []

        def count(_conn, _cursor, statement, *_args):
            if "FROM trajectories" in statement:
                statements.append(statement)
        event.listen(db.engine, "before_cursor_execute", count)
        try:
            page = self.app.test_client().get(f"/benchmarks/{self.b.id}").data.decode()
        finally:
            event.remove(db.engine, "before_cursor_execute", count)
        self.assertEqual(page.count("data-conversation-recovery="), 3)
        self.assertEqual(len(statements), 1, statements)


class AzureAssessmentTests(unittest.TestCase):
    """Azure rows go through scoring.assess_turn like multi-turn turns (coordinator request)."""

    def scan(self, scored):
        import os
        import sys
        from app.engine import azure_attack

        class RedTeam:
            def __init__(self, **_kw):
                pass

            async def scan(self, **_kw):
                return {"attack_details": [{"conversation": [{"role": "user", "content": "attack"},
                                                             {"role": "assistant", "content": "answer"}],
                                            "risk_category": "violence", "attack_strategy": "Baseline",
                                            "attack_success": True}]}
        red_team = types.ModuleType("azure.ai.evaluation.red_team")
        red_team.RedTeam = RedTeam
        red_team.RiskCategory = types.SimpleNamespace(Violence="violence")
        red_team.AttackStrategy = types.SimpleNamespace(Baseline="baseline")
        identity = types.ModuleType("azure.identity")
        identity.DefaultAzureCredential = lambda: None
        modules = {"azure": types.ModuleType("azure"), "azure.ai": types.ModuleType("azure.ai"),
                   "azure.ai.evaluation": types.ModuleType("azure.ai.evaluation"),
                   "azure.ai.evaluation.red_team": red_team, "azure.identity": identity}
        assess = AsyncMock(return_value=scored)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"AZURE_AI_PROJECT_ENDPOINT": "offline"}), \
             patch.object(azure_attack, "assess_turn", assess):
            rows = asyncio.run(azure_attack.run_azure_redteam_attack(
                attack={"slug": "az", "risk_categories": ["Violence"]}, target=object(), judge="judge",
                guard=runner.ConsecutiveFailureGuard(3)))
        self.assertEqual(assess.call_args.kwargs, {"objective": "attack", "prompt_text": "attack",
                                                   "response_text": "answer"})
        return rows[0]

    def test_assessment_is_carried_and_two_label_never_borrows_azure_success(self):
        row = self.scan({"verdict": None, "severity": None, "succeeded": False, "assessment": {"stage": "x"}})
        self.assertEqual(row["assessment"], {"stage": "x"})
        self.assertFalse(row["succeeded"])
        self.assertIsNone(row["severity"])
        self.assertEqual(row["severity_source"], "none")

    def test_severity_mode_is_unchanged(self):
        row = self.scan({"verdict": "SCORE: 4", "severity": 4, "succeeded": True, "assessment": None})
        self.assertEqual((row["severity"], row["succeeded"], row["assessment"]), (4, True, None))
        self.assertTrue(row["verdict"].startswith("azure:True | SCORE: 4"))
        failed_judge = self.scan({"verdict": "judge-error", "severity": None, "succeeded": False, "assessment": None})
        self.assertTrue(failed_judge["succeeded"])  # unchanged fallback to Azure's own verdict


if __name__ == "__main__":
    unittest.main()
