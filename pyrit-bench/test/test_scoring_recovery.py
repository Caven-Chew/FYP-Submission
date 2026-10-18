"""Offline regression for static judge-only recovery (Task 8.1/R2).

Run: .venv/bin/python -B test/test_scoring_recovery.py
The imported fixture installs PyRIT stubs, a throwaway SQLite database and a
disposable artifact path before the application is imported. No model or target
request is permitted by these tests.
"""
import asyncio
import csv
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_run_model_record as fixture  # noqa: F401; isolation before app import
from sqlalchemy import inspect, text
from app import _mark_interrupted_benchmarks, _migrate_add_columns
from app.engine import llms, retry, scoring_recovery, static_attack
from app.extensions import db
from app.models import Benchmark, Prompt, PromptStatus, ScoringEvaluation, Target
from app.web.rollups import benchmark_rollups


class _Guard:
    def record(self, error):
        assert error is None


class ScoringRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fixture.create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        target = Target(name="offline target", provider="openai", model_name="unused")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="recovery", target_id=target.id, status="completed",
                           attack_slugs_json=json.dumps([fixture.STATIC["slug"]]))
        db.session.add(self.b)
        db.session.commit()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def _row(self, **changes):
        values = dict(benchmark_id=self.b.id, attack_slug=fixture.STATIC["slug"],
                      sequence=0, attempt=0, prompt_text="Please explain a harmless topic",
                      response="A benign answer.", status=PromptStatus.OK,
                      severity_source="none", verdict="judge-error: RuntimeError")
        values.update(changes)
        p = Prompt(**values)
        db.session.add(p)
        db.session.flush()
        scoring_recovery.save_context(p, "Please explain a harmless topic", None, "leak")
        db.session.commit()
        return p

    def _run_static_judge_error(self):
        attack = dict(fixture.STATIC, converter=None, sweep=None, attempts=1)
        target = type("TargetStub", (), {"last_debug": {}})()
        async def fail_after_committed_capture(*_args, **_kwargs):
            # A separate connection sees only committed data. The target reply
            # and pending evaluation must survive even if this call never does.
            with db.engine.connect() as conn:
                captured = conn.execute(text(
                    "SELECT response FROM prompts WHERE benchmark_id = :bid"),
                    {"bid": self.b.id}).scalar_one()
                pending = conn.execute(text(
                    "SELECT status FROM scoring_evaluations WHERE prompt_id IN "
                    "(SELECT id FROM prompts WHERE benchmark_id = :bid)"),
                    {"bid": self.b.id}).scalar_one()
            self.assertEqual((captured, pending), ("captured reply", "running"))
            return "judge-error: TimeoutError", None, False
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["benign seed"])), \
             patch.object(static_attack, "send_prompt", AsyncMock(return_value=("captured reply", None))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), \
             patch.object(static_attack, "judge_response", side_effect=fail_after_committed_capture):
            asyncio.run(static_attack.run_static_attack(self.b, attack, target, object(), 1, _Guard()))
        db.session.expire_all()
        return Prompt.query.filter_by(benchmark_id=self.b.id).one()

    def test_first_judge_error_commits_target_and_evaluation(self):
        p = self._run_static_judge_error()
        self.assertEqual((p.prompt_text, p.response, p.status),
                         ("benign seed", "captured reply", PromptStatus.OK))
        self.assertEqual((p.severity, p.verdict, p.severity_source),
                         (None, "judge-error: TimeoutError", "none"))
        self.assertEqual(len(p.scoring_evaluations), 1)
        first = p.scoring_evaluations[0]
        self.assertEqual((first.status, first.verdict), ("error", p.verdict))
        self.assertEqual(json.loads(first.context_json), p.scoring_context)
        self.assertEqual(self.b.compute_totals()["counted"], 1)
        self.assertEqual(self.b.compute_totals()["scored"], 0)
        self.assertTrue(p.can_retry_scoring)

    def test_retry_uses_exact_saved_request_and_preserves_evidence_and_history(self):
        p = self._run_static_judge_error()
        original = (p.prompt_text, p.response, p.converter_used, p.latency_ms,
                    p.status, p.scoring_context_json)
        saved = p.scoring_context["request"]
        first_id = p.scoring_evaluations[0].id
        attempt = scoring_recovery.claim_retry(self.b, p)
        self.assertEqual(self.b.status, "scoring")
        judge = object()
        async def verdict(_judge, request, *, success_threshold):
            self.assertIs(_judge, judge)
            self.assertEqual(request, saved)
            self.assertEqual(success_threshold, p.scoring_context["success_threshold"])
            return "REASONING: benign control\nSCORE: 1", 1, False
        # The retry must be independent of today's model, rubric template,
        # dataset and converter, and cannot reach any target path.
        with patch.dict(os.environ, {"JUDGE_MODEL": "different-now"}), \
             patch.object(scoring_recovery, "build_judge_llm", return_value=judge), \
             patch.object(scoring_recovery, "judge_request", side_effect=verdict) as call, \
             patch.object(scoring_recovery, "render_judge_request", side_effect=AssertionError("template rebuilt")), \
             patch.object(static_attack, "load_dataset_prompts", side_effect=AssertionError("dataset read")), \
             patch.object(static_attack, "apply_converters", side_effect=AssertionError("converter called")), \
             patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
            asyncio.run(scoring_recovery.retry_evaluation(attempt.id))
        db.session.expire_all()
        p = db.session.get(Prompt, p.id)
        self.assertEqual(call.await_count, 1)
        self.assertEqual((p.prompt_text, p.response, p.converter_used, p.latency_ms,
                          p.status, p.scoring_context_json), original)
        self.assertEqual([(e.id, e.status) for e in p.scoring_evaluations],
                         [(first_id, "error"), (attempt.id, "scored")])
        self.assertEqual((p.severity, p.severity_source, p.succeeded), (1, "judge", False))
        self.assertEqual(self.b.status, "completed")
        self.assertFalse(p.can_retry_scoring)

    def test_invalid_retry_then_valid_keeps_all_attempts(self):
        p = self._row()
        first = scoring_recovery.new_evaluation(p)
        with patch.object(scoring_recovery, "judge_request", AsyncMock(return_value=("no numeric result", None, False))):
            asyncio.run(scoring_recovery.evaluate(p, first, object()))
        self.assertEqual(first.status, "invalid")
        retry = scoring_recovery.claim_retry(self.b, p)
        with patch.object(scoring_recovery, "build_judge_llm", return_value=object()), \
             patch.object(scoring_recovery, "judge_request", AsyncMock(return_value=("SCORE: 2", 2, False))):
            asyncio.run(scoring_recovery.retry_evaluation(retry.id))
        db.session.expire_all()
        self.assertEqual([(e.status, e.severity) for e in p.scoring_evaluations],
                         [("invalid", None), ("scored", 2)])
        self.assertEqual(p.severity, 2)

    def test_retry_restores_database_status_despite_stale_caller(self):
        for i, terminal in enumerate(scoring_recovery.TERMINAL):
            with self.subTest(terminal=terminal):
                p = self._row(sequence=i)
                original = (p.prompt_text, p.response, p.scoring_context_json)
                self.b.status = "scoring"
                db.session.commit()
                bid = self.b.id
                self.assertEqual(self.b.status, "scoring")
                # Another worker finishes after the request loaded its object.
                with db.engine.begin() as conn:
                    conn.execute(text("UPDATE benchmarks SET status=:status WHERE id=:id"),
                                 {"status": terminal, "id": bid})
                self.assertEqual(self.b.status, "scoring")
                attempt = scoring_recovery.claim_retry(self.b, p)
                self.assertEqual(attempt.previous_benchmark_status, terminal)
                with patch.object(scoring_recovery, "build_judge_llm", return_value=object()), \
                     patch.object(scoring_recovery, "judge_request", AsyncMock(return_value=("SCORE: 1", 1, False))), \
                     patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
                    self.assertTrue(asyncio.run(scoring_recovery.retry_evaluation(attempt.id)))
                db.session.expire_all()
                self.assertEqual(self.b.status, terminal)
                self.assertEqual((p.prompt_text, p.response, p.scoring_context_json), original)

    def test_retry_uses_current_terminal_status_and_rejects_lost_claim(self):
        p = self._row()
        bid = self.b.id
        self.assertEqual(self.b.status, "completed")
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE benchmarks SET status='stopped' WHERE id=:id"), {"id": bid})
        self.assertEqual(self.b.status, "completed")
        attempt = scoring_recovery.claim_retry(self.b, p)
        self.assertEqual(attempt.previous_benchmark_status, "stopped")
        scoring_recovery.fail_evaluation(attempt, "offline fixture")
        self.b.status = "completed"
        db.session.commit()
        count = ScoringEvaluation.query.count()
        query_type = type(Benchmark.query)
        update = query_type.update

        def change_before_update(query, *args, **kwargs):
            with db.engine.begin() as conn:
                conn.execute(text("UPDATE benchmarks SET status='stopped' WHERE id=:id"), {"id": bid})
            return update(query, *args, **kwargs)

        with patch.object(query_type, "update", autospec=True, side_effect=change_before_update):
            with self.assertRaises(ValueError):
                scoring_recovery.claim_retry(self.b, p)
        self.assertEqual(ScoringEvaluation.query.count(), count)
        self.assertEqual(db.session.get(Benchmark, bid).status, "stopped")

    def test_finished_worker_does_not_release_the_next_retry_claim(self):
        from app.activity import get_activity, set_activity, clear_activity
        first, second = self._row(), self._row(sequence=1)
        attempt = scoring_recovery.claim_retry(self.b, first)
        bid, pid = self.b.id, second.id
        run = scoring_recovery.retry_evaluation
        next_ids = []

        async def finish_then_claim(evaluation_id):
            released = await run(evaluation_id)
            with self.app.app_context():
                b, p = db.session.get(Benchmark, bid), db.session.get(Prompt, pid)
                next_ids.append(scoring_recovery.claim_retry(b, p).id)
                set_activity(bid, "judging", "next retry")
            return released

        with patch.object(scoring_recovery.threading, "Thread") as thread, \
             patch.object(scoring_recovery, "retry_evaluation", side_effect=finish_then_claim), \
             patch.object(scoring_recovery, "build_judge_llm", return_value=object()), \
             patch.object(scoring_recovery, "judge_request", AsyncMock(return_value=("SCORE: 1", 1, False))):
            scoring_recovery.start_scoring_retry(self.app, attempt.id)
            thread.call_args.kwargs["target"]()
        db.session.expire_all()
        self.assertEqual(db.session.get(Benchmark, bid).status, "scoring")
        self.assertEqual(db.session.get(ScoringEvaluation, next_ids[0]).status, "running")
        self.assertEqual(get_activity(bid)["detail"], "next retry")
        clear_activity(bid)

    def test_rejects_scored_fallback_legacy_dynamic_and_changed_evidence(self):
        p = self._row()
        for source in ("judge", "fallback"):
            p.severity, p.severity_source = 1, source
            db.session.commit()
            self.assertFalse(p.can_retry_scoring)
        p.severity = None
        p.scoring_context_json = None
        db.session.commit()
        self.assertIn("not saved", p.scoring_retry_blocker)
        dynamic = self._row(attack_slug="dynamic-legacy", sequence=1,
                            scoring_context_json=None)
        dynamic.scoring_context_json = None
        db.session.commit()
        self.assertFalse(dynamic.can_retry_scoring)
        p = self._row(sequence=2)
        p.response = "changed reply"
        db.session.commit()
        self.assertIn("no longer matches", p.scoring_retry_blocker)
        p.response = "A benign answer."
        p.prompt_text = "changed prompt"
        db.session.commit()
        self.assertIn("no longer matches", p.scoring_retry_blocker)

    def test_endpoint_identity_change_blocks_without_judge_call(self):
        with patch.dict(os.environ, {"LLM_ENDPOINT": "https://one.invalid/v1"}), \
             patch.object(scoring_recovery, "resolve_role", return_value={
                 "provider": "profllm", "model": "judge", "key_env": "LLM_API_KEY"}):
            p = self._row()
        attempt = scoring_recovery.claim_retry(self.b, p)
        with patch.dict(os.environ, {"LLM_ENDPOINT": "https://two.invalid/v1"}), \
             patch.object(scoring_recovery, "build_judge_llm", side_effect=AssertionError("judge built")), \
             patch.object(scoring_recovery, "judge_request", side_effect=AssertionError("judge called")):
            asyncio.run(scoring_recovery.retry_evaluation(attempt.id))
        db.session.expire_all()
        self.assertEqual(attempt.status, "error")
        self.assertIn("endpoint differs", attempt.verdict)
        self.assertEqual(self.b.status, "completed")
        self.assertEqual(p.response, "A benign answer.")

    def test_route_requires_csrf_parent_match_and_idle_claim(self):
        p = self._row()
        path = f"/benchmarks/{self.b.id}/prompts/{p.id}/retry-scoring"
        self.assertEqual(self.client.post(path).status_code, 403)
        with self.client.session_transaction() as session:
            session["csrf_token"] = "offline-token"
        other_target = Target(name="other", provider="openai", model_name="unused")
        db.session.add(other_target)
        db.session.flush()
        other = Benchmark(name="other", target_id=other_target.id, status="completed")
        db.session.add(other)
        db.session.commit()
        with patch("app.web.views_benchmarks.start_scoring_retry") as start:
            self.assertEqual(self.client.post(
                f"/benchmarks/{other.id}/prompts/{p.id}/retry-scoring",
                data={"csrf_token": "offline-token"}).status_code, 404)
            self.b.status = "running"
            db.session.commit()
            self.assertEqual(self.client.post(path, data={"csrf_token": "offline-token"}).status_code, 409)
            self.b.status = "completed"
            db.session.commit()
            self.assertEqual(self.client.post(path, data={"csrf_token": "offline-token"}).status_code, 302)
            self.assertEqual(start.call_count, 1)
            self.assertEqual(self.client.post(path, data={"csrf_token": "offline-token"}).status_code, 409)
            self.assertEqual(start.call_count, 1)

    def test_process_interruption_retains_retryable_capture(self):
        p = self._row()
        evaluation = scoring_recovery.claim_retry(self.b, p)
        self.assertEqual(evaluation.status, "running")
        _mark_interrupted_benchmarks()
        db.session.expire_all()
        self.assertEqual(evaluation.status, "interrupted")
        self.assertIn("outcome unknown", evaluation.verdict)
        self.assertEqual(self.b.status, "completed")
        self.assertTrue(p.can_retry_scoring)
        self.assertEqual(p.response, "A benign answer.")

    def test_legacy_column_migration_preserves_score_without_inventing_context(self):
        p = self._row(severity=2, severity_source="judge", verdict="SCORE: 2")
        pid = p.id
        db.session.remove()
        # This simulates the exact old table shape in the isolated test DB.
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE prompts DROP COLUMN scoring_context_json"))
        db.engine.dispose()  # startup would also create fresh connections
        self.assertNotIn("scoring_context_json", {c["name"] for c in inspect(db.engine).get_columns("prompts")})
        _migrate_add_columns()
        _migrate_add_columns()  # startup migration remains idempotent
        with db.engine.connect() as conn:
            row = conn.execute(text(
                "SELECT response, severity, verdict, scoring_context_json "
                "FROM prompts WHERE id = :pid"), {"pid": pid}).one()
            evaluations = conn.execute(text(
                "SELECT count(*) FROM scoring_evaluations WHERE prompt_id = :pid"),
                {"pid": pid}).scalar_one()
        self.assertEqual(tuple(row), ("A benign answer.", 2, "SCORE: 2", None))
        self.assertEqual(evaluations, 0)

    def test_retry_builds_original_judge_despite_changed_environment(self):
        p = self._row()
        saved = p.scoring_context["judge"]
        attempt = scoring_recovery.claim_retry(self.b, p)
        seen = []
        async def inspect_judge(judge, request, *, success_threshold):
            seen.append((judge.kw, request, success_threshold))
            return "SCORE: 1", 1, False
        with patch.dict(os.environ, {
                 "JUDGE_PROVIDER": "venice", "JUDGE_MODEL": "different-model",
                 "JUDGE_API_KEY_ENV": "VENICE_API_KEY"}), \
             patch.object(llms, "ensure_pyrit_initialized", lambda: None), \
             patch.object(scoring_recovery, "judge_request", side_effect=inspect_judge), \
             patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
            asyncio.run(scoring_recovery.retry_evaluation(attempt.id))
        self.assertEqual(attempt.status, "scored")
        self.assertEqual(len(seen), 1)
        args, request, threshold = seen[0]
        self.assertEqual(args["model_name"], saved["model"])
        self.assertEqual(args["endpoint"], llms._OPENAI_COMPAT_ENDPOINT[saved["provider"]])
        self.assertEqual(args["api_key"], fixture.SECRET_OPENAI)
        self.assertEqual(request, p.scoring_context["request"])
        self.assertEqual(threshold, p.scoring_context["success_threshold"])

    def test_transient_judge_failure_stops_at_retry_limit_without_target_call(self):
        p = self._run_static_judge_error()
        original = (p.prompt_text, p.response, p.status, p.scoring_context_json)
        attempt = scoring_recovery.claim_retry(self.b, p)
        class FailingJudge:
            calls = 0
            requests = []
            async def send_prompt_async(self, *, message):
                self.calls += 1
                self.requests.append(message.prompt)
                raise TimeoutError("transient judge failure")
        judge = FailingJudge()
        with patch.object(scoring_recovery, "build_judge_llm", return_value=judge), \
             patch.object(retry, "RETRY_MAX_ATTEMPTS", 3), \
             patch.object(retry, "RETRY_BASE_DELAY", 0), \
             patch.object(retry, "RETRY_MAX_DELAY", 0), \
             patch.object(retry.asyncio, "sleep", new_callable=AsyncMock), \
             patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
            asyncio.run(scoring_recovery.retry_evaluation(attempt.id))
        db.session.expire_all()
        self.assertEqual(judge.calls, 3)
        self.assertEqual(judge.requests, [p.scoring_context["request"]] * 3)
        self.assertEqual((attempt.status, attempt.verdict),
                         ("error", "judge-error: TimeoutError"))
        self.assertEqual((p.prompt_text, p.response, p.status, p.scoring_context_json), original)
        self.assertEqual([e.status for e in p.scoring_evaluations], ["error", "error"])
        self.assertEqual(self.b.status, "completed")

    def test_worker_launch_failure_records_error_and_restores_status(self):
        p = self._row()
        path = f"/benchmarks/{self.b.id}/prompts/{p.id}/retry-scoring"
        with self.client.session_transaction() as session:
            session["csrf_token"] = "offline-token"
        with patch("app.web.views_benchmarks.start_scoring_retry",
                   side_effect=RuntimeError("thread start failed")):
            with self.assertRaises(RuntimeError):
                self.client.post(path, data={"csrf_token": "offline-token"})
        db.session.expire_all()
        self.assertEqual(self.b.status, "completed")
        self.assertEqual(len(p.scoring_evaluations), 1)
        self.assertEqual(p.scoring_evaluations[0].status, "error")
        self.assertIn("worker could not start", p.scoring_evaluations[0].verdict)
        self.assertEqual(p.response, "A benign answer.")
        self.assertTrue(p.can_retry_scoring)

    def test_worker_releases_claim_if_evaluation_disappears_before_start(self):
        p = self._row()
        evaluation = scoring_recovery.claim_retry(self.b, p)
        bid = self.b.id
        with patch.object(scoring_recovery.threading, "Thread") as thread:
            scoring_recovery.start_scoring_retry(self.app, evaluation.id)
            work = thread.call_args.kwargs["target"]
            db.session.delete(evaluation)
            db.session.commit()
            work()
        self.assertEqual(db.session.get(Benchmark, bid).status, "completed")

    def test_saved_rubric_survives_catalog_change(self):
        p = self._row(severity=2, severity_source="judge")
        from app.web import rollups
        with patch.object(rollups, "by_slug", return_value={"name": "changed", "category": "test", "scoring": "harm"}):
            result = rollups.benchmark_rollups(self.b)
        self.assertEqual(set(result["rubric_stats"]), {"leak"})
        self.assertEqual(result["rollup"][0]["rubric"], "leak")

    def test_captured_scored_denominators_and_exports(self):
        no_score = self._row()
        scoring_recovery.fail_evaluation(
            scoring_recovery.new_evaluation(no_score), "judge-error: TimeoutError")
        synthetic = self._row(sequence=1, response="Another captured reply",
                              severity=5, severity_source="synthetic", succeeded=True)
        excluded = self._row(sequence=2, status=PromptStatus.SEND_ERROR,
                             response="", severity=None, scoring_context_json=None)
        excluded.scoring_context_json = None
        db.session.commit()
        totals = self.b.compute_totals()
        self.assertEqual((totals["rows"], totals["counted"], totals["scored"],
                          totals["unscored"], totals["excluded"], totals["successes"]),
                         (3, 2, 0, 2, 1, 0))
        self.assertIsNone(self.b.risk_score)
        roll = benchmark_rollups(self.b)
        self.assertEqual(roll["totals"]["unscored"], 2)
        status = self.client.get(f"/benchmarks/{self.b.id}/status.json").get_json()
        self.assertEqual((status["captured"], status["scored"], status["unscored"]), (2, 0, 2))
        page = self.client.get(f"/benchmarks/{self.b.id}").get_data(as_text=True)
        self.assertIn("Responses captured", page)
        self.assertIn("Responses scored", page)
        self.assertIn("Retry scoring", page)
        self.assertIn("Scoring history (1)", page)
        rows = list(csv.DictReader(io.StringIO(
            self.client.get(f"/benchmarks/{self.b.id}/csv").get_data(as_text=True))))
        self.assertEqual([(r["counts_toward_stats"], r["has_valid_score"])
                          for r in rows], [("1", "0"), ("1", "0"), ("0", "0")])
        self.assertEqual(rows[0]["scoring_evaluations"], "1")
        self.assertEqual(self.client.get(f"/benchmarks/{self.b.id}/pdf").status_code, 200)
        from pypdf import PdfReader
        pdf = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(
            self.client.get(f"/benchmarks/{self.b.id}/pdf").data)).pages)
        self.assertIn("Responses captured", pdf)
        self.assertIn("Responses scored", pdf)
        self.assertIn("Captured, unscored", pdf)

    def test_multi_turn_turn_with_saved_request_is_retried_without_any_target_send(self):
        # Task 8: judge-only recovery covers conversation turns; their judge request is
        # saved when the row is written (scoring.assess_turn -> runner._dynamic_row).
        from app.attacks import ATTACKS
        from app.engine import runner, scoring
        dynamic = next(a for a in ATTACKS if a.get("kind") == "dynamic")
        result = {"sequence": 100, "prompt_text": "attacker turn", "response": "target reply", "error": None,
                  "succeeded": False, "severity": None, "verdict": "judge-error: TimeoutError",
                  "status": PromptStatus.OK, "severity_source": "none", "latency_measured": True,
                  "latency_ms": 10, "taxonomy_category": None, "objective_id": "obj-1",
                  "attempt_group": "obj-1",
                  "scoring_context": scoring.build_scoring_context(
                      "attacker turn", "target reply", "attacker turn", None, "harm")}
        p = runner._dynamic_row(self.b, dynamic["slug"], 0, "roster-label", result, 100)
        db.session.add(p)
        db.session.commit()
        self.assertEqual(p.scoring_context["converter"], "roster-label")
        self.assertIsNone(p.scoring_retry_blocker)
        attempt = scoring_recovery.claim_retry(self.b, p)
        judge = object()
        with patch.object(scoring_recovery, "build_judge_llm", return_value=judge), \
             patch.object(scoring_recovery, "judge_request",
                          AsyncMock(return_value=("REASONING: x\nSCORE: 2", 2, False))) as call, \
             patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
            asyncio.run(scoring_recovery.retry_evaluation(attempt.id))
        db.session.expire_all()
        p = db.session.get(Prompt, p.id)
        self.assertEqual(call.await_count, 1)
        self.assertEqual((p.severity, p.severity_source, p.response), (2, "judge", "target reply"))
        self.assertEqual(p.scoring_evaluations[-1].status, "scored")
        azure = next((a for a in ATTACKS if a.get("kind") == "azure_redteam"), None)
        if azure is not None:
            q = self._row(attack_slug=azure["slug"], sequence=3)
            self.assertIn("multi-turn turns only", q.scoring_retry_blocker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
