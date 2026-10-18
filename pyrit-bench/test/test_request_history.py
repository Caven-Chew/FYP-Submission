"""Offline regression for Task 8.3/R3 local request-attempt accounting.

Run: .venv/bin/python -B tools/run_isolated_app_test.py test_request_history.py
The imported fixture sets disposable storage before app import. Provider calls
are synthetic, and the isolated runner blocks network sockets.
"""
import asyncio
import csv
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_run_model_record as fixture  # noqa: F401; isolation before app import
import httpx
from sqlalchemy import text

from app.engine import request_history, retry, runner, scoring, scoring_recovery, static_attack
from app.extensions import db
from app.models import (Benchmark, Prompt, PromptStatus, RequestAttempt,
                        StaticExecutionArchive, Target)


class _Reply:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class _Target:
    last_debug = {}

    def __init__(self, reply):
        self.send_prompt_async = AsyncMock(return_value=reply)


class _Guard:
    def record(self, _error):
        pass


class RequestHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fixture.create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        target = Target(name="ledger target", provider="openai", model_name="unused")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="ledger", target_id=target.id, status="completed",
                           attack_slugs_json=json.dumps([fixture.STATIC["slug"]]))
        db.session.add(self.b)
        db.session.commit()
        self.attack = dict(fixture.STATIC, converter=None, sweep=None, attempts=1)

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def _rows(self):
        db.session.expire_all()
        return RequestAttempt.query.order_by(RequestAttempt.id).all()

    def _history(self, stage="target_send", **kwargs):
        return request_history.request_history(self.b.id, stage, **kwargs)

    def test_timeout_then_success_exact_attempts_timing_and_backoff(self):
        clock = [100.0]
        calls = [0]

        async def invoke():
            calls[0] += 1
            clock[0] += 0.040 if calls[0] == 1 else 0.070
            if calls[0] == 1:
                raise TimeoutError("timeout with private provider detail")
            return "ok"

        async def fake_sleep(seconds):
            clock[0] += seconds

        with patch.object(retry, "RETRY_MAX_ATTEMPTS", 3), \
             patch.object(retry, "RETRY_BASE_DELAY", 1), \
             patch.object(retry, "RETRY_MAX_DELAY", 10), \
             patch.object(retry.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(retry.asyncio, "sleep", side_effect=fake_sleep), \
             patch("random.uniform", return_value=0):
            with self._history() as observer:
                result = asyncio.run(retry.retry_async(invoke, observer=observer))
        self.assertEqual(result, "ok")
        rows = self._rows()
        self.assertEqual([(r.attempt_number, r.max_attempts, r.status) for r in rows],
                         [(1, 3, "error"), (2, 3, "returned")])
        self.assertTrue(all(abs(actual - expected) <= 1
                            for actual, expected in zip([r.elapsed_ms for r in rows], [40, 70])))
        self.assertEqual((rows[0].reason, rows[0].delivery, rows[0].retryable),
                         ("TimeoutError", "unknown", True))
        self.assertEqual((rows[0].backoff_planned_ms, rows[0].backoff_elapsed_ms),
                         (1000, 1000))
        self.assertEqual(rows[1].delivery, "helper_returned")
        self.assertEqual(rows[0].call_id, rows[1].call_id)
        self.assertNotIn("private provider detail", json.dumps([r.as_dict() for r in rows]))

    def test_exhausted_timeout_has_no_final_backoff(self):
        calls = AsyncMock(side_effect=TimeoutError("not delivered? unknown"))
        with patch.object(retry, "RETRY_MAX_ATTEMPTS", 3), \
             patch.object(retry, "RETRY_BASE_DELAY", 0), \
             patch("random.uniform", return_value=0):
            with self._history() as observer:
                with self.assertRaises(TimeoutError):
                    asyncio.run(retry.retry_async(calls, observer=observer))
        self.assertEqual(calls.await_count, 3)
        rows = self._rows()
        self.assertEqual([r.status for r in rows], ["error"] * 3)
        self.assertEqual([r.delivery for r in rows], ["unknown"] * 3)
        self.assertEqual([r.backoff_planned_ms for r in rows], [0, 0, None])
        self.assertIsNone(rows[-1].backoff_elapsed_ms)

    def test_auth_400_is_one_nonretryable_response(self):
        req = httpx.Request("POST", "https://private.example.invalid")
        response = httpx.Response(400, request=req)
        error = httpx.HTTPStatusError("secret detail", request=req, response=response)
        calls = AsyncMock(side_effect=error)
        with self._history() as observer:
            with self.assertRaises(httpx.HTTPStatusError):
                asyncio.run(retry.retry_async(calls, observer=observer))
        self.assertEqual(calls.await_count, 1)
        row = self._rows()[0]
        self.assertEqual((row.http_status, row.retryable, row.delivery, row.status),
                         (400, False, "response_received", "error"))
        self.assertEqual(row.reason, "HTTPStatusError")
        self.assertNotIn("private.example", json.dumps(row.as_dict()))
        self.assertNotIn("secret detail", json.dumps(row.as_dict()))

    def test_pending_row_is_committed_before_provider_call(self):
        async def inspect_during_call():
            with db.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT stage, status, attempt_number, finished_at FROM request_attempts "
                    "WHERE benchmark_id = :bid"), {"bid": self.b.id}).one()
            self.assertEqual(tuple(row), ("target_send", "started", 1, None))
            return "ok"

        with self._history() as observer:
            asyncio.run(retry.retry_async(inspect_during_call, observer=observer))
        self.assertEqual(self._rows()[0].status, "returned")

    def test_persistence_failure_never_duplicates_provider_call(self):
        call = AsyncMock(return_value="ok")
        with self._history() as observer:
            with patch.object(observer, "begin", side_effect=RuntimeError("storage unavailable")):
                with self.assertRaises(retry.RequestAccountingError):
                    asyncio.run(retry.retry_async(call, observer=observer))
        call.assert_not_awaited()
        with self._history() as observer:
            with patch.object(observer, "finish", side_effect=RuntimeError("storage unavailable")):
                with self.assertRaises(retry.RequestAccountingError):
                    asyncio.run(retry.retry_async(call, observer=observer))
        call.assert_awaited_once()
        self.assertEqual(len(self._rows()), 1)
        self.assertEqual(self._rows()[0].status, "started")

    def test_finish_commit_failure_rolls_back_and_target_wrapper_rethrows(self):
        target = _Target([_Reply("saved nowhere")])
        with self._history() as observer:
            with patch.object(db.session, "commit", side_effect=RuntimeError("disk full")), \
                 patch.object(db.session, "rollback", wraps=db.session.rollback) as rollback:
                with self.assertRaises(retry.RequestAccountingError):
                    observer.begin(1, 1)
                rollback.assert_called_once()
        target.send_prompt_async.assert_not_awaited()

        with self._history() as observer:
            with patch.object(observer, "finish", side_effect=RuntimeError("disk full")):
                with self.assertRaises(retry.RequestAccountingError):
                    asyncio.run(retry.send_prompt(target, "one request"))
        target.send_prompt_async.assert_awaited_once()
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].status, rows[0].delivery), ("started", "unknown"))

    def test_judge_accounting_failure_does_not_fabricate_judge_error(self):
        p = Prompt(benchmark_id=self.b.id, attack_slug=self.attack["slug"], sequence=0,
                   prompt_text="saved prompt", response="saved reply", status=PromptStatus.OK)
        db.session.add(p)
        db.session.flush()
        scoring_recovery.save_context(p, "saved prompt", None, "leak")
        db.session.commit()
        judge = _Target([_Reply("SCORE: 1")])
        with self._history("judge", prompt=p) as observer:
            with patch.object(observer, "finish", side_effect=RuntimeError("ledger failed")):
                with self.assertRaises(retry.RequestAccountingError):
                    asyncio.run(scoring.judge_request(judge, "exact saved request"))
        judge.send_prompt_async.assert_awaited_once()
        self.assertIsNone(p.verdict)

        # The saved-response worker records a local interruption, leaving the
        # captured target response and its score untouched for later inspection.
        evaluation = scoring_recovery.claim_retry(self.b, p)
        with patch.object(scoring_recovery, "build_judge_llm", return_value=judge), \
             patch.object(request_history.RequestHistory, "finish", side_effect=RuntimeError("ledger failed")):
            asyncio.run(scoring_recovery.retry_evaluation(evaluation.id))
        db.session.expire_all()
        self.assertEqual(evaluation.status, "interrupted")
        self.assertIn("Local request accounting failed", evaluation.verdict)
        self.assertEqual((p.response, p.severity, p.verdict), ("saved reply", None, None))
        self.assertEqual(judge.send_prompt_async.await_count, 2)

    def test_runner_paths_propagate_local_accounting_failure(self):
        error = retry.RequestAccountingError("local ledger failed")
        with patch.object(runner, "run_static_attack", AsyncMock(side_effect=error)), \
             patch.object(runner, "ATTACK_DELAY", 0):
            with self.assertRaises(retry.RequestAccountingError):
                asyncio.run(runner._run_attacks(self.b, object(), object(), 1))
        self.assertEqual(Prompt.query.count(), 0)
        self.b.status = "completed"
        db.session.commit()
        with patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "build_target", return_value=object()), \
             patch.object(runner, "run_static_attack", AsyncMock(side_effect=error)), \
             patch.object(runner, "ATTACK_DELAY", 0):
            with self.assertRaises(retry.RequestAccountingError):
                asyncio.run(runner._recover_async(self.b.id))
        self.assertEqual(Prompt.query.count(), 0)

    def test_failed_and_blank_static_sends_are_committed_before_next_seed(self):
        calls = [0]

        async def send_prompt_async(**_kwargs):
            calls[0] += 1
            if calls[0] > 1:
                with db.engine.connect() as conn:
                    rows = conn.execute(text(
                        "SELECT p.status, r.prompt_id, r.status FROM prompts p "
                        "JOIN request_attempts r ON r.prompt_id = p.id "
                        "WHERE p.benchmark_id = :bid ORDER BY p.sequence"),
                        {"bid": self.b.id}).all()
                self.assertEqual(len(rows), calls[0] - 1)
                self.assertTrue(all(p_status == "send_error" and prompt_id is not None
                                    for p_status, prompt_id, _ in rows))
            if calls[0] == 1:
                return []
            if calls[0] == 2:
                raise TimeoutError("provider outcome unknown")
            return [_Reply("reply")]

        target = _Target([])
        target.send_prompt_async = AsyncMock(side_effect=send_prompt_async)
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["a", "b", "c"])), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), \
             patch.object(retry, "RETRY_MAX_ATTEMPTS", 1):
            asyncio.run(static_attack.run_static_attack(
                self.b, self.attack, target, object(), 3, _Guard()))
        db.session.expire_all()
        self.assertEqual(Prompt.query.count(), 3)
        attempts = self._rows()
        self.assertEqual([(a.status, a.delivery) for a in attempts],
                         [("returned", "helper_returned"), ("error", "unknown"),
                          ("returned", "helper_returned")])
        self.assertEqual(len({a.prompt_id for a in attempts}), 3)

    def test_bad_retry_after_values_use_bounded_fallback(self):
        for header in ("nan", "-3", "inf"):
            with self.subTest(header=header):
                RequestAttempt.query.delete()
                db.session.commit()
                req = httpx.Request("POST", "https://private.example.invalid")
                response = httpx.Response(429, headers={"retry-after": header}, request=req)
                error = httpx.HTTPStatusError("rate limited", request=req, response=response)
                call = AsyncMock(side_effect=[error, "ok"])
                with patch.object(retry, "RETRY_MAX_ATTEMPTS", 2), \
                     patch.object(retry, "RETRY_BASE_DELAY", 0.5), \
                     patch.object(retry, "RETRY_MAX_DELAY", 2), \
                     patch.object(retry.asyncio, "sleep", new_callable=AsyncMock) as sleep, \
                     patch("random.uniform", return_value=0):
                    with self._history() as observer:
                        result = asyncio.run(retry.retry_async(call, observer=observer))
                self.assertEqual(result, "ok")
                self.assertEqual(call.await_count, 2)
                sleep.assert_awaited_once_with(0.5)
                self.assertEqual(self._rows()[0].backoff_planned_ms, 500)

    def test_empty_helper_return_does_not_claim_response_delivery(self):
        target = _Target([])
        with self._history() as observer:
            result = asyncio.run(retry.send_prompt(target, "empty"))
        self.assertEqual(result, ("", None))
        self.assertEqual(self._rows()[0].delivery, "helper_returned")

    def test_skip_failed_does_not_resend_unattached_prior_request(self):
        with request_history.request_history(self.b.id, "target_send",
                attack_slug=self.attack["slug"], sequence=0, attempt_slot=0) as observer:
            observer.begin(1, 1)
        request_history.mark_interrupted_requests()
        target = _Target([_Reply("next response")])
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["prior", "next"])), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), \
             patch.object(runner, "build_target", return_value=target), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id, retry_failed=False))
        target.send_prompt_async.assert_awaited_once()
        self.assertEqual(Prompt.query.one().sequence, 1)
        self.assertEqual(self.b.status, "failed")
        self.assertNotIn(self.attack["slug"], self.b.completed_attacks)
        self.assertIsNone(self._rows()[0].prompt_id)
        self.assertEqual(self._rows()[0].delivery, "unknown")

    def test_restart_marks_started_delivery_unknown_without_invented_end(self):
        with self._history() as observer:
            started = observer.begin(1, 3)
        finished = RequestAttempt(benchmark_id=self.b.id, stage="judge", call_id="completed",
                                  attempt_number=1, max_attempts=1, status="returned",
                                  delivery="response_received")
        db.session.add(finished)
        db.session.commit()
        request_history.mark_interrupted_requests()
        db.session.expire_all()
        self.assertEqual((started.status, started.reason, started.delivery),
                         ("interrupted", "process_interrupted", "unknown"))
        self.assertIsNone(started.finished_at)
        self.assertIsNone(started.elapsed_ms)
        self.assertEqual(finished.status, "returned")

    def test_static_retry_moves_prior_request_rows_to_archive(self):
        p = Prompt(benchmark_id=self.b.id, attack_slug=self.attack["slug"], sequence=0,
                   attempt=0, prompt_text="old prompt", response="", status=PromptStatus.SEND_ERROR)
        db.session.add(p)
        db.session.flush()
        old = RequestAttempt(benchmark_id=self.b.id, stage="target_send", call_id="old",
                             prompt_id=p.id, attempt_number=1, max_attempts=1,
                             status="error", delivery="unknown", reason="TimeoutError")
        db.session.add(old)
        db.session.commit()
        target = _Target([_Reply("new target response")])
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed"])), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0):
            asyncio.run(static_attack.run_static_attack(
                self.b, self.attack, target, object(), 1, _Guard(), retry_prompt=p))
        db.session.expire_all()
        archive = StaticExecutionArchive.query.one()
        self.assertEqual(old.static_execution_archive_id, archive.id)
        self.assertEqual(len(p.current_request_attempts), 1)
        current = p.current_request_attempts[0]
        self.assertNotEqual(current.id, old.id)
        self.assertEqual((current.stage, current.status, current.attempt_number),
                         ("target_send", "returned", 1))
        self.assertEqual(target.send_prompt_async.await_count, 1)
        client = self.app.test_client()
        history = client.get(f"/benchmarks/{self.b.id}/request-attempts.json").json
        self.assertEqual(len(history["request_attempts"]), 2)
        self.assertEqual({a["static_execution_archive_id"] for a in history["request_attempts"]},
                         {None, archive.id})
        detail = client.get(f"/benchmarks/{self.b.id}/prompts/{p.id}/executions.json").json
        self.assertEqual(len(detail["request_attempts"]), 2)
        rows = list(csv.DictReader(io.StringIO(client.get(f"/benchmarks/{self.b.id}/csv").data.decode())))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observed_target_invocations"], "1")
        self.assertEqual(rows[0]["observed_judge_invocations"], "")
        self.assertIn("Automatic request history", client.get(f"/benchmarks/{self.b.id}").data.decode())

    def test_judge_only_scoring_retry_has_no_target_attempt(self):
        p = Prompt(benchmark_id=self.b.id, attack_slug=self.attack["slug"], sequence=0,
                   prompt_text="saved prompt", response="saved reply", status=PromptStatus.OK)
        db.session.add(p)
        db.session.flush()
        scoring_recovery.save_context(p, "saved prompt", None, "leak")
        db.session.commit()
        evaluation = scoring_recovery.claim_retry(self.b, p)
        judge = _Target([_Reply("SCORE: 1")])
        with patch.object(scoring_recovery, "build_judge_llm", return_value=judge), \
             patch.object(static_attack, "send_prompt", side_effect=AssertionError("target called")):
            asyncio.run(scoring_recovery.retry_evaluation(evaluation.id))
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].stage, rows[0].evaluation_id, rows[0].prompt_id,
                          rows[0].status), ("judge", evaluation.id, p.id, "returned"))
        self.assertEqual(judge.send_prompt_async.await_count, 1)
        self.assertEqual(p.severity, 1)


if __name__ == "__main__":
    unittest.main()
