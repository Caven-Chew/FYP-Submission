"""Failed prompt retry regression tests; isolated database and no network calls."""
import sys, types, os, asyncio, tempfile, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A throwaway database, set BEFORE any app import. app/config.py reads
# DATABASE_URL into a class attribute at import time, so setting this later —
# even a few lines later, after an incidental `from app.engine import ...` has
# already pulled the package in — silently runs the tests against
# instance/app.db and leaves test targets and benchmarks in real result data.
os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db")


# --- stub out pyrit just enough that the engine modules import ---------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()


for mod in ["pyrit", "pyrit.models", "pyrit.datasets", "pyrit.prompt_target",
            "pyrit.prompt_converter", "pyrit.score", "pyrit.memory",
            "pyrit.executor", "pyrit.executor.attack", "pyrit.common",
            "pyrit.models.conversation_reference"]:
    stub(mod)


class _Msg(_Any):
    @staticmethod
    def from_prompt(prompt=None, role=None):
        return types.SimpleNamespace(prompt=prompt, role=role)


sys.modules["pyrit.models"].Message = _Msg
for n in ("Base64Converter", "LeetspeakConverter", "ROT13Converter", "PersuasionConverter"):
    setattr(sys.modules["pyrit.prompt_converter"], n, _Any)


import unittest
from unittest.mock import AsyncMock, patch
from app import create_app
from app.extensions import db
from app.models import Benchmark, Prompt, PromptStatus, Target, ConversationRetry
from app.attacks import ATTACKS
import app.attacks as attack_catalog
from app.engine import runner, static_attack


class PromptRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        self.attack = next(a for a in ATTACKS if a.get("kind") == "static" and a.get("scoring") == "leak")
        target = Target(name="retry test", provider="openai", model_name="test")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="retry", target_id=target.id, status="completed",
                           attack_slugs_json=json.dumps([self.attack["slug"]]))
        db.session.add(self.b)
        db.session.flush()
        self.p = Prompt(benchmark_id=self.b.id, attack_slug=self.attack["slug"],
                        sequence=1, attempt=0, prompt_text="original converted prompt",
                        response="", status=PromptStatus.SEND_ERROR, error="timeout")
        self.good = Prompt(benchmark_id=self.b.id, attack_slug=self.attack["slug"],
                           sequence=0, prompt_text="keep me", response="safe", status="ok",
                           severity=1, succeeded=False)
        db.session.add_all([self.p, self.good])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def retry(self, reply=("leaked", None)):
        target = types.SimpleNamespace(aclose=AsyncMock())
        send = AsyncMock(return_value=reply)
        judge = AsyncMock(return_value=("SCORE: 5", 5, True))
        with patch.object(runner, "build_target", return_value=target), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed zero", "seed one"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", judge), \
             patch.object(static_attack, "REQUEST_DELAY", 0):
            asyncio.run(runner._rerun_prompt(self.b.id, self.p.id, "completed"))
        target.aclose.assert_awaited_once()
        return send, judge

    def test_retry_replaces_only_failed_row_and_refreshes_totals(self):
        pid = self.p.id
        send, judge = self.retry()
        send.assert_awaited_once()
        self.assertEqual(send.call_args.args[1], "original converted prompt")
        self.assertEqual(judge.call_args.args[1], "seed one")
        self.assertEqual(judge.call_args.kwargs["rubric"], "leak")
        self.assertEqual(Prompt.query.count(), 2)
        self.assertEqual(self.p.id, pid)
        self.assertEqual(self.p.status, "ok")
        self.assertIsNone(self.p.error)
        self.assertEqual(self.good.response, "safe")
        self.assertEqual(self.b.total_prompts, 2)
        self.assertEqual(self.b.total_successes, 1)
        self.assertFalse(self.p.can_rerun)

    def test_repeated_failure_stays_retryable(self):
        self.retry((None, "still offline"))
        self.assertEqual(self.p.error, "still offline")
        self.assertTrue(self.p.can_rerun)
        self.assertEqual(self.b.total_prompts, 1)
        self.assertEqual(Prompt.query.count(), 2)

    def test_converter_failure_reruns_conversion(self):
        self.p.status = PromptStatus.CONVERTER_ERROR
        self.p.prompt_text = "seed one"
        db.session.commit()
        convert = AsyncMock(return_value=("new converted prompt", None, None))
        with patch.object(static_attack, "apply_converters", convert):
            send, _ = self.retry()
        convert.assert_awaited_once()
        self.assertEqual(send.call_args.args[1], "new converted prompt")
        self.assertEqual(self.p.prompt_text, "new converted prompt")
        self.assertEqual(self.p.status, "ok")

    def test_route_csrf_ownership_and_duplicate_click(self):
        client = self.app.test_client()
        url = f"/benchmarks/{self.b.id}/prompts/{self.p.id}/rerun"
        self.assertEqual(client.post(url).status_code, 403)
        with client.session_transaction() as sess:
            sess["csrf_token"] = "test"
        data = {"csrf_token": "test"}
        with patch("app.web.views_benchmarks.start_prompt_retry") as start:
            self.assertEqual(client.post(url, data=data).status_code, 302)
            self.assertEqual(client.post(url, data=data).status_code, 302)
            self.assertEqual(start.call_count, 1)
        self.assertEqual(client.post(f"/benchmarks/{self.b.id}/prompts/{self.good.id}/rerun", data=data).status_code, 400)
        self.assertEqual(client.post(f"/benchmarks/{self.b.id}/prompts/99999/rerun", data=data).status_code, 404)

    def setup_conversation(self):
        # The course UI has no FYP dynamic attacks. Retain regression coverage
        # for stored legacy conversation retry using a test-local catalogue entry.
        # patch.object restores the real catalogue after each test, including errors.
        attack = {"slug": "test_legacy_conversation", "kind": "dynamic",
                  "name": "Synthetic legacy conversation", "category": "Test fixture",
                  "orchestrator": "red_teaming",
                  "taxonomy_category": "instruction_override", "max_turns": 3}
        catalog_patch = patch.object(attack_catalog, "ATTACKS", [*ATTACKS, attack])
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        self.p.attack_slug = attack["slug"]
        self.p.objective_id = "objective-two"
        self.good.attack_slug = attack["slug"]
        self.good.objective_id = "objective-one"
        self.b.attack_slugs_json = json.dumps([attack["slug"]])
        self.b.attacker_roster_json = json.dumps([
            {"provider": "openai", "model": "first"},
            {"provider": "openai", "model": "second"}])
        self.p.attempt = 1
        self.p.converter_used = "openai/second"
        db.session.commit()
        return attack

    def run_conversation(self, results):
        target = types.SimpleNamespace(aclose=AsyncMock())
        dynamic = AsyncMock(return_value=results)
        objectives = [{"id": "objective-one", "objective": "keep"},
                      {"id": "objective-two", "objective": "retry whole objective"}]
        with patch.object(runner, "build_target", return_value=target) as build, \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "build_attacker_llm", return_value=object()), \
             patch.object(runner, "resolve_objectives", AsyncMock(return_value=objectives)), \
             patch.object(runner, "run_dynamic_attack", dynamic):
            asyncio.run(runner._rerun_prompt(self.b.id, self.p.id, "completed"))
            self.assertTrue(build.call_args.kwargs["multi_turn"])
        target.aclose.assert_awaited_once()
        self.assertEqual(dynamic.call_args.kwargs["objectives"], [objectives[1]])
        return dynamic

    def test_conversation_retry_archives_and_replaces_whole_objective(self):
        attack = self.setup_conversation()
        partial = Prompt(benchmark_id=self.b.id, attack_slug=attack["slug"],
                         objective_id="objective-two", attempt=1, converter_used="openai/second",
                         prompt_text="old partial turn", response="partial reply", status="ok")
        other_attacker = Prompt(benchmark_id=self.b.id, attack_slug=attack["slug"],
                         objective_id="objective-two", attempt=0, converter_used="openai/first",
                         prompt_text="other attacker", response="keep too", status="ok")
        db.session.add_all([partial, other_attacker])
        db.session.commit()
        keep_ids = [self.good.id, other_attacker.id]
        results = [dict(sequence=i, prompt_text=f"new turn {i}", response="reply",
                        objective_id="objective-two", status="ok", severity=1,
                        succeeded=False, severity_source="judge", latency_measured=False)
                   for i in (1, 3)]
        self.run_conversation(results)
        current = Prompt.query.filter_by(objective_id="objective-two", attempt=1).all()
        self.assertEqual([r.prompt_text for r in current], ["new turn 1", "new turn 3"])
        self.assertEqual([r.sequence for r in current], [1000101, 1000103])
        self.assertTrue(all(db.session.get(Prompt, pid) for pid in keep_ids))
        archive = ConversationRetry.query.one()
        self.assertEqual(len(archive.original_rows), 2)
        self.assertEqual(self.b.total_prompts, 4)
        page = self.app.test_client().get(f"/benchmarks/{self.b.id}")
        self.assertIn(b"Previous conversation results", page.data)
        self.assertIn(b"old partial turn", page.data)

    def test_empty_conversation_retry_keeps_original_results(self):
        self.setup_conversation()
        original_id = self.p.id
        with self.assertRaises(ValueError):
            self.run_conversation([])
        self.assertIsNotNone(db.session.get(Prompt, original_id))
        self.assertEqual(ConversationRetry.query.count(), 0)

    def test_failed_conversation_retry_can_be_retried_again(self):
        self.setup_conversation()
        self.run_conversation([dict(sequence=0, prompt_text="objective", response="",
                              objective_id="objective-two", status="send_error",
                              error="offline", succeeded=False)])
        result = Prompt.query.filter_by(objective_id="objective-two").one()
        self.assertTrue(result.can_rerun)
        self.assertEqual(result.retry_label, "Rerun conversation")
        self.assertEqual(ConversationRetry.query.count(), 1)

    def test_conversation_button_and_valid_defence(self):
        self.setup_conversation()
        page = self.app.test_client().get(f"/benchmarks/{self.b.id}")
        self.assertIn(b"Rerun conversation", page.data)
        self.p.status = PromptStatus.OK
        self.p.response = "I cannot help with that"
        self.assertFalse(self.p.can_rerun)
        self.p.status = PromptStatus.ATTACKER_REFUSED
        self.assertFalse(self.p.can_rerun)

    def test_execution_errors_mark_finished_run_failed(self):
        async def failed_attack(bench, attack, target, judge, limit, guard, *, pending_scores=None):
            db.session.add(Prompt(benchmark_id=bench.id, attack_slug=attack["slug"],
                                  prompt_text="", status=PromptStatus.SETUP_ERROR,
                                  error="converter setup failed"))
            db.session.commit()
        with patch.object(runner, "run_static_attack", failed_attack), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._run_attacks(self.b, object(), object(), 1))
        self.assertEqual(self.b.status, "failed")
        self.assertIn("converter setup failed", self.b.status_detail)

    def test_manual_stop_stays_stopped_even_with_prior_errors(self):
        runner.finish_stopped(self.b.id)
        self.assertEqual(self.b.status, "stopped")
        self.assertIn("Stopped manually", self.b.status_detail)

    def test_retry_error_is_failed(self):
        self.retry((None, "offline"))
        self.assertEqual(self.b.status, "failed")
        self.assertIn("offline", self.b.status_detail)

    def test_bulk_static_retries_failed_and_fills_missing(self):
        self.b.status = "running"
        self.b.llm_config_json = json.dumps({"MAX_PROMPTS_PER_ATTACK": 3})
        db.session.commit()
        send = AsyncMock(return_value=("safe", None))
        target = types.SimpleNamespace(aclose=AsyncMock())
        with patch.object(runner, "build_target", return_value=target), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["zero", "one", "two"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id))
        self.assertEqual([call.args[1] for call in send.call_args_list], ["original converted prompt", "two"])
        self.assertEqual(self.good.response, "safe")
        self.assertEqual(Prompt.query.count(), 3)
        self.assertEqual(self.b.status, "completed")
        self.assertIsNone(self.b.status_detail)
        self.assertIn(self.attack["slug"], self.b.completed_attacks)
        target.aclose.assert_awaited_once()

    def test_bulk_skips_completed_attack_but_runs_unstarted_attack(self):
        self.p.status = "ok"
        self.p.response = "safe"
        slug = self.attack["slug"]
        other = next(a for a in ATTACKS if a.get("kind") == "static" and a["slug"] != slug)
        self.b.attack_slugs_json = json.dumps([slug, other["slug"]])
        self.b.completed_attacks_json = json.dumps([slug])
        db.session.commit()
        run = AsyncMock(return_value=True)
        with patch.object(runner, "run_static_attack", run), \
             patch.object(runner, "build_target", return_value=object()), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id))
        run.assert_awaited_once()
        self.assertEqual(run.call_args.args[1]["slug"], other["slug"])
        self.assertTrue(run.call_args.kwargs["resume"])
        self.assertEqual(Prompt.query.count(), 2)

    def test_bulk_dynamic_deduplicates_failed_conversation(self):
        attack = self.setup_conversation()
        # One attacker keeps this test focused on objective scheduling.
        self.b.attacker_roster_json = "[]"
        self.p.attempt = 0
        self.p.converter_used = None
        extra = Prompt(benchmark_id=self.b.id, attack_slug=attack["slug"],
                       objective_id="objective-two", prompt_text="failed second turn",
                       status="send_error", attempt=0)
        db.session.add(extra)
        db.session.commit()
        objectives = [{"id": "objective-one", "objective": "keep"},
                      {"id": "objective-two", "objective": "retry"},
                      {"id": "objective-three", "objective": "new"}]
        async def result(**kwargs):
            oid = kwargs["objectives"][0]["id"]
            return [dict(sequence=1, prompt_text=oid, response="safe", objective_id=oid,
                         status="ok", severity=1, succeeded=False)]
        dynamic = AsyncMock(side_effect=result)
        with patch.object(runner, "resolve_objectives", AsyncMock(return_value=objectives)), \
             patch.object(runner, "build_target", return_value=object()), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "build_attacker_llm", return_value=object()), \
             patch.object(runner, "run_dynamic_attack", dynamic), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id))
        self.assertEqual([c.kwargs["objectives"][0]["id"] for c in dynamic.call_args_list],
                         ["objective-two", "objective-three"])
        self.assertEqual(self.b.status, "completed")
        self.assertEqual(ConversationRetry.query.count(), 1)
        self.assertEqual(Prompt.query.count(), 3)

    def test_bulk_best_of_n_keeps_valid_attempts_and_retries_errors(self):
        self.good.sequence = 1
        self.good.attempt = 0
        self.good.succeeded = True
        self.p.sequence = 1
        self.p.attempt = 1
        db.session.commit()
        attack = {**self.attack, "attempts": 3}
        send = AsyncMock(return_value=("safe", None))
        with patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["zero", "one"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0):
            asyncio.run(static_attack.run_static_attack(self.b, attack, object(), object(), 2,
                runner.ConsecutiveFailureGuard(10), resume=True))
        self.assertEqual([c.args[1] for c in send.call_args_list],
                         ["zero", "zero", "zero", "original converted prompt"])
        self.assertTrue(self.good.succeeded)
        self.assertEqual(Prompt.query.filter_by(sequence=1).count(), 2)

    def test_bulk_repeated_target_errors_continue_with_target_error_status(self):
        target = types.SimpleNamespace(aclose=AsyncMock())
        send = AsyncMock(return_value=(None, "offline"))
        with patch.object(runner, "build_target", return_value=target), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "MAX_CONSECUTIVE_TARGET_FAILURES", 1), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["zero", "one", "two"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "REQUEST_DELAY", 0), patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id))
        self.assertEqual(send.await_count, 2)
        self.assertEqual(self.b.status, "completed_with_target_errors")
        self.assertIn("offline", self.b.status_detail)
        self.assertEqual(self.good.response, "safe")
        self.assertEqual(Prompt.query.count(), 3)
        target.aclose.assert_awaited_once()

    def test_bulk_route_blocks_csrf_and_double_click(self):
        client = self.app.test_client()
        url = f"/benchmarks/{self.b.id}/retry-failed"
        self.assertEqual(client.post(url).status_code, 403)
        with client.session_transaction() as sess:
            sess["csrf_token"] = "test"
        with patch("app.web.views_benchmarks.start_benchmark_recovery") as start:
            for _ in range(2):
                self.assertEqual(client.post(url, data={"csrf_token": "test"}).status_code, 302)
            start.assert_called_once()

    def test_bulk_cancel_preserves_results(self):
        runner.control.request_cancel(self.b.id)
        try:
            with patch.object(runner, "build_judge_llm", return_value=object()):
                with self.assertRaises(runner.control.BenchmarkCancelled):
                    asyncio.run(runner._recover_async(self.b.id))
            self.assertEqual(Prompt.query.count(), 2)
        finally:
            runner.control.clear_cancel(self.b.id)

    def test_skip_failed_runs_only_unattempted_static_prompts(self):
        original_error = self.p.error
        send = AsyncMock(return_value=("safe", None))
        with patch.object(runner, "build_target", return_value=object()), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["zero", "one", "two"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id, retry_failed=False))
        send.assert_awaited_once()
        self.assertEqual(send.call_args.args[1], "two")
        self.assertEqual(self.p.error, original_error)
        self.assertEqual(self.p.status, "send_error")
        self.assertEqual(self.b.status, "completed_with_target_errors")
        self.assertEqual(Prompt.query.count(), 3)

    def test_skip_failed_conversation_continues_next_objective(self):
        self.setup_conversation()
        self.b.attacker_roster_json = "[]"
        self.p.attempt = 0
        self.p.converter_used = None
        db.session.commit()
        objectives = [{"id": "objective-one", "objective": "keep"},
                      {"id": "objective-two", "objective": "failed"},
                      {"id": "objective-three", "objective": "new"}]
        dynamic = AsyncMock(return_value=[dict(sequence=1, prompt_text="new", response="safe",
                           objective_id="objective-three", status="ok", severity=1, succeeded=False)])
        with patch.object(runner, "resolve_objectives", AsyncMock(return_value=objectives)), \
             patch.object(runner, "build_target", return_value=object()), \
             patch.object(runner, "build_judge_llm", return_value=object()), \
             patch.object(runner, "build_attacker_llm", return_value=object()), \
             patch.object(runner, "run_dynamic_attack", dynamic), patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._recover_async(self.b.id, retry_failed=False))
        dynamic.assert_awaited_once()
        self.assertEqual(dynamic.call_args.kwargs["objectives"], [objectives[2]])
        self.assertEqual(self.p.error, "timeout")
        self.assertEqual(ConversationRetry.query.count(), 0)
        self.assertEqual(self.b.status, "completed_with_target_errors")

    def test_skip_route_and_retry_route_are_distinct(self):
        client = self.app.test_client()
        page = client.get(f"/benchmarks/{self.b.id}")
        self.assertIn(f'/benchmarks/{self.b.id}/retry-failed'.encode(), page.data)
        self.assertIn(f'/benchmarks/{self.b.id}/skip-failed'.encode(), page.data)
        url = f"/benchmarks/{self.b.id}/skip-failed"
        self.assertEqual(client.post(url).status_code, 403)
        with client.session_transaction() as sess:
            sess["csrf_token"] = "test"
        with patch("app.web.views_benchmarks.start_benchmark_recovery") as start:
            self.assertEqual(client.post(url, data={"csrf_token": "test"}).status_code, 302)
            self.assertEqual(client.post(url, data={"csrf_token": "test"}).status_code, 302)
            start.assert_called_once()
            self.assertFalse(start.call_args.kwargs["retry_failed"])

    def test_normal_run_continues_after_consecutive_failures(self):
        second = next(a for a in ATTACKS if a.get("kind") == "static" and a["slug"] != self.attack["slug"])
        slugs = [self.attack["slug"], second["slug"]]
        self.b.attack_slugs_json = json.dumps(slugs)
        self.b.status = "running"
        db.session.delete(self.p)
        db.session.delete(self.good)
        db.session.commit()
        calls = []
        async def failed_send(target, text):
            calls.append(text)
            self.assertEqual(self.b.status, "running")
            return None, "offline"
        with patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["first", "next", "last"])), \
             patch.object(static_attack, "send_prompt", failed_send), \
             patch.object(static_attack, "REQUEST_DELAY", 0), \
             patch.object(runner, "MAX_CONSECUTIVE_TARGET_FAILURES", 1), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._run_attacks(self.b, object(), object(), 3))
        self.assertEqual(calls, ["first", "next", "last"] * 2)
        self.assertEqual(Prompt.query.filter_by(status="send_error").count(), 6)
        self.assertEqual(self.b.completed_attacks, slugs)
        self.assertEqual(self.b.status, "completed_with_target_errors")
        self.assertIn("6 unresolved execution error", self.b.status_detail)

    def test_non_stopping_guard_tracks_failures(self):
        guard = runner.ConsecutiveFailureGuard(1, stop_on_limit=False)
        guard.record("first failure")
        guard.record("second failure")
        self.assertEqual(guard.consecutive, 2)
        self.assertFalse(guard.tripped)
        guard.record(None)
        self.assertEqual(guard.consecutive, 0)

    def test_detail_offers_retry(self):
        page = self.app.test_client().get(f"/benchmarks/{self.b.id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Rerun prompt", page.data)


if __name__ == "__main__":
    unittest.main()
