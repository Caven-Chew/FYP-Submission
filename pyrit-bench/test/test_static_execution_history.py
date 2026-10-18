"""Offline regression for Task 8.2/R1 static execution history.

Run: .venv/bin/python -B tools/run_isolated_app_test.py test_static_execution_history.py
The imported fixture sets a disposable database and artifact directory before
app import. All target and judge calls are mocked; the runner blocks sockets.
"""
import asyncio
from datetime import datetime
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_run_model_record as fixture  # noqa: F401; isolation before app import
from sqlalchemy import inspect, text

from app import _migrate_add_columns
from app.engine import runner, static_attack
from app.extensions import db
from app.models import (Benchmark, Prompt, PromptStatus, ScoringEvaluation,
                        StaticExecutionArchive, Target)
from app.web.rollups import benchmark_rollups


class _Guard:
    def record(self, _error):
        pass


class StaticExecutionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fixture.create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        target = Target(name="history target", provider="openai", model_name="unused")
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name="history", target_id=target.id, status="completed",
                           attack_slugs_json=json.dumps([fixture.STATIC["slug"]]))
        db.session.add(self.b)
        db.session.flush()
        self.p = Prompt(
            benchmark_id=self.b.id, attack_slug=fixture.STATIC["slug"], sequence=0,
            attempt=0, attempt_group="stable-group", prompt_text="old sent prompt",
            converter_used="old converter", response="", latency_ms=123,
            refused=False, succeeded=False, severity=None, taxonomy_category="old-tax",
            status=PromptStatus.SEND_ERROR, note="old note", error="old timeout",
            verdict="old verdict", severity_source="none", latency_measured=True,
            capture_ref="oldcapture", scoring_context_json='{"old":true}',
            created_at=datetime(2024, 1, 2, 3, 4, 5),
        )
        self.good = Prompt(
            benchmark_id=self.b.id, attack_slug=fixture.STATIC["slug"], sequence=1,
            attempt=0, attempt_group="good-group", prompt_text="valid prompt",
            response="valid refusal", status=PromptStatus.OK, severity=1,
            severity_source="judge", succeeded=False,
        )
        db.session.add_all([self.p, self.good])
        db.session.commit()
        self.client = self.app.test_client()
        self.attack = dict(fixture.STATIC, converter=None, sweep=None, attempts=1)

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def _run(self, reply=("new reply", None), *, retry_prompt=None, resume=False,
             retry_failed=True):
        target = type("TargetStub", (), {"last_debug": {}})()
        send = AsyncMock(return_value=reply)
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed 0", "seed 1"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("score 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0):
            asyncio.run(static_attack.run_static_attack(
                self.b, self.attack, target, object(), 2, _Guard(),
                retry_prompt=retry_prompt, resume=resume, retry_failed=retry_failed))
        db.session.expire_all()
        return send

    def test_exact_snapshot_repeated_retry_and_current_only_summary(self):
        original = self.p.execution_snapshot()
        self._run(reply=(None, "offline again"), retry_prompt=self.p)
        first_current = self.p.execution_snapshot()
        self.assertEqual(self.p.id, original["id"])
        self.assertEqual(self.p.attempt_group, "stable-group")
        self.assertEqual(self.p.attempt, 0)
        self.assertNotEqual(self.p.created_at.isoformat(), original["created_at"])
        self.assertEqual(self.p.execution_history[0].original_row, original)
        self._run(reply=("new reply", None), retry_prompt=self.p)
        self.assertEqual([a.original_row for a in self.p.execution_history],
                         [original, first_current])
        self.assertEqual(Prompt.query.count(), 2)
        self.assertEqual(self.p.response, "new reply")
        self.assertEqual(self.good.response, "valid refusal")
        totals = benchmark_rollups(self.b)["totals"]
        self.assertEqual(totals["counted"], 2)
        self.assertEqual(totals["scored"], 2)
        self.assertEqual(self.b.compute_totals()["counted"], 2)

    def test_prior_scoring_evaluation_follows_its_archived_execution(self):
        old = ScoringEvaluation(prompt_id=self.p.id, context_json='{"prior":"evidence"}',
                                status="error", verdict="old judge failure")
        db.session.add(old)
        db.session.commit()
        old_id = old.id
        self._run(retry_prompt=self.p)
        archive = self.p.execution_history[0]
        self.assertEqual(old.static_execution_archive_id, archive.id)
        self.assertEqual([e.id for e in archive.scoring_evaluations], [old_id])
        self.assertEqual(len(self.p.current_scoring_evaluations), 1)
        self.assertNotEqual(self.p.current_scoring_evaluations[0].id, old_id)
        payload = self.client.get(
            f"/benchmarks/{self.b.id}/prompts/{self.p.id}/executions.json").json
        self.assertEqual(payload["scoring_evaluations"][0]["static_execution_archive_id"], archive.id)
        self.assertEqual(payload["scoring_evaluations"][1]["static_execution_archive_id"], None)

    def test_archive_and_replacement_roll_back_together(self):
        original = self.p.execution_snapshot()
        target = type("TargetStub", (), {"last_debug": {}})()
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed 0"])), \
             patch.object(static_attack, "send_prompt", AsyncMock(return_value=("new reply", None))), \
             patch.object(static_attack, "new_evaluation", side_effect=RuntimeError("before commit")):
            with self.assertRaisesRegex(RuntimeError, "before commit"):
                asyncio.run(static_attack.run_static_attack(
                    self.b, self.attack, target, object(), 1, _Guard(), retry_prompt=self.p))
        db.session.rollback()
        db.session.expire_all()
        self.assertEqual(self.p.execution_snapshot(), original)
        self.assertEqual(StaticExecutionArchive.query.count(), 0)

    def test_resume_and_skip_failed_preserve_existing_rows(self):
        self._run(reply=(None, "offline again"), retry_prompt=self.p)
        archive_id = self.p.execution_history[0].id
        before = self.p.execution_snapshot()
        send = self._run(resume=True, retry_failed=False)
        send.assert_not_awaited()
        self.assertEqual(self.p.execution_snapshot(), before)
        self.assertEqual([a.id for a in self.p.execution_history], [archive_id])
        self._run(resume=True, retry_failed=True)
        self.assertEqual(len(self.p.execution_history), 2)
        self.assertEqual(self.good.response, "valid refusal")
        self.assertEqual(self.good.execution_history, [])

    def test_runner_resume_keeps_partial_static_history(self):
        self._run(reply=(None, "offline again"), retry_prompt=self.p)
        self.b.status = "interrupted"
        self.b.completed_attacks_json = "[]"
        db.session.commit()
        original_id = self.p.id
        original_archive = self.p.execution_history[0].original_row
        target = type("TargetStub", (), {"last_debug": {}})()
        send = AsyncMock(return_value=("new reply", None))
        with patch.object(static_attack, "build_converters", return_value=[]), \
             patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed 0", "seed 1"])), \
             patch.object(static_attack, "send_prompt", send), \
             patch.object(static_attack, "judge_response", AsyncMock(return_value=("score 1", 1, False))), \
             patch.object(static_attack, "REQUEST_DELAY", 0), \
             patch.object(runner, "ATTACK_DELAY", 0):
            asyncio.run(runner._run_attacks(self.b, target, object(), 2))
        db.session.expire_all()
        send.assert_awaited_once()
        self.assertEqual(self.p.id, original_id)
        self.assertEqual(self.p.response, "new reply")
        self.assertEqual(len(self.p.execution_history), 2)
        self.assertEqual(self.p.execution_history[0].original_row, original_archive)
        self.assertEqual(self.good.execution_history, [])

    def test_detail_json_csv_and_capture_retention(self):
        capture_dir = fixture.ARTIFACTS.ARTIFACTS_DIR / str(self.b.id)
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "oldcapture.png").write_bytes(b"png bytes")
        (capture_dir / "oldcapture.html.gz").write_bytes(b"gzip bytes")
        self._run(retry_prompt=self.p)
        page = self.client.get(f"/benchmarks/{self.b.id}").data.decode()
        self.assertIn("Previous executions (1)", page)
        self.assertIn("old timeout", page)
        self.assertIn("oldcapture", page)
        self.assertIn("Screenshot", page)
        self.assertIn("HTML capture", page)
        capture_url = f"/benchmarks/{self.b.id}/capture/oldcapture/png"
        capture = self.client.get(capture_url)
        try:
            self.assertEqual(capture.data, b"png bytes")
        finally:
            capture.close()
        payload = self.client.get(
            f"/benchmarks/{self.b.id}/prompts/{self.p.id}/executions.json").json
        self.assertEqual(payload["previous"][0]["original_row"]["error"], "old timeout")
        self.assertEqual(payload["current"]["response"], "new reply")
        self.assertEqual(len(payload["previous"]), 1)
        csv_text = self.client.get(f"/benchmarks/{self.b.id}/csv").data.decode()
        self.assertIn("archived_executions", csv_text)
        self.assertNotIn("old timeout", csv_text)
        (capture_dir / "oldcapture.png").unlink()
        (capture_dir / "oldcapture.html.gz").unlink()
        page = self.client.get(f"/benchmarks/{self.b.id}").data.decode()
        self.assertIn("unavailable under capture retention", page)
        self.assertNotIn(capture_url, page)

    def test_additive_table_migration_and_cascade_delete(self):
        original = self.p.execution_snapshot()
        ScoringEvaluation.__table__.drop(db.engine)
        with db.engine.begin() as conn:
            conn.execute(text("""CREATE TABLE scoring_evaluations (
                id INTEGER PRIMARY KEY, prompt_id INTEGER NOT NULL,
                context_json TEXT NOT NULL, status VARCHAR(24) NOT NULL,
                previous_benchmark_status VARCHAR(40), started_at DATETIME,
                finished_at DATETIME, elapsed_ms INTEGER, verdict TEXT,
                severity INTEGER, severity_source VARCHAR(12), succeeded BOOLEAN)"""))
        StaticExecutionArchive.__table__.drop(db.engine)
        self.assertNotIn("static_execution_archives", inspect(db.engine).get_table_names())
        db.create_all()
        _migrate_add_columns()
        _migrate_add_columns()
        self.assertIn("static_execution_archive_id", {c["name"] for c in inspect(db.engine).get_columns("scoring_evaluations")})
        self.assertEqual(self.p.execution_snapshot(), original)
        self._run(retry_prompt=self.p)
        self.assertEqual(StaticExecutionArchive.query.count(), 1)
        db.session.delete(self.b)
        db.session.commit()
        self.assertEqual(StaticExecutionArchive.query.count(), 0)
        self.assertEqual(ScoringEvaluation.query.count(), 0)


if __name__ == "__main__":
    unittest.main()
