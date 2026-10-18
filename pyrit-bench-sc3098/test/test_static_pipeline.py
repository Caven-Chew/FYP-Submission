"""Offline checks for captured replies progressing while judge calls are pending.

Run: ../.venv/bin/python -m unittest discover -s test -p test_static_pipeline.py -v
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import test_student_engine as fixtures
from app.extensions import db
from app.models import Prompt, PromptStatus
from app.engine.static_attack import run_static_attack
from app.engine.retry import ConsecutiveFailureGuard
from app.engine.control import BenchmarkCancelled


class StaticPipelineTests(unittest.TestCase):
    setUp = fixtures.StudentEngineTests.setUp
    tearDown = fixtures.StudentEngineTests.tearDown

    def run_attack(self, *, send, score, **kwargs):
        with patch('app.engine.static_attack.send_prompt', send), \
             patch('app.engine.static_attack.judge_response', score), \
             patch('app.engine.static_attack.close_after_reply', self.close_reply):
            return asyncio.run(run_static_attack(
                self.bench, fixtures.STATIC, object(), object(), 10,
                ConsecutiveFailureGuard(3, stop_on_limit=False), **kwargs))

    async def close_reply(self, target):
        await asyncio.sleep(0)

    def test_next_seed_starts_with_first_reply_saved_and_pending(self):
        async def send(target, text):
            if text == 'second':
                first = Prompt.query.filter_by(sequence=0).one()
                self.assertEqual(first.response, 'answer:first')
                self.assertEqual(first.status, PromptStatus.SCORING)
                self.release.set()
            return f'answer:{text}', None

        async def score(judge, seed, response, **kwargs):
            if seed == 'first':
                await self.release.wait()
            return f'judged:{seed}', 2, False

        self.release = asyncio.Event()
        self.assertTrue(self.run_attack(send=send, score=score))
        rows = Prompt.query.order_by(Prompt.sequence).all()
        self.assertEqual([p.verdict for p in rows], ['judged:first', 'judged:second'])
        self.assertTrue(all(p.status == PromptStatus.OK for p in rows))

    def test_runner_overlaps_static_attacks_and_marks_only_scored_captures(self):
        from app.engine.runner import _run_attacks
        attacks = {
            slug: {'slug': slug, 'kind': 'static', 'dataset': slug}
            for slug in ('advbench', 'harmbench')
        }
        self.bench.attack_slugs_json = json.dumps(list(attacks))
        self.bench.status = 'running'
        db.session.commit()
        release = asyncio.Event()
        calls = []

        async def load(dataset, limit):
            return [dataset]

        async def send(target, text):
            calls.append(text)
            if text == 'harmbench':
                first = Prompt.query.filter_by(attack_slug='advbench').one()
                self.assertEqual(first.status, PromptStatus.SCORING)
                self.assertEqual(first.response, 'answer:advbench')
                self.assertEqual(self.bench.completed_attacks, [])
                self.assertEqual(self.bench.status, 'running')
                release.set()
            return f'answer:{text}', None

        async def score(judge, seed, response, **kwargs):
            if seed == 'advbench':
                # A regression must fail promptly instead of hanging the suite.
                await asyncio.wait_for(release.wait(), 1)
            return f'judged:{seed}', 1, False

        with patch('app.engine.runner.by_slug', side_effect=attacks.get), \
             patch('app.engine.runner.ATTACK_DELAY', 0), \
             patch('app.engine.static_attack.load_dataset_prompts', load), \
             patch('app.engine.static_attack.send_prompt', send), \
             patch('app.engine.static_attack.judge_response', score), \
             patch('app.engine.static_attack.close_after_reply', self.close_reply):
            asyncio.run(_run_attacks(self.bench, object(), object(), 1))
        self.assertEqual(calls, ['advbench', 'harmbench'])
        self.assertEqual(self.bench.completed_attacks, ['advbench', 'harmbench'])
        self.assertEqual(self.bench.status, 'completed')
        self.assertEqual(self.bench.total_prompts, 2)
        self.assertEqual(self.bench.compute_totals()['scored'], 2)
        self.assertTrue(all(row.status == PromptStatus.OK for row in Prompt.query.all()))

    def test_best_of_n_waits_and_skips_attempts_after_success(self):
        calls = []

        async def send(target, text):
            calls.append(text)
            return f'answer:{text}', None

        async def score(judge, seed, response, **kwargs):
            await asyncio.sleep(0.01)
            return 'success', 5, True

        with patch('app.engine.static_attack.attempts_for', return_value=3):
            self.run_attack(send=send, score=score)
        self.assertEqual(calls, ['first', 'second'])
        self.assertEqual(Prompt.query.count(), 2)
        self.assertTrue(all(p.succeeded for p in Prompt.query.all()))

    def test_judge_failure_retains_reply_for_judge_only_retry(self):
        async def send(target, text):
            return f'answer:{text}', None

        async def score(*args, **kwargs):
            raise ValueError('offline judge failure')

        self.run_attack(send=send, score=score)
        rows = Prompt.query.all()
        self.assertTrue(all(p.status == PromptStatus.JUDGE_ERROR for p in rows))
        self.assertTrue(all(p.can_rescore for p in rows))
        self.assertEqual([p.response for p in rows], ['answer:first', 'answer:second'])

        async def never_send(*args):
            self.fail('Judge-only retry must not contact the target')

        async def recovered_score(*args, **kwargs):
            return 'recovered judge', 3, True

        self.run_attack(send=never_send, score=recovered_score, retry_prompt=rows[0])
        self.assertEqual(rows[0].response, 'answer:first')
        self.assertEqual(rows[0].severity, 3)
        self.assertEqual(Prompt.query.count(), 2)

    def test_cancel_between_seeds_drains_captured_reply(self):
        checks = 0

        async def checkpoint(benchmark_id):
            nonlocal checks
            checks += 1
            if checks == 3:
                raise BenchmarkCancelled()

        async def send(target, text):
            return 'captured', None

        async def score(*args, **kwargs):
            await asyncio.sleep(0.01)
            return 'saved evidence', 1, False

        with patch('app.engine.static_attack.wait_while_paused', checkpoint):
            with self.assertRaises(BenchmarkCancelled):
                self.run_attack(send=send, score=score)
        row = Prompt.query.one()
        self.assertEqual(row.response, 'captured')
        self.assertEqual(row.status, PromptStatus.OK)
        self.assertEqual(row.verdict, 'saved evidence')


if __name__ == '__main__':
    unittest.main()
