"""Offline overlap, closure, recovery and bounded-queue regression checks."""
import asyncio
import json
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

import test_student_engine as engine
from test_student_engine import ConvChat, SCRIPT, db, Prompt, PromptStatus
from app.engine.pending_scores import PendingScores, close_after_reply, prepare_reply_for_close
from app.targets.browser.target import BrowserUITarget


class ScriptedPipelineTests(unittest.TestCase):
    setUp = engine.StudentEngineTests.setUp
    tearDown = engine.StudentEngineTests.tearDown
    run_script = engine.StudentEngineTests.run_script
    enable_per_turn_scores = engine.StudentEngineTests.enable_per_turn_scores

    def test_followup_and_next_chat_before_judge_finishes(self):
        self.enable_per_turn_scores()
        self.bench.custom_conversations_json = json.dumps([['one', 'two'], ['three']])
        db.session.commit()
        outer = self

        async def scenario():
            release = asyncio.Event()
            snapshots = []
            class Chat(ConvChat):
                async def reply(self, prompt, history):
                    if prompt == 'two':
                        first = Prompt.query.filter_by(sequence=0).one()
                        outer.assertEqual(first.status, PromptStatus.SCORING)
                        outer.assertIn('one', first.response)
                        outer.assertEqual(history[1]['content'], first.response)
                    if prompt == 'three':
                        outer.assertEqual(history, [])
                        release.set()
                    return await super().reply(prompt, history)
            async def judge(_judge, prompt, response, history, **kwargs):
                snapshots.append((prompt, list(history)))
                await asyncio.wait_for(release.wait(), 2)
                return 'SCORE: 1', 1, False
            from app.engine.scripted_attack import run_scripted_attack
            from app.engine.retry import ConsecutiveFailureGuard
            with patch('app.engine.scripted_attack.judge_interactive_response', judge):
                await run_scripted_attack(self.bench, SCRIPT, Chat(), object(), 10,
                                         ConsecutiveFailureGuard(3, stop_on_limit=False))
            outer.assertEqual([p for p, _ in snapshots], ['one', 'two', 'three'])
            outer.assertEqual(snapshots[0][1], [])
            outer.assertEqual(snapshots[2][1], [])
        asyncio.run(scenario())
        self.assertTrue(all(row.status == PromptStatus.OK for row in Prompt.query.all()))

    def test_queue_serializes_and_drains_on_cooperative_stop(self):
        async def scenario():
            active, peak, completed = 0, 0, []
            async def job(n):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.01)
                active -= 1
                completed.append(n)
            class Stop(BaseException):
                pass
            with self.assertRaises(Stop):
                async with PendingScores() as scores:
                    for n in range(5):
                        await scores.submit(lambda n=n: job(n))
                        self.assertLessEqual(scores.pending_count, 2)
                    raise Stop()
            self.assertEqual(peak, 1)
            self.assertEqual(completed, list(range(5)))
        asyncio.run(scenario())

    def test_browser_prepares_confirmation_then_closes_api_does_not_wait(self):
        async def scenario():
            target = object.__new__(BrowserUITarget)
            target._session = {'page': object()}
            target.close_conversation = AsyncMock()
            with patch('app.engine.pending_scores.asyncio.sleep', new_callable=AsyncMock) as sleep:
                prepare_reply_for_close(target)
                self.assertTrue(target._confirm_before_close)
                await close_after_reply(target)
                sleep.assert_not_awaited()
                target.close_conversation.assert_awaited_once()
                sleep.reset_mock()
                await close_after_reply(ConvChat())
                sleep.assert_not_awaited()
        asyncio.run(scenario())

    def test_restart_retains_capture_for_judge_only_retry(self):
        self.bench.status = 'running'
        self.bench.llm_config_json = json.dumps({'JUDGE_METHOD': 'dual_v1_v2'})
        row = Prompt(benchmark_id=self.bench.id, attack_slug='harmbench', sequence=0,
                     prompt_text='saved prompt', response='saved reply', status=PromptStatus.SCORING)
        db.session.add(row)
        db.session.commit()
        self.assertFalse(row.can_rescore)
        from app import _mark_interrupted_benchmarks
        _mark_interrupted_benchmarks()
        self.assertEqual(row.status, PromptStatus.JUDGE_ERROR)
        self.assertTrue(row.can_rescore)
        self.assertEqual(row.response, 'saved reply')

    def test_resume_partial_script_preserves_old_capture_and_finishes_turns(self):
        self.enable_per_turn_scores()
        self.bench.custom_conversations_json = json.dumps([['one', 'two']])
        db.session.commit()
        from app.engine.control import BenchmarkCancelled
        checkpoints = 0
        async def checkpoint(bid):
            nonlocal checkpoints
            checkpoints += 1
            if checkpoints == 2:
                raise BenchmarkCancelled()
        with patch('app.engine.scripted_attack.wait_while_paused', checkpoint):
            with self.assertRaises(BenchmarkCancelled):
                self.run_script()
        original = Prompt.query.one().response
        target, _ = self.run_script(resume=True, retry_failed=False)
        self.assertEqual(len(target.calls), 2)
        self.assertEqual(len(Prompt.query.all()), 2)
        archive = engine.ConversationRetry.query.one()
        self.assertEqual(json.loads(archive.original_rows_json)[0]['response'], original)

    def test_real_browser_closes_before_pending_judge_and_next_chat(self):
        self.enable_per_turn_scores()
        self.bench.custom_conversations_json = json.dumps([['one'], ['two']])
        db.session.commit()
        with tempfile.TemporaryDirectory(prefix='sc3098-pipeline-browser-') as tmp:
            page = Path(tmp) / 'chat.html'
            page.write_text('''<input id="msg"><button id="send" onclick="setTimeout(()=>{let p=document.createElement('p');p.className='bot';p.textContent='Early reply';document.body.append(p);setTimeout(()=>{p.textContent='Complete updated reply'},600)},100)">Send</button>''')
            target = BrowserUITarget({'ui_url': page.as_uri(), 'ui_input_selector': '#msg',
                'ui_submit_selector': '#send', 'ui_response_selector': '.bot',
                'ui_headless': 'true', 'ui_initial_settle_ms': 0,
                'ui_stream_settle_ms': 100, 'ui_response_max_ms': 5000})
            captured, closed, started_sends = [], [], []
            send, close = target.send_prompt_async, target.close_conversation
            async def scenario():
                release = asyncio.Event()
                async def sending(**kwargs):
                    if captured:
                        self.assertIsNone(target._session)
                        self.assertEqual(Prompt.query.filter_by(sequence=0).one().status, PromptStatus.SCORING)
                    started_sends.append(time.perf_counter())
                    result = await send(**kwargs)
                    captured.append(time.perf_counter())
                    if len(captured) == 2:
                        release.set()
                    return result
                async def closing():
                    closed.append(time.perf_counter())
                    await close()
                async def judge(*args, **kwargs):
                    await asyncio.wait_for(release.wait(), 10)
                    return 'SCORE: 1', 1, False
                target.send_prompt_async, target.close_conversation = sending, closing
                from app.engine.scripted_attack import run_scripted_attack
                from app.engine.retry import ConsecutiveFailureGuard
                try:
                    with patch('app.engine.scripted_attack.judge_interactive_response', judge):
                        await run_scripted_attack(self.bench, SCRIPT, target, object(), 10,
                            ConsecutiveFailureGuard(3, stop_on_limit=False))
                finally:
                    await target.aclose()
            asyncio.run(scenario())
        self.assertEqual(len(closed), 2)
        for reply_at, close_at in zip(captured, closed):
            self.assertLess(close_at - reply_at, .5)
        for start, close_at in zip(started_sends, closed):
            self.assertGreaterEqual(close_at - start, 2.1)
        self.assertTrue(all(row.severity == 1 and row.response == 'Complete updated reply'
                            and 600 <= row.latency_ms < 1500 for row in Prompt.query.all()))

    def test_confirmation_restarts_after_updates_and_ignores_own_echo(self):
        async def scenario():
            target = object.__new__(BrowserUITarget)
            target.response_max_ms, target.last_debug = 5000, {}
            clock = [0.0]
            async def sleep(seconds):
                clock[0] += seconds
            async def snapshot():
                text = 'partial' if clock[0] < .5 else 'second' if clock[0] < 1 else 'final'
                return ['welcome', 'prompt', text]
            with patch('app.targets.browser.target.asyncio.sleep', sleep), \
                 patch('app.targets.browser.target.time.perf_counter', lambda: clock[0]):
                text, stamp = await target._confirm_closing_reply(snapshot, ['welcome'],
                    lambda text: text == 'prompt', 'partial', 0)
            self.assertEqual(text, 'final')
            self.assertGreaterEqual(clock[0] - stamp, 1.5)
            self.assertLess(clock[0] - stamp, 1.7)
            self.assertEqual(target.last_debug['close_confirmation']['wait_ms'], 1500)
        asyncio.run(scenario())

    def test_confirmation_times_out_on_continuous_changes(self):
        async def scenario():
            target = object.__new__(BrowserUITarget)
            target.response_max_ms, target.last_debug = 1500, {}
            clock = [0.0]
            async def sleep(seconds):
                clock[0] += seconds
            async def snapshot():
                return [str(clock[0])]
            with patch('app.targets.browser.target.asyncio.sleep', sleep), \
                 patch('app.targets.browser.target.time.perf_counter', lambda: clock[0]):
                with self.assertRaisesRegex(TimeoutError, 'response budget'):
                    await target._confirm_closing_reply(snapshot, [], lambda _: False, 'partial', 0)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
