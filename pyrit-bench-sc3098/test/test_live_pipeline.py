"""Offline live-session pipeline checks.

Run from SC3098: ../.venv/bin/python -B test/test_live_pipeline.py -v
"""
import threading
import time
import unittest
import test_interactive as fixtures


class LivePipelineTests(unittest.TestCase):
    setUp = fixtures.InteractiveTests.setUp
    tearDown = fixtures.InteractiveTests.tearDown
    start = fixtures.InteractiveTests.start
    snapshot = fixtures.InteractiveTests.snapshot
    wait = fixtures.InteractiveTests.wait
    post = fixtures.InteractiveTests.post
    send = fixtures.InteractiveTests.send

    def test_next_turn_uses_captured_history_while_previous_judge_waits(self):
        self.judge_block = threading.Event()
        self.start()
        self.send('first question', 'first')
        data = self.wait(lambda d: d['state'] == 'ready' and bool(d['turns'][0]['response']))
        self.assertEqual(data['turns'][0]['status'], 'scoring')
        self.assertEqual(data['turns_scored'], 0)
        self.assertEqual(self.send('follow-up question', 'second').status_code, 200)
        data = self.wait(lambda d: len(d['turns']) == 2 and bool(d['turns'][1]['response']))
        self.assertEqual(len(self.target.calls), 2)
        self.assertEqual(self.target.calls[1][1], [
            {'role': 'user', 'content': 'first question'},
            {'role': 'assistant', 'content': 'first reply'}])
        self.assertEqual(self.target.closed, 0)
        self.assertEqual(data['turns_scored'], 0)
        self.assertEqual(len(self.judge_calls), 1)  # judge requests are serialised
        self.assertEqual(data['state'], 'scoring')  # bounded queue holds two replies
        self.assertEqual(self.send('excess pending turn', 'third').status_code, 409)
        self.judge_block.set()
        self.wait(lambda d: d['state'] == 'ready' and d['turns_scored'] == 2)
        self.assertEqual(self.judge_calls[1][2], self.target.calls[1][1])

    def test_end_closes_browser_before_judge_finishes_and_waits_to_complete(self):
        self.judge_block = threading.Event()
        self.start()
        self.send('capture and close', 'first')
        self.wait(lambda d: d['state'] == 'ready' and bool(d['turns'][0]['response']))
        self.post('end')
        deadline = time.monotonic() + 2
        while self.target.closed == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.target.closed, 1)
        data = self.snapshot()
        self.assertEqual(data['state'], 'ending')
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['turns_scored'], 0)
        self.assertTrue(self.engine.worker_busy(self.bid))
        self.judge_block.set()
        data = self.wait(lambda d: not d['active'])
        self.assertEqual(data['status'], 'completed')
        self.assertEqual(data['turns_scored'], 1)
        self.assertFalse(self.engine.worker_busy(self.bid))

    def test_score_completion_does_not_reopen_composer_during_target_send(self):
        self.judge_block = threading.Event()
        self.start()
        self.send('first question', 'first')
        self.wait(lambda d: d['state'] == 'ready' and bool(d['turns'][0]['response']))
        self.target.block = threading.Event()
        self.send('second question', 'second')
        self.wait(lambda d: len(self.target.calls) == 2 and d['state'] == 'sending')
        self.judge_block.set()
        data = self.wait(lambda d: d['turns_scored'] == 1)
        self.assertEqual(data['state'], 'sending')
        self.assertEqual(self.send('must wait for target', 'third').status_code, 409)
        self.target.block.set()
        self.wait(lambda d: d['state'] == 'ready' and d['turns_scored'] == 2)

    def test_idle_closes_target_while_captured_reply_is_still_scoring(self):
        self.app.config['INTERACTIVE_IDLE_SECONDS'] = 1
        self.judge_block = threading.Event()
        self.start()
        self.send('idle after reply', 'first')
        self.wait(lambda d: d['state'] == 'ready' and bool(d['turns'][0]['response']))
        self.send('second pending reply', 'second')
        self.wait(lambda d: d['state'] == 'scoring' and len(d['turns']) == 2)
        self.wait(lambda d: d['state'] == 'ending', timeout=3)
        self.assertEqual(self.target.closed, 1)
        self.judge_block.set()
        data = self.wait(lambda d: not d['active'])
        self.assertEqual(data['turns_scored'], 2)
        self.assertEqual(data['detail'], 'Closed after inactivity.')

    def test_turn_limit_closes_target_without_waiting_for_final_score(self):
        self.app.config['INTERACTIVE_MAX_TURNS'] = 1
        self.judge_block = threading.Event()
        self.start()
        self.send('only turn', 'first')
        self.wait(lambda d: d['state'] == 'ending')
        self.assertEqual(self.target.closed, 1)
        self.assertEqual(self.snapshot()['turns'][0]['response'], 'first reply')
        self.judge_block.set()
        data = self.wait(lambda d: not d['active'])
        self.assertEqual(data['turns_scored'], 1)
        self.assertEqual(data['detail'], 'Turn limit reached.')


if __name__ == '__main__':
    unittest.main()
