"""Offline engine tests using real PyRIT 0.14 and in-memory databases.

Run from SC3098: ../.venv/bin/python -m unittest discover -s test -p test_student_engine.py -v
No provider calls or paid services are used.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['ATTACKER_BACKUP_MODEL'] = ''

# PyRIT creates a log/cache on import; keep those inside a disposable test root.
import atexit
_test_runtime = tempfile.TemporaryDirectory(prefix='sc3098-pyrit-test-')
atexit.register(_test_runtime.cleanup)
_runtime_patch = patch('appdirs.user_data_dir',
                       side_effect=lambda name, *a, **k: str(Path(_test_runtime.name) / name))
_runtime_patch.start()
atexit.register(_runtime_patch.stop)

from flask import Flask
from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from app.extensions import db
from app.models import Benchmark, Target, Prompt, PromptStatus, ConversationRetry
from app.engine.scoring import RUBRICS, judge_response, parse_strict_severity
from app.engine.scripted_attack import run_scripted_attack
from app.engine.static_attack import run_static_attack
from app.engine.retry import ConsecutiveFailureGuard
from app.web.rollups import benchmark_rollups

CentralMemory.set_memory_instance(SQLiteMemory(db_path=':memory:', silent=True))


class MockChat(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(capabilities=TargetCapabilities(
        supports_multi_turn=True, supports_editable_history=True))

    def __init__(self, fail_at=None):
        super().__init__()
        self.calls = []
        self.fail_at = fail_at

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        current = normalized_conversation[-1]
        self.calls.append((current.get_piece().conversation_id,
                           [(m.get_piece().role, m.get_value()) for m in normalized_conversation]))
        if len(self.calls) == self.fail_at:
            raise ValueError('offline test failure')
        return [construct_response_from_request(request=current.get_piece(),
                response_text_pieces=[f'reply {len(normalized_conversation)}: {current.get_value()}'])]


class ConvChat:
    """Conversation target with the app's contract (HttpTarget.send_turn_async).

    Mirrors MockChat's replies: "reply N: <prompt>", where N counts the messages
    sent so far including the new prompt. A new conversation starts with empty
    history, so each call records a fresh conversation key when history is empty.
    """

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at
        self._conv = 0

    async def reply(self, prompt, history):
        return f'reply {len(history) + 1}: {prompt}'

    async def send_turn_async(self, prompt, history):
        if not history:
            self._conv += 1
        hist = [(h['role'], h['content']) for h in history] + [('user', prompt)]
        self.calls.append((f'conv-{self._conv}', hist))
        if len(self.calls) == self.fail_at:
            raise ValueError('offline test failure')
        return await self.reply(prompt, history)


class BrowserLikeChat:
    """Browser-style target: keeps one chat per conversation_id, like BrowserUITarget."""

    def __init__(self):
        self.calls = []
        self.pages = {}

    async def send_prompt_async(self, *, message):
        cid = message.conversation_id
        page = self.pages.setdefault(cid, [])
        page.append(message.get_value())
        self.calls.append((cid, list(page)))
        from app.targets.common import wrap_response
        return wrap_response(f'page reply {len(page)}: {message.get_value()}')


class Judge:
    def __init__(self, verdict='REASONING: test evidence\nSCORE: 1'):
        self.verdict = verdict
        self.prompts = []

    async def send_prompt_async(self, *, message):
        self.prompts.append(message.get_value())
        return [construct_response_from_request(request=message.get_piece(), response_text_pieces=[self.verdict])]


SCRIPT = {'slug': 'custom_conversation', 'kind': 'scripted', 'dataset': 'custom'}
STATIC = {'slug': 'custom_prompt_list', 'kind': 'static', 'dataset': 'custom'}


class StudentEngineTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        t = Target(name='offline', provider='openai', model_name='mock')
        db.session.add(t)
        db.session.flush()
        self.bench = Benchmark(name='offline', target_id=t.id,
                               custom_conversations_json=json.dumps([['first', 'second', 'third'], ['other', 'last']]),
                               custom_prompts_json=json.dumps(['first', 'second']),
                               llm_config_json=json.dumps({'CUSTOM_RUBRIC': 'over_refusal',
                                                           'FAILURE_CRITERION': 'Unjustified refusal of legitimate requests',
                                                           'REFERENCE_FACTS': 'Allowed: explain diversification.'}))
        db.session.add(self.bench)
        db.session.commit()
        self.delay = patch('app.engine.scripted_attack.REQUEST_DELAY', 0)
        self.delay.start()
        self.static_delay = patch('app.engine.static_attack.REQUEST_DELAY', 0)
        self.static_delay.start()

    def tearDown(self):
        self.delay.stop()
        self.static_delay.stop()
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def run_script(self, target=None, judge=None, **kwargs):
        target = target or ConvChat()
        judge = judge or Judge()
        asyncio.run(run_scripted_attack(self.bench, SCRIPT, target, judge, 10,
                                       ConsecutiveFailureGuard(3, stop_on_limit=False), **kwargs))
        db.session.expire(self.bench, ['prompts'])
        return target, judge

    def enable_per_turn_scores(self):
        cfg = self.bench.llm_config
        cfg.update({'SCRIPTED_SCORE_UNIT': 'turn', 'JUDGE_METHOD': 'j0_severity'})
        self.bench.llm_config_json = json.dumps(cfg)
        db.session.commit()

    def test_scripted_per_turn_scores_and_prior_context_are_isolated(self):
        self.enable_per_turn_scores()

        class TurnJudge(Judge):
            async def send_prompt_async(self, *, message):
                self.verdict = f'REASONING: latest reply evidence\nSCORE: {len(self.prompts) + 1}'
                return await super().send_prompt_async(message=message)

        target, judge = self.run_script(judge=TurnJudge())
        rows = Prompt.query.order_by(Prompt.sequence).all()
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(judge.prompts), 5)
        self.assertEqual([p.sequence for p in rows], list(range(5)))
        self.assertEqual([p.objective_id for p in rows], ['script-0'] * 3 + ['script-1'] * 2)
        self.assertEqual([p.prompt_text for p in rows], ['first', 'second', 'third', 'other', 'last'])
        self.assertEqual([p.severity for p in rows], [1, 2, 3, 4, 5])
        self.assertEqual([p.succeeded for p in rows], [False, False, True, True, True])
        self.assertEqual([len(history) for _, history in target.calls], [1, 3, 5, 1, 3])
        self.assertTrue(all(p.latency_ms is not None and p.latency_ms >= 0 for p in rows))
        self.assertTrue(all(len(p.transcript) == 1 for p in rows))
        self.assertEqual([p.transcript[0]['turn'] for p in rows], [1, 2, 3, 1, 2])
        snapshots = [json.loads(p.judge_input_json) for p in rows]
        self.assertEqual([len(s['history']) for s in snapshots], [0, 2, 4, 0, 2])
        self.assertEqual(snapshots[1]['history'], [
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'reply 1: first'}])
        self.assertEqual(snapshots[3]['history'], [])
        self.assertNotIn('reply 5: third', judge.prompts[0])
        self.assertNotIn('reply 1: first', judge.prompts[3])
        self.assertEqual(self.bench.compute_totals()['counted'], 5)

    def test_scripted_per_turn_judge_retry_never_calls_target(self):
        self.enable_per_turn_scores()

        class OneInvalidJudge(Judge):
            async def send_prompt_async(self, *, message):
                self.verdict = ('invalid verdict' if len(self.prompts) == 1
                                else 'REASONING: latest reply evidence\nSCORE: 1')
                return await super().send_prompt_async(message=message)

        self.run_script(judge=OneInvalidJudge())
        failed = Prompt.query.filter_by(sequence=1).one()
        self.assertEqual(failed.status, PromptStatus.JUDGE_ERROR)
        before = [(p.id, p.response, p.severity) for p in Prompt.query.order_by(Prompt.sequence)]
        saved_context = json.loads(failed.judge_input_json)['history']
        target, judge = self.run_script(judge=Judge('REASONING: re-scored latest reply\nSCORE: 4'),
                                        retry_prompt=failed)
        self.assertEqual(target.calls, [])
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(failed.status, PromptStatus.OK)
        self.assertEqual(failed.severity, 4)
        self.assertEqual(json.loads(failed.judge_input_json)['history'], saved_context)
        self.assertEqual(ConversationRetry.query.count(), 0)
        after = [(p.id, p.response, p.severity) for p in Prompt.query.order_by(Prompt.sequence)]
        self.assertEqual(before[:1] + before[2:], after[:1] + after[2:])

    def test_scripted_per_turn_send_retry_archives_only_failed_group(self):
        self.enable_per_turn_scores()
        self.run_script(ConvChat(fail_at=2))
        original = Prompt.query.filter_by(objective_id='script-0').order_by(Prompt.sequence).all()
        failed = next(p for p in original if p.status == PromptStatus.SEND_ERROR)
        untouched = [(p.id, p.response, p.severity) for p in
                     Prompt.query.filter_by(objective_id='script-1').order_by(Prompt.sequence)]
        original_ids = [p.id for p in original]
        target, judge = self.run_script(retry_prompt=failed)
        self.assertEqual([len(history) for _, history in target.calls], [1, 3, 5])
        self.assertEqual(len(judge.prompts), 3)
        self.assertEqual(Prompt.query.count(), 5)
        self.assertTrue(all(p.status == PromptStatus.OK for p in Prompt.query.all()))
        archive = ConversationRetry.query.one()
        self.assertEqual(archive.objective_id, 'script-0')
        self.assertEqual([p['id'] for p in archive.original_rows], original_ids)
        self.assertEqual([p['status'] for p in archive.original_rows],
                         [PromptStatus.OK, PromptStatus.SEND_ERROR])
        self.assertEqual(untouched, [(p.id, p.response, p.severity) for p in
                         Prompt.query.filter_by(objective_id='script-1').order_by(Prompt.sequence)])

    def test_scripted_per_turn_progress_counts_replies_not_scripts(self):
        from app.settings import expected_rows
        self.enable_per_turn_scores()
        self.assertEqual(expected_rows(SCRIPT, 1, self.bench), 5)
        self.bench.custom_conversations_json = json.dumps([['one'], ['two', 'three']])
        self.assertEqual(expected_rows(SCRIPT, 10, self.bench), 3)
        self.bench.custom_conversations_json = '[]'
        self.assertEqual(expected_rows(SCRIPT, 10, self.bench), 0)

    def test_scripted_per_turn_cancel_retains_each_captured_score(self):
        from app.engine.control import request_cancel, clear_cancel, BenchmarkCancelled
        self.enable_per_turn_scores()
        bench = self.bench
        outer = self

        class CancelChat(ConvChat):
            async def reply(self, prompt, history):
                if len(self.calls) == 2:
                    first = Prompt.query.filter_by(sequence=0).one()
                    outer.assertIn(first.status, {PromptStatus.OK, PromptStatus.SCORING})
                    outer.assertTrue(first.response)
                    request_cancel(bench.id)
                return await super().reply(prompt, history)

        try:
            with self.assertRaises(BenchmarkCancelled):
                self.run_script(CancelChat())
            rows = Prompt.query.order_by(Prompt.sequence).all()
            self.assertEqual([p.prompt_text for p in rows], ['first', 'second'])
            self.assertTrue(all(p.status == PromptStatus.OK for p in rows))
            self.assertEqual(self.bench.compute_totals()['counted'], 2)
        finally:
            clear_cancel(bench.id)

    def test_scripted_per_turn_resume_scores_captured_reply_without_replay(self):
        self.enable_per_turn_scores()
        self.run_script()
        last = Prompt.query.filter_by(sequence=4).one()
        last.status, last.severity, last.verdict = PromptStatus.INCOMPLETE, None, None
        db.session.commit()
        original_ids = [p.id for p in Prompt.query.order_by(Prompt.sequence)]
        target, judge = self.run_script(resume=True, retry_failed=False)
        self.assertEqual(target.calls, [])
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(last.status, PromptStatus.OK)
        self.assertEqual(last.severity, 1)
        self.assertEqual(original_ids, [p.id for p in Prompt.query.order_by(Prompt.sequence)])
        self.assertEqual(ConversationRetry.query.count(), 0)

    def test_scripted_per_turn_one_reply_displays_as_conversation_turn(self):
        self.enable_per_turn_scores()
        self.bench.custom_conversations_json = json.dumps([['one']])
        self.run_script()
        block = benchmark_rollups(self.bench)['display_groups']['custom_conversation'][0]
        self.assertEqual(block['kind'], 'conversation')
        self.assertEqual(block['objective_id'], 'script-0')
        self.assertEqual(len(block['turns']), 1)
        self.assertEqual(block['turns'][0].transcript[0]['turn'], 1)

    def test_scripted_per_turn_both_judges_score_every_reply(self):
        self.enable_per_turn_scores()
        cfg = self.bench.llm_config
        cfg.update({'JUDGE_METHOD': 'dual_v1_v2', 'ASSESSMENT_RULES':
                    'Teach investing. Refuse harmful requests and advice about individual shares.'})
        self.bench.llm_config_json = json.dumps(cfg)
        self.bench.custom_conversations_json = json.dumps([['first', 'second'], ['independent']])
        v1 = Judge('REASONING: latest reply safe\nSCORE: 1')
        v2 = Judge('\n'.join(f'CRITERION_{n}: 2 | ambiguous evidence' for n in range(1, 6)) + '\nSCORE: 2')
        self.run_script(judge={'v1': v1, 'v2': v2})
        self.assertEqual(len(v1.prompts), 3)
        self.assertEqual(len(v2.prompts), 3)
        self.assertTrue(all('Evaluate ONLY the latest assistant reply' in p
                            for p in v1.prompts + v2.prompts))
        rows = Prompt.query.order_by(Prompt.sequence).all()
        for row in rows:
            results = json.loads(row.verdict)['judge_version_results']
            self.assertEqual(results['v1']['score'], 1)
            self.assertEqual(results['v2']['score'], 2)
            self.assertEqual(row.severity, 1)
        self.assertNotIn('reply 1: first', v1.prompts[2])
        self.assertNotIn('reply 1: first', v2.prompts[2])

    def test_one_turn_and_mixed_scripts_are_isolated_and_scored(self):
        self.bench.custom_conversations_json = json.dumps([['single\nmultiline'], ['first', 'follow-up'], ['independent']])
        target, judge = self.run_script()
        self.assertEqual([len(history) for _, history in target.calls], [1, 1, 3, 1])
        self.assertEqual([len(row.transcript) for row in self.bench.prompts], [1, 2, 1])
        self.assertEqual(len(judge.prompts), 3)
        self.assertTrue(all(row.counts_toward_stats for row in self.bench.prompts))

    def test_one_turn_scripts_support_real_pyrit_message_targets(self):
        self.bench.custom_conversations_json = json.dumps([['first'], ['independent']])
        target, judge = self.run_script(MockChat())
        self.assertEqual(len(target.calls), 2)
        self.assertNotEqual(target.calls[0][0], target.calls[1][0])
        self.assertEqual([len(history) for _, history in target.calls], [1, 1])
        self.assertEqual(len(judge.prompts), 2)

    def test_real_pyrit_order_history_isolation_full_judging(self):
        target, judge = self.run_script()
        self.assertEqual(len(target.calls), 5)
        self.assertEqual([len(hist) for _, hist in target.calls], [1, 3, 5, 1, 3])
        self.assertEqual(len({cid for cid, _ in target.calls[:3]}), 1)
        self.assertNotEqual(target.calls[0][0], target.calls[3][0])
        self.assertEqual([hist[-1][1] for _, hist in target.calls], ['first', 'second', 'third', 'other', 'last'])
        rows = self.bench.prompts
        self.assertEqual([len(p.transcript) for p in rows], [3, 2])
        self.assertTrue(all(p.counts_toward_stats for p in rows))
        self.assertEqual(self.bench.compute_totals()['counted'], 2)
        self.assertEqual(len(judge.prompts), 2)
        self.assertIn('reply 1: first', judge.prompts[0])
        self.assertIn('reply 5: third', judge.prompts[0])
        self.assertIn('TRUSTED ASSESSMENT CONFIGURATION', judge.prompts[0])
        self.assertIn('Allowed: explain diversification.', judge.prompts[0])
        self.assertEqual(benchmark_rollups(self.bench)['rubric_stats'].keys(), {'over_refusal'})

    def test_incomplete_excluded_retry_whole_fresh_chat_and_archive(self):
        target, judge = self.run_script(ConvChat(fail_at=2))
        first = self.bench.prompts[0]
        self.assertEqual(first.status, PromptStatus.SEND_ERROR)
        self.assertEqual(len(first.transcript), 2)
        self.assertFalse(first.counts_toward_stats)
        self.assertEqual(len(judge.prompts), 1)
        old_cid = first.transcript[0]['conversation_id']
        target2, _ = self.run_script(resume=True)
        self.assertEqual(len(target2.calls), 3)
        self.assertNotEqual(old_cid, first.transcript[0]['conversation_id'])
        self.assertTrue(first.counts_toward_stats)
        self.assertEqual(ConversationRetry.query.count(), 1)
        self.assertEqual(ConversationRetry.query.first().original_rows[0]["status"], PromptStatus.SEND_ERROR)

    def test_resume_without_retry_preserves_partial_fills_missing(self):
        self.run_script(ConvChat(fail_at=2))
        self.run_script(resume=True, retry_failed=False)
        self.assertEqual(Prompt.query.count(), 2)
        self.assertEqual(self.bench.prompts[0].status, PromptStatus.SEND_ERROR)

    def test_invalid_judge_never_defended(self):
        self.run_script(judge=Judge('Cannot judge; refer to section 4.'))
        self.assertTrue(all(p.status == PromptStatus.JUDGE_ERROR for p in self.bench.prompts))
        self.assertEqual(self.bench.compute_totals()['counted'], 0)
        self.assertIsNone(benchmark_rollups(self.bench)['rollup'][0]['rate'])

    def test_single_and_list_use_chosen_rubric(self):
        # Decision S24: students' own prompts always run in full; the cap is for datasets.
        for limit, expected in [(1, 2), (2, 2)]:
            Prompt.query.delete()
            target, judge = MockChat(), Judge('REASONING: unjustified refusal\nSCORE: 4')
            asyncio.run(run_static_attack(self.bench, STATIC, target, judge, limit,
                                         ConsecutiveFailureGuard(3)))
            self.assertEqual(Prompt.query.count(), expected)
            self.assertTrue(all(p.succeeded and p.rubric == 'over_refusal' for p in Prompt.query.all()))
            self.assertIn('Unjustified refusal', judge.prompts[0])
            self.assertEqual(len({cid for cid, _ in target.calls}), expected)

    def test_written_policy_is_judge_only_for_static_and_scripted(self):
        rules = 'POLICY-ONLY-93: teach investing; never recommend individual stocks.'
        self.bench.llm_config_json = json.dumps({'CUSTOM_RUBRIC': 'off_topic', 'ASSESSMENT_RULES': rules})
        db.session.commit()
        for scripted in (False, True):
            with self.subTest(scripted=scripted):
                Prompt.query.delete()
                target, judge = (ConvChat() if scripted else MockChat()), Judge()
                if scripted:
                    self.run_script(target, judge)
                else:
                    asyncio.run(run_static_attack(self.bench, STATIC, target, judge, 2,
                                                 ConsecutiveFailureGuard(3)))
                self.assertTrue(judge.prompts)
                self.assertTrue(all(rules in text for text in judge.prompts))
                self.assertTrue(all('TRUSTED ASSESSMENT CONFIGURATION' in text for text in judge.prompts))
                self.assertNotIn('POLICY-ONLY-93', json.dumps(target.calls))
                self.assertTrue(all(row.rubric == 'off_topic' for row in Prompt.query.all()))

    def test_policy_validation_and_legacy_assessment_settings(self):
        from app.engine.scoring import own_prompt_assessment, assessment_settings
        rules = 'x' * 8000
        self.assertEqual(own_prompt_assessment('injection', rules)['ASSESSMENT_RULES'], rules)
        for rubric in ('off_topic', 'over_refusal', 'injection', 'rule_override'):
            with self.assertRaises(ValueError):
                own_prompt_assessment(rubric, ' \n ')
        with self.assertRaises(ValueError):
            own_prompt_assessment('injection', rules + 'x')
        self.bench.llm_config_json = json.dumps({'CUSTOM_RUBRIC': 'harm_disclosure'})
        self.assertEqual(assessment_settings(self.bench, STATIC)['rubric'], 'harm_disclosure')
        self.assertEqual(assessment_settings(self.bench, STATIC)['assessment_rules'], '')
        self.bench.llm_config_json = json.dumps({'JUDGE_METHOD': 'j0_severity', 'CUSTOM_RUBRIC': 'off_topic',
                                               'ASSESSMENT_RULES': 'must not reach J0'})
        settings = assessment_settings(self.bench, STATIC)
        self.assertEqual(settings['method'], 'j0_severity')
        self.assertNotIn('assessment_rules', settings)

    def test_all_rubrics_parse_pass_and_fail(self):
        for rubric in RUBRICS:
            for severity in (1, 5):
                judge = Judge(f'REASONING: test evidence\nSCORE: {severity}')
                _, score, success = asyncio.run(judge_response(judge, 'test', 'response', rubric=rubric, strict=True))
                self.assertEqual(score, severity)
                self.assertEqual(success, severity == 5)
                self.assertNotIn('<<<PROMPT>>>', judge.prompts[0])
        for verdict in ('SCORE: 10', 'SCORE: 0', 'SCORE: 6', 'section 4', 'SCORE: 1\nSCORE: 5', 'SCORE: 3.5'):
            self.assertIsNone(parse_strict_severity(verdict))

    def test_custom_progress_counts_saved_cases_not_cap_or_turns(self):
        from app.settings import expected_rows
        self.bench.custom_conversations_json = json.dumps([['one', 'two', 'three']])
        self.assertEqual(expected_rows(SCRIPT, 2, self.bench), 1)
        self.assertEqual(expected_rows(STATIC, 10, self.bench), 2)
        self.assertEqual(expected_rows(STATIC, 1, self.bench), 2)  # S24: never capped
        self.bench.custom_conversations_json = '[]'
        self.assertEqual(expected_rows(SCRIPT, 10, self.bench), 0)
        self.assertEqual(expected_rows({'kind': 'static', 'dataset': 'baseline'}, 7, self.bench), 7)

    def test_blank_target_response_never_counted_or_judged(self):
        class EmptyChat(ConvChat):
            async def reply(self, prompt, history):
                return ''

        _, judge = self.run_script(EmptyChat())
        self.assertEqual(judge.prompts, [])
        self.assertEqual(self.bench.compute_totals()['counted'], 0)
        self.assertTrue(all(p.status == PromptStatus.SEND_ERROR for p in self.bench.prompts))
        self.assertTrue(all(not p.response for p in self.bench.prompts))

    def test_old_rows_remain_readable_and_counted(self):
        row = Prompt(benchmark_id=self.bench.id, attack_slug='old', prompt_text='old',
                     response='reply', status=PromptStatus.OK, severity=1)
        db.session.add(row)
        db.session.commit()
        self.assertEqual(row.transcript, [])
        self.assertIsNone(row.rubric)
        self.assertTrue(row.counts_toward_stats)
        self.assertEqual(benchmark_rollups(self.bench)['rollup'][0]['rubric'], 'harm')

    def test_canary_is_response_only_evidence_no_override(self):
        cfg = self.bench.llm_config
        cfg['CANARY'] = 'MARKER'
        self.bench.llm_config_json = json.dumps(cfg)
        from app.engine.scoring import detect_canary
        self.assertFalse(detect_canary(self.bench, ['marker']))
        self.assertTrue(detect_canary(self.bench, ['found MARKER']))
        self.bench.custom_conversations_json = json.dumps([['MARKER', 'second']])
        self.run_script()
        self.assertTrue(self.bench.prompts[0].canary_detected)
        self.assertFalse(self.bench.prompts[0].succeeded)

    def test_refusal_can_be_failure_and_static_invalid_is_unmeasured(self):
        class RefusingChat(MockChat):
            async def _send_prompt_to_target_async(self, *, normalized_conversation):
                return [construct_response_from_request(request=normalized_conversation[-1].get_piece(),
                        response_text_pieces=["I'm sorry, I can't assist with diversification."])]
        self.bench.custom_prompts_json = json.dumps(['explain diversification'])
        for verdict, expected in [('REASONING: unjustified refusal\nSCORE: 5', True), ('invalid section 4', False)]:
            Prompt.query.delete()
            asyncio.run(run_static_attack(self.bench, STATIC, RefusingChat(), Judge(verdict), 1,
                                         ConsecutiveFailureGuard(3)))
            row = Prompt.query.one()
            self.assertTrue(row.refused)
            self.assertEqual(row.succeeded, expected)
            self.assertEqual(row.counts_toward_stats, expected)

    def test_partial_turn_saved_before_next_send_and_cancel_retained(self):
        from app.engine.control import request_cancel, clear_cancel, BenchmarkCancelled
        bench = self.bench
        class CancelChat(ConvChat):
            async def reply(self, prompt, history):
                if len(self.calls) == 2:
                    partial = Prompt.query.one()
                    self_outer.assertEqual(len(partial.transcript), 1)
                    self_outer.assertEqual(partial.status, PromptStatus.INCOMPLETE)
                    self_outer.assertFalse(partial.counts_toward_stats)
                    request_cancel(bench.id)
                return await super().reply(prompt, history)
        self_outer = self
        try:
            with self.assertRaises(BenchmarkCancelled):
                self.run_script(CancelChat())
            row = Prompt.query.one()
            self.assertEqual(len(row.transcript), 2)
            self.assertFalse(row.counts_toward_stats)
            self.assertEqual(self.bench.best_of_n['groups'], 0)
            self.assertEqual(self.bench.reach_curve, [])
        finally:
            clear_cancel(bench.id)

    def test_additive_migration_preserves_older_rows(self):
        from sqlalchemy import text, inspect
        from app import _migrate_add_columns
        row = Prompt(benchmark_id=self.bench.id, attack_slug='old', prompt_text='old',
                     response='retained', status=PromptStatus.OK, severity=1)
        db.session.add(row)
        db.session.commit()
        row_id = row.id
        with db.engine.begin() as conn:
            for table, column in [('prompts', 'transcript_json'), ('prompts', 'rubric'),
                                  ('prompts', 'canary_detected'), ('benchmarks', 'custom_conversations_json')]:
                conn.execute(text(f'ALTER TABLE {table} DROP COLUMN {column}'))
        _migrate_add_columns()
        _migrate_add_columns()
        db.session.expire_all()
        self.assertEqual(db.session.get(Prompt, row_id).response, 'retained')
        self.assertEqual(db.session.get(Prompt, row_id).transcript, [])
        self.assertIn('custom_conversations_json', {c['name'] for c in inspect(db.engine).get_columns('benchmarks')})

    def test_runner_fresh_retry_and_resume_use_scripted_path(self):
        from app.engine import runner
        self.app.config['MAX_PROMPTS_PER_ATTACK'] = 10
        self.bench.attack_slugs_json = json.dumps(['custom_conversation'])
        target = ConvChat(fail_at=2)
        with patch.object(runner, 'build_conversation_target', return_value=target), patch.object(runner, 'ATTACK_DELAY', 0):
            asyncio.run(runner._run_attacks(self.bench, target, Judge(), 10))
        self.assertEqual(self.bench.status, 'completed_with_target_errors')
        failed = Prompt.query.filter_by(sequence=0).one()
        replacement = ConvChat()
        with patch.object(runner, 'build_conversation_target', return_value=replacement), patch.object(runner, 'build_judge_llm', return_value=Judge()):
            asyncio.run(runner._rerun_prompt(self.bench.id, failed.id, self.bench.status))
        self.assertEqual(len(replacement.calls), 3)
        self.assertTrue(failed.counts_toward_stats)
        self.assertEqual(self.bench.status, "completed")
        self.assertEqual(ConversationRetry.query.count(), 1)
        # Mark one whole conversation unfinished; recovery must start its chat afresh.
        failed.status = PromptStatus.INCOMPLETE
        db.session.commit()
        resumed = ConvChat()
        with patch.object(runner, 'build_conversation_target', return_value=resumed), patch.object(runner, 'build_judge_llm', return_value=Judge()):
            asyncio.run(runner._recover_async(self.bench.id))
        self.assertEqual(len(resumed.calls), 3)
        self.assertEqual(self.bench.status, 'completed')
        self.assertEqual(Prompt.query.count(), 2)


if __name__ == '__main__':
    unittest.main()
