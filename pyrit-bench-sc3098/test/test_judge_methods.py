"""Judge-method contracts and dual-judge mounted aliases.

Checks saved dual results, J0 prompt/parser compatibility, method preservation during
re-score and retry, judge configuration, method labels and shared application storage.
Temporary SQLite and mocked target/judge; dotenv is disabled and no provider is called.
Run from the repository root: .venv/bin/python -B test/test_judge_methods.py -v
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = tempfile.TemporaryDirectory(prefix='sc3098-methods-')
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['DEFAULT_JUDGE'] = 'v1'
os.environ['DATABASE_URL'] = 'sqlite:///' + str(Path(_tmp.name) / 'test.db')
os.environ.update({'JUDGE_PROVIDER': 'profllm', 'JUDGE_MODEL': 'offline-test', 'LLM_ENDPOINT': 'http://127.0.0.1:9/v1',
                   'LLM_API_KEY': 'offline-test-key', 'JUDGE_API_KEY_ENV': 'LLM_API_KEY',
                   'LLM_REQUEST_DELAY': '0', 'LLM_ATTACK_DELAY': '0'})
for _name in ('SHOW_PUBLIC_BASELINES', 'IN_DOCKER', 'MAX_PROMPTS_PER_ATTACK', 'SUCCESS_SEVERITY_THRESHOLD'):
    os.environ.pop(_name, None)
import atexit  # noqa: E402
_runtime_patch = patch('appdirs.user_data_dir', side_effect=lambda name, *a, **k: str(Path(_tmp.name) / name))
_runtime_patch.start()
atexit.register(_runtime_patch.stop)

import app as app_module  # noqa: E402
from app import V1_CONFIG, create_served_app  # noqa: E402
from app.engine import runner  # noqa: E402
from app.engine.llms import build_judge_llm, llm_overrides  # noqa: E402
from app.engine.retry import ConsecutiveFailureGuard  # noqa: E402
from app.engine.scoring import (J0_METHOD, JUDGE_PROMPT_TEMPLATE, OWN_PROMPT_RUBRIC, RUBRICS,  # noqa: E402
                                SC3098_METHOD, DUAL_METHOD, V2_RUBRIC)
from app.engine.scripted_attack import judge_views, run_scripted_attack  # noqa: E402
from app.engine.static_attack import run_static_attack  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import Benchmark, Prompt, PromptStatus, Target  # noqa: E402
from app.targets.common import wrap_response  # noqa: E402
from app.web import evidence  # noqa: E402

REMOVED_BANNER = 'Both v1 and v2 score each reply'
# judge_template_sha256 of reports/final_review/rubric_calibration_v1/study_s1/public_human_01/
# judge_main_h1s_01/run_manifest.json (the H1S run).
H1S_TEMPLATE_SHA256 = 'd4d0e43f90fa8795a3a0c11d80916b5a62f3bac36c93ee23b04f05a3a93e7695'
# No SCORE line: J0's parser takes the last 1-5 digit (4); SC3098's strict parser finds no score.
LOOSE_VERDICT = 'REASONING: the reply lists 4 Paris sights.'
STATIC = {'slug': 'custom_prompt_list', 'kind': 'static', 'dataset': 'custom'}
SCRIPTED = {'slug': 'custom_conversation', 'kind': 'scripted', 'dataset': 'custom'}
BOT_REPLY, TURN_REPLY = 'A stored reply from the bot.', 'A stored turn reply.'
JUDGE_CFG = {'JUDGE_PROVIDER': 'profllm', 'JUDGE_MODEL': 'offline-test', 'SUCCESS_SEVERITY_THRESHOLD': 3}
ASSESSMENT = {'CUSTOM_RUBRIC': 'off_topic', 'FAILURE_CRITERION': 'Fails if it plans a trip.',
              'REFERENCE_FACTS': 'Scope: investing only.'}


def j0_request(prompt, response):
    """The H1S study's rendering (tools/study_s1_judge.py severity_prompt)."""
    return (JUDGE_PROMPT_TEMPLATE.replace('<<<PROMPT>>>', prompt).replace('<<<RESPONSE>>>', response)
            .replace('<<<DECODED_BLOCK>>>', ''))


class CapturingJudge:
    def __init__(self, verdict=LOOSE_VERDICT):
        self.verdict, self.seen = verdict, []

    async def send_prompt_async(self, *, message):
        self.seen.append(message.get_value())
        return wrap_response(self.verdict)


class StubTarget:
    async def send_prompt_async(self, *, message):
        return wrap_response(BOT_REPLY)

    async def send_turn_async(self, prompt, history):
        return TURN_REPLY


class V1Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('app.artifacts.ARTIFACTS_DIR', Path(_tmp.name) / 'artifacts'):
            cls.app = create_served_app()
        cls.v1 = cls.app.wsgi_app.mounts['/v1']
        cls.v2 = cls.app.wsgi_app.mounts['/v2']
        for flask_app in (cls.app, cls.v1, cls.v2):
            flask_app.config['TESTING'] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        t = Target(name='Demo target', provider='openai', model_name='mock', endpoint_url='http://127.0.0.1:8999/v1')
        db.session.add(t)
        db.session.commit()
        self.tid = t.id
        self.c = self.app.test_client()
        with self.c.session_transaction() as s:
            s['csrf_token'] = 'test-token'

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def post(self, path, **data):
        return self.c.post(path, data={'csrf_token': 'test-token', 'target_id': str(self.tid), **data})

    def bench(self, config=None, slug='custom_prompt_list', prompts=('Plan my Paris trip',), conversations=(),
              name='run', status='completed'):
        b = Benchmark(name=name, target_id=self.tid, status=status, attack_slugs_json=json.dumps([slug]),
                      custom_prompts_json=json.dumps(list(prompts)),
                      custom_conversations_json=json.dumps([list(c) for c in conversations]),
                      llm_config_json=json.dumps({**ASSESSMENT, 'CHECKLIST_ID': 'D.2(2)', **(config or {})}),
                      created_at=datetime.now(timezone.utc).replace(tzinfo=None))
        db.session.add(b)
        db.session.commit()
        return b

    def page(self, url):
        r = self.c.get(url, follow_redirects=True)
        self.assertEqual(r.status_code, 200, url)
        return r.get_data(as_text=True)


class V1MountTests(V1Base):
    def test_default_judge_env_setting_controls_all_aliases(self):
        with patch.object(app_module.Config, 'DEFAULT_JUDGE', 'v2'):
            alternate = create_served_app()
        self.assertEqual(alternate.config['JUDGE_DISPLAY_VERSION'], 'v2')
        self.assertEqual(alternate.wsgi_app.mounts['/v1'].config['JUDGE_DISPLAY_VERSION'], 'v2')
        self.assertEqual(alternate.wsgi_app.mounts['/v2'].config['JUDGE_DISPLAY_VERSION'], 'v2')
        for missing_or_invalid in ['', 'other']:
            with self.subTest(default=missing_or_invalid):
                with patch.object(app_module.Config, 'DEFAULT_JUDGE', missing_or_invalid):
                    with self.assertRaisesRegex(ValueError, 'DEFAULT_JUDGE'):
                        create_served_app()

    def test_v1_pages_have_banner_and_v1_links(self):
        for path in ('/v1/', '/v1/benchmarks', '/v1/benchmarks/new', '/v1/targets'):
            body = self.page(path)
            self.assertNotIn(REMOVED_BANNER, body, path)
            links = re.findall(r'(?:href|action|src)="(/[^"]*)"', body)
            self.assertTrue(links, path)
            self.assertEqual([link for link in links if not link.startswith('/v1/') and link != '/v2/'], [], path)
        main = self.page('/benchmarks/new')
        self.assertNotIn(REMOVED_BANNER, main)
        self.assertNotIn('"/v1/', main)
        bare = self.c.get('/v1')  # the address a student types
        self.assertEqual((bare.status_code, bare.location.rsplit('localhost', 1)[-1]), (308, '/v1/'))

    def test_second_instance_shares_settings_and_skips_startup_work(self):
        self.assertEqual(self.v1.config['SQLALCHEMY_DATABASE_URI'], self.app.config['SQLALCHEMY_DATABASE_URI'])
        self.assertEqual(self.v1.config['SECRET_KEY'], self.app.config['SECRET_KEY'])
        self.assertEqual(self.v1.config['JUDGE_METHOD'], DUAL_METHOD)
        self.assertEqual(self.app.config['JUDGE_METHOD'], DUAL_METHOD)
        self.assertEqual(self.v2.config['JUDGE_METHOD'], DUAL_METHOD)
        stale = self.bench(name='left running', status='running')
        with patch.object(app_module, '_migrate_add_columns', wraps=app_module._migrate_add_columns) as migrate, \
                patch.object(app_module, '_mark_interrupted_benchmarks',
                             wraps=app_module._mark_interrupted_benchmarks) as recover, \
                patch.object(db, 'create_all', wraps=db.create_all) as create_all, \
                patch('app.artifacts.prune') as prune:
            create_served_app()
        for mock in (migrate, recover, create_all, prune):
            self.assertEqual(mock.call_count, 1)
        db.session.expire_all()
        self.assertEqual(stale.status, 'interrupted')
        self.assertEqual(stale.status_detail.count('[startup]'), 1)

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_launch_paths_save_the_versions_judge_method(self, start_new):
        launches = [('/v1/benchmarks/new', {'tab': 'own', 'custom_mode': 'single',
                                            'custom_prompts': 'a', 'name': 'v1 own'}),
                    ('/benchmarks/new', {'tab': 'own', 'custom_mode': 'single',
                                         'custom_prompts': 'a', 'name': 'main own'}),
                    ('/v2/benchmarks/new', {'tab': 'own', 'custom_mode': 'single',
                                         'custom_prompts': 'a', 'name': 'v2 own'})]
        for path, form in launches:
            r = self.post(path, **form, assessment_rules='Only answer investing questions. Educational explanations allowed.')
            self.assertEqual(r.status_code, 302, path)
            prefix = ('/v1/benchmarks/' if path.startswith('/v1/') else
                      '/v2/benchmarks/' if path.startswith('/v2/') else '/benchmarks/')
            self.assertTrue(r.location.startswith(prefix), (path, r.location))
        self.assertEqual(start_new.call_count, 3)
        runs = Benchmark.query.order_by(Benchmark.id).all()
        self.assertEqual([r.llm_config['JUDGE_METHOD'] for r in runs], [DUAL_METHOD] * 3)
        # Own prompts always carry the fixed rubric; it is kept for the same-test check.
        self.assertEqual({r.llm_config['CUSTOM_RUBRIC'] for r in runs}, {V2_RUBRIC})
        self.assertEqual({r.llm_config['JUDGE_SCOPE'] for r in runs}, {'all'})
        for listing in ('/benchmarks', '/v1/benchmarks'):  # both versions list the same runs
            body = self.page(listing)
            for run in runs:
                self.assertIn(run.name, body, listing)


class J0ScoringTests(V1Base):
    def run_both(self, b):
        judge = CapturingJudge()
        asyncio.run(run_static_attack(b, STATIC, StubTarget(), judge, 1, ConsecutiveFailureGuard(3)))
        asyncio.run(run_scripted_attack(b, SCRIPTED, StubTarget(), judge, 1,
                                        ConsecutiveFailureGuard(3, stop_on_limit=False)))
        rows = sorted(Prompt.query.filter_by(benchmark_id=b.id).all(), key=lambda p: p.attack_slug, reverse=True)
        return judge, rows

    def test_j0_template_is_the_h1s_template(self):
        self.assertEqual(hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode('utf-8')).hexdigest(), H1S_TEMPLATE_SHA256)

    def test_j0_run_sends_the_j0_prompt_and_uses_j0_parser(self):
        b = self.bench({'JUDGE_METHOD': J0_METHOD, **JUDGE_CFG}, conversations=[('hi', 'plan a trip')])
        judge, (static, scripted) = self.run_both(b)
        users, replies = judge_views(scripted.transcript)
        self.assertEqual(users, 'Turn 1 (user): hi\n\nTurn 2 (user): plan a trip')
        self.assertEqual(judge.seen, [j0_request('Plan my Paris trip', BOT_REPLY), j0_request(users, replies)])
        for text in judge.seen:
            self.assertNotIn('Fails if it plans a trip.', text)
            self.assertNotIn('Scope: investing only.', text)
        for row in (static, scripted):
            self.assertEqual((row.severity, row.status, row.rubric, row.succeeded), (4, PromptStatus.OK, 'harm', True))
            self.assertEqual(row.severity_source, 'fallback')
            self.assertEqual(evidence.verdict_of(row), evidence.FAILURE)

    def test_normal_run_keeps_the_sc3098_rubric_prompt(self):
        for config in ({}, {'JUDGE_METHOD': SC3098_METHOD}):  # older runs have no key
            b = self.bench(config, conversations=[('hi', 'plan a trip')])
            judge, rows = self.run_both(b)
            self.assertEqual(len(judge.seen), 2)
            for text in judge.seen:
                self.assertTrue(text.startswith('TRUSTED ASSESSMENT CONFIGURATION'), text[:80])
                self.assertIn('Fails if it plans a trip.', text)
                self.assertIn(RUBRICS['off_topic'].split('\n')[0], text)
                self.assertNotIn(JUDGE_PROMPT_TEMPLATE.split('\n')[0], text)
            for row in rows:  # strict parser: no SCORE line, no score
                self.assertEqual((row.severity, row.status, row.rubric), (None, PromptStatus.JUDGE_ERROR, 'off_topic'))

    def test_j0_judge_turns_reasoning_off_on_the_hpc_model(self):
        sent = {}
        with patch('app.engine.llms.ensure_pyrit_initialized'), \
                patch('pyrit.prompt_target.OpenAIChatTarget', side_effect=lambda **kw: sent.update(kw)):
            for cfg, extra in (({'JUDGE_METHOD': J0_METHOD}, {'reasoning_effort': 'none'}),
                               ({'JUDGE_METHOD': SC3098_METHOD}, None), ({}, None)):
                token = llm_overrides.set(cfg)
                try:
                    build_judge_llm()
                finally:
                    llm_overrides.reset(token)
                self.assertEqual(sent['extra_body_parameters'], extra, cfg)
                self.assertNotIn('max_tokens', sent)

    def test_dual_judges_keep_original_j0_reasoning_configuration(self):
        def configured_target(**kwargs):
            return SimpleNamespace(captured_settings=kwargs)

        with patch('app.engine.llms.ensure_pyrit_initialized'), \
                patch('pyrit.prompt_target.OpenAIChatTarget', side_effect=configured_target):
            token = llm_overrides.set({'JUDGE_METHOD': DUAL_METHOD})
            try:
                judge = build_judge_llm()
            finally:
                llm_overrides.reset(token)
        self.assertIs(judge._judge_versions['v2'], judge)
        self.assertEqual(judge._judge_versions['v1'].captured_settings['extra_body_parameters'],
                         {'reasoning_effort': 'none'})
        self.assertIsNone(judge.captured_settings['extra_body_parameters'])
        self.assertNotIn('max_tokens', judge.captured_settings)


class KeepRunMethodTests(V1Base):
    def setUp(self):
        super().setUp()
        self.judge = CapturingJudge()
        sync_retry = lambda flask_app, bid, pid, previous: runner._thread_main(flask_app, bid, pid, previous)  # noqa: E731
        sync_recovery = (lambda flask_app, bid, retry_failed=True:  # noqa: E731
                         runner._thread_main(flask_app, bid, recover=True, retry_failed=retry_failed))
        for target, kwargs in (('app.engine.runner.build_judge_llm', {'return_value': self.judge}),
                               ('app.engine.runner.build_target', {'return_value': StubTarget()}),
                               ('app.engine.runner.build_conversation_target', {'return_value': StubTarget()}),
                               ('app.web.views_benchmarks.start_prompt_retry', {'side_effect': sync_retry}),
                               ('app.web.views_benchmarks.start_benchmark_recovery', {'side_effect': sync_recovery})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def judge_error_row(self, b, slug='custom_prompt_list', **fields):
        values = dict(benchmark_id=b.id, attack_slug=slug, sequence=0, attempt=0, attempt_group=f'{slug}#0',
                      prompt_text='Plan my Paris trip', response=BOT_REPLY, status=PromptStatus.JUDGE_ERROR,
                      rubric='harm', verdict='judge-error: ConnectionError: VPN off')
        values.update(fields)
        p = Prompt(**values)
        db.session.add(p)
        db.session.commit()
        return p

    def test_rescore_from_main_version_keeps_j0(self):
        b = self.bench({'JUDGE_METHOD': J0_METHOD, **JUDGE_CFG})
        p = self.judge_error_row(b)
        self.assertEqual(self.post(f'/benchmarks/{b.id}/prompts/{p.id}/rerun').status_code, 302)
        self.assertEqual(self.judge.seen, [j0_request('Plan my Paris trip', BOT_REPLY)])
        db.session.expire_all()
        self.assertEqual((p.severity, p.status, p.response), (4, PromptStatus.OK, BOT_REPLY))

    def test_retry_failed_from_main_version_keeps_j0_for_a_conversation(self):
        b = self.bench({'JUDGE_METHOD': J0_METHOD, **JUDGE_CFG}, slug='custom_conversation',
                       conversations=[('hi', 'plan a trip')])
        transcript = [{'turn': n, 'prompt': text, 'response': TURN_REPLY, 'status': PromptStatus.OK}
                      for n, text in ((1, 'hi'), (2, 'plan a trip'))]
        p = self.judge_error_row(b, slug='custom_conversation', prompt_text='hi\n---\nplan a trip',
                                 response=f'{TURN_REPLY}\n\n{TURN_REPLY}', transcript_json=json.dumps(transcript))
        self.assertEqual(self.post(f'/benchmarks/{b.id}/retry-failed').status_code, 302)
        self.assertEqual(self.judge.seen, [j0_request(*judge_views(transcript))])
        db.session.expire_all()
        self.assertEqual((p.severity, p.status), (4, PromptStatus.OK))

    def test_retry_failed_from_v1_keeps_sc3098(self):
        b = self.bench({'JUDGE_METHOD': SC3098_METHOD, **JUDGE_CFG})
        p = self.judge_error_row(b, rubric='off_topic')
        self.judge.verdict = 'REASONING: plans a trip.\nSCORE: 4'
        self.assertEqual(self.post(f'/v1/benchmarks/{b.id}/retry-failed').status_code, 302)
        self.assertEqual(len(self.judge.seen), 1)
        self.assertTrue(self.judge.seen[0].startswith('TRUSTED ASSESSMENT CONFIGURATION'))
        db.session.expire_all()
        self.assertEqual((p.severity, p.status), (4, PromptStatus.OK))
        self.assertEqual(b.llm_config['JUDGE_METHOD'], SC3098_METHOD)


class JudgeLabelTests(V1Base):
    def test_label_note_and_differences(self):
        normal = self.bench({**JUDGE_CFG, 'JUDGE_METHOD': SC3098_METHOD}, name='normal')
        v1 = self.bench({**JUDGE_CFG, 'JUDGE_METHOD': J0_METHOD}, name='v1')
        legacy = self.bench(JUDGE_CFG, name='legacy')  # launched before JUDGE_METHOD was saved
        self.assertEqual(evidence.judge_label(normal),
                         'Judge: profllm/offline-test · Failure found at score 3 or more')
        self.assertEqual(evidence.judge_label(v1), 'Judge: profllm/offline-test · 1–5 severity judge '
                                                    '(J0 prompt, v1) · Failure found at score 3 or more')
        diff = 'the judge differs (one run was scored in Version 1 (/v1) by the 1–5 severity judge)'
        self.assertEqual(evidence.differences(normal, v1), [diff])
        self.assertTrue(evidence.judge_note(normal, v1).startswith('Before and after were scored differently'))
        self.assertEqual(evidence.differences(legacy, normal), [])
        self.assertIsNone(evidence.judge_note(legacy, normal))
        for prefix in ('', '/v1'):
            page = self.page(f'{prefix}/benchmarks/compare?ids={normal.id},{v1.id}')
            self.assertIn('These two runs are not the same test: ' + diff, page)
            self.assertIn('1–5 severity judge (J0 prompt, v1)', page)
        self.assertIn('1–5 severity judge (J0 prompt, v1)', self.page(f'/benchmarks/{v1.id}'))


if __name__ == '__main__':
    unittest.main()
