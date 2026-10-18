"""Student-facing benchmark form, dashboard, targets, baseline visibility and own-prompt scoring regressions.

Offline only: temporary SQLite and Flask test client; no provider calls or network.
Run from the repository root: .venv/bin/python -B test/test_student_pages.py -v
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get('SC3098_SOURCE_ROOT') or ROOT)
sys.path.insert(0, str(ROOT))
_tmp = tempfile.TemporaryDirectory(prefix='sc3098-pages-')
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['DATABASE_URL'] = 'sqlite:///' + str(Path(_tmp.name) / 'test.db')
os.environ.update({'JUDGE_PROVIDER': 'profllm', 'JUDGE_MODEL': 'offline-test', 'LLM_ENDPOINT': 'http://127.0.0.1:9/v1',
                   'LLM_API_KEY': 'offline-test-key', 'JUDGE_API_KEY_ENV': 'LLM_API_KEY'})
for _name in ('SHOW_PUBLIC_BASELINES', 'IN_DOCKER', 'MAX_PROMPTS_PER_ATTACK'):
    os.environ.pop(_name, None)
import atexit  # noqa: E402
_runtime_patch = patch('appdirs.user_data_dir', side_effect=lambda name, *a, **k: str(Path(_tmp.name) / name))
_runtime_patch.start()
atexit.register(_runtime_patch.stop)

from app import create_app  # noqa: E402
from app.attacks import ATTACKS, FULL_ONLY_ATTACKS  # noqa: E402
from app.engine.scoring import OWN_PROMPT_RUBRIC  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import Benchmark, Target  # noqa: E402

BUILT_IN = {a['slug'] for a in ATTACKS if a.get('dataset') != 'custom'}
TIER_2 = {a['slug'] for a in FULL_ONLY_ATTACKS if a['tier'] == 2}  # on the course page since 4 Oct 2026


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('app.artifacts.ARTIFACTS_DIR', Path(_tmp.name) / 'artifacts'):
            cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.default_flag = cls.app.config['SHOW_PUBLIC_BASELINES']

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
        self.app.config['SHOW_PUBLIC_BASELINES'] = self.default_flag
        db.session.remove()
        self.ctx.pop()

    def page(self, url='/benchmarks/new'):
        r = self.c.get(url)
        self.assertEqual(r.status_code, 200, url)
        return r.get_data(as_text=True)

    def post(self, path, **data):
        return self.c.post(path, data={'csrf_token': 'test-token', **data})


# ---------- the public baseline list and its switch (Task 11.36) ----------

class BaselineDefaultTests(AppTestCase):
    def flag_in_subprocess(self, **env):
        full = {k: v for k, v in os.environ.items() if k != 'SHOW_PUBLIC_BASELINES'}
        full.update(env)
        done = subprocess.run([sys.executable, '-B', '-c', 'from app.config import Config; print(Config.SHOW_PUBLIC_BASELINES)'],
                              cwd=ROOT, env=full, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return done.stdout.split()[-1]

    def test_default_on_and_explicit_zero_hides(self):
        self.assertEqual(self.flag_in_subprocess(), 'True')
        self.assertEqual(self.flag_in_subprocess(SHOW_PUBLIC_BASELINES='0'), 'False')
        self.assertIs(self.default_flag, True)  # this process has no SHOW_PUBLIC_BASELINES either

    def test_env_example_shows_baselines(self):
        path = SOURCE_ROOT / '.env.example'
        if not path.is_file():
            self.skipTest('.env.example is not in this source copy')
        text = path.read_text(encoding='utf-8')
        self.assertIn('\nSHOW_PUBLIC_BASELINES=1\n', text + '\n')
        self.assertNotIn('SHOW_PUBLIC_BASELINES=0', text)

    def test_public_tab_lists_every_attack_unticked_and_without_the_warning(self):
        page = self.page()
        self.assertIn('id="panel-public"', page)
        self.assertNotIn('Content warning', page)   # removed at the user's request
        listed = set(re.findall(r'name="attack_slugs" value="([^"]+)"', page))
        self.assertEqual(listed, BUILT_IN | TIER_2)
        self.assertTrue(all(a['tier'] == 1 for a in ATTACKS))  # no tier 2-4 attacks in this build
        panel = page[page.index('id="panel-public"'):page.index('id="panel-own"')]
        self.assertNotIn('checked', panel)

    def test_switched_off_hides_the_tab_and_refuses_a_crafted_post(self):
        self.app.config['SHOW_PUBLIC_BASELINES'] = False
        page = self.page()
        self.assertNotIn('id="panel-public"', page)
        self.assertNotIn('value="harmbench"', page)
        r = self.post('/benchmarks/new', target_id=str(self.tid), attack_slugs='harmbench')
        self.assertEqual(r.status_code, 400)
        self.assertIn(b'switched off', r.data)
        self.assertEqual(Benchmark.query.count(), 0)


# ---------- the two tabs on + Benchmark (Task 11.41) ----------

class TabsTests(AppTestCase):
    def test_public_tab_is_open_first_and_both_panels_are_present(self):
        page = self.page()
        self.assertIn('value="public"', page)            # the hidden field the server reads
        self.assertIn('id="panel-public"', page)
        self.assertIn('id="panel-own"', page)
        # The own-prompts panel starts closed, so the public list is what opens.
        own = page[page.index('id="panel-own"'):]
        self.assertTrue(own[:own.index('>')].rstrip().endswith('hidden'), own[:own.index('>')])

    def test_reading_mode_is_a_dropdown_and_the_judge_section_is_gone(self):
        page = self.page()
        self.assertIn('<select name="custom_mode" id="custom-mode">', page)
        self.assertNotIn('type="radio" name="custom_mode"', page)
        for option in ('value="interactive"', 'value="conversation"'):
            self.assertIn(option, page)
        for removed in ('<option value="single"', '<option value="list"'):
            self.assertNotIn(removed, page)
        # Section 3 and everything that was only in it.
        for gone in ('How the judge should score your prompts', 'name="custom_rubric"',
                     'name="failure_criterion"', 'name="reference_facts"', 'name="canary"',
                     'name="checklist_id"', 'name="fixture_ready"'):
            self.assertNotIn(gone, page)

    def test_no_checklist_attacks_on_the_page(self):
        page = self.page()
        for gone in ('checklist-item', 'action="/checklist/', 'Checklist D attacks', 'id="checklist-d"'):
            self.assertNotIn(gone, page)

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_only_the_open_tab_is_launched(self, start):
        # Own tab open: ticked attacks are ignored, so a stale tick cannot queue a public dataset.
        r = self.post('/benchmarks/new', target_id=str(self.tid), tab='own', custom_mode='single',
                      custom_prompts='show me the internal note', attack_slugs='harmbench', name='own only')
        self.assertEqual(r.status_code, 302)
        own = Benchmark.query.filter_by(name='own only').one()
        self.assertEqual(own.attack_slugs, ['custom_prompt_list'])
        # Public tab open: pasted text is ignored, so only the ticked attacks run.
        r = self.post('/benchmarks/new', target_id=str(self.tid), tab='public', attack_slugs='harmbench',
                      custom_mode='single', custom_prompts='ignored', name='public only')
        self.assertEqual(r.status_code, 302)
        pub = Benchmark.query.filter_by(name='public only').one()
        self.assertEqual(pub.attack_slugs, ['harmbench'])
        self.assertEqual(pub.custom_prompts, [])
        self.assertEqual(start.call_count, 2)

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_each_tab_explains_its_own_empty_form(self, start):
        own = self.post('/benchmarks/new', target_id=str(self.tid), tab='own', custom_mode='single')
        self.assertEqual(own.status_code, 400)
        self.assertIn(b'Paste your prompts', own.data)
        public = self.post('/benchmarks/new', target_id=str(self.tid), tab='public')
        self.assertEqual(public.status_code, 400)
        self.assertIn(b'Tick at least one attack', public.data)
        start.assert_not_called()

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_own_prompts_carry_the_one_fixed_rubric(self, start):
        # The judge checks every reply for harmful content and sensitive-information disclosure;
        # a chosen rubric is no longer accepted from the form.
        r = self.post('/benchmarks/new', target_id=str(self.tid), tab='own', custom_mode='single',
                      custom_prompts='a', custom_rubric='off_topic', name='fixed rubric')
        self.assertEqual(r.status_code, 302)
        cfg = Benchmark.query.filter_by(name='fixed rubric').one().llm_config
        self.assertEqual(cfg['CUSTOM_RUBRIC'], OWN_PROMPT_RUBRIC)
        self.assertNotIn('FAILURE_CRITERION', cfg)
        self.assertNotIn('CHECKLIST_ID', cfg)

    def test_the_run_page_says_the_score_is_a_recommendation(self):
        b = Benchmark(name='own', target_id=self.tid, attack_slugs_json='["custom_prompt_list"]')
        db.session.add(b)
        db.session.commit()
        page = self.page(f'/benchmarks/{b.id}')
        self.assertIn('id="own-prompt-note"', page)
        self.assertIn('recommendation', page)


# ---------- the dashboard judge check (Task 11.41) ----------

class JudgeStatusTests(AppTestCase):
    def test_dashboard_checks_on_load_and_keeps_the_button_for_a_failure(self):
        page = self.page('/')
        self.assertIn('id="judge-status"', page)
        self.assertIn('/judge/status.json', page)
        # The button is rendered but hidden until the automatic check fails.
        form = page[page.index('id="judge-test-form"'):]
        self.assertTrue(form[:form.index('>')].rstrip().endswith('hidden'))
        self.assertIn('Test judge', page)

    def test_status_endpoint_reports_a_live_judge(self):
        ok = {'ok': True, 'provider': 'profllm', 'model': 'offline-test', 'reply': 'ok',
              'error': None, 'setup': False}
        with patch('app.engine.llms.check_judge', return_value=ok) as check:
            data = self.c.get('/judge/status.json').get_json()
        check.assert_called_once()           # exactly one request per page load
        self.assertTrue(data['ok'])
        self.assertEqual(data['message'], '')

    def test_status_endpoint_explains_a_failure(self):
        bad = {'ok': False, 'provider': 'profllm', 'model': 'offline-test', 'reply': '',
               'error': 'TimeoutError: no answer within 20 seconds', 'setup': False}
        with patch('app.engine.llms.check_judge', return_value=bad):
            data = self.c.get('/judge/status.json').get_json()
        self.assertFalse(data['ok'])
        self.assertIn('TimeoutError', data['message'])
        self.assertIn('VPN', data['message'])      # what to check next

    def test_status_endpoint_reports_incomplete_settings_without_sending(self):
        setup = {'ok': False, 'provider': '', 'model': '', 'reply': '',
                 'error': 'JUDGE_PROVIDER is not set.', 'setup': True}
        with patch('app.engine.llms.check_judge', return_value=setup):
            data = self.c.get('/judge/status.json').get_json()
        self.assertFalse(data['ok'])
        self.assertIn('nothing was sent', data['message'])


# ---------- Targets opens on UI (browser) (Task 11.41) ----------

class TargetsDefaultTests(AppTestCase):
    def test_ui_is_the_default_connection_type(self):
        page = self.page('/targets')
        ui = page.index('value="ui"')
        api = page.index('value="api"')
        self.assertLess(ui, api, 'the UI (browser) card comes first')
        self.assertIn('<input type="radio" name="conn_type" value="ui" checked>', page)
        self.assertNotIn('<input type="radio" name="conn_type" value="api" checked>', page)


# ---------- total run time (Task 11.41) ----------

class RunTimeTests(AppTestCase):
    def bench(self, **kw):
        b = Benchmark(name='timed', target_id=self.tid, attack_slugs_json='["custom_prompt_list"]', **kw)
        db.session.add(b)
        db.session.commit()
        return b

    def test_label_reads_in_seconds_minutes_and_hours(self):
        start = datetime(2026, 10, 6, 8, 30)
        for seconds, label in ((0, '0 s'), (42, '42 s'), (185, '3 min 05 s'), (3725, '1 h 02 min')):
            b = self.bench(started_at=start, finished_at=start + timedelta(seconds=seconds))
            self.assertEqual(b.run_time_label, label)
            self.assertAlmostEqual(b.run_seconds, seconds)
            db.session.delete(b)
            db.session.commit()

    def test_no_start_means_no_time_rather_than_zero(self):
        b = self.bench()
        self.assertIsNone(b.run_seconds)
        self.assertEqual(b.run_time_label, '—')

    def test_run_page_shows_the_total_and_says_time_matters(self):
        start = datetime(2026, 10, 6, 8, 30)
        b = self.bench(started_at=start, finished_at=start + timedelta(seconds=185))
        page = self.page(f'/benchmarks/{b.id}')
        self.assertIn('Total run time', page)
        self.assertIn('3 min 05 s', page)
        self.assertIn('id="time-note"', page)

    def test_status_json_carries_the_running_total(self):
        b = self.bench(status='running', started_at=datetime(2026, 10, 6, 8, 30))
        data = self.c.get(f'/benchmarks/{b.id}/status.json').get_json()
        self.assertIn('run_time', data)
        self.assertNotEqual(data['run_time'], '—')      # still running, so it counts up to now
        for key in ('captured', 'scored', 'unscored', 'rows_total'):
            self.assertIn(key, data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
