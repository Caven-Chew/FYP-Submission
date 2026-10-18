"""Offline route, validation and evidence-export checks. No model calls.
Run from SC3098: ../.venv/bin/python test/test_student_web.py -v
"""
import io
import asyncio
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp=tempfile.TemporaryDirectory(prefix='sc3098-web-')
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['DATABASE_URL']='sqlite:///'+str(Path(_tmp.name)/'test.db')
os.environ.update({'JUDGE_PROVIDER':'profllm','JUDGE_MODEL':'offline-test','LLM_ENDPOINT':'http://127.0.0.1:9/v1',
                   'LLM_API_KEY':'offline-test-key','JUDGE_API_KEY_ENV':'LLM_API_KEY'})
import atexit
_runtime_patch=patch('appdirs.user_data_dir',side_effect=lambda name,*a,**k:str(Path(_tmp.name)/name))
_runtime_patch.start(); atexit.register(_runtime_patch.stop)
from app import create_app
from app.extensions import db
from app.models import Benchmark, Prompt, Target
from app.web.views_benchmarks import parse_student_input
from checklist_packs import checklist_items

class StudentWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('app.artifacts.ARTIFACTS_DIR',Path(_tmp.name)/'artifacts'):
            cls.app=create_app()
        cls.app.config['TESTING']=True
    def setUp(self):
        self.ctx=self.app.app_context(); self.ctx.push()
        db.drop_all(); db.create_all()
        t=Target(name='Demo target', provider='openai',model_name='mock',endpoint_url='http://127.0.0.1:8999/chat')
        db.session.add(t); db.session.commit(); self.tid=t.id
        self.c=self.app.test_client()
        with self.c.session_transaction() as s: s['csrf_token']='test-token'
    def tearDown(self): db.session.remove(); self.ctx.pop()
    def post(self,path,**data):
        return self.c.post(path,data={'csrf_token':'test-token','target_id':str(self.tid),**data},follow_redirects=True)
    def test_browser_advanced_capture_and_save(self):
        """Collapsed native details retain captured settings in the real form submission."""
        from patchright.async_api import async_playwright
        root = Path(__file__).resolve().parents[1]
        selectors = {'ui_input_selector': '#message', 'ui_submit_selector': '#send',
                     'ui_response_selector': '.assistant',
                     'ui_pre_actions': '[{"type":"click","selector":"#start-chat"}]'}
        html = self.c.get('/targets').get_data(as_text=True)
        t = db.session.get(Target, self.tid)
        t.provider = 'ui_browser'
        t.extra_json = json.dumps({'ui_url': 'http://offline.invalid/chat', **selectors})
        db.session.commit()
        duplicate_html = self.c.get(f'/targets/{self.tid}/duplicate').get_data(as_text=True)
        css = (root / 'app/static/styles.css').read_text(encoding='utf-8')
        submissions = []

        async def exercise():
            submitted = asyncio.Event()
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()

                    async def serve(route):
                        path = urlsplit(route.request.url).path
                        if path.endswith('/capture'):
                            await route.fulfill(json={'ok': True, 'selectors': selectors})
                        elif route.request.method == 'POST':
                            submissions.append(parse_qs(route.request.post_data))
                            await route.fulfill(status=204)
                            submitted.set()
                        elif path.endswith('/styles.css'):
                            await route.fulfill(body=css, content_type='text/css')
                        else:
                            await route.fulfill(body=duplicate_html if path.endswith('/duplicate') else html,
                                                content_type='text/html')

                    await page.route('**/*', serve)
                    await page.goto('http://offline.invalid/targets')
                    await page.locator('#targetForm').evaluate('form => form.parentElement.open = true')
                    await page.locator('[name="name"]').fill('Captured browser target')
                    await page.locator('[name="extra_config_ui_url"]').fill('http://offline.invalid/chat')
                    advanced = page.locator('#uiAdvanced')
                    self.assertFalse(await advanced.evaluate('e => e.open'))
                    self.assertTrue(await page.locator('[name="extra_config_ui_url"]').is_visible())
                    for key in selectors:
                        self.assertFalse(await page.locator(f'[name="extra_config_{key}"]').is_visible())
                    await page.locator('#captureBtn').click()
                    await page.wait_for_function("document.getElementById('captureResult').textContent.includes('Filled browser settings')")
                    self.assertFalse(await advanced.evaluate('e => e.open'))
                    for key, value in selectors.items():
                        field = page.locator(f'[name="extra_config_{key}"]')
                        self.assertEqual(await field.input_value(), value)
                        self.assertFalse(await field.is_visible())
                    await page.get_by_role('button', name='Create target', exact=True).click()
                    await asyncio.wait_for(submitted.wait(), timeout=5)
                    self.assertEqual(len(submissions), 1)
                    for key, value in selectors.items():
                        self.assertEqual(submissions[0][f'extra_config_{key}'], [value])
                    await advanced.locator(':scope > summary').click()
                    for key in selectors:
                        self.assertTrue(await page.locator(f'[name="extra_config_{key}"]').is_visible())
                    await page.locator('[name="extra_config_ui_response_selector"]').fill('.corrected-assistant')
                    submitted.clear()
                    await page.get_by_role('button', name='Create target', exact=True).click()
                    await asyncio.wait_for(submitted.wait(), timeout=5)
                    self.assertEqual(len(submissions), 2)
                    self.assertEqual(submissions[1]['extra_config_ui_response_selector'], ['.corrected-assistant'])
                    await page.goto(f'http://offline.invalid/targets/{self.tid}/duplicate')
                    self.assertFalse(await page.locator('#uiAdvanced').evaluate('e => e.open'))
                    for key, value in selectors.items():
                        field = page.locator(f'[name="extra_config_{key}"]')
                        self.assertEqual(await field.input_value(), value)
                        self.assertFalse(await field.is_visible())
                finally:
                    await browser.close()

        asyncio.run(exercise())
        # The backend reads assembled extra_config, not the individual browser fields.
        for index, submitted in enumerate(submissions):
            payload = {key: values[0] for key, values in submitted.items()}
            payload['name'] = f'Captured browser saved {index}'
            response = self.c.post('/targets', data=payload)
            self.assertEqual(response.status_code, 302)
            saved = Target.query.filter_by(name=payload['name']).one()
            config = json.loads(saved.extra_json)
            for key, value in selectors.items():
                expected = '.corrected-assistant' if index == 1 and key == 'ui_response_selector' else value
                self.assertEqual(config[key], expected)

    def test_modes(self):
        self.assertEqual(parse_student_input('a\nb','single'),(['a\nb'],[]))
        self.assertEqual(parse_student_input('a\n\nb','list'),(['a','b'],[]))
        self.assertEqual(parse_student_input('a\n---\nb\n===\nc\n---\nd','conversation'),([],[['a','b'],['c','d']]))
        self.assertEqual(parse_student_input('a\nb','conversation'), ([], [['a\nb']]))
        self.assertEqual(parse_student_input('a\n===\nb','conversation'), ([], [['a'], ['b']]))
        for text,mode in [('', 'single'),('\n===\n','conversation'),('a','invalid')]:
            with self.assertRaises(ValueError): parse_student_input(text,mode)
    @patch('app.web.views_benchmarks.start_benchmark')
    def test_launch_three_modes(self,start):
        for mode,text in [('single','a\nb'),('list','a\nb'),('conversation','a\n---\nb\n---\nc')]:
            response=self.post('/benchmarks/new',custom_mode=mode,custom_prompts=text)
            self.assertEqual(response.status_code,200)
        self.assertEqual(start.call_count,3)
        self.assertEqual(Benchmark.query.order_by(Benchmark.id.desc()).first().custom_conversations,[['a','b','c']])
    @patch('app.web.views_benchmarks.start_benchmark')
    def test_instruction_assessments_persist_in_all_own_modes(self, start):
        from app.engine.scoring import OWN_PROMPT_ASSESSMENTS
        from types import SimpleNamespace
        rules = 'Allow investment education; prohibit stock picks. POLICY-ONLY-82'
        for rubric, (_, required) in OWN_PROMPT_ASSESSMENTS.items():
            for mode in ('single', 'list', 'conversation', 'interactive'):
                with self.subTest(rubric=rubric, mode=mode):
                    data = dict(tab='own', custom_mode=mode, assessment=rubric,
                                assessment_rules=rules, start_id=f'{rubric}-{mode}',
                                custom_prompts='first\n---\nsecond' if mode == 'conversation' else 'first\nsecond')
                    with patch('app.engine.interactive.start_session', return_value=SimpleNamespace(id=999)) as live:
                        response = self.c.post('/benchmarks/new', data={
                            'csrf_token':'test-token', 'target_id':str(self.tid), **data})
                    self.assertEqual(response.status_code, 302, response.data)
                    cfg = live.call_args.args[3] if mode == 'interactive' else Benchmark.query.order_by(Benchmark.id.desc()).first().llm_config
                    self.assertEqual(cfg['CUSTOM_RUBRIC'], rubric)
                    self.assertEqual(cfg['ASSESSMENT_RULES'], rules if required else '')

    def test_browser_assessment_visibility_required_tabs_and_escape(self):
        from patchright.async_api import async_playwright
        malicious = '</textarea><script>window.policyExecuted = true</script><b>policy</b>'
        shown = self.app.config.get('SHOW_PUBLIC_BASELINES')
        previous = self.app.config.get('JUDGE_METHOD')
        self.app.config['SHOW_PUBLIC_BASELINES'] = True
        try:
            response = self.post('/benchmarks/new', tab='own', custom_mode='single',
                                 custom_prompts='test', target_id='999', assessment='off_topic',
                                 assessment_rules=malicious)
            self.assertEqual(response.status_code, 400)
            html = response.get_data(as_text=True)
            self.assertIn('&lt;/textarea&gt;', html)
            self.assertNotIn('<script>window.policyExecuted', html)
            self.app.config['JUDGE_METHOD'] = 'j0_severity'
            v1_html = self.c.get('/benchmarks/new').get_data(as_text=True)
        finally:
            self.app.config['SHOW_PUBLIC_BASELINES'] = shown
            self.app.config['JUDGE_METHOD'] = previous

        async def exercise():
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()
                    async def serve(route):
                        await route.fulfill(body=v1_html if '/v1/' in route.request.url else html,
                                            content_type='text/html')
                    await page.route('**/*', serve)
                    await page.goto('http://offline.invalid/benchmarks/new')
                    prompts = page.locator('#custom-text')
                    await page.locator('#custom-mode').select_option('conversation')
                    example = await prompts.get_attribute('placeholder')
                    parsed = parse_student_input(example, 'conversation')
                    self.assertEqual([len(turns) for turns in parsed[1]], [2, 2])
                    self.assertIn('\n---\n', example)
                    self.assertIn('\n===\n', example)
                    self.assertEqual(await prompts.input_value(), 'test')
                    self.assertIn('with 1 turn(s)', await page.locator('#detected').inner_text())
                    self.assertEqual(await page.locator('#custom-mode option').evaluate_all('(options) => options.map(option => option.value)'), ['interactive', 'conversation'])
                    await page.locator('#custom-mode').select_option('interactive')
                    self.assertFalse(await prompts.is_visible())
                    self.assertEqual(await prompts.input_value(), 'test')
                    field = page.locator('#assessment-rules')
                    self.assertEqual(await field.input_value(), malicious)
                    self.assertFalse(await page.evaluate('Boolean(window.policyExecuted)'))
                    for rubric in ('off_topic', 'over_refusal', 'injection', 'rule_override'):
                        await page.locator('#assessment').select_option(rubric)
                        self.assertTrue(await field.is_visible())
                        self.assertTrue(await field.evaluate('e => e.required'))
                        await page.locator('#tab-public').click()
                        self.assertFalse(await field.evaluate('e => e.required'))
                        await page.locator('#tab-own').click()
                        self.assertTrue(await field.is_visible())
                        self.assertTrue(await field.evaluate('e => e.required'))
                    for rubric in ('harm_disclosure', 'toxic'):
                        await page.locator('#assessment').select_option(rubric)
                        self.assertFalse(await field.is_visible())
                        self.assertFalse(await field.evaluate('e => e.required'))
                    await page.goto('http://offline.invalid/v1/benchmarks/new')
                    self.assertEqual(await page.locator('#assessment').count(), 0)
                    self.assertEqual(await page.locator('#assessment-rules').count(), 0)
                finally:
                    await browser.close()
        asyncio.run(exercise())

    def test_batch_pdf_exports_assessment_and_full_length_policy(self):
        from pypdf import PdfReader
        rules = 'POLICY-START-95 ' + ('Allow education and prohibit stock picks. ' * 200)
        rules = rules[:7980].ljust(7980, ' ') + ' POLICY-END-95'
        rules = rules.ljust(8000, ' ')
        self.assertEqual(len(rules), 8000)
        b = Benchmark(name='Long saved policy', target_id=self.tid, status='completed',
                      attack_slugs_json='["custom_prompt_list"]',
                      llm_config_json=json.dumps({'CUSTOM_RUBRIC': 'off_topic', 'ASSESSMENT_RULES': rules}))
        db.session.add(b)
        db.session.commit()
        response = self.c.get(f'/benchmarks/{b.id}/pdf')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        reader = PdfReader(io.BytesIO(response.data))
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        self.assertIn('Assessment', text)
        self.assertIn('off_topic', text)
        self.assertIn('Saved target rules', text)
        self.assertIn('POLICY-START-95', text)
        self.assertIn('POLICY-END-95', text)
        self.assertGreater(len(reader.pages), 1)

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_invalid_instruction_assessments_never_start_target(self, start):
        for mode in ('single', 'list', 'conversation', 'interactive'):
            for assessment, rules in [('off_topic', '  '), ('over_refusal', ''),
                                      ('injection', ''), ('rule_override', ''),
                                      ('off_topic', 'x' * 8001), ('leak', 'private instructions'),
                                      ('hallucination', 'facts'), ('not-a-rubric', '')]:
                with self.subTest(mode=mode, assessment=assessment, length=len(rules)):
                    with patch('app.engine.interactive.start_session') as live:
                        response = self.post('/benchmarks/new', tab='own', custom_mode=mode,
                            custom_prompts='first\n---\nsecond' if mode == 'conversation' else 'first',
                            assessment=assessment, assessment_rules=rules, start_id='rejected')
                    self.assertEqual(response.status_code, 400, response.data)
                    live.assert_not_called()
        self.assertEqual(Benchmark.query.count(), 0)
        start.assert_not_called()

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_v1_rejects_other_assessments_in_all_modes(self, start):
        previous = self.app.config.get('JUDGE_METHOD')
        self.app.config['JUDGE_METHOD'] = 'j0_severity'
        try:
            for mode in ('single', 'list', 'conversation', 'interactive'):
                with patch('app.engine.interactive.start_session') as live:
                    response = self.post('/benchmarks/new', tab='own', custom_mode=mode,
                        custom_prompts='first\n---\nsecond' if mode == 'conversation' else 'first',
                        assessment='toxic', assessment_rules='', start_id='v1-rejected')
                self.assertEqual(response.status_code, 400)
                self.assertIn(b'original harm-severity judge', response.data)
                live.assert_not_called()
            self.post('/benchmarks/new', tab='own', custom_mode='single', custom_prompts='first')
            cfg = Benchmark.query.one().llm_config
            self.assertEqual(cfg['JUDGE_METHOD'], 'j0_severity')
            self.assertEqual(cfg['CUSTOM_RUBRIC'], 'harm_disclosure')
        finally:
            self.app.config['JUDGE_METHOD'] = previous
    @patch('app.web.views_benchmarks.start_benchmark')
    def test_invalid_input_and_legacy_block(self,start):
        for data in [dict(custom_mode='conversation',custom_prompts='\n---\n'.join(['one']*21)),dict(custom_prompts='safe'),dict(custom_mode='single',custom_prompts='safe',target_id='999'),dict(attack_slugs='harmbench',prompts_per_attack='0')]:
            r=self.post('/benchmarks/new',**data)
            self.assertEqual(r.status_code,400)
        self.assertEqual(Benchmark.query.count(),0); start.assert_not_called()
        # Kept input on error: the pasted text comes back in the form.
        r=self.post('/benchmarks/new',custom_prompts='keep me')
        self.assertIn(b'keep me',r.data)
        # Baseline datasets are restored (user decision, 1 Oct 2026), unticked by default.
        # The course build shows them only with SHOW_PUBLIC_BASELINES on (finding #48).
        shown=self.app.config.get('SHOW_PUBLIC_BASELINES'); self.app.config['SHOW_PUBLIC_BASELINES']=True
        try:
            page=self.c.get('/benchmarks/new').data
            self.assertIn(b'value="harmbench"',page); self.assertNotIn(b'value="harmbench" checked',page)
            self.post('/benchmarks/new',attack_slugs='harmbench')
        finally: self.app.config['SHOW_PUBLIC_BASELINES']=shown
        self.assertEqual(Benchmark.query.one().attack_slugs,['harmbench']); start.assert_called_once()
        r=self.c.post('/benchmarks/new',data={'custom_prompts':'safe'}); self.assertEqual(r.status_code,403)
    @patch('app.web.views_benchmarks.start_benchmark')
    def test_upload_validation(self,start):
        self.post('/benchmarks/new',custom_prompt_file=(io.BytesIO(b'\xff'),'test.txt'))
        self.post('/benchmarks/new',custom_prompt_file=(io.BytesIO(b'a'),'test.csv'))
        self.post('/benchmarks/new',custom_prompts='a',custom_prompt_file=(io.BytesIO(b'b'),'test.txt'))
        self.assertEqual(Benchmark.query.count(),0)
        self.post('/benchmarks/new',custom_mode='list',custom_prompt_file=(io.BytesIO(b'a\nb'),'test.txt'))
        self.assertEqual(Benchmark.query.one().custom_prompts,['a','b'])
    def test_full_csv_zero_values_reasoning_and_unmeasured(self):
        b=Benchmark(name='Long evidence',target_id=self.tid,total_prompts=0,total_successes=0)
        db.session.add(b);db.session.flush()
        p=Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=0,attempt=0,
                 prompt_text='=formula',response='x'*800+'END',status='judge_error',rubric='pii',
                 canary_detected=False,verdict='UNIQUE JUDGE REASON',transcript_json='[]')
        db.session.add(p);db.session.commit()
        import csv
        response=self.c.get(f'/benchmarks/{b.id}/csv')
        row=list(csv.DictReader(io.StringIO(response.data.decode())))[0]
        self.assertEqual(row['seq'],'0');self.assertEqual(row['counts_toward_stats'],'0')
        self.assertEqual(row['succeeded'],'0');self.assertEqual(row['response'],'x'*800+'END')
        self.assertEqual(row['prompt'],"'=formula");self.assertEqual(row['rubric'],'pii')
        self.assertEqual(row['canary_detected'],'False');self.assertEqual(row['transcript_json'],'[]')
        page=self.c.get(f'/benchmarks/{b.id}').data
        # Task 11.41: the FYP run page shows the saved judge text and the captured reply.
        for value in (b'UNIQUE JUDGE REASON',b'Judge explanation',b'END'):
            self.assertIn(value,page)


    @patch('app.web.views_benchmarks.start_benchmark')
    def test_one_turn_scripts_accept_generic_http_without_history(self, start):
        t = db.session.get(Target, self.tid)
        t.provider = 'http_generic'
        t.extra_json = json.dumps({'body_template': '{"input": "{prompt}"}', 'response_path': 'output'})
        db.session.commit()
        response = self.post('/benchmarks/new', tab='own', custom_mode='conversation', custom_prompts='first\n===\nsecond')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Benchmark.query.one().custom_conversations, [['first'], ['second']])
        start.assert_called_once()

    @patch('app.web.views_benchmarks.start_benchmark')
    def test_generic_http_conversation_rejected_before_queue(self,start):
        t=db.session.get(Target,self.tid); t.provider='http_generic'
        t.extra_json=json.dumps({'body_template':'{"input": "{prompt}"}','response_path':'output'});db.session.commit()
        response=self.post('/benchmarks/new',custom_mode='conversation',custom_prompts='a\n---\nb')
        self.assertIn(b'cannot carry earlier turns',response.data)
        start.assert_not_called();self.assertEqual(Benchmark.query.count(),0)
        t.extra_json=json.dumps({'body_template':'{"message": "{prompt}", "history": {history}}','response_path':'reply'});db.session.commit()
        self.post('/benchmarks/new',custom_mode='conversation',custom_prompts='a\n---\nb')
        self.assertEqual(Benchmark.query.count(),1); start.assert_called_once()

    @patch('app.web.views_benchmarks.render_benchmark_pdf')
    def test_unmeasured_list_compare_and_student_pdf_guard(self, render_pdf):
        runs=[]
        for status,slug in [('judge_error','custom_prompt_list'),('incomplete','custom_conversation')]:
            b=Benchmark(name='Unmeasured '+status,target_id=self.tid,total_prompts=0,total_successes=0,
                        status='failed',attack_slugs_json=json.dumps([slug]))
            db.session.add(b);db.session.flush()
            db.session.add(Prompt(benchmark_id=b.id,attack_slug=slug,prompt_text='test',
                                  response='partial evidence',status=status,rubric='pii'))
            runs.append(b)
        db.session.commit()
        dashboard=self.c.get('/').data
        self.assertEqual(dashboard.count(b'unmeasured'),3)
        self.assertNotIn(b'(0.0%)',dashboard)
        listing=self.c.get('/benchmarks').data
        self.assertEqual(listing.count(b'unmeasured'),2)
        comparison=self.c.get('/benchmarks/compare?ids='+','.join(str(b.id) for b in runs)).data
        self.assertEqual(comparison.count(b'unmeasured'),4)
        self.assertNotIn(b'0/0 (0.0%)',comparison)
        self.assertNotIn('0.0–0.0'.encode(),comparison)
        render_pdf.return_value=b'%PDF-1.4 test stub'
        for b in runs:  # PDF restored for student runs (decision S27)
            self.assertEqual(self.c.get(f'/benchmarks/{b.id}/pdf').status_code,200)
            self.assertIn(b'Download PDF',self.c.get(f'/benchmarks/{b.id}').data)
        render_pdf.reset_mock()
        # Legacy dataset reports remain available for old benchmark records.
        historic=Benchmark(name='Historical dataset',target_id=self.tid,attack_slugs_json='["harmbench"]')
        db.session.add(historic);db.session.commit()
        render_pdf.return_value=b'%PDF-1.4 test stub'
        self.assertEqual(self.c.get(f'/benchmarks/{historic.id}/pdf').status_code,200)
        render_pdf.assert_called_once()


    def test_old_benchmark_renders(self):
        b=Benchmark(name='Old run',target_id=self.tid);db.session.add(b);db.session.commit()
        self.assertEqual(self.c.get(f'/benchmarks/{b.id}').status_code,200)

    def checklist_run(self, name, item_id='D.2(2)', **kwargs):
        b=Benchmark(name=name,target_id=self.tid,status='completed',
                    attack_slugs_json='["custom_prompt_list"]',custom_prompts_json='["Explain fees."]',
                    llm_config_json=json.dumps({'CHECKLIST_ID':item_id}),**kwargs)
        db.session.add(b);db.session.flush()
        return b

    def checklist_card(self, page, item_id):
        # Task 11.36: the Checklist D attacks are <details> cards at the bottom of + Benchmark.
        start=page.index(f'<details class="card checklist-item" id="{item_id}"')
        end=page.find('<details class="card checklist-item"',start+1)
        return page[start:end if end!=-1 else page.index('</section>',start)]

    def test_old_checklist_link_opens_the_new_run_page(self):
        # Task 11.41: the Checklist D attacks left the app; the example prompts are in the slides
        # and the user guide. Old links still land somewhere useful.
        with patch('app.engine.llms.judge_setup_problem') as judge, \
             patch('app.targets.registry.build_conversation_target') as build:
            old=self.c.get('/checklist')
            self.assertEqual(old.status_code,302)
            self.assertTrue(old.headers['Location'].endswith('/benchmarks/new'))
            page=self.c.get('/benchmarks/new')
        self.assertEqual(page.status_code,200)
        self.assertNotIn('checklist-item',page.text)
        self.assertNotIn('action="/checklist/',page.text)
        judge.assert_not_called();build.assert_not_called()
        db.session.delete(db.session.get(Target,self.tid));db.session.commit()
        page=self.c.get('/checklist',follow_redirects=True)
        self.assertEqual(page.status_code,200)
        self.assertIn('Add a target first',page.text)





if __name__=='__main__': unittest.main()
