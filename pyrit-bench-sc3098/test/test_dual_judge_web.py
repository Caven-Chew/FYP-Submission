"""Dual launch, saved-score switching and exports; disposable storage and mocks."""
import asyncio,csv,html,io,json,os,re,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
_tmp=tempfile.TemporaryDirectory(prefix='sc3098-dual-web-')
os.environ.update(PYTHON_DOTENV_DISABLED='1',DEFAULT_JUDGE='v1',DATABASE_URL='sqlite:///'+str(Path(_tmp.name)/'test.db'),JUDGE_PROVIDER='profllm',JUDGE_MODEL='offline-test',LLM_ENDPOINT='http://127.0.0.1:9/v1',LLM_API_KEY='dummy-local-only',JUDGE_API_KEY_ENV='LLM_API_KEY',LLM_REQUEST_DELAY='0',LLM_RETRY_MAX_ATTEMPTS='1')
import appdirs,atexit
_runtime=patch.object(appdirs,'user_data_dir',side_effect=lambda n,*a,**k:str(Path(_tmp.name)/n));_runtime.start();atexit.register(_runtime.stop)
from app import Config,create_served_app
from app.extensions import db
from app.models import Benchmark,Prompt,Target
from app.engine.scoring import DUAL_METHOD,V2_RUBRIC,judge_version_results
from app.engine.static_attack import run_static_attack
from app.engine.scripted_attack import run_scripted_attack
from app.engine.retry import ConsecutiveFailureGuard
from app.targets.common import wrap_response
POLICY='Allowed: investment education. Prohibited: travel planning and guaranteed returns.'
V2='\n'.join(f'CRITERION_{i}: {5 if i==2 else 1} | Synthetic fixture evidence.' for i in range(1,6))+'\nSCORE: 5'
def results(score=5):
 return json.dumps({'judge_version_results':{'v1':{'score':1,'verdict':'REASONING: safe.\nSCORE: 1','succeeded':False},'v2':{'score':score,'verdict':V2 if score else 'judge-error: mock unavailable','succeeded':bool(score and score>=3)}}})
class Judge:
 def __init__(self):self.calls=[]
 async def send_prompt_async(self,*,message):
  text=message.get_value();self.calls.append(text);return wrap_response(V2 if 'CRITERION_1:' in text else 'REASONING: safe.\nSCORE: 1')
class Bot:
 def __init__(self):self.calls=[]
 async def send_prompt_async(self,*,message):self.calls.append(message.get_value());return wrap_response('Synthetic response.')
 async def send_turn_async(self,text,history):self.calls.append((text,list(history)));return 'Synthetic response.'
class DualWebTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with patch('app.artifacts.ARTIFACTS_DIR',Path(_tmp.name)/'artifacts'):cls.app=create_served_app()
  cls.app.config['TESTING']=True
 def setUp(self):
  self.app.config['V2_TARGET_RULES']=POLICY
  self.ctx=self.app.app_context();self.ctx.push();db.drop_all();db.create_all()
  t=Target(name='Mock robo-adviser',provider='openai',model_name='mock',endpoint_url='http://127.0.0.1:9/v1');db.session.add(t);db.session.commit();self.tid=t.id;self.c=self.app.test_client()
  with self.c.session_transaction() as s:s['csrf_token']='test-token'
 def tearDown(self):db.session.remove();self.ctx.pop()
 def post(self,**data):return self.c.post('/benchmarks/new',data={'csrf_token':'test-token','target_id':self.tid,**data})
 def bench(self,scripted=False,verdict=None):
  b=Benchmark(name='Both saved judges',target_id=self.tid,status='completed',attack_slugs_json=json.dumps(['custom_conversation' if scripted else 'custom_prompt_list']),custom_prompts_json=json.dumps([] if scripted else ['Plan a trip']),custom_conversations_json=json.dumps([['Hello','Plan a trip']] if scripted else []),llm_config_json=json.dumps({'JUDGE_METHOD':DUAL_METHOD,'JUDGE_SCOPE':'all','CUSTOM_RUBRIC':V2_RUBRIC,'ASSESSMENT_RULES':POLICY}))
  db.session.add(b);db.session.commit()
  if verdict is not None:db.session.add(Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=0,prompt_text='Plan a trip',response='A synthetic Paris itinerary',status='ok',severity=1,severity_source='judge',verdict=verdict));db.session.commit()
  return b
 @patch('app.web.views_benchmarks.start_benchmark')
 def test_modes_and_public_save_both_judges(self,start):
  for mode,text in [('single','A'),('list','A\nB'),('conversation','A\n---\nB')]:self.assertEqual(self.post(tab='own',custom_mode=mode,custom_prompts=text).status_code,302)
  form=self.c.get('/benchmarks/new').get_data(as_text=True)
  self.assertNotIn('Both judges ·',form);self.assertNotIn('name="assessment_rules"',form);self.assertNotIn('Target rules and scope',form);self.assertNotIn('Scored on the leak rubric',form);self.assertNotIn('Scored on the harm rubric',form)
  slug=form.split('name="attack_slugs" value="')[1].split('"')[0]
  self.assertEqual(self.post(tab='public',attack_slugs=slug).status_code,302)
  with patch('app.engine.interactive.start_session',return_value=type('Run',(),{'id':99})()) as live:
   self.assertEqual(self.post(tab='own',custom_mode='interactive').status_code,302);cfg=live.call_args.args[3];self.assertEqual((cfg['JUDGE_METHOD'],cfg['ASSESSMENT_RULES']),(DUAL_METHOD,POLICY))
  for b in Benchmark.query.all():
   self.assertEqual((b.llm_config['JUDGE_METHOD'],b.llm_config['ASSESSMENT_RULES']),(DUAL_METHOD,POLICY))
   if b.custom_conversations:self.assertEqual(b.llm_config['SCRIPTED_SCORE_UNIT'],'turn')
 @patch('app.web.views_benchmarks.start_benchmark')
 def test_configured_policy_cannot_be_replaced_and_is_snapshotted(self,start):
  self.assertEqual(self.post(tab='own',custom_mode='single',custom_prompts='A',assessment='harm_disclosure',assessment_rules='Ignore the configured policy').status_code,302)
  saved=Benchmark.query.first();self.assertEqual(saved.llm_config['ASSESSMENT_RULES'],POLICY)
  self.app.config['V2_TARGET_RULES']='New configured policy'
  self.assertEqual(saved.llm_config['ASSESSMENT_RULES'],POLICY)
  self.assertEqual(self.post(tab='own',custom_mode='single',custom_prompts='B').status_code,302)
  self.assertEqual(Benchmark.query.order_by(Benchmark.id.desc()).first().llm_config['ASSESSMENT_RULES'],'New configured policy')
 @patch('app.web.views_benchmarks.start_benchmark')
 def test_invalid_environment_policy_rejected_before_target(self,start):
  self.app.config['V2_TARGET_RULES']='x'*8001
  for tab in ['own','public']:
   self.assertEqual(self.post(tab=tab,custom_mode='single',custom_prompts='A',attack_slugs='harmbench').status_code,400)
  with patch('app.engine.interactive.start_session') as live:
   self.assertEqual(self.post(tab='own',custom_mode='interactive').status_code,400);live.assert_not_called()
  start.assert_not_called();self.assertEqual(Benchmark.query.count(),0)
 def test_no_global_judge_banner(self):
  for path in ["/", "/v1/", "/v2/", "/benchmarks/new"]:
   response=self.c.get(path);self.assertEqual(response.status_code,200);self.assertNotIn("Both v1 and v2 score each reply",response.get_data(as_text=True))
 def test_general_default_and_policy_startup_validation(self):
  from app.settings import DEFAULT_V2_TARGET_RULES
  self.assertIn('educational',DEFAULT_V2_TARGET_RULES)
  self.assertIn('role-play',DEFAULT_V2_TARGET_RULES)
  with patch.object(Config,'V2_TARGET_RULES','x'*8001),patch('app._migrate_add_columns') as migrate:
   with self.assertRaisesRegex(ValueError,'V2_TARGET_RULES'):create_served_app()
   migrate.assert_not_called()
 def test_environment_default_blank_missing_and_custom(self):
  import runpy
  from app.settings import DEFAULT_V2_TARGET_RULES
  settings=ROOT/'app/settings.py'
  with patch.dict(os.environ,{'V2_TARGET_RULES':''}):
   self.assertEqual(runpy.run_path(str(settings))['V2_TARGET_RULES'],DEFAULT_V2_TARGET_RULES)
   os.environ.pop('V2_TARGET_RULES')
   self.assertEqual(runpy.run_path(str(settings))['V2_TARGET_RULES'],DEFAULT_V2_TARGET_RULES)
  with patch.dict(os.environ,{'V2_TARGET_RULES':'  '+POLICY+'  '}):
   self.assertEqual(runpy.run_path(str(settings))['V2_TARGET_RULES'],POLICY)
  # The isolated runner copies only the public template, never a configured .env.
  template=ROOT/'.env.example'
  from dotenv import dotenv_values
  self.assertEqual(dotenv_values(template)['V2_TARGET_RULES'],DEFAULT_V2_TARGET_RULES)
 def test_default_required_and_resolved_for_aliases(self):
  for value in ['', 'v3']:
   with patch.object(Config,'DEFAULT_JUDGE',value),patch('app._migrate_add_columns') as migrate:
    with self.assertRaisesRegex(ValueError,'DEFAULT_JUDGE'):create_served_app()
    migrate.assert_not_called()
  with patch.object(Config,'DEFAULT_JUDGE','v2'):app=create_served_app()
  self.assertEqual(app.config['JUDGE_DISPLAY_VERSION'],'v2');self.assertTrue(all(a.config['JUDGE_DISPLAY_VERSION']=='v2' for a in app.wsgi_app.mounts.values()))
 def test_one_target_exchange_two_judge_calls(self):
  for scripted in [False,True]:
   b=self.bench(scripted);bot=Bot();judge=Judge();attack={'slug':'custom_conversation' if scripted else 'custom_prompt_list','kind':'scripted' if scripted else 'static','dataset':'custom'}
   asyncio.run((run_scripted_attack if scripted else run_static_attack)(b,attack,bot,judge,1,ConsecutiveFailureGuard(3)))
   self.assertEqual(len(bot.calls),2 if scripted else 1);self.assertEqual(len(judge.calls),2);p=b.prompts[0];saved=judge_version_results(p.verdict);self.assertEqual((p.severity,saved['v1']['score'],saved['v2']['score']),(1,1,5));self.assertNotIn(POLICY,str(bot.calls))
 def test_scripted_each_reply_saves_both_judges_and_renders_individual_results(self):
  b=self.bench(scripted=True);cfg=b.llm_config;cfg['SCRIPTED_SCORE_UNIT']='turn';b.llm_config_json=json.dumps(cfg);db.session.commit()
  bot=Bot();judge=Judge()
  asyncio.run(run_scripted_attack(b,{'slug':'custom_conversation','kind':'scripted','dataset':'custom'},bot,judge,1,ConsecutiveFailureGuard(3)))
  db.session.expire(b,['prompts'])
  self.assertEqual((len(bot.calls),len(judge.calls),len(b.prompts)),(2,4,2))
  self.assertEqual([len(history) for _,history in bot.calls],[0,2])
  for p in b.prompts:
   saved=judge_version_results(p.verdict);self.assertEqual((saved['v1']['score'],saved['v2']['score']),(1,5))
  page=self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True)
  self.assertIn('Each reply is judged separately',page)
  self.assertEqual(page.count('data-prompt-id='),2)
  self.assertEqual(page.count('data-judge-versions='),2)
  self.assertIn('Conversation <code>script-0</code>',page)
  self.assertIn('<td>1</td>',page);self.assertIn('<td>2</td>',page)
  rows=list(csv.DictReader(io.StringIO(self.c.get(f'/benchmarks/{b.id}/csv').get_data(as_text=True))))
  self.assertEqual(len(rows),2);self.assertTrue(all((r['v1_score'],r['v2_score'])==('1','5') for r in rows))
  b.status='running';db.session.commit()
  live=self.c.get(f'/benchmarks/{b.id}/status.json').get_json()
  self.assertEqual(live['prompt_results_html'].count('data-prompt-id='),2)

 def test_scripted_judge_retry_does_not_construct_a_target(self):
  from app.engine.runner import _rerun_prompt
  b=self.bench(scripted=True);cfg=b.llm_config;cfg['SCRIPTED_SCORE_UNIT']='turn';b.llm_config_json=json.dumps(cfg);db.session.commit()
  p=Prompt(benchmark_id=b.id,attack_slug='custom_conversation',sequence=0,objective_id='script-0',
           prompt_text='Hello',response='Saved reply',status='judge_error',
           transcript_json=json.dumps([{'turn':1,'prompt':'Hello','response':'Saved reply','status':'ok'}]),
           judge_input_json=json.dumps({'assessment':{'method':DUAL_METHOD,'rubric':V2_RUBRIC,'assessment_rules':POLICY},'history':[]}))
  db.session.add(p);db.session.commit()
  with patch('app.engine.runner.build_conversation_target',side_effect=AssertionError('must not construct target')) as target, patch('app.engine.runner.build_judge_llm',return_value=Judge()):
   asyncio.run(_rerun_prompt(b.id,p.id,'failed'))
  target.assert_not_called();self.assertEqual(p.response,'Saved reply');self.assertEqual(p.severity,1)

 def test_partial_failure_retry_never_resends(self):
  b=self.bench(verdict=results(None));p=b.prompts[0];self.assertTrue(p.can_rescore);self.assertTrue(p.can_rerun);self.assertEqual(p.retry_label,'Re-score');bot=Bot();judge=Judge()
  asyncio.run(run_static_attack(b,{'slug':'custom_prompt_list','kind':'static','dataset':'custom'},bot,judge,1,ConsecutiveFailureGuard(3),retry_prompt=p));self.assertEqual(bot.calls,[]);self.assertEqual(len(judge.calls),2);self.assertEqual(judge_version_results(p.verdict)['v2']['score'],5)
 def test_converter_failures_are_unsent_inputs_in_page_csv_and_pdf(self):
  from pypdf import PdfReader
  b=self.bench();b.attack_slugs_json=json.dumps(['harmbench'])
  for i,status in enumerate(['converter_error','converter_refused']):
   db.session.add(Prompt(benchmark_id=b.id,attack_slug='harmbench',sequence=i,prompt_text='Original converter input',response='',status=status,converter_used='persuasion:expert_endorsement',error='converter-error: invalid JSON' if i==0 else None,note='Converter declined' if i else None,latency_measured=False))
  db.session.add(Prompt(benchmark_id=b.id,attack_slug='harmbench',sequence=2,prompt_text='Actual rewritten target prompt',response='Actual target reply',status='ok',converter_used='persuasion:expert_endorsement',verdict=results(),severity=1));db.session.commit()
  page=self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True)
  self.assertEqual(page.count('Converter input — not sent to target'),2)
  self.assertIn('Converter error — show',page);self.assertIn('Converter declined',page)
  for row in re.findall(r'<tr data-prompt-id=.*?</tr>',page,re.S):
   if 'Original converter input' in row:
    self.assertNotIn('data-judge-explanation',row);self.assertIn('Not sent',row)
  rows=list(csv.DictReader(io.StringIO(self.c.get(f'/benchmarks/{b.id}/csv').get_data(as_text=True))))
  self.assertEqual([r['prompt_role'] for r in rows],['converter_input_not_sent','converter_input_not_sent','target_prompt'])
  self.assertEqual(rows[2]['prompt'],'Actual rewritten target prompt')
  text=' '.join(p.extract_text() for p in PdfReader(io.BytesIO(self.c.get(f'/benchmarks/{b.id}/pdf').data)).pages)
  self.assertIn('Converter input',text);self.assertIn('No target request or response.',text)

 def test_attack_sections_follow_completion_order_initial_and_live(self):
  b=self.bench();b.status='running';b.attack_slugs_json=json.dumps(['advbench','harmbench'])
  # Reverse alphabetical/capture order so only completion history is decisive.
  b.completed_attacks_json=json.dumps(['harmbench','advbench'])
  for i,slug in enumerate(['advbench','harmbench']):
   db.session.add(Prompt(benchmark_id=b.id,attack_slug=slug,sequence=i,prompt_text=slug,response='Saved reply',status='ok'))
  db.session.commit()
  for page in [self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True),self.c.get(f'/benchmarks/{b.id}/status.json').get_json()['prompt_results_html']]:
   self.assertLess(page.index('id="a-harmbench"'),page.index('id="a-advbench"'))
  b.completed_attacks_json=json.dumps(['advbench','harmbench']);db.session.commit()
  page=self.c.get(f'/benchmarks/{b.id}/status.json').get_json()['prompt_results_html']
  self.assertLess(page.index('id="a-advbench"'),page.index('id="a-harmbench"'))

 def test_csv_pdf_preserve_both_results(self):
  from pypdf import PdfReader
  b=self.bench(verdict=results());row=next(csv.DictReader(io.StringIO(self.c.get(f'/benchmarks/{b.id}/csv').get_data(as_text=True))));self.assertEqual((row['v1_score'],row['v2_score']),('1','5'))
  text='\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(self.c.get(f'/benchmarks/{b.id}/pdf').data)).pages);self.assertIn('v1 score: 1',text);self.assertIn('v2 score: 5',text)
 def test_escaped_credentials_masked_in_both_result_envelope(self):
  from app.web.evidence import mask_keys
  from pypdf import PdfReader
  secret = 'test' + 'A1' * 16
  payload = json.loads(results())
  payload['judge_version_results']['v2']['verdict'] += '\npassword="' + secret + '"'
  raw = json.dumps(payload)
  self.assertNotIn(secret, mask_keys(raw))
  b = self.bench(verdict=raw)
  for suffix in ['', '/csv']:
   self.assertNotIn(secret, self.c.get(f'/benchmarks/{b.id}' + suffix).get_data(as_text=True))
  pdf=self.c.get(f'/benchmarks/{b.id}/pdf').data
  self.assertNotIn(secret, '\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages))
  # Interactive uses this same shared mask before its JSON API and PDF.
  from app.web.views_interactive import _display
  with patch('app.engine.interactive.snapshot',return_value={'turns':[{'verdict':raw}]}):
   shown=_display(b)
  self.assertNotIn(secret,json.dumps(shown));self.assertEqual(shown['turns'][0]['judge_versions']['v2']['score'],5)
 def test_running_status_renders_persisted_rows_and_updated_judgement(self):
  b=self.bench();b.status='running';db.session.commit()
  snapshot=lambda:self.c.get(f'/benchmarks/{b.id}/status.json').get_json()
  self.assertIn('No prompts recorded yet.',snapshot()['prompt_results_html'])
  secret='test'+'A1'*16
  payload=json.loads(results());payload['judge_version_results']['v2']['verdict']+='\npassword="'+secret+'" <script>fixture</script>'
  p=Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=0,prompt_text='<b>Live prompt</b>',response='<script>fixture</script> password="'+secret+'"',status='ok',severity=1,severity_source='judge',verdict=json.dumps(payload));db.session.add(p);db.session.commit()
  d=snapshot();fragment=d['prompt_results_html'];self.assertEqual(d['status'],'running');self.assertEqual(d['captured'],1)
  self.assertIn('&lt;b&gt;Live prompt&lt;/b&gt;',fragment);self.assertIn('&lt;script&gt;fixture&lt;/script&gt;',fragment);self.assertNotIn('<script>',fragment);self.assertNotIn(secret,fragment)
  saved=json.loads(html.unescape(re.search(r"data-judge-versions='([^']+)'",fragment).group(1)))
  self.assertEqual((saved['v1']['score'],saved['v2']['score']),(1,5));self.assertNotIn(secret,saved['v2']['verdict'])
  payload['judge_version_results']['v2'].update(score=2,verdict='Updated persisted judgement.\nSCORE: 2',succeeded=False);p.verdict=json.dumps(payload)
  db.session.add(Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=1,prompt_text='No-reply fixture',response=None,status='target_error',error='Synthetic target unavailable'));db.session.commit()
  updated=snapshot();self.assertEqual(updated['status'],'running');self.assertEqual(updated['captured'],1);self.assertIn('Updated persisted judgement.',updated['prompt_results_html']);self.assertIn('No-reply fixture',updated['prompt_results_html']);self.assertIn('Synthetic target unavailable',updated['prompt_results_html'])
 def test_live_poll_keeps_selected_judge_and_open_disclosures(self):
  from patchright.async_api import async_playwright
  b=self.bench();b.status='running';db.session.commit()
  initial=self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True)
  snapshot=lambda:self.c.get(f'/benchmarks/{b.id}/status.json').get_json()
  empty=snapshot()
  p=Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=0,prompt_text='Live browser fixture',response='Persisted live reply',status='ok',severity=1,severity_source='judge',verdict=results(),error='Synthetic judge warning');db.session.add(p);db.session.commit();first=snapshot()
  changed=json.loads(results());changed['judge_version_results']['v2'].update(score=2,verdict='Updated browser judgement.\nSCORE: 2',succeeded=False);p.verdict=json.dumps(changed)
  db.session.add(Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=1,prompt_text='No-reply browser fixture',response=None,status='target_error',error='Synthetic target unavailable'));db.session.commit();updated=snapshot()
  async def exercise():
   async with async_playwright() as pw:
    browser=await pw.chromium.launch(headless=True);page=await browser.new_page();live=empty;polls=0;documents=0
    async def serve(route):
     nonlocal polls,documents
     url=route.request.url
     if url.endswith('/status.json'):polls+=1;body=json.dumps(live);ctype='application/json'
     elif url.endswith('dual_judge.js'):body=(ROOT/'app/static/dual_judge.js').read_text();ctype='application/javascript'
     elif url.endswith('styles.css'):body=(ROOT/'app/static/styles.css').read_text();ctype='text/css'
     else:documents+=1;body=initial;ctype='text/html'
     await route.fulfill(body=body,content_type=ctype)
    await page.route('**/*',serve)
    try:
     await page.goto(f'http://offline.invalid/benchmarks/{b.id}');await page.wait_for_function('Boolean(window.DualJudge)');await page.locator('#judge-version').select_option('v2')
     live=first;await page.wait_for_function("document.querySelector('#prompt-results').textContent.includes('Persisted live reply')")
     self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'5/5')
     judge=f'[data-judge-explanation="{p.id}"]';error=f'[data-prompt-error="{p.id}"]'
     await page.locator(judge+' summary').click();await page.locator(error+' summary').click()
     await page.evaluate("window.savedLiveRow=document.querySelector('#prompt-results tbody tr')")
     previous=polls;live={**first,'run_time':'Unchanged poll fixture'}
     await page.wait_for_function("document.querySelector('#m-runtime').textContent === 'Unchanged poll fixture'")
     self.assertGreater(polls,previous);self.assertTrue(await page.evaluate("window.savedLiveRow === document.querySelector('#prompt-results tbody tr')"))
     live=updated;await page.wait_for_function("document.querySelector('#prompt-results').textContent.includes('Updated browser judgement.')")
     self.assertEqual(await page.locator('#judge-version').input_value(),'v2');self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'2/5')
     self.assertTrue(await page.locator(judge).evaluate('(node)=>node.open'));self.assertTrue(await page.locator(error).evaluate('(node)=>node.open'))
     self.assertIn('No-reply browser fixture',await page.locator('#prompt-results').inner_text());self.assertEqual(documents,1);self.assertEqual(await page.locator('#status').inner_text(),'running')
    finally:await browser.close()
  asyncio.run(exercise())
 @patch('app.web.views_benchmarks.start_benchmark_recovery')
 @patch('app.web.views_benchmarks.start_benchmark')
 def test_public_static_resume_preserves_lost_judge_capture(self,start,recover):
  b=self.bench();b.attack_slugs_json='["harmbench"]';b.status='interrupted'
  row=Prompt(benchmark_id=b.id,attack_slug='harmbench',sequence=0,prompt_text='Saved',response='Retained reply',status='judge_error')
  db.session.add(row);db.session.commit()
  self.assertEqual(self.c.post(f'/benchmarks/{b.id}/resume',data={'csrf_token':'test-token'}).status_code,302)
  recover.assert_called_once();self.assertFalse(recover.call_args.kwargs['retry_failed']);start.assert_not_called()
  self.assertEqual(db.session.get(Prompt,row.id).response,'Retained reply')
 def test_pending_reply_visible_and_scoring_label_survives_switch(self):
  from patchright.async_api import async_playwright
  b=self.bench();b.status='running'
  db.session.add(Prompt(benchmark_id=b.id,attack_slug='custom_prompt_list',sequence=0,prompt_text='Pending prompt',response='Captured pending reply',status='scoring'))
  db.session.commit();html=self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True)
  self.assertIn('Captured pending reply',html)
  async def exercise():
   async with async_playwright() as pw:
    browser=await pw.chromium.launch(headless=True)
    try:
     page=await browser.new_page()
     async def serve(route):
      url=route.request.url
      body=(ROOT/'app/static/dual_judge.js').read_text() if url.endswith('dual_judge.js') else html
      await route.fulfill(body=body,content_type='application/javascript' if url.endswith('dual_judge.js') else 'text/html')
     await page.route('**/*',serve);await page.goto('http://offline.invalid/benchmarks/1')
     await page.wait_for_function('Boolean(window.DualJudge)')
     self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'Scoring…')
     await page.locator('#judge-version').select_option('v2')
     self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'Scoring…')
    finally:await browser.close()
  asyncio.run(exercise())
 def test_footer_switch_has_no_requests(self):
  from patchright.async_api import async_playwright
  b=self.bench(verdict=results());html=self.c.get(f'/benchmarks/{b.id}').get_data(as_text=True);self.assertGreater(html.index('id="judge-version"'),html.index('data-judge-versions'))
  async def exercise():
   async with async_playwright() as pw:
    browser=await pw.chromium.launch(headless=True);page=await browser.new_page();network=[]
    async def serve(route):
     url=route.request.url
     if url.endswith('dual_judge.js'):body=(ROOT/'app/static/dual_judge.js').read_text();ctype='application/javascript'
     elif url.endswith('styles.css'):body=(ROOT/'app/static/styles.css').read_text();ctype='text/css'
     else:body=html;ctype='text/html'
     await route.fulfill(body=body,content_type=ctype)
    await page.route('**/*',serve);await page.goto('http://offline.invalid/benchmarks/1');await page.wait_for_function('Boolean(window.DualJudge)');self.assertEqual(await page.locator('#judge-version').input_value(),'v1');self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'1/5');page.on('request',lambda request:network.append(request.url));await page.locator('#judge-version').select_option('v2');self.assertEqual(await page.locator('[data-dual-score]').first.inner_text(),'5/5');self.assertIn('CRITERION_2',await page.locator('[data-dual-explanation]').first.text_content());self.assertIn('1 candidate successes',await page.locator('#selected-judge-summary').inner_text());self.assertEqual(network,[]);await browser.close()
  asyncio.run(exercise())
if __name__=='__main__':unittest.main()
