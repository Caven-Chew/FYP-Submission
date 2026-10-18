"""Live result payload checks using isolated data and existing PyRIT stubs."""
import re
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

fixture = runpy.run_path(str(Path(__file__).with_name('test_prompt_retry.py')), run_name='live_results_fixture')
create_app, db = fixture['create_app'], fixture['db']
Benchmark, Target, Prompt = fixture['Benchmark'], fixture['Target'], fixture['Prompt']


class LivePromptResultsTests(unittest.TestCase):
    def setUp(self):
        with patch('app.artifacts.prune'):
            self.app = create_app()
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        target = Target(name='synthetic', provider='openai', model_name='no-network')
        db.session.add(target)
        db.session.flush()
        self.b = Benchmark(name='live', target_id=target.id, status='running', attack_slugs_json='["advbench"]')
        db.session.add(self.b)
        db.session.commit()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def test_saved_rows_appear_in_poll_and_remain_escaped(self):
        url = f'/benchmarks/{self.b.id}/status.json'
        self.assertNotIn('saved response', self.client.get(url).json['prompt_results_html'])
        db.session.add(Prompt(benchmark_id=self.b.id, attack_slug='advbench', sequence=0,
            prompt_text='<script>alert(1)</script>', response='saved response', status='ok', severity=1))
        db.session.commit()
        html = self.client.get(url).json['prompt_results_html']
        self.assertIn('AdvBench harmful prompts', html)
        self.assertIn('saved response', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        page = self.client.get(f'/benchmarks/{self.b.id}').text
        self.assertIn(html, page)

    def test_existing_row_update_is_in_next_poll(self):
        p = Prompt(benchmark_id=self.b.id, attack_slug='advbench', sequence=0,
                   prompt_text='synthetic', response='', status='send_error', error='synthetic timeout')
        db.session.add(p)
        db.session.commit()
        url = f'/benchmarks/{self.b.id}/status.json'
        self.assertIn('synthetic timeout', self.client.get(url).json['prompt_results_html'])
        p.response, p.error, p.status, p.severity = 'recovered synthetic reply', None, 'ok', 1
        db.session.commit()
        html = self.client.get(url).json['prompt_results_html']
        self.assertIn('recovered synthetic reply', html)
        self.assertNotIn('synthetic timeout', html)

    def test_judge_explanation_is_collapsed_escaped_and_marked_when_missing(self):
        # One single-turn table row with a verdict, one failed send without one,
        # and a two-turn conversation, so both result tables are covered.
        self.b.attack_slugs_json = '["advbench", "crescendo_dynamic"]'
        db.session.add_all([
            Prompt(benchmark_id=self.b.id, attack_slug='advbench', sequence=0, prompt_text='scored',
                   response='synthetic reply', status='ok', severity=2,
                   verdict='SCORE: 2\nReason: <b>partial</b> synthetic detail'),
            Prompt(benchmark_id=self.b.id, attack_slug='advbench', sequence=1, prompt_text='unsent',
                   response='', status='send_error', error='synthetic timeout'),
            *(Prompt(benchmark_id=self.b.id, attack_slug='crescendo_dynamic', sequence=i,
                     objective_id='obj-1', prompt_text=f'turn {i}', response='reply', status='ok',
                     severity=1, verdict=f'synthetic turn {i} reason') for i in range(2)),
        ])
        db.session.commit()
        page = self.client.get(f'/benchmarks/{self.b.id}').text
        poll = self.client.get(f'/benchmarks/{self.b.id}/status.json').json['prompt_results_html']
        for html in (page, poll):
            self.assertIn('Conversation <code>obj-1</code>', html)
            self.assertIn('SCORE: 2\nReason: &lt;b&gt;partial&lt;/b&gt; synthetic detail', html)
            self.assertNotIn('<b>partial</b>', html)
            self.assertIn('synthetic turn 0 reason', html)
            self.assertIn('synthetic turn 1 reason', html)
            self.assertEqual(html.count('Judge explanation: not recorded'), 1)
            self.assertEqual(html.count('<details class="small" data-judge-explanation='), 3)
            self.assertIsNone(re.search(r'<details[^>]*data-judge-explanation[^>]*\bopen\b', html))


if __name__ == '__main__':
    unittest.main()
