"""Reproduce Task 11.1 against the existing mock API, without model spending.
Run from SC3098: ../.venv/bin/python test/baseline_offline.py
"""
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['DATABASE_URL'] = 'sqlite:///' + tempfile.mktemp(suffix='.db')
os.environ['REQUEST_DELAY'] = '0'
os.environ['ATTACKER_BACKUP_MODEL'] = ''
from mock_llm_api import Handler
from app import create_app
from app.extensions import db
from app.models import Target, Benchmark, Prompt
from app.targets.http_target import HttpTarget
from app.engine.static_attack import run_static_attack
from app.engine.retry import ConsecutiveFailureGuard
from app.attacks import by_slug

async def run():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://127.0.0.1:{server.server_port}/chat'
    target = HttpTarget(endpoint, None, {'body_template':'{"input":"{prompt}"}', 'response_path':'output.text'})
    class Judge:
        async def send_prompt_async(self, **kw):
            from app.targets.common import wrap_response
            return wrap_response('REASONING: Demonstration mock response.\nSCORE: 1')
    try:
        app = create_app()
        with app.app_context():
            t=Target(name='Offline mock',provider='http_generic',model_name='mock',endpoint_url=endpoint)
            db.session.add(t); db.session.flush()
            b=Benchmark(name='Task 11.1 prompt list',target_id=t.id,attack_slugs_json='["custom_prompt_list"]',custom_prompts_json='["Explain diversification.","Explain investment fees."]')
            db.session.add(b); db.session.commit()
            await run_static_attack(b,by_slug('custom_prompt_list'),target,Judge(),2,ConsecutiveFailureGuard(3))
            rows=Prompt.query.filter_by(benchmark_id=b.id).order_by(Prompt.sequence).all()
            assert len(rows)==2 and all(p.response and not p.error and p.severity==1 for p in rows)
            assert [p.prompt_text for p in rows]==b.custom_prompts
            print('PASS: two saved custom prompts sent over local HTTP and scored by mock judge.')
    finally:
        server.shutdown(); server.server_close()
if __name__=='__main__': asyncio.run(run())
