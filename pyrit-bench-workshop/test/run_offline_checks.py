#!/usr/bin/env python3
"""Validate preserved historical source in disposable copies; no hosted calls.
Run: python -B test/run_offline_checks.py --bot-python brewbot/.venv/bin/python
"""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def blocked(*a,**k):raise AssertionError('Network is disabled in historical snapshot checks')
def app_checks(work):
    os.chdir(work/'pyrit');sys.path.insert(0,str(Path.cwd()))
    import appdirs
    appdirs.user_data_dir=lambda name,*a,**k:str(work/'runtime'/name)
    from datetime import date
    import app
    from app import expiry
    from app.extensions import db
    from app.models import Target,Benchmark
    with patch('socket.socket.connect',blocked),patch('socket.create_connection',blocked):
        with patch('app.ensure_workshop_active',side_effect=lambda:expiry.ensure_workshop_active(date(2026,9,8))):
            try:app.create_app()
            except RuntimeError as e:assert 'expired' in str(e).lower()
            else:raise AssertionError('Post-cutoff startup must refuse')
        with patch('app.ensure_workshop_active',side_effect=lambda:expiry.ensure_workshop_active(date(2026,9,7))):
            instance=app.create_app()
        instance.config['TESTING']=True
        client=instance.test_client()
        paths=['/','/targets','/benchmarks','/benchmarks/new','/static/styles.css']
        for path in paths:
            response=client.get(path,follow_redirects=True)
            assert response.status_code==200,(path,response.status_code)
        with instance.app_context():assert Target.query.count()==0 and Benchmark.query.count()==0;db.session.remove()
    print(json.dumps({'post_cutoff_start_refused':True,'historical_test_date':'2026-09-07','page_checks':paths,'empty_database':True}))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bot-python',default=sys.executable);p.add_argument('--app-child',type=Path);p.add_argument('--bot-child',type=Path);a=p.parse_args()
    if a.app_child:app_checks(a.app_child);return
    if a.bot_child:
        import runpy
        os.chdir(a.bot_child/'brewbot');sys.argv=['unittest','discover','-s','tests','-v']
        with patch('socket.socket.connect',blocked),patch('socket.create_connection',blocked):runpy.run_module('unittest',run_name='__main__')
        return
    bot_python=str(Path(a.bot_python).absolute());env={k:v for k,v in os.environ.items() if not k.startswith(('JUDGE_','ATTACKER_','VISION_','LLM_','OPENAI_','ANTHROPIC_','BREWBOT_','FYP_WORKSHOP_','DATABASE_URL'))}
    env.update(PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1',SECRET_KEY='offline-snapshot-test')
    with tempfile.TemporaryDirectory(prefix='historical-workshop-check-') as tmp:
        work=Path(tmp)
        for name in ['pyrit','brewbot']:shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('.env','.venv','venv','instance','__pycache__','*.pyc'))
        env['DATABASE_URL']='sqlite:///'+str(work/'app-test.db')
        for command in [[sys.executable,'-B',str(work/'pyrit/test/test_expiry.py')],[sys.executable,'-B',str(Path(__file__).resolve()),'--app-child',str(work)],[bot_python,'-B',str(Path(__file__).resolve()),'--bot-child',str(work)]]:
            subprocess.run(command,cwd=work,env=env,check=True,timeout=180)
    print('PASS historical expiry, disposable page rendering and mocked/offline BrewBot checks')
if __name__=='__main__':main()
