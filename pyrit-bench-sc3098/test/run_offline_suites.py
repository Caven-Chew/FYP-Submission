#!/usr/bin/env python3
"""Run the nineteen standalone application offline suites in a disposable source copy, without .env or real data.
From this repository: .venv/bin/python -B test/run_offline_suites.py
Logs/results: a new timestamped test/runs/offline-suites-* directory.
--output must name a new directory, so retained evidence is never overwritten.
This creates a temporary test workspace, never a distribution package.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SUITES = ['test_workshop_preset.py', 'test_scoring_changes.py', 'test_prompt_retry.py', 'test_student_engine.py', 'test_student_web.py', 'test_input_and_evidence.py', 'baseline_offline.py', 'test_engine_lifecycle.py', 'test_judge_configuration_and_redaction.py', 'test_results_and_exports.py', 'test_student_pages.py', 'test_judge_methods.py', 'test_interactive.py', 'test_ui_latency.py', 'test_v2_judge.py', 'test_dual_judge_web.py', 'test_static_pipeline.py', 'test_scripted_pipeline.py', 'test_live_pipeline.py']
BOOT = '''import os,runpy,sys
from pathlib import Path
import appdirs
appdirs.user_data_dir=lambda name,*a,**k:str(Path(os.environ['SC3098_TEST_RUNTIME'])/name)
sys.path.insert(0,str(Path('test').resolve()))
sys.argv=[sys.argv[1]]+sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'test/runs'/
                   ('offline-suites-'+datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d-%H%M%S-%f')))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'started_sgt':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
            'method':f'{len(SUITES)} registered offline suites in a disposable app/test source copy with the public .env.example; dotenv disabled, mock calls only.',
            'source_sha256':{},'suites':[]}
    for path in [ROOT/'app/data/gdp_checklist.json',ROOT/'.env.example',ROOT/'test/template_safety.py',Path(__file__),*[ROOT/'test'/n for n in SUITES]]:
        report['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='sc3098-offline-suites-') as tmp:
        work=Path(tmp)
        for area in ['app','test']:
            shutil.copytree(ROOT/area,work/area,ignore=shutil.ignore_patterns('__pycache__','runs','.venv'))
        shutil.copy2(ROOT/'.env.example',work/'.env.example')
        env={k:v for k,v in os.environ.items() if not k.startswith(
            ('JUDGE_', 'ATTACKER_', 'VISION_', 'LLM_', 'OPENAI_', 'ANTHROPIC_', 'GEMINI_', 'DATABASE_URL', 'V2_TARGET_RULES'))}
        env.update({'PYTHON_DOTENV_DISABLED':'1','PYTHONDONTWRITEBYTECODE':'1',
                    'SC3098_TEST_RUNTIME':str(work/'runtime'),
                    # Read-only access to the public source template for attack catalogue checks.
                    'SC3098_SOURCE_ROOT':str(ROOT)})
        for name in SUITES:
            logfile=args.output/(Path(name).stem+'.log')
            command=[sys.executable,'-B','-c',BOOT,'test/'+name]
            if name in SUITES[3:6] or name == 'test_results_and_exports.py':command.append('-v')
            with logfile.open('w') as output:
                done=subprocess.run(command,cwd=work,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=300)
            report['suites'].append({'suite':name,'exit_code':done.returncode,'log':logfile.name})
            print(('PASS' if done.returncode==0 else 'FAIL')+': '+name,flush=True)
    report['passed']=all(x['exit_code']==0 for x in report['suites'])
    report['finished_sgt']=datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    (args.output/'suite_results.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
