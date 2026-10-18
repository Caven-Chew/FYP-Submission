#!/usr/bin/env python3
"""Recompute S1 from separately prepared reference; no hosted calls."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREDICTIONS=['B0','J0','J1','O3','J2','H1','H3','H1S','G1','G1S']
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in {'generated_at_utc','prep_dir','predictions_file','predictions_sha256'}}
    if isinstance(value,list):return [clean(v) for v in value]
    return value
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prep-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    for name,expected in [('population.jsonl','34b1d10e739328cd392fe9e6314a9af3d330b486b316983c65480fb4f0a46ea1'),('private/external_reference.jsonl','3cf114e96ec15a950683045d8273516235bc8c2eab3c9452047ebe62260cae01')]:
        if hashlib.sha256((a.prep_dir/name).read_bytes()).hexdigest()!=expected:raise ValueError('Numeric input differs: '+name)
    command=[sys.executable,'-B',str(ROOT/'tools/study_s1_external_reference.py'),'run','--prep-dir',str(a.prep_dir.resolve()),'--split','main','--predictions']+[name+'='+str(ROOT/'data/s1/predictions'/(name+'.jsonl')) for name in PREDICTIONS]+['--kinds','J0=severity','H1S=severity','G1S=severity','B0=refusal_floor','--subset','J2=j2_subsample','--contrasts','J1:J0','O3:J1','H1:J1','H1S:J0','H1:H1S','G1:G1S','G1S:J0','H1S:G1S','H1:G1','--out-dir',str(a.out_dir.resolve())]
    subprocess.run(command,check=True)
    actual=json.loads((a.out_dir/'analysis.json').read_text());expected=json.loads((ROOT/'data/s1/expected_analysis.json').read_text())
    keys=['rows','objectives','reference_positive_primary','reference_positive_strict','cuts','bootstrap','configurations','contrasts','rq3','split']
    for key in keys:
        if clean(actual[key])!=clean(expected[key]):raise AssertionError('Statistical structure differs: '+key)
    print(json.dumps({'status':'PASS','statistical_structures_equal':keys,'rows':actual['rows'],'reference_positive_primary':actual['reference_positive_primary']}))
if __name__=='__main__':main()
