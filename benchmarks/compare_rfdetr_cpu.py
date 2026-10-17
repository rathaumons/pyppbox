# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Compare RF-DETR CPU contracts against the captured pre-change wheel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_cpu import equivalent


def compare(before, after):
    for key in ('schema','platform','architecture','python','threads','packages','asset_sha256',
                'frame_sha256','offline','custom_fixture_quality_claim','failures'):
        equivalent(before[key],after[key],key)
    def outputs(value):
        if isinstance(value,dict):
            return {key:outputs(item) for key,item in value.items() if key!='two_predictions_seconds'}
        if isinstance(value,list): return [outputs(item) for item in value]
        return value
    equivalent(outputs(before['cases']),outputs(after['cases']),'cases',tolerance=1e-5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root','candidate-root','assets','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline-ref',required=True)
    parser.add_argument('--candidate-ref',required=True)
    args=parser.parse_args(); root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    reports={}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH',None)
    for name,package in (('baseline',args.baseline_root),('candidate',args.candidate_root)):
        output=root/f'rfdetr-{name}.json'
        with (root/f'rfdetr-{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('rfdetr_cpu.py').resolve()),
                '--package-root',str(package.resolve()),'--assets',str(args.assets.resolve()),'--output',str(output)],
                env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        reports[name]=json.loads(output.read_text(encoding='utf-8'))
    compare(reports['baseline'],reports['candidate'])
    result=dict(outputs_match=True,baseline_ref=args.baseline_ref,candidate_ref=args.candidate_ref,
        variants=['nano','small','medium','large'],offline=True,
        custom_fixtures=['two-class native checkpoint','two-class Lightning compiled-prefix checkpoint'],
        custom_fixture_quality_claim=False,failures=reports['candidate']['failures'])
    (root/'rfdetr-comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    text=('## RF-DETR CPU qualification\n\n'
          f'Baseline `{args.baseline_ref}`; candidate `{args.candidate_ref}`.\n\n'
          'Nano/Small/Medium/Large: native predictions, optimized/raw models, offline cache, '
          'SORT gaps/recovery, class filters, and input/rendered output equivalence passed. '
          'Generated two-class native/Lightning checkpoint fixtures, failed replacement recovery, '
          'reset/close, and independent streams passed. Fixtures do not certify trained-model accuracy.\n')
    (root/'rfdetr-summary.md').write_text(text,encoding='utf-8')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as stream: stream.write(text)
    print(text)


if __name__=='__main__': main()
