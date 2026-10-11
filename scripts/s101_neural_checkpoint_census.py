"""S101: inspect saved S38/S40/S41 neural checkpoints, not accuracy.

Historical checkpoints are a prerequisite for exact deployment; manifest
presence alone does not establish full S80 inference. Never load truth.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def inspect(root,out):
    inventory={}
    expected={'S38':16,'S40':4,'S41':19}
    for family,minimum in expected.items():
        directory=root/family
        files=sorted(p for p in directory.rglob('*') if p.is_file())
        ckpt=[p for p in files if p.suffix in {'.pt','.joblib'}]
        reports=[p for p in files if p.name=='report.json']
        info={'files':len(files),'checkpoints':len(ckpt),
              'checkpoint_files':[str(p.relative_to(directory)) for p in ckpt],
              'report_files':[str(p.relative_to(directory)) for p in reports],
              'expected_checkpoint_count':minimum,
              'checkpoint_count_valid':len(ckpt)>=minimum,
              'original_new_WAV_forward_verified':False}
        inventory[family]=info
    result={'status':'S101_ORIGINAL_NEURAL_CHECKPOINT_INVENTORY',
            'inventory':inventory,'all_checkpoint_counts_valid':all(x['checkpoint_count_valid'] for x in inventory.values()),
            'truth_used':False,'GOAT_forward_verified':False,
            'new_S45_families_ready':0,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['all_checkpoint_counts_valid']:
        raise ValueError('Missing original neural checkpoints; fail closed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    inspect(a.root,a.out)
