"""Describe every preserved risk variant by fold and retain paired event coordinates."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.verify_v273_learned_selection import read_npz
from scripts.verify_v273_selector_repair_artifacts import require,digest,named_pairs
from scripts.v273_contextual_risk_contract import retention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require(not args.output.exists(),'refusing to replace outcome audit');args.output.mkdir(parents=True)
    path=args.evidence/'preserved-candidates.npz';d=read_npz(path)
    ids,y,b,f,eligible,parent=(d[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index','parent_K'])
    lookup={int(g):i for i,g in enumerate(ids)};pos=np.array([lookup[int(g)] for g in eligible])
    keys=list(d['variant_ids']);pred=d['predictions']
    require(pred.shape==(59309,5),'five preserved variants')
    result=dict(source_memory_sha256=digest(path),folds={},transitions={},independent_validation=False)
    for j,key in enumerate(keys):
        result['folds'][str(key)]={str(fold):dict(versus_freeze=named_pairs(y[f==fold],b[f==fold],pred[f==fold,j]),
            versus_parent=named_pairs(y[f==fold],parent[f==fold],pred[f==fold,j])) for fold in (0,1,2,4)}
        added=(y==b)&(parent==b)&(pred[:,j]!=b)
        result['transitions'][str(key)]=dict(new_regressions=int(added.sum()),
            new_regressions_by_initial_to_proposed={f'{source}->{target}':int((added&(b==source)&(pred[:,j]==target)).sum())
                for source in (2,3,4) for target in range(2,7) if source!=target})
    with (args.evidence/'paired-cases.csv').open() as handle:cases=list(csv.DictReader(handle))
    require([int(c['global_index']) for c in cases]==list(map(int,eligible)),'event coordinates aligned')
    good=keys.index('catalogue8_parent_risk_new_correction');inverse=keys.index('catalogue8_new_risk_parent_correction')
    output=args.output/'paired-factor-cases.csv'
    with output.open('w',newline='') as handle:
        fields=['global_index','fold','recording_id','start_sample','true_K','frozen_K','parent_K','candidate_K',
            'parent_profile','parent_risk_new_correction_K','new_risk_parent_correction_K','effect_good_crossover_vs_parent',
            'parent_baseline_correct_probability','candidate_baseline_correct_probability']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for i,c in enumerate(cases):
            row={key:c[key] for key in fields if key in c};at=pos[i]
            row.update(parent_risk_new_correction_K=int(pred[at,good]),new_risk_parent_correction_K=int(pred[at,inverse]),
                effect_good_crossover_vs_parent=int(pred[at,good]==y[at])-int(parent[at]==y[at]))
            writer.writerow(row)
    result['paired_cases']=dict(file=output.name,rows=len(cases),sha256=digest(output))
    result['risk_swap_with_new_correction_fixed']=retention(y,b,pred[:,0],pred[:,good])
    (args.output/'diagnostics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:{f:v['versus_parent']['global']['net'] for f,v in folds.items()} for k,folds in result['folds'].items()}),flush=True)


if __name__=='__main__':main()
