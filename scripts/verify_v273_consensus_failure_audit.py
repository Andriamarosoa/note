"""Cross-check published audit cases directly against frozen prediction arrays."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.verify_v273_selector_repair_artifacts import digest,require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--consensus',type=Path,required=True)
    p.add_argument('--archived-global',type=Path,required=True)
    args=p.parse_args();a=json.loads((args.audit/'audit.json').read_text())
    require(digest(args.consensus/'predictions.npz')==a['source_sha256']['consensus'],'source SHA')
    require(digest(args.archived_global/'predictions.npz')==a['source_sha256']['archived_global'],'comparison SHA')
    with np.load(args.consensus/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(args.archived_global/'predictions.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    index={v:i for i,v in enumerate(d['global_index'])};at=np.array([index[v] for v in d['eligible_global_index']])
    ids=d['eligible_global_index'];y=d['true_K'][at];b=d['frozen_baseline_K'][at];pred=d['predicted_K'][at]
    possible=(d['group_proposal']==y[:,None]).any(1);changed=pred!=b
    selections={
        'regressions-522.csv':changed&(y==b),
        'corrections-582.csv':changed&(y==pred),
        'corrections-perdues-171.csv':(b!=y)&(old['predicted_K'][at]==y)&(pred!=y),
        'opportunites-manquees-1042.csv':(b!=y)&possible&(pred!=y),
    }
    candidate_index={v:i for i,v in enumerate(ids)}
    for name,take in selections.items():
        path=args.audit/name;require(digest(path)==a['files'][name]['sha256'],'CSV SHA')
        with path.open() as handle:rows=list(csv.DictReader(handle))
        require(len(rows)==int(take.sum())==a['files'][name]['rows'],'case count')
        require({int(r['global_index']) for r in rows}==set(ids[take]),'case population')
        for record in rows:
            i=candidate_index[int(record['global_index'])]
            require((int(record['true_K']),int(record['frozen_K']),int(record['predicted_K']))==(y[i],b[i],pred[i]),'case labels')
            if 'true_K_reachable' in record:require((record['true_K_reachable']=='True')==bool(possible[i]),'case availability')
            if changed[i] and 'selected_gain' in record:
                m=d['chosen_group_mask'][i]-1
                require(m>=0 and abs(float(record['selected_gain'])-float(d['group_expected_gain'][i,m]))<1e-6,'case selected gain')
    at_changed=np.flatnonzero(changed);chosen=d['chosen_group_mask'][changed]-1
    issue=d['group_outcome_probability'][at_changed,chosen].astype(float)
    accounting=a['gain_accounting']
    require(abs(issue[:,0].sum()-accounting['expected_corrections'])<1e-4,'expected corrections from group outputs')
    require(abs(issue[:,1].sum()-accounting['expected_regressions'])<1e-4,'expected regressions from group outputs')
    actual=(pred==y).astype(int)-(b==y).astype(int)
    require(int(actual.sum())==accounting['observed_net']==60,'observed net')
    require(sum(a['state_counts'].values())==len(ids),'state coverage')
    require(sum(t['regressions'] for t in a['transitions'])==522 and sum(t['corrections'] for t in a['transitions'])==582,'transition coverage')
    result=dict(status='verified',audit_sha256=digest(args.audit/'audit.json'),source_sha256=a['source_sha256'],
        case_counts={k:int(v.sum()) for k,v in selections.items()},group_probability_gain_recomputed=True,
        labels_availability_and_gains_checked_for_every_exported_case=True,
        changed_rows=int(changed.sum()),net=int(actual.sum()),no_retraining=True)
    (args.audit/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
