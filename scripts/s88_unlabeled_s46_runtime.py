"""S88: exact S46 193-policy decision runtime without ANY annotation access.

Consumes a strictly label-free NPZ:
row_id, parent_K, family_ids, proposal_K [N,16],
risks [N,16,2], independent_votes [N,16].
All variants are original S46 decisions, not a surrogate. Producing these
16 family inputs for NEW audio remains the responsibility of S45 producers.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.audit_v273_s46_classwise_compromise import (
    FAMILY_GROUPS,LAM_POLY,CUT_POLY,VOTES,STRUCTURE,select
)
FORBIDDEN={'true_K','truth','y','label','labels','target','correct','regression','fix'}
def require(value,why):
    if not value:raise ValueError(why)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def run(inp,output):
    require(not output.exists(),'refuse output overwrite')
    with np.load(inp,allow_pickle=False) as x:
        require(not FORBIDDEN.intersection(x.files),'external S46 inference cannot read truth/target')
        req={'row_id','parent_K','family_ids','proposal_K','risks','independent_votes'}
        require(set(x.files)==req,'missing/extra S46 contract columns')
        data={k:x[k].copy() for k in x.files}
    ids=data['row_id'];parent=data['parent_K'].astype(np.int8)
    families=list(map(str,data['family_ids']))
    props=data['proposal_K'].astype(np.int8);risks=data['risks'];votes=data['independent_votes']
    n=len(parent)
    require(n>0 and len(ids)==n and len(np.unique(ids))==n,'invalid row IDs')
    require(len(families)==16 and len(set(families))==16,'family IDs incomplete')
    require(props.shape==(n,16) and risks.shape==(n,16,2) and votes.shape==(n,16),'family output shape')
    require(np.isin(parent,range(7)).all() and np.isin(props,range(7)).all(),'invalid K0..6')
    changed=props!=parent[:,None]
    require(np.isfinite(risks[changed]).all() and np.isfinite(votes).all(),
            'invalid S45 risk for changed proposals; unchanged can have NaN risk')
    require(not any('h9' in k.lower() or 'yourmt3' in k.lower() for k in families),
            'H9/competitor prohibited')
    result={'series18_parent':parent.copy()}
    group_index={}
    for group,members in FAMILY_GROUPS.items():
        if members is None:group_index[group]=list(range(16))
        else:
            require(all(k in families for k in members),'missing S45 family '+group)
            group_index[group]=[families.index(k) for k in members]
        for lam in LAM_POLY:
            for cut in CUT_POLY:
                for minvote in VOTES:
                    for structure in STRUCTURE:
                        key=f'series46__{group}__polyLambda{lam:g}__cut{cut:g}__votes{minvote}__{structure}'
                        y,_=select(parent,props,risks,votes,group_index[group],
                                   lam,cut,minvote,structure)
                        result[key]=y
    require(len(result)==193,'193 policies contract violated')
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',row_id=ids,
        variant_ids=np.asarray(list(result)),
        predictions=np.column_stack(list(result.values())).astype(np.int8))
    report={'status':'S46_UNLABELED_193_POLICY_RUNTIME_SUCCESS',
            'row_count':n,'families':families,'variants':len(result),
            'labels_opened':False,'real_audio_inference':False,
            'source_input_sha256':digest(inp),
            'output_sha256':digest(output/'predictions.npz'),
            'external_S80_validation':False,
            'required_upstream':'Original 16 S45 families must provide all input arrays for new recordings'}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('status','row_count','variants','labels_opened')},sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
