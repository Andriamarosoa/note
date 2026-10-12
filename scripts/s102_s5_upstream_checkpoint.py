"""S102: inspect the frozen S5 models needed by S45's 32 vote features.

Trusted repository checkpoints only. Do not read evaluation truth or
manufacture source K decisions. Diagnose whether exact S5 inference can be
reconstructed from label-free acoustic evidence and upstream posteriors.
"""
from __future__ import annotations
import json,hashlib,argparse
from pathlib import Path
import joblib
import numpy as np

def scan(root,out):
    rows=[]
    for f in (0,1,2,4):
        p=root/f'fold{f}'
        checkpoint=p/'model.joblib'
        report=json.loads((p/'report.json').read_text())
        if report['model_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError(f'frozen fold {f} SHA256 mismatch')
        obj=joblib.load(checkpoint) # only this trusted, pinned repository artifact
        fields=sorted(obj.keys()) if isinstance(obj,dict) else []
        details={}
        if isinstance(obj,dict):
            for key,val in obj.items():
                if key in ('X','y','train_ids','held_ids','truth'):continue
                desc={'type':type(val).__name__}
                if hasattr(val,'n_features_in_'):desc['n_features_in']=int(val.n_features_in_)
                if hasattr(val,'shape'):desc['shape']=list(val.shape)
                if hasattr(val,'classes_'):desc['classes']=[str(x) for x in val.classes_]
                details[str(key)]=desc
        else:
            details['root']={'type':type(obj).__name__,'n_features_in':getattr(obj,'n_features_in_',None)}
        rows.append({'fold':f,'model_sha256_verified':True,
                     'fields':fields,'model_members':details,
                     'train_folds':report['train_folds'],
                     'independent_validation':False})
    result={'status':'S102_S5_UPSTREAM_MODEL_PROVENANCE_AUDITED',
            'frozen_models':rows,'loaded_checkpoint_count':len(rows),
            'source_audio_inference_verified':False,
            'S45_proposal_families_inferred':0,
            'S80_independent_score':None,'promotion':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(result,indent=2,default=str))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('analysis/evidence/v273-open-k0-k6/series5/models'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();scan(a.root,a.out)
