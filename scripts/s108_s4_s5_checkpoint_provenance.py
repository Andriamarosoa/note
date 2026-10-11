"""S108 forensic comparison of original S4 and S5 checkpoint feature schemas.
Metadata only: no labels, no generated audio predictions, no promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np

def load(path):
    saved=joblib.load(path)
    model=saved.get('model') if isinstance(saved,dict) else saved
    keys=sorted(saved) if isinstance(saved,dict) else []
    primitive={}
    if isinstance(saved,dict):
        for k,v in saved.items():
            if isinstance(v,(str,int,float,bool,type(None))):
                primitive[k]=v
            elif isinstance(v,np.ndarray):
                primitive[k]={'type':'array','shape':list(v.shape),'dtype':str(v.dtype)}
            elif isinstance(v,(list,tuple)):
                primitive[k]={'type':type(v).__name__,'length':len(v)}
            elif isinstance(v,dict):
                primitive[k]={'type':'dict','keys':sorted(map(str,v))[:30]}
    return model,dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      saved_type=type(saved).__name__,saved_keys=keys,
                      saved_primitive_metadata=primitive,
                      estimator_type=type(model).__name__,
                      feature_dimensions=int(getattr(model,'n_features_in_',-1)),
                      classes=np.asarray(getattr(model,'classes_',[])).tolist())

def run(a):
    s4=[];s5=[];missing=[]
    for fold in (0,1,2,4):
        p=a.root/'series5'/'models'/f'fold{fold}'
        rep=json.loads((p/'report.json').read_text())
        info=load(p/'model.joblib')
        if info['sha256']!=rep['model_sha256']:raise ValueError('S5 pinned checkpoint drift')
        if info['feature_dimensions']!=527:raise ValueError('S5 527 contract drift')
        s5.append(dict(fold=fold,**info))
        for variant in ('control','owned'):
            for poly in (1,2):
                file=a.root/'series4'/'models'/f'fold{fold}__{variant}__poly{poly}.joblib'
                if not file.is_file():
                    missing.append(str(file.relative_to(a.root)))
                    continue
                info=load(file)
                s4.append(dict(fold=fold,variant=variant,poly=poly,**info))
    report={'status':'S108_S4_S5_FROZEN_CHECKPOINT_PROVENANCE',
            'S4_feature_dimensions':sorted(set(q['feature_dimensions'] for q in s4)),
            'S5_feature_dimensions':sorted(set(q['feature_dimensions'] for q in s5)),
            'S4_model_count':len(s4),'S5_model_count':len(s5),
            'S4_saved_keys':sorted(set(k for q in s4 for k in q['saved_keys'])),
            'S5_saved_keys':sorted(set(k for q in s5 for k in q['saved_keys'])),
            'S4_models':s4,'S5_models':s5,'missing_S4_checkpoints':missing,
            'source_feature_builder_verified':False,
            'external_S80_predictions':0,'promotion':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('status','S4_feature_dimensions','S5_feature_dimensions',
          'S4_model_count','S5_model_count','S4_saved_keys','S5_saved_keys')},indent=2))
    print(json.dumps({'S5_fold0_metadata':s5[0]['saved_primitive_metadata'],
                      'S4_fold0_owned_metadata':next(q['saved_primitive_metadata'] for q in s4 if q['fold']==0 and q['variant']=='owned' and q['poly']==1)},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
