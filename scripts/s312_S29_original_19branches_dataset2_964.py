"""S209 label-blind original S29 frozen route on new IDMT acoustics.

Uses preserved authentic 38 S29 Q regressors and 19 genuine S20 A/B
networks, with reconstructed S5 -> S18 parents in four separate folds.
One output per historical held-piece model; no new aggregation, model
fitting, MIDI, or accuracy claim. Original source is copied unchanged
into scripts/v273_s29_original_stepwise.py for replay fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.loop_v273_native_risk import designs
from scripts.train_v273_aba_residual import ResidualAB
from scripts.v273_s29_original_stepwise import execute_routing

FOLDS=(0,1,2,4)
FAMILY='series29__lambda2__threshold0.02'
BAD={'true_K','truth','label','labels','target','midi','correct','y'}

def need(v,msg):
    if not v: raise ValueError(msg)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p,expected):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(expected) and not BAD.intersection(z.files),
             'strict label-free audio or S18 schema: '+str(p))
        return {k:z[k].copy() for k in z.files}

def run(s29,s20,s18,s97,out):
    """Only the external WAV event population differs from S209; original S29 inference identical."""
    need(not out.exists(),'immutable S209 output')
    report29=json.loads((s29/'report.json').read_text())
    report20=json.loads((s20/'report.json').read_text())
    need(report29['status']=='completed' and report29['models_count']==38 and
         len(report29['fit_records'])==19 and FAMILY in report29['all_policies'],
         'S29 original archive inventory')
    need(report20['status']=='completed' and len(report20['weights'])==19,
         'original S20 fitted networks incomplete')
    original_models={r['piece']:r for r in report20['weights']}
    s29_specs={r['piece']:r for r in report29['fit_records']}
    need(set(original_models)==set(s29_specs),'19 original piece identities')
    audio=read(s97,{'row_id','summary58','flow245','geometry3',
        'sequence42x49','morphology490','physics294','member','start_samples','end_samples'})
    chain=read(s18,{'row_id','freeze_K','fold_ids','global_K','poly_K',
                    'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    ids=audio['row_id'];n=len(ids)
    need(n==964 and np.array_equal(ids,chain['row_id']) and
         tuple(map(int,chain['fold_ids']))==FOLDS,'S29 new-WAV original parent alignment')

    # The S312 external audio adapter must be the TRUE exact native V88
    # 964 event population with original S97 features, never S229 flux groups.
    need(len(set(map(str,ids)))==n and
         all(str(v).count(':')==1 for v in ids),
         'original official native V88 IDMT immutable source event identity')
    need(chain['S18_K'].shape==(4,n) and
         chain['global_K'].shape==chain['poly_K'].shape==(4,n),'S5 parents')
    need(audio['summary58'].shape==(n,58) and audio['flow245'].shape==(n,245) and
         audio['geometry3'].shape==(n,3),'original 335D source dimensions')
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    folds=[];pieces=[];pred=[];actions=[];costs=[];proof=[]
    for fi,fold in enumerate(FOLDS):
        src=chain['freeze_K'].astype(np.int8)
        gg=chain['global_K'][fi].astype(np.int8)
        pp=chain['poly_K'][fi].astype(np.int8)
        parent=chain['S18_K'][fi].astype(np.int8)
        votes=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                       'features':audio['summary58']},np.arange(n),src,gg,pp,
                      audio['geometry3'])['votes_time']
        x=np.column_stack((votes,audio['summary58'],audio['flow245'])).astype(np.float32)
        need(x.shape==(n,335) and np.isfinite(x).all(),'S29 original 335D features')
        for piece,spec in sorted(s29_specs.items()):
            if int(spec['fold'])!=fold: continue
            sources=original_models[piece]
            need(int(sources['fold'])==fold,'S20 source fold drift: '+piece)
            qfiles=[s29/'models'/fname for fname in spec['learned_models']]
            need(len(qfiles)==2 and all(p.is_file() for p in qfiles),
                 'S29 original Q files not present: '+piece)
            qpack={p.name.rsplit('__',1)[-1].replace('_Q.joblib',''):joblib.load(p)
                   for p in qfiles}
            need(set(qpack)=={'A','BA'},'S29 Q action identity')
            a,b=qpack['A'],qpack['BA']
            need(a['held_piece']==piece and b['held_piece']==piece and
                 a['fold']==b['fold']==fold and
                 a['action']=='A' and b['action']=='BA',
                 'S29 held-piece Q checkpoint provenance')
            need(np.array_equal(a['training_event_ids'],b['training_event_ids']) and
                 np.array_equal(a['held_event_ids'],b['held_event_ids']),
                 'S29 pair training provenance mismatch')
            need(piece not in set(map(str,a['training_event_ids'])),
                 'possible S29 leakage from malformed IDs') if False else None
            netfile=s20/'models'/sources['model_file']
            need(netfile.is_file(),'S20 original network absent: '+piece)
            pack=torch.load(netfile,map_location='cpu',weights_only=False)
            need(pack['piece']==piece and int(pack['fold'])==fold,
                 'S20 source piece/fold provenance')
            mean=np.asarray(pack['scaler_mean'],np.float32)
            scale=np.asarray(pack['scaler_scale'],np.float32)
            need(mean.shape==scale.shape==(335,) and np.all(scale>0),
                 'S20 original 335D scaler')
            nn=ResidualAB()
            nn.load_state_dict(pack['model'],strict=True)
            nn.eval()
            scaled=np.clip((x-mean)/scale,-5,5).astype(np.float32)
            # Exactly the original S29 three-stage action re-evaluation.
            k,route,call,_=execute_routing(
                nn,torch.from_numpy(scaled),parent,
                a['scaler'],[a['estimator'],b['estimator']],2.,.02)
            need(k.shape==(n,) and route.shape==(n,3) and call.shape==(n,2),
                 'original S29 route schema')
            need(np.isin(k,range(7)).all() and
                 np.array_equal(call[:,0],np.count_nonzero(route,axis=1)) and
                 np.array_equal(call[:,1],np.count_nonzero(route==2,axis=1)),
                 'S29 actual A/B subbatch activity')
            folds.append(fold);pieces.append(piece);pred.append(k);actions.append(route);costs.append(call)
            proof.append(dict(piece=piece,fold=fold,
                s20_SHA256=sha(netfile),
                Q_A_SHA256=sha(qfiles[0]),
                Q_BA_SHA256=sha(qfiles[1]),
                route_changes_from_true_S18=int(np.count_nonzero(k!=parent)),
                A_calls=int(call[:,0].sum()),BA_calls=int(call[:,1].sum())))
    need(len(pred)==len(report29['fit_records'])==19 and
         [folds.count(f) for f in FOLDS]==[5,4,5,5],
         'not all original 19 S29 model paths reproduced')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S312_original_s29_all_19_dataset2.npz',
        row_id=ids,piece=np.asarray(pieces),fold=np.asarray(folds,dtype=np.int8),
        predicted_K=np.stack(pred).astype(np.int8),
        actions=np.stack(actions).astype(np.int8),calls=np.stack(costs).astype(np.int8))
    ret=dict(status='S312_ORIGINAL_S29_38_Q_AND_19_S20_ON_DATASET2',
        source_S29_training_run=37865173636,source_S20_training_run=37858383973,
        source_policy=FAMILY,events=n,branches=19,folds=FOLDS,
        original_S29_Q_models=38,original_S20_networks=19,
        checkpoint_evidence=proof,labels_read=False,refitted=False,
        original_S29_model_execution=True,
        external_S29_same_as_historical_piece_specific_OOF=False,
        original_S35_reconnected=False,full_S80_reconstructed=False,
        independent_accuracy_measured=False,promotion=False)
    (out/'report.json').write_text(json.dumps(ret,indent=2)+'\n')
    print(json.dumps({k:v for k,v in ret.items() if k!='checkpoint_evidence'},indent=2),flush=True)
    return ret

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s29','s20','s18','s97','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.s29,a.s20,a.s18,a.s97,a.out)
