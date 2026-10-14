"""S296: label-blind original S71/S72 trained-model runtime on native new WAV.

Run original S224 four S71 risk heads on S223 genuine S70 interventions,
then original S225 four S72 temporal veto heads. 46D geometry rebuilt
EXACTLY with original S82 ownership formula (42 time offsets + 4 stats).
Each original historical held-fold model retains its trained threshold.
For unseen songs a NEW identity-neutral rule applies 3-of-4 veto quorum;
all original model weights unchanged, no truth opened. Real S70 parent
may produce ZERO eligible changes: then downstream abstains, as it must.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,joblib
from scripts.s82_external_pr16_neural_inference import ownership
from scripts.audit_v273_s71_acoustic_risk import EDGES
from scripts.audit_v273_s72_temporal_poly_guard import GROUP
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','midi','target','correct','regression'}
def need(v,msg):
    if not v:raise ValueError(msg)
def load(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(fields) and not BAD.intersection(z.files),
             'S296 strict label-free source only '+str(path))
        return {k:z[k].copy() for k in z.files}
def geometry46(audio):
    """S82 geometry: ownership mask at the 42 original PR16 timestamps."""
    members=audio['member'].astype(str)
    starts=audio['start_samples'].astype(np.int64)
    ends=audio['end_samples'].astype(np.int64)
    n=len(starts);geom=np.empty((n,46),np.float32)
    times=np.arange(42)*256*1000./44100.-80.
    offset=np.rint(times*44100./1000.).astype(np.int64)
    for member in sorted(set(members)):
        ordered=np.flatnonzero(members==member)
        ordered=ordered[np.argsort(starts[ordered],kind='stable')]
        s=starts[ordered];e=ends[ordered]
        need(np.all(s[1:]>e[:-1]),'original native PR16 event overlap: '+member)
        for local,i in enumerate(ordered):
            query=int(s[local])+offset
            mask=ownership(s,e,local,query)
            before=(s[local]-e[local-1])/10584. if local else 1.
            after=(s[local+1]-s[local])/10584. if local+1<len(s) else 1.
            geom[i]=np.r_[mask,(e[local]-s[local])/1764.,
                np.clip(before,0,1),np.clip(after,0,1),mask.mean()]
    need(np.isfinite(geom).all(),'S82 ownership geometry NaN')
    return geom
def temporal(summary,sequence):
    t=sequence
    need(t.ndim==3 and t.shape[1:]==(42,49),'original temporal trajectory length')
    x=np.column_stack([summary,t[:,:6].mean(1),t[:,6:14].mean(1),
        t[:,14:28].mean(1),t[:,28:].mean(1),t.std(1),t.max(1),
        t[:,-1]-t[:,0],np.abs(np.diff(t,axis=1)).mean(1)]).astype(np.float32)
    need(x.shape==(len(t),450),'original S72 temporal view wrong dimension')
    return x
def run(s69file,s70file,s97file,s71models,s72models,out):
    need(not out.exists(),'immutable S296 result')
    a=load(s69file,{'row_id','S18_parent_K','balanced_s46_K',
        'balanced_protected_K','polymax_s46_K','polymax_protected_K',
        'balanced_fold_veto','balanced_veto_three_folds',
        'polymax_fold_veto','polymax_veto_three_folds'})
    b=load(s70file,{'row_id','s69_polymax_K','s69_balanced_K','s18_K',
        'proposed_poly_K','proposed_poly_fold_K','proposed_poly_fold_active',
        'proposed_poly_vote_accepted','downward_poly_K','downward_poly_fold_K',
        'downward_poly_fold_active','downward_poly_vote_accepted'})
    audio=load(s97file,{'row_id','summary58','flow245','geometry3',
        'sequence42x49','morphology490','physics294','member',
        'start_samples','end_samples'})
    ids=a['row_id'];n=len(ids)
    need(n==506 and np.array_equal(ids,b['row_id']) and
         np.array_equal(ids,audio['row_id']) and
         np.array_equal(b['s69_polymax_K'],a['polymax_protected_K']),
         'genuine 506 S69 S70 S97 native IDMT event alignment')
    parent=a['polymax_protected_K'].astype(np.int8)
    s70=b['proposed_poly_K'].astype(np.int8)
    summary=audio['summary58'].astype(np.float32)
    sequence=audio['sequence42x49'].astype(np.float32)
    geom=geometry46(audio)
    need(summary.shape==(n,58) and geom.shape==(n,46),'original PR16 geometry source')
    s71input=np.column_stack([summary,geom]).astype(np.float32)
    t450=temporal(summary,sequence)
    changed=np.flatnonzero(s70!=parent)
    fold_veto71=np.zeros((4,n),bool)
    modelproof71=[]
    if len(changed):
        old=parent[changed];new=s70[changed]
        X=np.column_stack([s71input[changed],np.eye(7)[old],
                           np.eye(7)[new],new-old]).astype(np.float32)
        need(X.shape==(len(changed),119),'original S71 119D model')
    for fi,fold in enumerate(FOLDS):
        path=s71models/'models'/f'original_S71_risk_heldfold{fold}.joblib'
        st=joblib.load(path)
        need(st['original_fold']==fold and st['original_feature_width']==119,
             'original S71 saved checkpoint identity/feature-width')
        modelproof71.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if not len(changed):continue
        prob=st['model'].predict_proba(X)[:,1]
        for source,target in EDGES:
            key=f'{source}->{target}'
            cut=float(st['edge_cuts'][key])
            mask=(old==source)&(new==target)&(prob>=cut)
            fold_veto71[fi,changed[mask]]=True
    veto71=fold_veto71.sum(0)>=3
    s71risk=s70.copy();s71risk[veto71]=parent[veto71]
    veto71poly=veto71&~((parent==3)&(s70==2))
    s71poly=s70.copy();s71poly[veto71poly]=parent[veto71poly]
    positions=np.flatnonzero(s71risk!=parent)
    fold_veto72=np.zeros((4,n),bool)
    modelproof72=[]
    if len(positions):
        old=parent[positions];new=s71risk[positions]
        x=np.column_stack([t450[positions],np.eye(7)[old],
                           np.eye(7)[new],new-old]).astype(np.float32)
        need(x.shape==(len(positions),465),'original S72 465D model')
    for fi,fold in enumerate(FOLDS):
        path=s72models/'models'/f'original_S72_temporal_veto_fold{fold}.joblib'
        st=joblib.load(path)
        need(st['original_held_fold']==fold and
             st['original_trained_feature_width']==465,
             'original S72 saved model schema')
        modelproof72.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if not len(positions) or st['threshold'] is None:continue
        info=st['model_info']
        if 'model' in info:
            scores=info['model'].predict_proba(x)[:,1]
        else:scores=np.full(len(positions),info['constant_probability'])
        group=np.logical_or.reduce([(old==i)&(new==j) for i,j in GROUP])
        fold_veto72[fi,positions[group&(scores>=st['threshold'])]]=True
    veto72=fold_veto72.sum(0)>=3
    poly_veto=veto72&(parent>=2)
    result={}
    for label,src in (('risk',s71risk),('poly',s71poly)):
        for suffix,gate in (('all',veto72),('poly_guard',poly_veto)):
            outK=src.copy();outK[gate]=parent[gate]
            result[f's72_{label}_{suffix}_K']=outK
    out.mkdir(parents=True)
    np.savez_compressed(out/'S296_original_S71_S72_native_IDMT.npz',
        row_id=ids,s69_polymax_K=parent,s70_proposed_poly_K=s70,
        s71_risk_K=s71risk,s71_poly_K=s71poly,
        s71_veto_three_folds=veto71,
        s72_veto_three_folds=veto72,
        s71_fold_veto=fold_veto71,s72_fold_veto=fold_veto72,
        original_pr16_geometry46=geom,**result)
    rep=dict(status='S296_ORIGINAL_S71_S72_RECOVERED_TRAINED_CHECKPOINTS_IDMT',
        events=n,real_original_model_count=8,
        model_S71_count=4,model_S72_count=4,
        original_S71_input_width=119,original_S72_input_width=465,
        real_S70_source_eligible_actions=len(changed),
        post_S71_changed_K=int(np.count_nonzero(s71risk!=parent)),
        post_S72_changed_K=int(np.count_nonzero(result['s72_poly_poly_guard_K']!=parent)),
        label_free_S82_geom46_recomputed=True,
        thresholds_retained_as_historical=True,
        original_model_sha_S71=modelproof71,original_model_sha_S72=modelproof72,
        adaptation='3/4 fold veto without historic held-piece oracle',
        original_historical_outofpiece_identity_not_claimed=True,
        new_external_audio_accuracy_measured=False,original_S80_tail_not_yet_executed=True,
        labels_read=False,new_models_trained=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rep.items()
          if not k.startswith('original_model_sha_')},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('s69','s70','s97','s71','s72','out'):
        ap.add_argument('--'+k,type=Path,required=True)
    q=ap.parse_args()
    run(q.s69,q.s70,q.s97,q.s71,q.s72,q.out)
