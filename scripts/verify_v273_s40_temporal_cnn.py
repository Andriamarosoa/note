"""Independently replay four S40 temporal CNN checkpoints and six Exact-K policies."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_s40_temporal_cnn import TemporalKNet,load_native_trajectory
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions
from scripts.train_v273_extended_dynamic_router import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,require

def sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

def run(a):
    require(not a.output.exists(),'source verification already exists')
    torch.set_num_threads(2)
    d=prepare(a)
    archive=read(a.input/'predictions.npz')
    orig=read(a.input/'temporal_probs.npz')
    meta=json.loads((a.input/'report.json').read_text())
    require(len(d['ids'])==59309 and
            np.array_equal(archive['global_index'],d['ids']) and
            np.array_equal(orig['global_index'],d['ids']),
            'different original corpus')
    require(np.array_equal(archive['true_K'],d['y']) and
            np.array_equal(archive['fold'],d['fold']),
            'native truth/fold mismatch')
    require(meta['status']=='completed' and
            meta['H9_excluded'] and len(meta['trained_models'])==4,
            'CNN provenance cannot be verified')
    old=d['parent'].astype(np.int8)
    require(metrics(d['y'],old)['correct']==49178,
            'S18 baseline changed')
    seq,_=load_native_trajectory(a.features,d)
    out=np.full((len(old),7),np.nan,np.float32)
    validated={}
    for fold in FOLDS:
        spec=next((p for p in meta['trained_models']
                    if p['fold']==fold),None)
        require(spec is not None,'missing neural fold')
        path=a.input/spec['file']
        require(path.exists(),'CNN weights missing')
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        fit=np.flatnonzero((d['fold']!=fold)&(d['y']>=1))
        test=np.flatnonzero(d['fold']==fold)
        require(ckpt['held_fold']==fold and
            np.array_equal(ckpt['train_native_ids'],d['ids'][fit]) and
            np.array_equal(ckpt['held_native_ids'],d['ids'][test]),
            'CNN training contains evaluation fold')
        require(ckpt['fit_sha256']==sha(d['ids'][fit]) and
            ckpt['held_sha256']==sha(d['ids'][test]),
            'CNN train/test native proof mismatch')
        net=TemporalKNet()
        net.load_state_dict(ckpt['state_dict'])
        net.eval()
        X=np.clip((d['X'][test]-
            ckpt['audio_mean'])/ckpt['audio_scale'],-5,5).astype(np.float32)
        S=np.clip((seq[test]-ckpt['time_mean'])/
                  ckpt['time_std'],-5,5).transpose(0,2,1).copy()
        values=[]
        with torch.no_grad():
            for start in range(0,len(test),512):
                y=net(torch.from_numpy(S[start:start+512]),
                      torch.from_numpy(X[start:start+512]))
                values.append(torch.softmax(y,dim=1).numpy())
        out[test]=np.column_stack([
            np.zeros(len(test),np.float32),np.vstack(values)])
        validated[str(fold)]=dict(fit_count=len(fit),held_count=len(test),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    require(np.allclose(out,orig['probabilities'],atol=5e-6),
            'new forward passes differ from archived CNN')
    predicted=out.argmax(axis=1).astype(np.int8)
    variants=list(map(str,archive['variant_ids']))
    require(len(variants)==7 and variants[0]=='series18_parent',
            'missing six frozen interventions')
    require(np.array_equal(archive['predictions'][:,0],old),
            'reference S18 mutated')
    for domain in ('source_K1to6','source_poly_K2to6'):
        for gate in ('argmax','delta_gt0.10','delta_gt0.25'):
            key=f'series40__temporalCNN__{domain}__{gate}'
            require(key in variants,'policy absent '+key)
            val=score_predictions(old,predicted,out,domain,gate)
            require(np.array_equal(archive['predictions'][:,variants.index(key)],
                                   val),
                    'CNN event-level decision differs '+key)
    a.output.mkdir(parents=True)
    result=dict(status='verified',trained_outer_fold_exclusion=True,
        independent_replay_all_four_neural_models=True,
        all_six_event_level_policy_vectors_exact=True,
        H9_excluded=True,not_validated_on_truly_new_compositions=True,
        no_model_promotion=True,
        checkpoint_provenance=validated)
    (a.output/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('PASS independent 4-fold CNN replay and six Exact-K decisions; H9 excluded')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
