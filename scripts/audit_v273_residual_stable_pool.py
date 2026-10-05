"""Measure only the stable F0 tie-order change on frozen internal populations."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.v273_residual_audit import (
    FOLDS,FEATURES,analyze,fit_classifier,require,sha256_file,verify_export,write_json,
)


def run(a):
    require(not a.output.exists(),'refusing overwrite')
    inputs={};records={};old_values=defaultdict(list)
    for fold in FOLDS:
        folder=a.exports/f'fold-{fold}'
        verify_export(folder)
        with np.load(folder/'replay.npz',allow_pickle=False) as z:arr=dict(z)
        inputs[fold]=(arr,json.loads((folder/'report.json').read_text()))
        for split in ['fit','val']:
            selected=arr[split+'_b_low']&(arr[split+'_base_k']==3)
            good=arr[split+'_valid']
            ids=arr[split+'_ids'][selected][good]
            members=arr[split+'_recording'][selected][good]
            starts=arr[split+'_start_sample'][selected][good]
            folds=arr[split+'_fold'][selected][good]
            X=arr[split+'_X'][good]
            for i,member,start,f,x in zip(ids,members,starts,folds,X):
                require(int(f) in FOLDS,'outer fold')
                value=(str(member),int(start))
                require(int(i) not in records or records[int(i)]==value,'row identity changed')
                records[int(i)]=value;old_values[int(i)].append(x.copy())
    wanted={m for m,s in records.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(wanted==set(tracks),'missing audio')
    computed={}
    for member in sorted(wanted):
        t=tracks[member]
        audio=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
        samples=np.asarray(audio.samples,np.float64)/32768
        for row,(m,start) in records.items():
            if m!=member:continue
            freq,x=h.transition_spectrum(samples,start)
            features=h.extract_one(freq,x)
            require(features is not None,'validity changed')
            computed[row]=[features[k] for k in (*FEATURES,'median_triplet_f0')]
        print(member,len(computed),flush=True)
    a.output.mkdir(parents=True)
    ids=np.array(sorted(computed),dtype=np.int64)
    values=np.array([computed[int(i)] for i in ids])
    np.savez_compressed(a.output/'stable-features.npz',row_id=ids,features=values)
    outputs={};folds=[]
    for fold,(arr,previous) in inputs.items():
        data=[]
        old_X=[]
        for split in ['fit','val']:
            selected=arr[split+'_b_low']&(arr[split+'_base_k']==3)
            good=arr[split+'_valid'];row_ids=arr[split+'_ids'][selected][good]
            y=arr[split+'_true_k'][selected][good]
            new=np.array([computed[int(i)] for i in row_ids])
            data.extend([new[:,:2],y,new[:,2]])
            old_X.append(arr[split+'_X'][good])
        train=np.isin(data[1],(2,3))
        control=fit_classifier(old_X[0][train],(data[1][train]==2).astype(int))
        p_control=control.predict_proba(old_X[1])[:,1]
        err=float(np.max(np.abs(p_control-arr['val_probability'])))
        require(err<1e-10,'unchanged-feature refit drift')
        result,state,pred=analyze(*data)
        write_json(a.output/f'model-fold-{fold}.json',state)
        for name,value in pred.items():outputs[f'fold_{fold}_{name}']=value
        folds.append({'fold':fold,'control_refit_max_abs_probability_error':err,
                      'old':previous['global_action'],'stable':result['global_action'],
                      'old_auc':previous['joint']['val_auc'],'stable_auc':result['joint']['val_auc'],
                      'changed_validation_decisions':int(np.sum((pred['val_probability']>=.5)!=(arr['val_probability']>=.5))),
                      'changed_fit_features':int(np.sum(np.max(np.abs(data[0]-old_X[0]),axis=1)>1e-8)),
                      'changed_val_features':int(np.sum(np.max(np.abs(data[3]-old_X[1]),axis=1)>1e-8))})
    np.savez_compressed(a.output/'scores.npz',**outputs)
    report={'status':'completed','source_run':37356100423,'outer_fold_3_used':False,
            'neural_training':False,'diagnostic_lr_refit':True,
            'single_change':'stable increasing-grid-order tie break in F0 salience sorting before NMS',
            'unique_audio_rows':len(records),'recordings':len(wanted),'folds':folds,
            'source_inconsistent_rows':sum(np.max(np.ptp(np.array(xs),axis=0))>1e-8 for xs in old_values.values()),
            'totals':{arm:{k:sum(f[arm][k] for f in folds) for k in
                          ['applied','corrections','regressions','other_k_actions','global_net']} for arm in ['old','stable']},
            'source_sha256':{p:sha256_file(Path(__file__).parent/p) for p in
                             ['audit_v273_residual_stable_pool.py','audit_v273_internal_b_low_harmonic_strata.py']},
            'automatic_promotion':False}
    report['source_inconsistent_rows']=int(report['source_inconsistent_rows'])
    write_json(a.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['exports','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
