"""S407 — SONG-HELD-OUT acoustic onset timing refinement, S80 K unchanged.

Baseline S406: place each S80 K onset at native group start or midpoint.
S407 estimates ONE onset time per group from source-only 42x49 spectral
birth/decay morphology with learned residual around its midpoint. Training
targets (median JAMS onset assigned to the group) are used ONLY for training
of other song compositions. Each test composition remains completely unseen
by its timing regressor. The K new-note count is never changed. It is only
a first timestamp experiment, not a complete per-note multiple-onset model.

Previously exposed GuitarSet S80 references mean this is NOT independent
whole-system validation. The native group can contain several attacks with
different true positions; S407 outputs one shared onset timestamp for K notes.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path,PurePosixPath
import zipfile
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.s406_score_S80_gate_native_onsets_real_GuitarSet import (
    npz,req,onset_match,calc_pred,aggregate,N,GT_TOTAL_EXPECTED,S80_CORRECT_EXPECTED)
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

SR=44100
TIME=np.arange(42,dtype=np.float64)*256/SR*1000-80
WINDOW=(TIME>=-20)&(TIME<=95)
TIMES=TIME[WINDOW]
SEED=407
def build_features(audio,k):
    """Audio + K only; NO annotations; fixed 20D descriptors."""
    a=audio['sequence42x49'];n=len(k)
    start=audio['start_samples'].astype(np.int64)
    end=audio['end_samples'].astype(np.int64)
    X=np.empty((n,18),np.float32)
    for i in range(0,n,256):
        j=min(n,i+256)
        v=np.expm1(a[i:j].astype(np.float64))
        q=v.sum(axis=2)
        flux=np.maximum(np.diff(v,axis=1),0.).sum(axis=2)
        pos=np.maximum(np.diff(v,axis=1),0.).mean(axis=2)
        onset=flux[:,1:][:,WINDOW] if False else flux[:,np.flatnonzero(WINDOW)-1]
        energy=q[:,WINDOW]
        energy=energy/(np.max(energy,axis=1,keepdims=True)+1e-5)
        onset=onset/(np.max(onset,axis=1,keepdims=True)+1e-5)
        s=onset.sum(1)+1e-5
        e=energy.sum(1)+1e-5
        onset_center=(onset @ TIMES)/s
        energy_center=(energy @ TIMES)/e
        onset_peak=TIMES[np.argmax(onset,axis=1)]
        energy_peak=TIMES[np.argmax(energy,axis=1)]
        onset_width=np.sqrt((onset*((TIMES[None,:]-onset_center[:,None])**2)).sum(1)/s)
        width=(end[i:j]-start[i:j]).astype(np.float64)/SR*1000
        # Richer dynamics of attack-vs-ringing. Fixed acoustic calculations.
        cols=[k[i:j],width,onset_center,energy_center,
           onset_peak,energy_peak,onset_width,
           onset.max(1),energy.max(1),
           energy[:,:4].mean(1),energy[:,4:8].mean(1),
           onset[:,:4].mean(1),onset[:,4:8].mean(1),
           onset[:,-4:].mean(1),
           np.log1p(v.sum((1,2))),
           np.log1p(np.maximum(v[:,1:]-v[:,:-1],0).sum((1,2))),
           np.log1p(np.maximum(v[:,:-1]-v[:,1:],0).sum((1,2))),
           np.log1p(v.max(axis=(1,2)))]
        X[i:j]=np.stack(cols,axis=1).astype(np.float32)
    req(np.isfinite(X).all() and X.shape==(n,18),'S407 source-only 18D onset refinement features')
    return X
def run(audio,ref,truth,annotations,out):
    req(not out.exists(),'immutable S407 result only')
    a=npz(audio);r=npz(ref);t=npz(truth)
    row=a['row_id'].astype(str)
    req(len(row)==N and np.array_equal(row,r['row_id'].astype(str))
       and np.array_equal(row,t['row_id'].astype(str)),
       'S407 original ID order mismatch')
    k=r['s80_K'].astype(np.int8);yK=t['true_K'].astype(np.int8)
    req(np.isin(k,range(7)).all() and int((k==yK).sum())==S80_CORRECT_EXPECTED,
        'wrong S80 reference')
    member=a['member'].astype(str)
    start=a['start_samples'].astype(np.int64)
    end=a['end_samples'].astype(np.int64)
    # Source-only feature computation before reading any JAMS annotation.
    X=build_features(a,k)
    baseline_mid=start+(end-start)//2
    baseline_start=start.copy()
    groups=np.array([composition_key(x) for x in member])
    folds=np.array([composition_fold(g) for g in groups],dtype=np.int8)
    # Note annotations used ONLY to assign training-only onset regression
    # targets and for held-composition scoring AFTER each fold predict().
    by_row=collections.defaultdict(list)
    truth_by_track={}
    total=0
    with zipfile.ZipFile(annotations,'r') as z:
        index=collections.defaultdict(list)
        for item in z.namelist():
            if item.endswith('.jams'):
                index[PurePosixPath(item).name].append(item)
        for m in sorted(set(member)):
            fn=index[PurePosixPath(m).name]
            req(len(fn)==1,'JAMS membership collision '+str(m))
            ev=read_note_events(z.read(fn[0]))
            times=np.asarray([x[0] for x in ev],np.int64)
            truth_by_track[m]=times;total+=len(times)
            ii=np.flatnonzero(member==m)
            local=ii[np.argsort(start[ii],kind='stable')]
            _,index=assign_onsets(times,start[local],end[local])
            for t0,j in zip(times,index):
                if j>=0: by_row[int(local[j])].append(int(t0))
    req(total==GT_TOTAL_EXPECTED,'original 190-track JAMS ground truth changed')
    regtarget=np.full(N,np.nan,np.float32)
    for i,gt in by_row.items():
        regtarget[i]=float(np.median(gt)-baseline_mid[i])/SR*1000.
    ready=np.isfinite(regtarget)&(k>0)
    refined=baseline_mid.astype(np.int64).copy()
    proof=[]
    out.mkdir(parents=True)
    for f in range(4):
        held=folds==f
        training=~held & ready
        test=np.flatnonzero(held&(k>0))
        req(len(test)>1000 and training.sum()>1000
           and not (set(groups[training])&set(groups[test])),
           'song-disjoint timing fold contamination')
        network=HistGradientBoostingRegressor(
           max_iter=120,max_leaf_nodes=15,min_samples_leaf=45,
           learning_rate=.06,l2_regularization=5.0,
           random_state=SEED+f,loss='absolute_error',
           early_stopping=False)
        network.fit(X[training],regtarget[training])
        lag=network.predict(X[test]) # ms relative group midpoint
        # Temporal gate bounds are fixed, not optimized on test annotations.
        candidate=baseline_mid[test]+np.rint(lag*SR/1000).astype(np.int64)
        refined[test]=np.clip(candidate,start[test]-882,end[test]+882)
        cp=out/f'S407_timing_song_fold{f}.joblib'
        joblib.dump(dict(model=network,train_feature_width=18,holdout_fold=f,
             target='median_group_onset_minus_native_midpoint_ms'),
             cp,compress=3)
        proof.append(dict(held_fold=f,training_rows=int(training.sum()),
          held_events=int(held.sum()),held_positive_K_events=int(len(test)),
          training_compositions=len(set(groups[training])),
          held_compositions=len(set(groups[test])),
          composition_overlap=0,checkpoint_sha256=sha(cp)))
    req(np.array_equal(refined[k==0],baseline_mid[k==0]),
        'K0 should be unaffected and emit zero onsets')
    scores={}
    for tol,radius in [('20ms',882),('50ms',2205)]:
        tracks=collections.defaultdict(list)
        for m,ts in truth_by_track.items():
            ii=np.flatnonzero(member==m)
            z=k[ii]
            for policy,tm in (('raw',baseline_start[ii]),
                    ('S80_start',baseline_start[ii]),
                    ('S80_mid',baseline_mid[ii]),
                    ('S407_S80_songheld_refined',refined[ii])):
                kk=np.ones(len(ii),np.int8) if policy=='raw' else z
                pp=np.repeat(tm,kk.astype(np.int64))
                tracks[policy].append(onset_match(pp,ts,radius))
        scores[tol]=aggregate(tracks)
    np.savez_compressed(out/'S407_frozen_S80_K_and_songheld_timing.npz',
          row_id=row,s80_K=k,starting_samples=start,
          midpoint_samples=baseline_mid,
          refined_onset_samples=refined,held_composition_fold=folds)
    result=dict(status='S407_REAL_GUITARSET_SONGHELD_ACOUSTIC_ONSET_TIMING',
      S80_reference='S80_S397_K5_GUARDED',tracks=len(truth_by_track),
      n_events=N,source_only_audio_features=True,
      no_trueK_model_input=True,S80_onset_count_unchanged=True,
      GT_for_training_excludes_all_held_compositions=True,
      S80_count_reference_training_overlap=True,
      same_GuitarSet_experiment_looked_at_prior_S406_results=True,
      timestamp_model_source='only 42x49 original acoustic + S80 predicted K',
      output_one_shared_onset_time_per_group=True,
      offset_not_modelled=True,
      songheld_proof=proof, metrics=scores,
      source_S80_SHA256=sha(ref),source_audio_SHA256=sha(audio),
      prediction_sha256=sha(out/'S407_frozen_S80_K_and_songheld_timing.npz'),
      official_promotion=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('S407 FOLD_PROOF',json.dumps(proof),flush=True)
    print('S407 TIMING SCORES',json.dumps(scores),flush=True)
    print('S407 PASS: four-songfold heldout learned onset timestamps; S80 K unchanged',flush=True)
def selftest():
    fake={'sequence42x49':np.zeros((5,42,49),np.float32),
          'start_samples':np.arange(5)*1764,
          'end_samples':np.arange(5)*1764+882}
    x=build_features(fake,np.array([0,1,2,3,6]))
    req(x.shape==(5,18) and np.isfinite(x).all(),
        'S407 temporal features contract')
    print('S407 SELFTEST PASS: timestamp features 18D finite, K at input')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for name in ('audio','s80','truth','annotations','out'):
        p.add_argument('--'+name,type=Path)
    v=p.parse_args()
    if v.self_test:selftest()
    else:
        for name in ('audio','s80','truth','annotations','out'):
            req(getattr(v,name) is not None,'missing '+name)
        run(v.audio,v.s80,v.truth,v.annotations,v.out)
