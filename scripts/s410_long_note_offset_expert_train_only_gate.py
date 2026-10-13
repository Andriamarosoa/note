"""S410 — Frozen S408 Offset + train-only LONG NOTE correction gate.

S409 diagnostics: S408 scores no ~1s notes within ±50ms, so evaluate a
conditional long-sustain expert. Features are ONLY prior-source predictions
S80 K, S402 MIDI pitch, S407 onset, S408 predicted durations; no raw
labels or hidden GT are used to generate any heldout features.
For each outer composition fold, a separate INTERNAL composition split
selects a classifier threshold based on offset-50ms correctness; that
threshold is fixed before inspecting its heldout fold.

Crucial: frozen S408 has already been developed on GuitarSet. This new
experiment is posthoc development, not a prospective independent benchmark.
No official S80/S402/S407/S408 reference weights or outputs change.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference,metrics_note_offset)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

SR=44100
SEED=410
N=33475
LONG_MS=500
THRESHOLDS=(.10,.15,.20,.25,.30,.35,.40,.50,.60)
def require(x,s):
    if not x:raise ValueError(s)
def input_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def create_features(p):
    n=len(p['S402_midi_pitch'])
    pitch=p['S402_midi_pitch'].astype(int)
    onset=p['S407_onset_samples'].astype(np.int64)
    k=p['S80_K'].astype(np.int8)
    base=(p['S408_offset_samples']-onset)/SR*1000
    blind=(p['no_future_offset_samples']-onset)/SR*1000
    median=(p['median_pitch_offset_samples']-onset)/SR*1000
    globalmed=(p['median_global_offset_samples']-onset)/SR*1000
    track=p['track_member'].astype(str)
    rows=np.zeros((n,23),np.float32)
    rowtime=np.zeros((n,6),np.float32)
    for m in sorted(set(track)):
        ix=np.flatnonzero(track==m)
        ts=onset[ix]
        order=np.argsort(ts,kind='stable')
        sorted_t=ts[order]
        note_pitch=pitch[ix[order]]
        # After a note onset, simultaneous notes are not subsequent events.
        for ix0 in ix:
            t=int(onset[ix0]);midi=int(pitch[ix0])
            nxt=np.searchsorted(sorted_t,t,side='right')
            prev=np.searchsorted(sorted_t,t,side='left')-1
            dt_next_any=(sorted_t[nxt]-t)/SR*1000 if nxt<len(sorted_t) else 5000
            dt_prev_any=(t-sorted_t[prev])/SR*1000 if prev>=0 else 5000
            following=sorted_t[nxt:]
            following_p=note_pitch[nxt:]
            q=np.flatnonzero(following_p==midi)
            dt_next_same=(following[q[0]]-t)/SR*1000 if len(q) else 5000
            rowtime[ix0]=[min(5000,dt_next_any),
                min(5000,dt_prev_any),min(5000,dt_next_same),
                np.searchsorted(sorted_t,t+int(.25*SR),side='right')-nxt,
                np.searchsorted(sorted_t,t+int(.75*SR),side='right')-nxt,
                np.searchsorted(sorted_t,t+int(2.*SR),side='right')-nxt]
    # Predicted-only features, no reference durations or GT K.
    X=np.column_stack([
        k/6., pitch/88.,(pitch%12)/12.,pitch//12/7.,
        np.log1p(base),np.log1p(blind),np.log1p(median),np.log1p(globalmed),
        base-blind,base-median,base/globalmed,base/(blind+1.),
        (base>500).astype(float),(blind>500).astype(float),
        (base>1000).astype(float),np.log1p(np.abs(base-blind)),
        np.log1p(rowtime[:,:3]),
        rowtime[:,3:],
        np.abs(np.sin(pitch*np.pi/6.))
    ]).astype(np.float32)
    require(X.shape==(n,23) and np.isfinite(X).all(),
            'S410 features shape or validity failed')
    return X,base
def read_labels(p,annotations):
    track=p['track_member'].astype(str)
    on=p['S407_onset_samples'].astype(np.int64)
    pitch=p['S402_midi_pitch'].astype(int)
    ms=np.full(len(pitch),np.nan,np.float32)
    offset=np.full(len(pitch),-1,np.int64)
    labels={}
    with zipfile.ZipFile(annotations,'r') as z:
        lookup=collections.defaultdict(list)
        for item in z.namelist():
            if item.endswith('.jams'):lookup[PurePosixPath(item).name].append(item)
        for m in sorted(set(track)):
            paths=lookup[PurePosixPath(m).name]
            require(len(paths)==1,'GuitarSet unique JAMS basename failed')
            labels[m]=get_gt(z.read(paths[0]))
    require(len(labels)==190 and sum(len(v) for v in labels.values())==34954,
            'S410 dataset label identity changed')
    for m,gt in labels.items():
        ii=np.flatnonzero(track==m)
        pred=[(int(on[i]),int(pitch[i]),int(i)) for i in ii]
        for u,g in match_predicted_to_reference(pred,gt,2205):
            idx=pred[u][2]
            onset,off,gt_pitch,string=gt[g]
            ms[idx]=(off-onset)/SR*1000
            offset[idx]=off
    require(int(np.isfinite(ms).sum())==25120,
            'S408 observed 25120 pitch+onset matched note labels drift')
    return ms,offset,labels

def fit_experts(X,known,real_duration,fold):
    require(int(known.sum())>3000,'insufficient S410 train labels')
    long=real_duration[known]>=LONG_MS
    require(int(long.sum())>500 and int((~long).sum())>3000,
            'S410 long note class missing')
    cls=HistGradientBoostingClassifier(
       max_iter=90,max_leaf_nodes=12,min_samples_leaf=65,
       learning_rate=.05,l2_regularization=5.0,
       early_stopping=False,random_state=SEED+fold)
    cls.fit(X[known],long.astype(int))
    reg=HistGradientBoostingRegressor(
       max_iter=90,max_leaf_nodes=10,min_samples_leaf=35,
       learning_rate=.05,l2_regularization=3.0,
       loss='squared_error',early_stopping=False,
       random_state=SEED+fold)
    reg.fit(X[known][long],np.log(real_duration[known][long]))
    return cls,reg

def correct(base_on,base_off,X,cl,reg,threshold):
    prob=cl.predict_proba(X)[:,1]
    longdur=np.clip(np.exp(reg.predict(X)),500.,4000.)
    base=(base_off-base_on)/SR*1000
    proposed=np.maximum(base,longdur)
    corrected=base_on+np.rint(proposed*SR/1000).astype(np.int64)
    on=(prob>=threshold)&(proposed>base)
    return np.where(on,corrected,base_off),prob,on

def run(source,annotations,out):
    require(not out.exists(),'S410 immutable output path')
    p=input_npz(source)
    n=len(p['S402_midi_pitch'])
    require(n==N and len(p['S80_K'])==n and
        np.isin(p['S80_K'],range(1,7)).all(),
        'S410 source cardinalities')
    X,base_ms=create_features(p)
    true_dur,true_off,labels=read_labels(p,annotations)
    rowtrack=p['track_member'].astype(str)
    comps=np.array([composition_key(x) for x in rowtrack])
    foldid=np.array([composition_fold(x) for x in comps],np.int8)
    require(np.array_equal(foldid,p['composition_fold'].astype(np.int8)),
            'S410 same held-song folds changed')
    source_on=p['S407_onset_samples'].astype(np.int64)
    source_off=p['S408_offset_samples'].astype(np.int64)
    gate_off=source_off.copy()
    chosen=[]
    out.mkdir(parents=True)
    for f in range(4):
        held=foldid==f
        train=(foldid!=f)&np.isfinite(true_dur)
        # Pre-declared composition split inside outer training: different
        # GuitarSet pieces, not random held note events.
        inner_val=train&np.array([
            int(hashlib.sha256(('S410-inner-'+str(c)).encode()).hexdigest()[:8],16)%5==0
            for c in comps])
        inner_train=train&~inner_val
        require(inner_train.sum()>3000 and inner_val.sum()>500
          and not (set(comps[inner_train])&set(comps[inner_val]))
          and not (set(comps[train])&set(comps[held])),
          'S410 inner/outer composition separation broken')
        model,expert=fit_experts(X,inner_train,true_dur,f+10)
        # Each candidate threshold is scored exclusively on INNER train's
        # held compositions, never on the final held outer composition.
        prob=model.predict_proba(X[inner_val])[:,1]
        longdur=np.clip(np.exp(expert.predict(X[inner_val])),500,4000)
        d=(source_off[inner_val]-source_on[inner_val])/SR*1000
        candidates=np.maximum(d,longdur)
        offset_baseline=source_off[inner_val]
        gt=true_off[inner_val]
        inner_results={}
        for threshold in THRESHOLDS:
            changed=np.where((prob>=threshold)&(candidates>d),
                  source_on[inner_val]+np.rint(candidates*SR/1000).astype(np.int64),
                  offset_baseline)
            hits=int(np.sum(np.abs(changed-gt)<=2205))
            inner_results[str(threshold)]=hits
        base_hits=int(np.sum(np.abs(offset_baseline-gt)<=2205))
        best=sorted(THRESHOLDS,key=lambda t:(-inner_results[str(t)],t))[0]
        active=inner_results[str(best)]>base_hits
        threshold=best if active else 1.1
        # Train-only selected threshold is now FROZEN. Fit model against all
        # other compositions and apply to outer TEST composition only.
        network,reg=fit_experts(X,train,true_dur,f+30)
        sel=held
        outoffs,pr,actions=correct(source_on[sel],source_off[sel],X[sel],
                                    network,reg,threshold)
        gate_off[sel]=outoffs
        cp=out/f'S410_long_note_expert_held_song_fold{f}.joblib'
        joblib.dump(dict(classifier=network,long_duration_model=reg,
            K0_cannot_generate_onset=True,chosen_threshold=threshold,
            composition_fold_heldout=f,internal_threshold_only=True),
            cp,compress=3)
        chosen.append(dict(fold=f,train=int(train.sum()),
          inner_train=int(inner_train.sum()),inner_valid=int(inner_val.sum()),
          internal_baseline_50ms=base_hits,internal_gate_50ms=inner_results[str(best)],
          threshold=float(threshold),correction_enabled=bool(active),
          corrected_test_notes=int(actions.sum()),
          outer_test_candidates=int(sel.sum()),
          train_compositions=len(set(comps[train])),
          outer_compositions=len(set(comps[held])),
          composition_overlap=0,checkpoint_sha256=sha(cp)))
    require(np.all(gate_off>=source_off) and np.all(gate_off>source_on),
            'S410 is a long-note-only correction; cannot shorten S408 Offsets')
    baseline=metrics_note_offset(rowtrack,p['S402_midi_pitch'],
                                 source_on,source_off,labels)
    result=metrics_note_offset(rowtrack,p['S402_midi_pitch'],
                                 source_on,gate_off,labels)
    require(baseline['offset_50ms_correct_given_note_onset_pitch']==13050 and
       result['note_onset_pitch_matched']==25120,
       'S410 offset baseline/integrity drift')
    duration_segments={}
    mat=collections.defaultdict(list)
    for m,gt in labels.items():
        ii=np.flatnonzero(rowtrack==m)
        pred=[(int(source_on[i]),int(p['S402_midi_pitch'][i]),int(i)) for i in ii]
        for j,q in match_predicted_to_reference(pred,gt,2205):
            u=pred[j][2];refstart,refend,_,_=gt[q]
            du=(refend-refstart)/SR*1000
            key=('<100ms' if du<100 else '100-499ms' if du<500 else
                 '500-999ms' if du<1000 else '1000ms+')
            mat[key].append((abs(int(source_off[u])-refend)<=2205,
                             abs(int(gate_off[u])-refend)<=2205,
                             abs(int(source_off[u])-refend)/SR*1000,
                             abs(int(gate_off[u])-refend)/SR*1000))
    for group,vals in mat.items():
        duration_segments[group]=dict(n=len(vals),
             S408_correct=sum(v[0] for v in vals),
             S410_correct=sum(v[1] for v in vals),
             corrections=sum(v[1] and not v[0] for v in vals),
             regressions=sum(v[0] and not v[1] for v in vals),
             median_S408_AE_ms=float(np.median([v[2] for v in vals])),
             median_S410_AE_ms=float(np.median([v[3] for v in vals])))
    npz=out/'S410_held_composition_LONG_NOTE_offset_gate_N33475.npz'
    np.savez_compressed(npz,row_id=p['source_row_id'],track_member=rowtrack,
        composition_fold=foldid,S80_K=p['S80_K'],
        midi_pitch=p['S402_midi_pitch'],onset_samples=source_on,
        S408_offset_samples=source_off,S410_offset_samples=gate_off)
    report=dict(status='S410_S408_LONG_NOTE_OFFSET_EXPERT_HELD_COMPOSITION',
       scope='GuitarSet development, not independent complete system',
       S408_source_SHA256=sha(source),GuitarSet_JAMS_SHA256=sha(annotations),
       count_and_note_references_untouched=True,
       K0_cannot_create_new_notes=True,
       every_S410_offset_is_after_S407_onset=True,
       train_only_per_fold_nested_composition_threshold=True,
       no_held_song_target_for_fitting_or_threshold=True,
       supervised_from_only_correctly_matched_training_notes=True,
       no_new_trueK_or_true_note_pitch_model_inputs=True,
       original_S408_frozen_baseline=baseline, measured_S410=result,
       per_duration_segments=duration_segments,fold_proofs=chosen,
       source_prediction_sha256=sha(npz),official_promotion=False,
       test_labels_previously_exposed_in_prior_S409_audit=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S410 FOLDS',json.dumps(chosen),flush=True)
    print('S410 RESULTS',json.dumps(dict(old=baseline,new=result,by_duration=duration_segments)),flush=True)
    print('S410 PASS: long-note train-only gate measured on 4 composition-held folds',flush=True)
def selftest():
    a={'S402_midi_pitch':np.array([48,60,63],np.int16),
      'S407_onset_samples':np.array([1000,900,2500],np.int64),
      'S80_K':np.array([1,2,2],np.int8),
      'S408_offset_samples':np.array([3000,7000,6000],np.int64),
      'no_future_offset_samples':np.array([2500,5500,7500],np.int64),
      'median_pitch_offset_samples':np.array([3500,9000,8500],np.int64),
      'median_global_offset_samples':np.array([2900,8000,8100],np.int64),
      'track_member':np.array(['a','a','a'])}
    x,d=create_features(a)
    require(x.shape==(3,23) and np.isfinite(x).all(),'S410 feature-only input contract')
    print('S410 SELFTEST PASS: source-only 23D model features, no GT leakage')
if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--self-test',action='store_true')
    for n in ('source','annotations','out'):arg.add_argument('--'+n,type=Path)
    z=arg.parse_args()
    if z.self_test:selftest()
    else:
        for n in ('source','annotations','out'):
            require(getattr(z,n) is not None,'missing --'+n)
        run(z.source,z.annotations,z.out)
