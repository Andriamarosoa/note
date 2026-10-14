"""S408 — Real per-note Offset supervision and held-composition evaluation.

Inputs are ORIGINAL GuitarSet source-only S258 42x49 time-pitch trajectories,
frozen S80 K, S402 song-held note MIDI, S407 song-held onset timestamp.
Model predicts DURATION (thus NoteOff), conditional on each predicted
MIDI pitch and longitudinal, *offline* event features.

The training targets are durations from JAMS note_midi annotations paired
to predicted notes only in TRAINING compositions, never exposed for the
held-out composition. No GT true K/pitch/duration is read into inference.
The preceding S80 count itself has GuitarSet training overlap, hence these
are DEVELOPMENT diagnostics, not external generalization proof.

No full waveform streaming is performed. Future sparse acoustic event frames
provide intermittent observations, not a true continuous spectrogram.
This is a supervised experiment, not promoted live offset.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import midi_value
from scripts.s406_score_S80_gate_native_onsets_real_GuitarSet import npz,req,N
from scripts.s402_k_input_first_supervised_midi_pitch_network import composition_key,composition_fold
from scripts.yourmt3_exactk_common import assign_onsets

SR=44100
HORIZONS_MS=(120,250,400,650,950,1450,2200)
MAX_GAP_MS=170
RANGE=49
NUM_FEATURES=29+len(HORIZONS_MS)*4
MIN_DURATION_MS=30
MAX_DURATION_MS=4000
SEED=408
def get_gt(raw):
    doc=json.loads(raw.decode('utf-8'))
    notes=[]
    for ann in doc.get('annotations',[]):
        if ann.get('namespace')!='note_midi':continue
        meta=ann.get('annotation_metadata') or {}
        slot=str(meta.get('data_source'))
        req(slot in set(map(str,range(6))),'GuitarSet physical slot invalid')
        for ob in ann['data']:
            onset=float(ob['time']);duration=float(ob['duration'])
            req(np.isfinite(onset) and onset>=0 and
                 np.isfinite(duration) and duration>0,'invalid JAMS onset duration')
            s=int(round(onset*SR));end=int(round((onset+duration)*SR))
            req(end>s,'invalid JAMS NoteOff timestamp')
            notes.append((s,end,midi_value(ob['value']),int(slot)))
    return sorted(notes)

def build_candidates(a,b,c,r):
    ids=a['row_id'].astype(str);n=len(ids)
    req(n==N and np.array_equal(ids,b['row_id'].astype(str))
        and np.array_equal(ids,c['row_id'].astype(str))
        and np.array_equal(ids,r['row_id'].astype(str)),
        'S408 acoustic / S402 / S407 / S80 original row_id mismatch')
    k=r['s80_K'].astype(np.int8)
    req(np.array_equal(k,b['s80_K']) and np.array_equal(k,c['s80_K']),
        'S408 original S80 K drift across source heads')
    slots=b['midi_pitch_slots'].astype(np.int16)
    times=c['refined_onset_samples'].astype(np.int64)
    req(slots.shape==(N,6) and
        np.array_equal((slots>=0).sum(1),k.astype(np.int64)) and
        np.array_equal((slots>=0).sum(1),c['s80_K']) and
        np.isin(slots[slots>=0],range(40,89)).all(),
        'S408 source note K / MIDI width mismatch')
    candidates=np.flatnonzero(slots>=0)
    row,slot=np.unravel_index(candidates,slots.shape)
    return ids,row.astype(np.int32),slot.astype(np.int8),slots[row,slot].astype(np.int16),times[row],k[row]

def build_features(a,row,pitch,start,k,score=None):
    """Build label-blind future acoustic descriptors; no note annotations.

    Audio windows centred on ORIGINAL event candidates are not contiguous
    surveillance. Mark missing future spectral observations explicitly.
    """
    member=a['member'].astype(str)
    native_start=a['start_samples'].astype(np.int64)
    native_end=a['end_samples'].astype(np.int64)
    req(len(row)==len(pitch)==len(start)==len(k),'S408 candidate feature rows invalid')
    amp=np.expm1(a['sequence42x49'].astype(np.float32))
    req(amp.shape==(N,42,49) and np.isfinite(amp).all(),
        'S408 original spectral activation source missing')
    before=amp[:,:13].mean(1)
    during=amp[:,13:24].mean(1)
    later=amp[:,24:].mean(1)
    birth=np.maximum(np.diff(amp,axis=1),0).sum(1)
    decay=np.maximum(-np.diff(amp,axis=1),0).sum(1)
    peak=amp.max(axis=1)
    kO=np.array(k,np.int8)
    features=np.zeros((len(row),NUM_FEATURES),np.float32)
    track_index={}
    for m in np.unique(member):
        ii=np.flatnonzero(member==m)
        order=ii[np.argsort(native_start[ii],kind='stable')]
        track_index[m]=(native_start[order],order)
    # Candidate-output pitch stays fixed across the note's acoustic lifetime.
    # No GT feature extraction. Need future audio up to 2.2 sec, offline only.
    score=np.asarray(score,np.float32) if score is not None else np.zeros((N,6),np.float32)
    req(score.shape==(N,6),'S402 onset pitch confidence expected Nx6')
    # Fast per-track near-neighbor source spectral lookup
    for n,(i,p,t,kk) in enumerate(zip(row,pitch,start,kO)):
        j=int(p)-40;ii=int(i)
        track=str(member[ii])
        group_start,sorted_ids=track_index[track]
        selfpre=float(before[ii,j]);selfon=float(during[ii,j]);selflate=float(later[ii,j])
        rowvals=[
          float(j)/48, float(kk)/6,
          float(np.log1p(max(0,t-native_start[ii])/SR*1000)),
          float((native_end[ii]-native_start[ii])/SR*1000/40),
          0.,  # reserved S402 confidence slot; explicitly absent from this ablation
          np.log1p(selfpre),np.log1p(selfon),np.log1p(selflate),
          np.log1p(float(birth[ii,j])),np.log1p(float(decay[ii,j])),
          np.log1p(float(peak[ii,j])),
          selflate/(selfon+1e-4),
          selfpre/(selfon+1e-4),
          float(during[ii,j]/(during[ii].sum()+1e-4)),
          float(birth[ii,j]/(birth[ii].sum()+1e-4)),
          float(decay[ii,j]/(decay[ii].sum()+1e-4)),
          float(np.log1p(during[ii].sum())),
          float(np.log1p(later[ii].sum())),
          float(np.log1p(birth[ii].sum())),
          float(np.log1p(decay[ii].sum())),
          float(np.log1p(peak[ii].max())),
          float(j%12)/12.,
          float(j//12)/4.,
        ]
        # 6 extra source-only neighboring spectral pitch descriptors
        for delta in (-12,-7,-1,1,7,12):
            x=j+delta
            rowvals.append(np.log1p(float(during[ii,x])) if 0<=x<49 else 0.)
        req(len(rowvals)==29,'S408 acoustic base descriptor dimensions changed')
        for h in HORIZONS_MS:
            target=int(t+int(h*SR/1000))
            pos=int(np.searchsorted(group_start,target))
            opts=([pos-1] if pos>0 else [])+([pos] if pos<len(group_start) else [])
            chosen=min(opts,key=lambda q:abs(int(group_start[q])-target)) if opts else None
            if chosen is not None and abs(int(group_start[chosen])-target)<=MAX_GAP_MS*SR/1000:
                o=int(sorted_ids[chosen])
                rowvals.extend([
                    1.,
                    np.log1p(float(during[o,j])),
                    np.log1p(float(later[o,j])),
                    float(np.log1p(birth[o,j]))])
            else:rowvals.extend([0.,0.,0.,0.])
        features[n]=np.asarray(rowvals,np.float32)
    req(np.isfinite(features).all(),'NaN/inf in source-only S408 future features')
    return features

def match_predicted_to_reference(predictions,ground_truth,radius=2205):
    """One-to-one by pitch and onset; used ONLY for evaluation/training labels.
    Each predicted note and each ground-truth note used at most once.
    """
    matches=[]
    bypitch_p=collections.defaultdict(list)
    bypitch_g=collections.defaultdict(list)
    for index,(on,pitch,_) in enumerate(predictions):
        bypitch_p[int(pitch)].append((int(on),index))
    for index,(on,off,pitch,slot) in enumerate(ground_truth):
        bypitch_g[int(pitch)].append((int(on),index))
    for pitch in sorted(set(bypitch_p)|set(bypitch_g)):
        ps=sorted(bypitch_p[pitch]);gs=sorted(bypitch_g[pitch])
        u=v=0
        while u<len(ps) and v<len(gs):
            diff=ps[u][0]-gs[v][0]
            if abs(diff)<=radius:
                matches.append((ps[u][1],gs[v][1]))
                u+=1;v+=1
            elif diff< -radius:u+=1
            else:v+=1
    return matches

def metrics_note_offset(member,pitch,onset,offset,annotations):
    """Count note identity+onset matched first; then NoteOff success."""
    predicted={}
    for i,m in enumerate(member):
        predicted.setdefault(str(m),[]).append(i)
    tp_note=0;offs50=0;offs_rel=0;sum_abs=0
    dur_errors=[]
    byK=collections.defaultdict(lambda:[0,0,0])
    gt_total=sum(len(v) for v in annotations.values())
    for m,allix in predicted.items():
        items=[(int(onset[i]),int(pitch[i]),int(i)) for i in allix]
        gt=annotations[m]
        matches=match_predicted_to_reference(items,gt,2205)
        tp_note+=len(matches)
        for pi,gi in matches:
            predindex=items[pi][2]
            start0,stop0,p,slot=gt[gi]
            abserr=abs(int(offset[predindex])-stop0)
            duration=stop0-start0
            offs50+=int(abserr<=2205)
            offs_rel+=int(abserr<=max(2205,int(.2*duration)))
            sum_abs+=abserr;dur_errors.append(abserr/SR*1000.)
    size=len(pitch)
    return dict(n_predictions=size,n_ground_truth=gt_total,
        note_onset_pitch_matched=tp_note,
        onset_pitch_precision=tp_note/size if size else None,
        onset_pitch_recall=tp_note/gt_total if gt_total else None,
        offset_50ms_correct_given_note_onset_pitch=offs50,
        offset_50ms_accuracy=offs50/tp_note if tp_note else None,
        offset_tolerance50ms_or_20pct_correct_given_note_onset_pitch=offs_rel,
        offset_tolerance50ms_or_20pct_accuracy=offs_rel/tp_note if tp_note else None,
        complete_note_F1_50ms=2*offs50/(size+gt_total) if size+gt_total else None,
        complete_note_F1_50ms_or_20pct=2*offs_rel/(size+gt_total) if size+gt_total else None,
        note_offset_MAE_ms_matched=sum_abs/tp_note/SR*1000 if tp_note else None,
        note_offset_median_AE_ms=float(np.median(dur_errors)) if dur_errors else None)

def run(audio,s80,s402,s407,jams,out):
    req(not out.exists(),'S408 may not overwrite existing offset artifact')
    a=npz(audio);r=npz(s80);notes=npz(s402);timestamp=npz(s407)
    rowIds,row,slot,pitch,onset,k=build_candidates(a,notes,timestamp,r)
    req(len(pitch)==33475,'S408 original S80 K note candidate count drift')
    member=a['member'].astype(str)
    comps=np.array([composition_key(x) for x in member])
    folds=np.array([composition_fold(x) for x in comps],np.int8)
    req(np.array_equal(timestamp['held_composition_fold'].astype(np.int8),folds),
        'S407 row composition fold mismatch')
    X=build_features(a,row,pitch,onset,k)
    # Freeze full candidate feature bank independent from target labels.
    req(len(X)==33475 and X.shape[1]==NUM_FEATURES,
        'S408 candidate features geometry')
    track_gt={}
    with zipfile.ZipFile(jams) as archive:
        index=collections.defaultdict(list)
        for n in archive.namelist():
            if n.endswith('.jams'):index[PurePosixPath(n).name].append(n)
        for m in sorted(set(member)):
            names=index[PurePosixPath(m).name]
            req(len(names)==1,'S408 song label member ambiguous '+m)
            track_gt[m]=get_gt(archive.read(names[0]))
    req(sum(len(v) for v in track_gt.values())==34954,
        'S408 GuitarSet JAMS note count altered')

    # Build ground-truth duration SUPERVISION for predicted notes only; held
    # validation labels are used exclusively in the final metric function.
    pred_member=member[row]
    ref_duration=np.full(len(pitch),np.nan,np.float32)
    for m,gt in track_gt.items():
        ids=np.flatnonzero(pred_member==m)
        if not len(ids):continue
        predictions=[(int(onset[i]),int(pitch[i]),int(i)) for i in ids]
        matches=match_predicted_to_reference(predictions,gt,2205)
        for pi,gi in matches:
            cid=predictions[pi][2]
            o,e,p,slotid=gt[gi]
            ref_duration[cid]=float((e-o)/SR*1000.)

    foldid=folds[row]
    learned_offset=np.zeros(len(pitch),np.int64)
    baseline_pitch_offset=np.zeros(len(pitch),np.int64)
    baseline_global_offset=np.zeros(len(pitch),np.int64)
    no_future_offset=np.zeros(len(pitch),np.int64)
    proofs=[]
    out.mkdir(parents=True)
    for f in range(4):
        tested=foldid==f
        train=(foldid!=f)&np.isfinite(ref_duration)
        req(int(train.sum())>1000 and int(tested.sum())>1000,
            'S408 insufficient heldout samples')
        req(not (set(comps[row[train]])&set(comps[row[tested]])),
            'S408 leaked same song to offset training')
        duration=ref_duration[train]
        targets=np.log(duration)
        global_median=float(np.median(duration))
        medians={int(p):float(np.median(ref_duration[train&(pitch==p)]))
                 for p in np.unique(pitch[train])}
        global_baseline=np.full(int(tested.sum()),global_median)
        pitch_baseline=np.array([medians.get(int(p),global_median)
                                 for p in pitch[tested]])
        # A precommitted 2-model ablation: same model without future sparse
        # acoustic evidence, to test whether future events add information.
        net=HistGradientBoostingRegressor(loss='absolute_error',
          max_iter=115,learning_rate=.06,max_leaf_nodes=15,
          min_samples_leaf=40,l2_regularization=3.,
          early_stopping=False,random_state=SEED+f)
        net.fit(X[train],targets)
        blind=HistGradientBoostingRegressor(loss='absolute_error',
          max_iter=115,learning_rate=.06,max_leaf_nodes=15,
          min_samples_leaf=40,l2_regularization=3.,
          early_stopping=False,random_state=SEED+f)
        blind.fit(X[train,:29],targets)
        guess=np.clip(np.exp(net.predict(X[tested])),MIN_DURATION_MS,MAX_DURATION_MS)
        naive=np.clip(np.exp(blind.predict(X[tested,:29])),MIN_DURATION_MS,MAX_DURATION_MS)
        convert=lambda d:onset[tested]+np.rint(d*SR/1000).astype(np.int64)
        learned_offset[tested]=convert(guess)
        no_future_offset[tested]=convert(naive)
        baseline_pitch_offset[tested]=convert(pitch_baseline)
        baseline_global_offset[tested]=convert(global_baseline)
        cp=out/f'S408_song_fold{f}_predict_note_duration.joblib'
        joblib.dump(dict(offset_model=net,no_future_model=blind,
           per_pitch_train_median_ms=medians,global_train_median_ms=global_median,
           feature_width=NUM_FEATURES,source_only_model_features=True,
           held_composition_fold=f),cp,compress=3)
        proofs.append(dict(fold=f,train_matched_predicted_notes=int(train.sum()),
            held_predicted_note_candidates=int(tested.sum()),
            train_compositions=int(len(set(comps[row[train]]))),
            held_compositions=int(len(set(comps[row[tested]]))),
            composition_overlap=0,checkpoint_sha256=sha(cp)))
    req(np.all(learned_offset>onset) and np.all(no_future_offset>onset),
        'S408 offset must follow corresponding onset')
    results={}
    for name,predicted in (
       ('global_duration_median_training_only',baseline_global_offset),
       ('per_pitch_duration_median_training_only',baseline_pitch_offset),
       ('S408_no_future_acoustic_ablation',no_future_offset),
       ('S408_sparse_future_acoustic_duration',learned_offset)):
        results[name]=metrics_note_offset(pred_member,pitch,onset,predicted,track_gt)
    npzpath=out/'S408_K_CONDITIONED_NOTE_OFFSET_SONGHELD_33475.npz'
    np.savez_compressed(npzpath,
       source_row_id=rowIds[row],source_row_index=row,source_slot=slot,
       track_member=pred_member,composition_fold=foldid,
       S80_K=k,S402_midi_pitch=pitch,S407_onset_samples=onset,
       S408_offset_samples=learned_offset,
       no_future_offset_samples=no_future_offset,
       median_pitch_offset_samples=baseline_pitch_offset,
       median_global_offset_samples=baseline_global_offset)
    report=dict(status='S408_SONGHELD_PITCH_CONDITIONED_NOTE_OFFSET_REAL_GUITARSET',
       predicted_before_scoring_held_song_JAMS=True,
       S80_reference_unchanged=True,S402_pitch_reference_unchanged=True,
       S407_timing_reference_unchanged=True,
       original_tracks=len(track_gt),original_events=N,
       predicted_new_note_instances=int(len(pitch)),
       annotated_gt_notes=int(sum(map(len,track_gt.values()))),
       score_uses_match_on_correct_MIDI_pitch_and_onset_within_50ms=True,
       true_note_pitch_never_used_to_form_test_prediction=True,
       true_note_duration_never_used_to_form_test_prediction=True,
       K0_new_note_prediction_count_zero=True,
       future_sparse_acoustic_events_only_not_full_continuous_audio=True,
       no_true_note_offset_model_input=True,
       new_note_retrigger_can_overlap_unhandled_in_runtime=True,
       no_actual_audio_tail_model_after_last_candidate_event=True,
       song_disjoint_model_fit=True,prior_GuitarSet_count_reference_training_overlap=True,
       prior_GuitarSet_experiment_selection_bias=True,
       offset_model_not_production_promoted=True,
       feature_width=NUM_FEATURES,horizon_ms=HORIZONS_MS,
       four_fold_proofs=proofs,
       metrics=results,prediction_sha256=sha(npzpath),
       sources=dict(audio=sha(audio),S80=sha(s80),S402=sha(s402),
                    S407=sha(s407),annotation=sha(jams)))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S408 FOLDS',json.dumps(proofs),flush=True)
    print('S408 METRICS',json.dumps(results),flush=True)
    print('S408 PASS: song-held train-only supervised offsets from S402 S407 note candidates',flush=True)

def selftest():
    gt=[(100,550,48,0),(200,850,52,1),(200,900,52,2)]
    pred=[(102,48,0),(200,52,1),(205,52,2)]
    matched=match_predicted_to_reference(pred,gt,20)
    req(len(matched)==3 and len(set(x[1] for x in matched))==3,
        'Must preserve duplicate same-pitch across independent notes')
    z=metrics_note_offset(np.array(['a','a','a']),np.array([48,52,52]),
       np.array([102,200,205]),np.array([550,850,900]),{'a':gt})
    req(z['complete_note_F1_50ms']==1.,
        'Offset complete-note metric mismatch')
    print('S408 SELFTEST PASS: duplicate notes matched one-to-one, offset follows onset, 50ms note triple F1')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for nm in ('audio','s80','s402','s407','annotations','out'):
        p.add_argument('--'+nm,type=Path)
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        for nm in ('audio','s80','s402','s407','annotations','out'):
            req(getattr(args,nm) is not None,'missing --'+nm)
        run(args.audio,args.s80,args.s402,args.s407,args.annotations,args.out)
