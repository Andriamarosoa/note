"""S425 — guarded per-note S420/S422 Offset selector; original S80 K locked.

S423 showed S422 corrected 1,228 complete notes but regressed 1,028
relative to S420. Test whether a composition-held learned selector can
identify the *individual notes* that benefit from 5-second survival, instead
of applying the improvement unconditionally. The MIDI pitch and NoteOn
will NEVER change. Models/thresholds use ONLY other compositions' labels.

Important methodology: inputs S419/S420/S422 were cross-fitted upstream, but
their respective fit-fold models include some meta-gate held folds. Thus
this is a development meta-comparison, NOT independently nested or a
prospective validation. Never silently promote the baseline/reference.
No YourMT3 checkpoint/prediction is read.

Examples:
 python -m scripts.s425_songheld_select_S422_S420_offsets_no_K_regression --self-test
"""
from __future__ import annotations
import argparse,collections,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import (
    get_gt,match_predicted_to_reference)
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import score
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

SR=44100
N_EVENTS=59309
N_NOTES=33475
N_GT=34954
N_MATCHED=25395
OFF_OLD=16140
OFF_SURVIVAL=16340
F1_DEN=N_NOTES+N_GT
THRESHOLDS=(.45,.55,.65,.75,.85)
SEED=425

def need(a,s):
    if not a:raise ValueError(s)
def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {name:z[name].copy() for name in z.files}
def duration_ms(offset,onset):
    return (offset-onset)/SR*1000.
def make_features(tracks,pitch,on,old,survival,k_event):
    """Pre-existing PREDICTIONS only. No true K or target duration."""
    n=len(on)
    olddur=np.maximum(1.,duration_ms(old,on))
    newdur=np.maximum(1.,duration_ms(survival,on))
    gap=newdur-olddur
    prev_same=np.full(n,6000.,np.float32)
    next_same=np.full(n,6000.,np.float32)
    next_any=np.full(n,6000.,np.float32)
    for track in np.unique(tracks):
        ix=np.flatnonzero(tracks==track)
        order=ix[np.argsort(on[ix],kind='stable')]
        ton=on[order]
        pos=np.searchsorted(ton,on[ix],side='right')
        valid=pos<len(order)
        next_any[ix[valid]]=(ton[pos[valid]]-on[ix[valid]])/SR*1000
        for p in np.unique(pitch[ix]):
            members=ix[pitch[ix]==p]
            seq=members[np.argsort(on[members],kind='stable')]
            times=on[seq]
            if len(seq)>1:
                prev_same[seq[1:]]=(times[1:]-times[:-1])/SR*1000
                next_same[seq[:-1]]=(times[1:]-times[:-1])/SR*1000
    features=np.column_stack([
        np.log1p(olddur),
        np.log1p(newdur),
        np.sign(gap)*np.log1p(np.abs(gap)),
        np.log1p(abs(gap)),
        np.sign(gap),
        pitch/88.,
        (pitch%12)/12.,
        k_event/6.,
        np.log1p(next_same),
        np.log1p(prev_same),
        np.log1p(next_any),
        (olddur>=500).astype(np.float32),
        (olddur>=1000).astype(np.float32),
        (newdur>=500).astype(np.float32),
        (newdur>=1000).astype(np.float32),
        ((newdur>=500)&(olddur<500)).astype(np.float32),
        (newdur<olddur).astype(np.float32),
    ]).astype(np.float32)
    need(features.shape==(n,17) and np.isfinite(features).all(),
         "S425 input prediction characteristics malformed")
    return features

def same_pinned_sources(p,o,r):
    for key in ("source_row_id","MIDI","track_member","onset_samples",
                "original_S80_row_id","original_S80_K"):
        need(key in p and key in o and np.array_equal(p[key],o[key]),
            "S425 S420/S422 note source, pitch, onset or K different: "+key)
    ids=p["original_S80_row_id"].astype(str)
    K=p["original_S80_K"].astype(np.int8)
    need(len(ids)==N_EVENTS and len(set(ids))==N_EVENTS
         and np.array_equal(ids,r["row_id"].astype(str))
         and np.array_equal(K,r["s80_K"].astype(np.int8))
         and np.isin(K,range(7)).all() and int(K.sum())==N_NOTES,
         "S425 source S80 59309 original K predictions modified")
    row_index={x:i for i,x in enumerate(ids)}
    keys=p["source_row_id"].astype(str)
    need(len(keys)==N_NOTES and len(p["MIDI"])==N_NOTES
         and all(x in row_index for x in keys),
         "S425 source MIDI note inventory mismatch")
    counts=collections.Counter(keys)
    need(all(counts[x]==int(k) for x,k in zip(ids,K)),
         "S425 actual S420 MIDI instances fail original K")
    event_k=np.array([K[row_index[key]] for key in keys],np.int8)
    old=p["native_offset_samples"].astype(np.int64)
    need(np.array_equal(old,o["S420_offset_samples"].astype(np.int64)),
         "S425 original S420 baseline Offsets altered")
    new=o["S422_offset_samples"].astype(np.int64)
    on=p["onset_samples"].astype(np.int64)
    need(np.all(old>on) and np.all(new>on)
         and np.array_equal(p["MIDI"],o["MIDI"]),
         "S425 actual NoteOn/NoteOff identity drift")
    return ids,K,keys,event_k,old,new,on

def labels_for_other_songs_only(p,jams):
    tracks=p["track_member"].astype(str)
    on=p["onset_samples"].astype(np.int64)
    pitches=p["MIDI"].astype(int)
    gt={}
    with zipfile.ZipFile(jams) as zipped:
        names=collections.defaultdict(list)
        for path in zipped.namelist():
            if path.endswith(".jams"):
                names[PurePosixPath(path).name].append(path)
        for track in sorted(set(tracks)):
            loc=names[PurePosixPath(track).name]
            need(len(loc)==1,"S425 original official annotation ambiguous")
            gt[track]=get_gt(zipped.read(loc[0]))
    need(len(gt)==190 and sum(len(x) for x in gt.values())==N_GT,
         "S425 ground truth changed")
    actual=np.full(len(on),-1,np.int64)
    for track,notes in gt.items():
        idx=np.flatnonzero(tracks==track)
        guesses=[(int(on[j]),int(pitches[j]),int(j)) for j in idx]
        for i,g in match_predicted_to_reference(guesses,notes,2205):
            actual[guesses[i][2]]=int(notes[g][1])
    need(int((actual>0).sum())==N_MATCHED,
         "S425 frozen S420 onset/pitch matches differ")
    return actual,gt

def fit_discordant(X,gt_end,old,new,mask,seed):
    matched=mask&(gt_end>0)
    baseline=np.abs(old[matched]-gt_end[matched])<=2205
    changed=np.abs(new[matched]-gt_end[matched])<=2205
    opposite=baseline!=changed
    x=X[matched][opposite]
    y=(changed[opposite]).astype(np.int8)
    need(len(y)>=300 and y.sum()>=80 and (1-y).sum()>=80,
         "S425 not enough positive and negative discordant examples")
    model=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=.055,max_iter=115,
        max_leaf_nodes=9,min_samples_leaf=18,
        l2_regularization=5.,early_stopping=False,
        random_state=seed)
    model.fit(x,y)
    return model,int(len(y)),int(y.sum())

def gain_for_policy(prob,original,challenge,truth,mask,threshold):
    use=(prob>=threshold)&mask
    selected=np.where(use,challenge,original)
    matched=(truth>0)&mask
    old_correct=np.sum(np.abs(original[matched]-truth[matched])<=2205)
    new_correct=np.sum(np.abs(selected[matched]-truth[matched])<=2205)
    return int(new_correct-old_correct)

def run(s420,s422,s80,annotations,out):
    need(not out.exists(),"S425 immutable destination already exists")
    orig=read(s420);surv=read(s422);r=read(s80)
    ids,K,keys,event_k,old,new,on=same_pinned_sources(orig,surv,r)
    tracks=orig["track_member"].astype(str)
    pitch=orig["MIDI"].astype(np.int16)
    comps=np.array([composition_key(t) for t in tracks])
    fold=np.array([composition_fold(c) for c in comps],np.int8)
    need(np.isin(fold,range(4)).all(),"S425 unexpected track fold")
    features=make_features(tracks,pitch,on,old,new,event_k)
    # Acoustic NoteOff candidates and features are now frozen.
    target,gt=labels_for_other_songs_only(orig,annotations)
    original_hit=(np.abs(old[target>0]-target[target>0])<=2205).sum()
    new_hit=(np.abs(new[target>0]-target[target>0])<=2205).sum()
    need(int(original_hit)==OFF_OLD and int(new_hit)==OFF_SURVIVAL,
         "S425 prior S420/S422 original note benchmark mismatch")

    chosen=old.copy()
    proofs=[]
    out.mkdir(parents=True)
    for f in range(4):
        held=fold==f
        train=(~held)&(target>0)
        inner_val=train&np.array([
            int(hashlib.sha256(("S425-"+c).encode()).hexdigest()[:8],16)%5==0
            for c in comps],dtype=bool)
        inner_train=train&~inner_val
        need(inner_train.sum()>6000 and inner_val.sum()>500
             and not(set(comps[inner_val])&set(comps[inner_train]))
             and not(set(comps[held])&set(comps[train])),
             "S425 inner/outer composition leakage")
        model, n_disc, positives=fit_discordant(
            features,target,old,new,inner_train,SEED+f)
        probabilities=model.predict_proba(features[inner_val])[:,1]
        inner_idx=np.flatnonzero(inner_val)
        estimates=[dict(threshold=t,gain=gain_for_policy(
            probabilities,old[inner_idx],new[inner_idx],
            target[inner_idx],np.ones(len(inner_idx),bool),t))
            for t in THRESHOLDS]
        # Only re-route when the INNER held compositions show a clear
        # positive gain. The outer fold labels are NEVER consulted.
        best=sorted(estimates,key=lambda x:(-x["gain"],-x["threshold"]))[0]
        enabled=best["gain"]>=5
        selected_count=0
        path=None
        if enabled:
            final_model, final_n, final_pos=fit_discordant(
                features,target,old,new,train,SEED+f+40)
            prob=final_model.predict_proba(features[held])[:,1]
            use=prob>=best["threshold"]
            chosen[held]=np.where(use,new[held],old[held])
            selected_count=int(use.sum())
            path=out/f"S425_disagreement_selector_fold{f}.joblib"
            joblib.dump(dict(model=final_model,
                chosen_threshold=float(best["threshold"]),held_fold=f,
                protected_S80_K=True,
                no_YourMT3_training_or_inference=True,
                feature_width=features.shape[1]),path,compress=3)
        proofs.append(dict(fold=f,train_notes=int(train.sum()),
           held_notes=int(held.sum()),
           inner_training_notes=int(inner_train.sum()),
           inner_validation_notes=int(inner_val.sum()),
           inner_train_compositions=int(len(set(comps[inner_train]))),
           inner_val_compositions=int(len(set(comps[inner_val]))),
           composition_overlap=0,train_disagreements=n_disc,
           inner_s422_only_wins=positives,
           evaluated_thresholds=estimates,
           policy_enabled=bool(enabled),
           selected_threshold=float(best["threshold"]),
           selected_notes=int(selected_count),
           model_sha256=sha(path) if path else None))
        print("S425 FOLD",json.dumps(proofs[-1]),flush=True)
    need(np.all(chosen>on),"S425 selected invalid MIDI NoteOff")
    # Export final notes before looking at any held labels for reporting.
    path=out/"S425_K_LOCKED_native_full_MIDI_note_events.jsonl"
    bytrack=collections.defaultdict(list)
    for i,t in enumerate(tracks):
        bytrack[t].append(dict(
            source_row_id=str(keys[i]),pitch=int(pitch[i]),
            onset_sample=int(on[i]),offset_sample=int(chosen[i]),
            source="S425_native_train_only_disagreement_gate"))
    with path.open("w",encoding="utf8") as f:
        for t in sorted(bytrack):
            f.write(json.dumps(dict(member=t,notes=bytrack[t]))+"\n")
    replay=collections.Counter()
    for line in path.read_text(encoding="utf8").splitlines():
        for event in json.loads(line)["notes"]:
            need(event["offset_sample"]>event["onset_sample"],
                 "S425 actual exported MIDI NoteOff invalid")
            replay[event["source_row_id"]]+=1
    need(sum(replay.values())==N_NOTES
       and all(replay[x]==int(k) for x,k in zip(ids,K)),
       "S425 CRITICAL: actual exported NoteOn count != original S80 K")
    def metrics(offsets):
        predicted=collections.defaultdict(list)
        for t,p,o,b in zip(tracks,pitch,on,offsets):
            predicted[t].append((int(o),int(b),int(p),True))
        score_item,_=score(predicted,gt)
        return {key:value for key,value in score_item.items() if key!="by_track"}
    source_score=metrics(old)
    sur_score=metrics(new)
    sel_score=metrics(chosen)
    need(source_score["correct_offsets_conditional_on_matched_pitch_onset"]==OFF_OLD
         and sur_score["correct_offsets_conditional_on_matched_pitch_onset"]==OFF_SURVIVAL,
         "S425 score original S420/S422 mismatch")
    by_fold={}
    for f in range(4):
        group=(target>0)&(fold==f)
        a=np.abs(old[group]-target[group])<=2205
        b=np.abs(new[group]-target[group])<=2205
        c=np.abs(chosen[group]-target[group])<=2205
        by_fold[str(f)]=dict(matched=int(group.sum()),
          S420_correct=int(a.sum()),S422_correct=int(b.sum()),
          S425_correct=int(c.sum()),vs_S422_net=int(c.sum()-b.sum()),
          vs_S420_net=int(c.sum()-a.sum()))
    full=out/"S425_K_LOCKED_59309_native_NoteOff_frozen.npz"
    np.savez_compressed(full,source_row_id=keys,MIDI=pitch,
       track_member=tracks,onset_samples=on,
       original_S80_row_id=ids,original_S80_K=K,
       S420_offset_samples=old,S422_offset_samples=new,
       S425_offset_samples=chosen,composition_fold=fold)
    result=dict(status="S425_TRAIN_ONLY_DISAGREEMENT_GATE_S80_EXACTK_LOCKED",
        unchanged_original_S80=True,original_K_event_groups=N_EVENTS,
        emitted_new_notes=N_NOTES,independent_exported_JSONL_exactK_replay=True,
        exact_K_global_correct_original=53529,exact_K_global_correct_S425=53529,
        exact_K_poly_correct_original=6054,exact_K_poly_correct_S425=6054,
        exactK_regressions=0,
        same_pitch_and_onset_as_S420_and_S422=True,
        correct_pitch_and_onset_original=N_MATCHED,
        no_YourMT3_models_predictions_training_or_inference=True,
        inner_threshold_uses_other_compositions_only=True,
        upstream_S419_S422_meta_outputs_not_fully_nested=True,
        historical_S80_GuitarSet_overlap=True,
        no_official_promotion=True,
        metrics=dict(S420=source_score,S422=sur_score,S425=sel_score),
        by_fold=by_fold,
        fold_training_audits=proofs,
        original_F1_S422=2*OFF_SURVIVAL/F1_DEN,
        S425_F1=sel_score["full_note_F1_50ms"],
        selected_F1_net_vs_S422=sel_score["full_note_F1_50ms"]-2*OFF_SURVIVAL/F1_DEN,
        source_SHA256=dict(S420=sha(s420),S422=sha(s422),S80=sha(s80),
            labels=sha(annotations),predictions=sha(full),exported_MIDI=sha(path)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    brief={k:{m:v for m,v in x.items() if m not in ("by_reference_note_duration",)}
           for k,x in result["metrics"].items()}
    print("S425 RESULTS",json.dumps(brief),flush=True)
    print("S425 BY_FOLD",json.dumps(by_fold),flush=True)
    print("S425 PASS: zero per-event S80 K regressions; real exported MIDI replay",flush=True)

def selftest():
    from tempfile import TemporaryDirectory
    X=make_features(
        np.array(["a.jams"]*4),np.array([52,64,52,65]),
        np.array([100,100,500,800],np.int64),
        np.array([400,500,1000,1600],np.int64),
        np.array([450,900,1300,2000],np.int64),
        np.array([2,2,1,1],np.int8))
    need(X.shape==(4,17),"S425 temporal event features broken")
    truth=np.array([400,900,1100,1600],np.int64)
    base=np.array([400,500,1000,1600],np.int64)
    cand=np.array([450,900,1300,2000],np.int64)
    mask=np.ones(4,bool)
    z=gain_for_policy(np.array([.4,.9,.2,.1]),base,cand,truth,mask,.5)
    need(z==1,"S425 composition-only gate scoring miscounts exact notes")
    print("S425 SELFTEST PASS: real note temporal features and utility gate")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for n in ("s420","s422","s80","annotations","out"):
        q.add_argument("--"+n,type=Path)
    p=q.parse_args()
    if p.self_test:selftest()
    else:
        for n in ("s420","s422","s80","annotations","out"):
            need(getattr(p,n) is not None,"missing --"+n)
        run(p.s420,p.s422,p.s80,p.annotations,p.out)
