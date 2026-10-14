"""S528: three-song held, genuine trained piano note-pitch heads on S527 audio.

Unlike S516/S519, NO canonical BP note onset is required at inference.
Every one of 575 ORIGINAL prelabel V88 time candidates including K0
receives independent 49-MIDI-pitch probabilities. Train only on OTHER
recordings, and report for the 322 known original positive MIDI attacks:
- whether 102 onsets removed by S498 now get a correctly named new pitch
- named-note exact chord, conditional positive-onset note P/R/F1 and K
- regression and source-wise action direction, zero tuning on held truth
- two predeclared feature subsets: raw 7-D vs raw+BP 11-D, thresholds 0.5/0.7.

No confirmed stream negatives among 253 V88 unmatched events; these are
UNKNOWN, not false onsets. Eight true MIDI attacks missing from V88
remain missed. The 3 historically exposed piano songs are NOT blind.
No promotion, no original S80/Live change and no full-stream F1 claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import collections
import numpy as np


def need(ok,msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(a):
    f=json.loads(a.features_report.read_text())
    t=json.loads(a.truth_report.read_text())
    l=json.loads(a.lost_report.read_text())
    need(f["status"]=="S527_PRELABEL_ALL_575_V88_PIANO_ONSET_CANDIDATES_49_RAW_PITCH_FEATURES"
         and f["frozen_feature_ledger_SHA256"]==sha(a.features)
         and f["no_original_true_MIDI_annotation_read"]
         and f["original_V88_K0_candidates_included"]==473
         and t["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and t["SHA256"]["paired_label_comparison"]==sha(a.truth)
         and l["status"]=="S526_BP_CANONICAL_LOST_TRUE_V88_K0_AND_KPOS_PIANO_ATTACKS"
         and l["every_event_ledger_SHA256"]==sha(a.lost),
         "S528 original prelabel 575 pitch features or MIDI audit SHA changed")
    rows=[json.loads(z) for z in a.features.read_text().splitlines()]
    truth=[json.loads(z) for z in a.truth.read_text().splitlines()]
    lost=[json.loads(z) for z in a.lost.read_text().splitlines()]
    ids=[r["V88_original_event_id"] for r in rows]
    idx={z:i for i,z in enumerate(ids)}
    need(len(rows)==len(idx)==575 and len(truth)==len(lost)==322
         and set(x["row_id"] for x in truth)==set(x["row_id"] for x in lost),
         "S528 source original 575 V88 events and exactly 322 evaluated positive MIDI attacks")
    X=np.asarray([r["per_midi_features_49x11"] for r in rows],np.float32)
    need(X.shape==(575,49,11) and np.isfinite(X).all(),
         "S528 exactly 49 real pitch features from all K0..K6 original events")
    rec=np.asarray([r["source_recording_sha"] for r in rows],str)
    need(len(set(rec))==3,"S528 original three composers held separately")
    y=np.zeros((575,49),bool)
    labeled=np.zeros(575,bool)
    missing_bp=np.zeros(575,bool)
    for trow in truth:
        i=idx[trow["row_id"]]
        need(not labeled[i],"S528 duplicate original MIDI attack label on V88 candidate")
        pitches=[int(p) for p in trow["true_new_onset_MIDI_pitches"]]
        need(len(pitches)>0 and all(40<=p<=88 for p in pitches),
             "S528 only true new-onset MIDI40..88 note targets supported")
        y[i,np.asarray(pitches,dtype=int)-40]=True
        labeled[i]=True
    for r in lost:
        i=idx[r["row_id"]]
        missing_bp[i]=bool(r["new_note_event_missing_from_training_rows_S505"])
    need(int(labeled.sum())==322 and
         int(np.sum(missing_bp&labeled))==102 and
         int(np.sum(~labeled))==253 and
         not np.any(missing_bp&~labeled),
         "S528 all 102 lost true piano attacks must receive genuine held labels")
    before=np.zeros((575,49),bool)
    for i,r in enumerate(rows):
        for p in r["old_basicpitch_birth_pitches"]:
            if 40<=int(p)<=88:before[i,int(p)-40]=True
    return f,t,l,rows,X,y,labeled,missing_bp,before,rec


def fit(X,y,train,kind,seed):
    from sklearn.ensemble import ExtraTreesClassifier
    cols=list(range(11)) if kind=="with_BP" else [0,1,2,3,4,5,10]
    x=X[train][:,:,cols].reshape(-1,len(cols))
    target=y[train].reshape(-1)
    need(len(x)>0 and int(target.sum())>=20 and int(np.sum(~target))>=20,
         "S528 acoustic note-pitch train source lacks confirmed both pitch classes")
    model=ExtraTreesClassifier(n_estimators=160,max_depth=15,
        min_samples_leaf=3,max_features=.8,
        class_weight="balanced_subsample",random_state=seed,n_jobs=2)
    model.fit(x,target)
    need(set(map(bool,model.classes_))=={False,True},
         "S528 no separately trained positive and negative pitch labels")
    return model,cols,int(target.sum()),int(np.sum(~target))


def predict(model,X,rows,cols):
    z=X[rows][:,:,cols].reshape(-1,len(cols))
    probability=model.predict_proba(z)[:,list(model.classes_).index(True)]
    return probability.reshape(len(rows),49).astype(np.float32)


def metrics(Y,Q,BP,miss):
    known=len(Y)
    count=Y.sum(axis=1)
    oldcorrect=np.all(Y==BP,axis=1)
    newcorrect=np.all(Y==Q,axis=1)
    oldk=BP.sum(axis=1)
    newk=Q.sum(axis=1)
    tp=int(np.sum(Y&Q));fp=int(np.sum(~Y&Q));fn=int(np.sum(Y&~Q))
    def subgroup(mask):
        yy=Y[mask];qq=Q[mask];bb=BP[mask]
        old=np.all(yy==bb,axis=1);new=np.all(yy==qq,axis=1)
        true_any=np.any(yy,axis=1);pred_any=np.any(qq,axis=1)
        return dict(n=int(mask.sum()),
            baseline_BP_birth_full_note_set=int(old.sum()),
            output_exact_full_note_set=int(new.sum()),
            complete_chord_fixes=int(np.sum(~old&new)),
            complete_chord_regressions=int(np.sum(old&~new)),
            estimated_pitch_identity_ever_correct=int(np.sum(np.any(yy&qq,axis=1))),
            output_nonzero_event_count=int(pred_any.sum()),
            output_zero_even_with_true_onset_count=int(np.sum(true_any&~pred_any)),
            TP_true_note_pitch=int(np.sum(yy&qq)),
            FP_pitch_at_known_true_onset=int(np.sum(~yy&qq)),
            FN_true_note_pitch=int(np.sum(yy&~qq)),
            actual_note_K_correct=int(np.sum(qq.sum(axis=1)==yy.sum(axis=1))),
            baseline_note_K_correct=int(np.sum(bb.sum(axis=1)==yy.sum(axis=1))))
    return dict(
        evaluated_TRUE_attacks_only=known,
        note_pitch_precision_conditional_on_positive_onsets=tp/max(1,tp+fp),
        note_pitch_recall_conditional_on_positive_onsets=tp/max(1,tp+fn),
        note_pitch_F1_conditional_on_positive_onsets=2*tp/max(1,2*tp+fp+fn),
        overall=subgroup(np.ones(known,bool)),
        originally_BP_missing_102=subgroup(miss),
        originally_BP_matched_220=subgroup(~miss),
        by_true_K={str(k):subgroup(count==k) for k in sorted(set(count))})


def run(a):
    need(not a.out.exists(),"S528 genuine held three-source trained evidence immutable")
    f,t,l,raw,X,Y,labeled,miss,before,rec=load(a)
    a.out.mkdir(parents=True)
    import joblib
    predictions={}
    folds=[]
    for kind in ("raw_only","with_BP"):
        outputs=np.full((575,49),np.nan,np.float32)
        for fold,held in enumerate(sorted(set(rec))):
            train=np.flatnonzero((rec!=held)&labeled)
            held_all=np.flatnonzero(rec==held)
            held_positive=np.flatnonzero((rec==held)&labeled)
            need(len(set(rec[train]))==2 and len(held_positive)>0
                 and len(held_all)>0 and not np.any(rec[train]==held),
                 "S528 entire original held recording excluded from fitting")
            model,cols,pos,neg=fit(X,Y,train,kind,seed=152800+fold+100*(kind=="with_BP"))
            outputs[held_all]=predict(model,X,held_all,cols)
            checkpoint=a.out/f"S528_{kind}_held{fold}_ACTUAL_PIANO_AUDIO_NOTE_HEAD.joblib"
            joblib.dump(dict(model=model,feature_columns=cols,held_sha=held),
                        checkpoint,compress=3)
            folds.append(dict(feature_policy=kind,held_original_recording_SHA=held,
               trained_on_OTHER_original_recording_count=2,
               train_true_positive_attack_events=int(len(train)),
               held_true_positive_attack_events=int(len(held_positive)),
               trained_positive_note_pitch_labels=pos,
               trained_negative_pitch_labels_on_TRUE_onsets=neg,
               all_V88_K0_candidates_in_held_inference=True,
               trained_model_SHA256=sha(checkpoint)))
        need(np.isfinite(outputs).all() and np.all((0<=outputs)&(outputs<=1)),
             "S528 must predict 49 note pitches for ALL 575 original V88 candidates")
        predictions[kind]=outputs
    result={}
    for kind in ("raw_only","with_BP"):
        result[kind]={}
        for threshold in (0.5,0.7):
            pred=predictions[kind]>=threshold
            summary=metrics(Y[labeled],pred[labeled],before[labeled],miss[labeled])
            by_rec={}
            for recid in sorted(set(rec)):
                sel=(rec==recid)&labeled
                by_rec[recid]=metrics(Y[sel],pred[sel],before[sel],miss[sel])
            result[kind][str(threshold)]=dict(overall=summary,by_recording=by_rec)
            print("S528 TRUE HELD AUDIO PIANO NOTE PITCH TEST",
                  json.dumps(dict(kind=kind,threshold=threshold,
                                  metrics=summary)),flush=True)
    ledger=a.out/"S528_ALL_V88_K0_AUDIO_GENERATED_NOTE_PROBABILITIES_49.jsonl"
    ledger.write_text("".join(json.dumps(dict(
       V88_original_event_id=raw[i]["V88_original_event_id"],
       source_original_recording_sha=rec[i],
       original_V88_K=raw[i]["original_V88_K_soft_prior_ONLY"],
       observed_TRUE_positive_onset_FOR_AUDIT=bool(labeled[i]),
       truth_MIDI_pitches_FOR_AUDIT=(np.flatnonzero(Y[i])+40).tolist() if labeled[i] else None,
       original_canonical_BP_missed_FOR_AUDIT=bool(miss[i]) if labeled[i] else None,
       raw_only_49_pitch_prob=predictions["raw_only"][i].round(6).tolist(),
       with_BP_49_pitch_prob=predictions["with_BP"][i].round(6).tolist(),
       evaluation_truth_never_used_for_held_inference=True))
       +"\n" for i in range(575)))
    report=dict(status="S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES",
        truly_trained_6_models=6,
        trained_note_heads_raw7_vs_with_BP11=True,
        original_V88_K0_K6_candidate_events_all_575_evaluated=True,
        original_322_positive_MIDI_onsets_only_for_training_AND_metrics=True,
        unpaired_253_piano_original_V88_candidates_NOT_labeled_negative=True,
        original_MAESTRO_true_8_onsets_without_V88_candidates_still_missed=True,
        three_recording_LOO_training_no_held_truth_in_fit=True,
        historic_three_piano_songs_exposed_NOT_new_blind_generalization=True,
        threshold_050_070_frozen_before_held_labels=True,
        note_F1_conditional_on_true_positive_attack_only_NOT_full_stream=True,
        unknown_negative_onsets_prevent_real_noteon_precision_F1=True,
        original_S80_Live_reference_and_K_unchanged=True,
        no_promotion=True,
        models=folds,scoring=result,
        original_S527_audio_feature_SHA=sha(a.features),
        original_S476_MIDI_labels_SHA=sha(a.truth),
        predictions_sha256=sha(ledger))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


def selftest():
    y=np.zeros((2,49),bool);y[0,12]=True;y[1,15]=True
    b=np.zeros_like(y);p=y.copy();p[0,13]=True
    r=metrics(y,p,b,np.array([True,False]))
    need(r["originally_BP_missing_102"]["n"]==1 and
         r["overall"]["TP_true_note_pitch"]==2 and
         r["overall"]["FP_pitch_at_known_true_onset"]==1 and
         r["overall"]["output_exact_full_note_set"]==1,
         "S528 all correct named notes distinguish extra false harmonic pitches")
    print("S528 SELFTEST PASS true note-pitch TP/FP/FN and original K independent")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("features","features-report","truth","truth-report",
                 "lost","lost-report","out"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:
        selftest()
    else:
        for name in ("features","features_report","truth","truth_report",
                     "lost","lost_report","out"):
            need(getattr(a,name)is not None,"S528 missing "+name)
        run(a)
