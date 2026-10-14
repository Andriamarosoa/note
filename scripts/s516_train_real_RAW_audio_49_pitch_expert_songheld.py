"""S516 — truly fitted multi-label per-MIDI fundamental/attack expert.

Use SHA-pinned NEW actual raw WAV harmonic/frame/onset evidence S515,
and authentic S505 frozen (13 BP/source pitch heads) note sets.
Eight leave-ONE-original-recording-out training folds. The recording's
true note pitches never appear in its model training or features.

This is intentionally retrospective development: the 5+3 recordings
were already used to investigate previous modules. No claim of blind
independent generalization, no full-stream false-onset precision without
verified negatives, and no official S80/Live modification.

Predeclare three outputs BEFORE labels are scored on each held fold:
  0.50 / 0.70 per-pitch threshold and conservative BP add/.85 remove/.15.
Reports note-level micro P/R/F1 CONDITIONAL on paired true onsets,
exact full named-note sets, Exact-K and *every* regression by source.
Missing canonical onsets stay in the overall true-onset denominator.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_sources(a):
    head=json.loads(a.head_report.read_text())
    acoustic=json.loads(a.acoustic_report.read_text())
    need(head["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
       and head["audio_proposal_truth_table_SHA256"]==sha(a.heads)
       and acoustic["status"]==
        "S515_TRUE_EIGHT_SOURCE_WAV_INDEPENDENT_MULTILABEL_NOTE_FEATURES_PRETRUTH"
       and acoustic["real_audio_feature_SHA256"]==sha(a.acoustic)
       and acoustic["no_original_MIDI_XML_labels_read_to_generate_features"]
       and acoustic["source_number"]==8,
       "S516 source actual acoustic evidence and real note truth must match prelabel original SHA")
    with np.load(a.heads,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    raw=[json.loads(l) for l in a.acoustic.read_text().splitlines()]
    idx={x["canonical_event_id"]:x for x in raw}
    events=d["event_id"].astype(str)
    need(len(idx)==len(raw)==294 and all(s in idx for s in events),
         "S516 60+234 ORIGINAL canonical events or paired 265 known true notes changed")
    feats=np.asarray([idx[e]["per_midi_features_49x11"] for e in events],np.float32)
    need(feats.shape==(len(events),49,11) and np.isfinite(feats).all(),
         "S516 original unexposed true audio 49 pitches x 11 features corrupted")
    for ev,src in zip(events,d["recording_sha"].astype(str)):
        need(idx[ev]["source_recording_sha"]==src,
             "S516 WAV SHA source identity drift or leakage")
    originals=d["head_note_bits"][np.arange(len(events)),
        d["selected_BP_baseline_head"].astype(int)].astype(bool)
    y=d["truth_note_bits"].astype(bool)
    need(not y[:,:40].any() and not y[:,89:].any(),
         "S516 true acoustic notes outside 40..88 not covered, do not silently erase")
    contexts=d["context"].astype(np.float32)
    domain=(d["corpus"].astype(str)=="guitar_S80").astype(np.float32)
    ctx=np.concatenate([contexts,domain[:,None]],axis=1)
    X=np.concatenate([feats,np.repeat(ctx[:,None,:],49,axis=1)],axis=2)
    truth=y[:,40:89]
    need(X.shape==(len(events),49,28) and truth.shape==(len(events),49),
         "S516 no 49 independent pitch generators vs real note truth")
    return head,d,originals,y,X,truth
def eval_notes(y,before,after,originalK):
    correct0=np.all(y==before,axis=1)
    correct1=np.all(y==after,axis=1)
    changed=np.any(after!=before,axis=1)
    trueK=y.sum(axis=1)
    K0=before.sum(1);K1=after.sum(1)
    def metrics(sel):
        tp=np.sum((after&y)[sel]);fp=np.sum((after&~y)[sel])
        fn=np.sum((~after&y)[sel])
        return dict(n=int(sel.sum()),
          old_exact_full_note_sets=int(np.sum(sel&correct0)),
          new_exact_full_note_sets=int(np.sum(sel&correct1)),
          fixes=int(np.sum(sel&~correct0&correct1&changed)),
          regressions=int(np.sum(sel&correct0&~correct1&changed)),
          wrong_to_wrong=int(np.sum(sel&~correct0&~correct1&changed)),
          changed=int(np.sum(sel&changed)),
          original_BP_ExactK=int(np.sum(sel&(K0==trueK))),
          new_ExactK=int(np.sum(sel&(K1==trueK))),
          original_source_S80orV88_ExactK=int(np.sum(sel&(originalK==trueK))),
          known_attack_note_TP=int(tp),known_attack_note_FP=int(fp),
          known_attack_note_FN=int(fn),
          note_precision_conditional=float(tp/max(tp+fp,1)),
          note_recall_conditional=float(tp/max(tp+fn,1)),
          note_F1_conditional=float(2*tp/max(2*tp+fp+fn,1)))
    return dict(overall=metrics(np.ones(len(y),bool)),
         poly=metrics(trueK>=2),
         by_trueK={str(k):metrics(trueK==k) for k in range(7)})
def decode(p,old,policy):
    if policy in ("threshold050","threshold070"):
        bound=.5 if policy=="threshold050" else .7
        output=p>=bound
    else:
        output=old.copy()
        output[(~old)&(p>=.85)]=True
        output[old&(p<=.15)]=False
    # Prevent acoustic generators incorrectly describing observed
    # note-onset as K0 merely due low probabilities.
    missing=np.flatnonzero(~output.any(axis=1))
    output[missing]=old[missing]
    return output
def run(a):
    need(not a.out.exists(),"S516 immutable trained note-head report exists")
    report,d,bp,y,X,yraw=load_sources(a)
    events=d["event_id"].astype(str)
    rec=d["recording_sha"].astype(str)
    source=d["corpus"].astype(str)
    n=len(events)
    from sklearn.ensemble import ExtraTreesClassifier
    import joblib
    a.out.mkdir(parents=True)
    predictions=np.zeros((n,49),np.float32)
    folds=[]
    for fold,held in enumerate(sorted(set(rec))):
        trainrows=np.flatnonzero(rec!=held);heldrows=np.flatnonzero(rec==held)
        need(len(trainrows)>0 and len(heldrows)>0 and
             not set(rec[trainrows])&set(rec[heldrows]),
             "S516 no source recording can be shared across training and held eval")
        trainx=X[trainrows].reshape(-1,28)
        trainy=yraw[trainrows].reshape(-1)
        testx=X[heldrows].reshape(-1,28)
        need(int(trainy.sum())>=20 and int(np.sum(~trainy))>=20,
             "S516 genuine note-present and note-absent classes must be trained")
        # Equalise *true note* imbalance using TRAIN labels only.
        clf=ExtraTreesClassifier(n_estimators=160,max_depth=13,
             min_samples_leaf=4,max_features=.65,n_jobs=2,
             class_weight="balanced_subsample",random_state=1516+fold)
        clf.fit(trainx,trainy)
        probabilities=clf.predict_proba(testx)[:,list(clf.classes_).index(True)]
        predictions[heldrows]=probabilities.reshape(len(heldrows),49)
        checkpoint=a.out/f"S516_fold{fold}_ACTUAL_RAW_NOTE_MULTILABEL_EXPERT.joblib"
        joblib.dump(clf,checkpoint,compress=3)
        record=dict(fold=fold,held_recording_SHA=held,
            train_original_recording_count=len(set(rec[trainrows])),
            held_true_positive_music_attacks=len(heldrows),
            trained_positive_individual_note_labels=int(trainy.sum()),
            trained_negative_pitch_candidates=int(np.sum(~trainy)),
            checkpoint_SHA256=sha(checkpoint),
            threshold_tuning_using_held_true_notes=False)
        folds.append(record)
        print("S516 REAL FRAME ONSET HARMO EXPERT FIT",json.dumps(record),flush=True)
    need(np.isfinite(predictions).all() and np.any(predictions>0),
         "S516 no real held-recording probabilities found")
    result={}
    for corpus in ("guitar_S80","piano_V88"):
        idx=np.flatnonzero(source==corpus)
        src_y=y[idx]
        old=bp[idx]
        K=d["source_K"][idx].astype(int)
        pred=predictions[idx]
        measured={}
        for policy in ("threshold050","threshold070","BP_guarded_add085_remove015"):
            proposed=np.zeros_like(old)
            proposed[:,40:89]=decode(pred,old[:,40:89],policy)
            measured[policy]=eval_notes(src_y,old,proposed,K)
            measured[policy]["missing_original_TRUE_onsets_no_canonical_note_group"]=(
                report["by_source"][corpus]["annotated_true_attacks_NOT_matched"])
            measured[policy]["all_original_true_attacks_including_missed"]=(
                report["by_source"][corpus]["truth_original_positive_attacks"])
            measured[policy]["unmatched_predicted_music_groups_have_UNKNOWN_truth"]=True
        result[corpus]=measured
        print("S516 REAL HELD-NOTE MODEL SCORE",json.dumps(dict(
            corpus=corpus,policies={k:{"overall":v["overall"],"poly":v["poly"]}
                                      for k,v in measured.items()})),flush=True)
    ledger=a.out/"S516_REAL_HELD_RECORDING_EVERY_PITCH_PROBABILITY.jsonl"
    ledger.write_text("".join(json.dumps(dict(
        canonical_event_id=events[i],source_rec=rec[i],corpus=source[i],
        probabilities_by_MIDI_40to88=predictions[i].round(5).tolist(),
        only_external_audio_features_in_inference=True))
        +"\n" for i in range(n)))
    final=dict(status="S516_NEW_TRAINED_RAW_WAV_MULTILABEL_NOTE_GENERATOR_EIGHT_RECORDING_OOF",
        truly_trained_new_independent_audio_pitch_generator=True,
        model="ExtraTrees on original 11 raw spectral/attack pitch features + source prior",
        full_8_recording_crossfit=True,
        per_fold_train_recordings_exclude_evaluated_recording=True,
        all_49_MIDI_pitch_candidates_trainable_not_just_basic_pitch_notes=True,
        true_pitch_supervision_only_from_other_original_recordings=True,
        known_positive_canonical_events_only_not_full_stream_precision=True,
        previously_exposed_corpora_NOT_prospective_blind=True,
        threshold_3_fixed_policies_no_test_tuning=True,
        missing_true_attack_onsets_preserved_in_denominator=True,
        unknown_model_only_events_not_assumed_false=True,
        original_S80_V88_Live_onset_offset_unchanged=True,no_promotion=True,
        folds=folds,by_source=result,
        predictions_sha256=sha(ledger),S515_original_audio_SHA256=sha(a.acoustic),
        original_head_labels_sha256=sha(a.heads))
    (a.out/"report.json").write_text(json.dumps(final,indent=2)+"\n")
def selftest():
    y=np.zeros((2,128),bool);y[:,52]=True
    b=y.copy();b[1,52]=False;b[1,55]=True
    t=eval_notes(y,y,b,np.array([1,1]))
    need(t["overall"]["regressions"]==1 and
         t["overall"]["new_ExactK"]==2,
         "S516 wrong note at correct Exact-K must be recorded as regression")
    print("S516 SELFTEST PASS truth note F1 and chord exact-K distinct regression accounting")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("acoustic","acoustic-report","heads","head-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("acoustic","acoustic_report","heads","head_report","out"):
            need(getattr(a,k)is not None,"S516 missing --"+k)
        run(a)
