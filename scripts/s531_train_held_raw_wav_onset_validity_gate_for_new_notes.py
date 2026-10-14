"""S531 real recording-held learned onset validity head for V88 piano K0.

Three MAESTRO recordings, original SHA-verified full MIDI used to label
V88 candidates by maximum-cardinality one-to-one +/-80ms. All 253 V88
events not paired to a MIDI note-on are *unmatched under this benchmark*:
189 may be duplicates near a true note, NOT necessarily silent audio.
Train ONSET_VALID from pre-annotation raw S527 WAV+BP features only on
two different source recordings, evaluate third. No S528 pitch model
probability enters the event-head training feature matrix (prevents
outer-held leakage through nonnested cross-fit). Record all OOF scores.

Test a pretrained S528 acoustic-only 0.70 pitch head, protected as S529:
never delete pre-existing BP birth note groups. For birth-empty sites
with predicted pitch(es), accept ONLY if independently trained onset
head passes 0.50/0.70. Evaluate pitch+onset 1-1 across *all* 330 original
MIDI groups/3 x 12 seconds, including false outputs and missing attacks.

Three historically exposed songs -> DEVELOPMENT, NOT blind test. No
reference promotion, full S80 replay or proven Live latency.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    need,sha,digest,read_rows,score_event_notes,score_event_birth,
    proposed_notes)

def event_features(rows):
    """Every feature is from frozen original audio or prelabel V88/BP, not true MIDI."""
    bank=np.asarray([r["per_midi_features_49x11"] for r in rows],np.float32)
    need(bank.shape==(575,49,11) and np.isfinite(bank).all(),
         "S531 original 575*49*11 prelabel source music evidence shape")
    features=[]
    byrec=collections.defaultdict(list)
    for i,r in enumerate(rows):
        byrec[r["source_recording_sha"]].append(i)
    delta_before=np.zeros(len(rows),np.float32)
    delta_after=np.zeros(len(rows),np.float32)
    for rec,ids in byrec.items():
        ids.sort(key=lambda i:float(rows[i]["original_V88_candidate_start_s"]))
        t=np.asarray([rows[i]["original_V88_candidate_start_s"] for i in ids],float)
        for j,i in enumerate(ids):
            delta_before[i]=min(.5,float(t[j]-t[j-1])) if j else .5
            delta_after[i]=min(.5,float(t[j+1]-t[j])) if j+1<len(ids) else .5
    for i,r in enumerate(rows):
        arr=bank[i,:,1:]  # 10 original pitch-local acoustic/BP quantities, excludes note index
        stats=np.concatenate((arr.mean(0),arr.max(0),arr.std(0),
                              np.quantile(arr,.9,axis=0)))
        additional=np.asarray([
            r["original_V88_K_soft_prior_ONLY"]/6.,
            len(r["old_basicpitch_birth_pitches"])/6.,
            len(r["old_basicpitch_active_pitches"])/6.,
            delta_before[i],delta_after[i],
            # Strongest harmonic and physical support are still raw prelabel observations.
            np.sum(arr[:,2]>.4)/49.,
            np.sum(arr[:,4]>.4)/49.,
        ],dtype=np.float32)
        features.append(np.concatenate((stats,additional)))
    x=np.asarray(features,np.float32)
    need(x.shape==(575,47) and np.isfinite(x).all(),
         "S531 note birth head must use exactly 47 original signal features")
    return x

def prepare(a):
    manifest=json.loads(a.manifest.read_text())
    audit=json.loads(a.s471_report.read_text())
    s527=json.loads(a.audio_report.read_text())
    s528=json.loads(a.pitch_report.read_text())
    s530=json.loads(a.full_report.read_text())
    need(manifest["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and audit["status"]=="S471_EXTERNAL_THIRD_REAL_MAESTRO_PIANO_V88_MIDI_POSTFREEZE"
         and audit["SHA256"]["S469_preregistration"]==sha(a.manifest)
         and s527["status"]=="S527_PRELABEL_ALL_575_V88_PIANO_ONSET_CANDIDATES_49_RAW_PITCH_FEATURES"
         and s527["frozen_feature_ledger_SHA256"]==sha(a.audio)
         and s528["status"]=="S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES"
         and s528["predictions_sha256"]==sha(a.pitch)
         and s530["status"]=="S530_FULL_36SECOND_MAESTRO_NOTEON_PITCH_AND_ATTACK_RETROSPECTIVE",
         "S531 immutable source audio, known true MIDI or fixed trained pitch models drifted")
    raw=read_rows(a.audio)
    pitch=read_rows(a.pitch)
    need(len(raw)==len(pitch)==575 and all(
         r["V88_original_event_id"]==p["V88_original_event_id"]
         for r,p in zip(raw,pitch)),
         "S531 external pitch outputs vs raw candidate order")
    rec=np.asarray([r["source_recording_sha"] for r in raw],str)
    times=np.asarray([r["original_V88_candidate_start_s"] for r in raw],float)
    need(len(set(rec))==3,"S531 three recordings need fully separate train/test")
    true={}
    label=np.full(575,-1,np.int8)
    byrec={z:np.flatnonzero(rec==z) for z in sorted(set(rec))}
    source={v["filename_SHA256"]:v for v in manifest["selected_sources"]}
    proof={v["piano_filename_SHA"]:v for v in audit["per_recording"]}
    need(set(source)==set(proof)==set(byrec),
         "S531 unique composer SHA source proof fails")
    paired_proof={}
    for z in sorted(byrec):
        meta=source[z]
        rawmidi=fetch_midi(manifest["audio_mirror_directory"],
                           meta["MIDI_filename_path_ONLY_not_opened"])
        need(digest(rawmidi)==proof[z]["MIDI_bytes_SHA256"],
             "S531 complete MIDI note-on annotations did not match original S471 source SHA")
        events=midi_events(rawmidi)
        offset=float(meta["audio_excerpt_start_seconds"])
        groups=onset_groups(events,offset,float(meta["audio_excerpt_duration_seconds"]))
        true[z]=[dict(time=float(g["onset"]),
                notes=new_notes(events,offset,float(g["onset"]))) for g in groups]
        ids=byrec[z]
        pairs=match_times(times[ids],np.asarray([g["time"] for g in true[z]],float))
        need(len(pairs)==proof[z]["matched_onsets_independent_80ms"]
             and len(groups)==proof[z]["MIDI_attack_groups_in_excerpt"],
             "S531 one-to-one V88 positive/duplicate labels mismatch original S471")
        label[ids]=0
        label[ids[np.asarray([i for i,j,dt in pairs],dtype=int)]]=1
        paired_proof[z]=dict(positive_paired_original_V88_events=len(pairs),
              negative_unpaired_under_80ms_MIDI_matching=len(ids)-len(pairs),
              original_true_MIDI_groups=len(groups),
              true_MIDI_groups_missing_V88=len(groups)-len(pairs))
    need(int(np.sum(label==1))==322 and int(np.sum(label==0))==253
         and all(len(v)>0 for v in true.values()),
         "S531 true 322 matches and 253 nonmatches must remain complete")
    return raw,pitch,rec,times,label,true,paired_proof,s530

def train_gate(X,labels,trainrows,kind):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y=labels[trainrows]
    need(set(y)=={0,1},"S531 onset expert must train real positive and negative events")
    if kind=="et":
        m=ExtraTreesClassifier(n_estimators=192,max_depth=8,
             min_samples_leaf=5,max_features=.75,class_weight="balanced",
             random_state=1531,n_jobs=2)
    elif kind=="logreg":
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.2,
            class_weight="balanced",max_iter=2000,random_state=1531))
    else:raise ValueError("S531 unknown onset-valid model family")
    m.fit(X[trainrows],y)
    return m

def proposed_pitches(audio,pitch,event_prob,kind,threshold):
    """Always retain BP note-births; learned event gate only on originally empty sites."""
    base=proposed_notes(audio,pitch,"BP_birth")
    if base:return base
    candidate=proposed_notes(audio,pitch,"raw_only:0.7")
    return candidate if float(event_prob)>=threshold else set()

def run(a):
    need(not a.out.exists(),"S531 immutable onset expert audit exists")
    raw,pitch,rec,times,y,true,proof,s530=prepare(a)
    X=event_features(raw)
    a.out.mkdir(parents=True)
    import joblib
    prob={}
    folds=[]
    for kind in ("et","logreg"):
        q=np.full(575,np.nan,np.float32)
        for fold,z in enumerate(sorted(set(rec))):
            train=np.flatnonzero(rec!=z)
            test=np.flatnonzero(rec==z)
            need(len(set(rec[train]))==2 and
                 not np.any(rec[train]==z),
                 "S531 held source cannot enter event classifier training")
            model=train_gate(X,y,train,kind)
            p=model.predict_proba(X[test])[:,list(model.classes_).index(1)]
            q[test]=p.astype(np.float32)
            modelpath=a.out/f"S531_{kind}_held{fold}_ACTUAL_ANNOTATED_NOTEON_EVENT_GATE.joblib"
            joblib.dump(model,modelpath,compress=3)
            folds.append(dict(model=kind,fold=fold,held_recording_sha=z,
                  train_other_recording_count=2,
                  train_MIDI_matched_positive_events=int(np.sum(y[train]==1)),
                  train_MIDI_unmatched_negative_events=int(np.sum(y[train]==0)),
                  evaluated_held_MIDI_positive_events=int(np.sum(y[test]==1)),
                  evaluated_held_MIDI_unmatched_events=int(np.sum(y[test]==0)),
                  no_held_MIDI_labels_or_crossfit_pitch_predictions_in_features=True,
                  weights_sha256=sha(modelpath)))
        need(np.isfinite(q).all() and np.all((q>=0)&(q<=1)),
             "S531 every original K0 and nonK0 event needs held probabilities")
        prob[kind]=q
    results={}
    for kind in ("et","logreg"):
        for th in (.5,.7):
            key=f"{kind}_p{th}"
            sums=collections.Counter();per={}
            for z in sorted(set(rec)):
                idx=np.flatnonzero(rec==z)
                pred=[]
                for i in idx:
                    note=proposed_pitches(raw[i],pitch[i],prob[kind][i],kind,th)
                    if note:pred.append(dict(time=float(times[i]),notes=sorted(note)))
                truth=true[z]
                sc=score_event_notes(pred,truth)
                onset=score_event_birth(pred,truth)
                per[z]=dict(noteon=sc,onsets=onset)
                sums.update(note_tp=sc["TP"],note_fp=sc["FP"],note_fn=sc["FN"],
                    note_groups_pred=onset["note_groups_predicted"],
                    onset_tp=onset["physical_true_onset_TP"],
                    onset_fp=onset["physical_predicted_onset_FP"],
                    onset_fn=onset["physical_true_onset_FN"],
                    full_chords=onset["matched_exact_full_note_chords"],
                    exactK=onset["matched_exactK"])
            o=dict(sums)
            o["F1_NoteOn_pitch_and_time"]=2*o["note_tp"]/max(1,2*o["note_tp"]+o["note_fp"]+o["note_fn"])
            o["precision_NoteOn"]=o["note_tp"]/max(1,o["note_tp"]+o["note_fp"])
            o["recall_NoteOn"]=o["note_tp"]/max(1,o["note_tp"]+o["note_fn"])
            o["F1_physical_onset"]=2*o["onset_tp"]/max(1,2*o["onset_tp"]+o["onset_fp"]+o["onset_fn"])
            results[key]=dict(total=o,per_recording=per)
            print("S531 TRAINED GENUINE EVENT GATE HELD-MAESTRO NOTE_ON",
                  json.dumps(dict(policy=key,totals=o)),flush=True)
    need(int(sum(len(x) for x in true.values()))==330
         and s530["scoring"]["raw_only:0.7"]["total"]["note_tp"]>=0,
         "S531 full 330 true MIDI attacks vs original S530 lost")
    ledger=a.out/"S531_ALL_575_CANDIDATE_HELD_ONSET_VALID_PROBABILITIES.jsonl"
    ledger.write_text("".join(json.dumps(dict(
        original_v88_id=raw[i]["V88_original_event_id"],
        recording_sha=rec[i],
        original_v88_K=raw[i]["original_V88_K_soft_prior_ONLY"],
        ET_onset_valid_probability=round(float(prob["et"][i]),6),
        LR_onset_valid_probability=round(float(prob["logreg"][i]),6),
        is_one_to_one_original_true_MIDI_matched_FOR_AUDIT_only=bool(y[i]),
        held_song_truth_never_used_to_fit_model=True))
        +"\n" for i in range(575)))
    report=dict(status="S531_TRUE_TRAINED_RAW_ACOUSTIC_NOTEON_GATE_HELD_SONG_MAESTRO",
        original_complete_MIDI_source_SHA_verified=True,
        physical_candidate_positives=322,
        physical_candidate_unmatched=253,
        MIDI_unmatched_native_candidate_is_benchmark_negative_NOT_silence_label=True,
        original_true_MIDI_groups=330,
        original_true_MIDI_8_groups_without_native_V88_still_missed=True,
        genuine_model_checkpoints=6,
        trained_gate_only_on_original_audio_BP_and_temporal_features=True,
        S528_pitch_crossfit_probabilities_NOT_used_in_event_gate_training=True,
        thresholds_050_070_fixed_and_not_fit_to_test=True,
        F1_all_true_MIDI_onsets_not_just_known_positive_matched=True,
        historical_MAESTRO_test_songs_EXPOSED_NOT_blind=True,
        original_S80_ExactK_and_Live_unchanged=True,no_promotion=True,
        train_splits=folds,by_recording_proof=proof,
        policies=results,
        predictions_sha256=sha(ledger),
        source_sha256=dict(S527=sha(a.audio),S528=sha(a.pitch),
              S471=sha(a.s471_report),S530=sha(a.full_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    X=np.zeros((575,49,11),np.float32)
    z=np.zeros((575,47),np.float32)
    base={"old_basicpitch_birth_pitches":[52]}
    empty={"old_basicpitch_birth_pitches":[]}
    pred={"raw_only_49_pitch_prob":[0]*12+[.81]+[0]*36}
    need(proposed_pitches(base,pred,0.01,"et",.7)=={52}
         and proposed_pitches(empty,pred,.8,"et",.7)=={52}
         and proposed_pitches(empty,pred,.3,"et",.7)==set(),
         "S531 onset gate may never suppress an already-present original BP birth")
    print("S531 SELFTEST PASS independent onset validity only gates new BP-missing note births")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","s471-report","audio","audio-report",
              "pitch","pitch-report","full-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","s471_report","audio","audio_report",
                  "pitch","pitch_report","full_report","out"):
            need(getattr(a,z)is not None,"S531 missing "+z)
        run(a)
