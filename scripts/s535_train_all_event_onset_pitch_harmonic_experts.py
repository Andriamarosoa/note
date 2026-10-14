"""S535 train note-pitch experts on ALL prelabel V88 original onset candidates.

Differences from S528: all 253 V88 events without one-to-one MIDI match
provide benchmark-negative NOTE_ON pitches (not assumed silent audio);
322 paired true MIDI groups supply positive pitches. Includes 8 real MIDI
attacks with NO V88 candidate as recall false negatives at evaluation.

Genuine leave-one-whole-piano-recording-out training for 2 independent
learned heads: per-pitch raw acoustic versus harmonic-neighbor competition.
Do not train on S528 held predictions, labels or model confidence.
Never replace any old nonempty Basic Pitch born set at inference.

Original MIDI was historically exposed; therefore a DEVELOPMENT
comparison, no claim of prospective generalization or reference promotion.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
   load_sources,score_event_notes,score_event_birth,need,sha,digest,
   proposed_notes)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
   fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes

def features(raw,mode):
    x=np.asarray([r["per_midi_features_49x11"] for r in raw],np.float32)
    need(x.shape==(575,49,11) and np.isfinite(x).all(),
         "S535 all 575 prelabel WAV event 49x11 independent acoustic inputs required")
    if mode=="raw":return x
    if mode!="harmonic":raise ValueError("S535 unknown architecture")
    extras=[]
    for semis in (12,19,24):
        low=np.zeros_like(x)
        low[:,semis:,:]=x[:,:49-semis,:]
        # [strength,birth burst,novelty,prelabel support] of potential
        # lower played fundamental, plus direct relative energy ratios.
        cols=(2,3,5,8)
        selected=low[:,:,cols]
        relative=x[:,:,cols]-selected
        extras.extend([selected,relative])
    # Per-note physical strength relative to competing *other* notes,
    # without using any ground-truth note identity or labels.
    for idx in (2,3,5):
        z=x[:,:,idx:idx+1]
        maxi=np.max(z,axis=1,keepdims=True)
        extras.append(z/np.maximum(maxi,.001))
    z=np.concatenate([x]+extras,axis=2).astype(np.float32)
    need(z.shape==(575,49,38) and np.isfinite(z).all(),
         "S535 harmonic-competition requires exactly 38 physical features per MIDI note")
    return z

def truth_labels(a,manifest,proof,raw):
    record=np.asarray([r["source_recording_sha"] for r in raw],str)
    times=np.asarray([r["original_V88_candidate_start_s"] for r in raw],float)
    y=np.zeros((575,49),np.int8)
    onsets=np.zeros(575,bool)
    truths={}
    summary={}
    for ident,info in sorted(manifest.items()):
        midi_blob=fetch_midi(a.source_url,info["MIDI_filename_path_ONLY_not_opened"])
        need(digest(midi_blob)==proof[ident]["MIDI_bytes_SHA256"],
             "S535 original true MIDI SHA of recording changed")
        notes=midi_events(midi_blob)
        offset=float(info["audio_excerpt_start_seconds"])
        groups=onset_groups(notes,offset,float(info["audio_excerpt_duration_seconds"]))
        truths[ident]=[dict(time=float(k["onset"]),
                 notes=new_notes(notes,offset,float(k["onset"]))) for k in groups]
        ids=np.flatnonzero(record==ident)
        matching=match_times(times[ids],np.asarray([k["time"] for k in truths[ident]],float))
        need(len(matching)==proof[ident]["matched_onsets_independent_80ms"]
             and len(groups)==proof[ident]["MIDI_attack_groups_in_excerpt"],
             "S535 correct source one-to-one audio candidate MIDI training matching drift")
        for i,j,lag in matching:
            k=int(ids[i])
            need(not onsets[k],"S535 cannot assign MIDI to same V88 event twice")
            onsets[k]=True
            for pitch in truths[ident][j]["notes"]:
                need(40<=int(pitch)<=88,
                     "S535 original full MIDI note outside supported MIDI40..88")
                y[k,int(pitch)-40]=1
        summary[ident]=dict(original_MIDI_true_onsets=len(groups),
                            V88_time_matched_onset_positives=len(matching),
                            V88_time_unmatched_candidate_negative_labels=len(ids)-len(matching),
                            V88_candidates=len(ids))
    need(onsets.sum()==322 and (len(onsets)-onsets.sum())==253
         and sum(len(q) for q in truths.values())==330
         and sum(np.sum(y))==408,
         "S535 full original MIDI 408 matched notes, 322 positives, 253 unmatched negatives drift")
    return record,y,onsets,truths,summary

def fit(X,y,record,architecture,held):
    from sklearn.ensemble import ExtraTreesClassifier
    train=np.flatnonzero(record!=held)
    test=np.flatnonzero(record==held)
    need(len(set(record[train]))==2 and
         not np.any(record[train]==held) and len(test)>0,
         "S535 whole held recording must never contribute training notes")
    xx=X[train].reshape(-1,X.shape[-1])
    yy=y[train].reshape(-1)
    need(int(yy.sum())>50 and int(np.sum(yy==0))>1000,
         "S535 original true notes/non-notes training pool lacks sufficient positives")
    model=ExtraTreesClassifier(n_estimators=128,max_depth=13,
       min_samples_leaf=3,class_weight="balanced_subsample",
       max_features=.8,random_state=53501 if architecture=="raw" else 53502,
       n_jobs=2)
    model.fit(xx,yy)
    prob=model.predict_proba(X[test].reshape(-1,X.shape[-1]))
    pos=int(np.flatnonzero(model.classes_==1)[0])
    q=prob[:,pos].reshape(len(test),49).astype(np.float32)
    return train,test,model,q

def choose(r,prob,th):
    bp=set(int(v) for v in r["old_basicpitch_birth_pitches"]
           if 40<=int(v)<=88)
    if bp:return bp
    return {j+40 for j,z in enumerate(prob) if float(z)>=th}

def run(a):
    need(not a.out.exists(),"S535 frozen true model evidence already exists")
    metadata,s471,s527,s528,s529,source,proof,raw,pretrained=load_sources(a)
    a.source_url=metadata["audio_mirror_directory"]
    record,labels,onsets,true,byrec=truth_labels(a,source,proof,raw)
    a.out.mkdir(parents=True)
    import joblib
    models=[];probs={}
    for mode in ("raw","harmonic"):
        X=features(raw,mode)
        all_p=np.full((575,49),np.nan,np.float32)
        for i,held in enumerate(sorted(set(record))):
            train,test,model,q=fit(X,labels,record,mode,held)
            all_p[test]=q
            filename=a.out/f"S535_{mode}_held{i}_REAL_CANDIDATE_ONSET_PITCH_HEAD.joblib"
            joblib.dump(dict(model=model,mode=mode,
                 training_no_holdout_recording=True,held_original_sha=held),
                 filename,compress=3)
            models.append(dict(model=mode,fold=i,held_sha=held,
                train_other_recordings=2,
                train_true_midi_pitches=int(labels[train].sum()),
                train_unmatched_event_candidates=int(np.sum(~onsets[train])),
                test_all_v88_candidate_events=len(test),
                test_true_pitch_count=int(labels[test].sum()),
                SHA256=sha(filename)))
        need(np.isfinite(all_p).all() and np.all((all_p>=0)&(all_p<=1)),
             "S535 all true trained held note candidates need probability")
        probs[mode]=all_p
    outcomes={}
    for mode in ("raw","harmonic"):
        for threshold in (.5,.7):
            key=f"{mode}_p{threshold}"
            sums=collections.Counter()
            per={}
            for rec,gt in sorted(true.items()):
                ids=np.flatnonzero(record==rec)
                pred=[]
                for j in ids:
                    got=choose(raw[j],probs[mode][j],threshold)
                    if got:
                        pred.append(dict(time=float(raw[j]["original_V88_candidate_start_s"]),
                                         notes=sorted(got)))
                note=score_event_notes(pred,gt)
                ons=score_event_birth(pred,gt)
                per[rec]=dict(noteon=note,ons=ons)
                sums.update(note_tp=note["TP"],note_fp=note["FP"],note_fn=note["FN"],
                    full_chords=ons["matched_exact_full_note_chords"],
                    exactK=ons["matched_exactK"],
                    original_true_groups=ons["note_groups_true"],
                    emitted_groups=ons["note_groups_predicted"],
                    onset_tp=ons["physical_true_onset_TP"],
                    onset_fp=ons["physical_predicted_onset_FP"],
                    onset_fn=ons["physical_true_onset_FN"])
            totals=dict(sums)
            totals["F1_NoteOn"]=2*totals["note_tp"]/max(1,2*totals["note_tp"]+totals["note_fp"]+totals["note_fn"])
            totals["precision_NoteOn"]=totals["note_tp"]/max(1,totals["note_tp"]+totals["note_fp"])
            totals["recall_NoteOn"]=totals["note_tp"]/max(1,totals["note_tp"]+totals["note_fn"])
            outcomes[key]=dict(summary=totals,per_recording=per)
            print("S535 6 ACTUALLY TRAINED ALL-EVENT HARMONIC-PITCH HEAD LOOP",
                  json.dumps(dict(policy=key,total=totals)),flush=True)
    # Snapshot *held predictions* but never true labels as an inference feature.
    file=a.out/"S535_OOF_GENUINELY_TRAINED_575X49_PITCH_PROBABILITIES.jsonl"
    file.write_text("".join(json.dumps(dict(
        V88_original_event_id=r["V88_original_event_id"],
        source_original_recording_sha=record[i],
        raw_onset_note_pitch_prob=probs["raw"][i].round(6).tolist(),
        harmonic_onset_note_pitch_prob=probs["harmonic"][i].round(6).tolist(),
        matched_true_positive_onset_FOR_AUDIT_ONLY=bool(onsets[i])))
        +"\n" for i,r in enumerate(raw)))
    report=dict(status="S535_HELD_FULL_EVENT_NEGATIVE_HARMONIC_PITCH_EXPERTS",
      actual_trained_checkpoints=6,
      training_49_pitch_predictions_at_all_575_V88_original_events=True,
      negative_candidates_253_use_full_original_MIDI_matched_event_rule=True,
      true_positive_MIDI_matched_onsets=322,
      true_full_MIDI_onset_groups=330,
      full_source_note_on_F1_counts_8_true_attacks_without_V88=True,
      original_note_BasicPitch_births_untouched=True,
      no_held_source_in_its_pitch_model_training=True,
      six_source_models_train_only_on_TWO_other_recordings=True,
      original_MAESTRO_three_already_explored_recordings_not_blind=True,
      no_original_S80_or_V88_K_Live_modification=True,
      no_promotion=True,
      by_recording=byrec,
      models=models,policies=outcomes,
      actual_frozen_OOF_probs_SHA256=sha(file))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    z=np.zeros((575,49,11),np.float32)
    raw=features([dict(per_midi_features_49x11=r.tolist()) for r in z],"raw")
    harmonic=features([dict(per_midi_features_49x11=r.tolist()) for r in z],"harmonic")
    need(raw.shape==(575,49,11) and harmonic.shape==(575,49,38),
         "S535 physical 38 feature fundamental-comparison ABI must be immutable")
    b={"old_basicpitch_birth_pitches":[52,64]}
    need(choose(b,np.zeros(49),.7)=={52,64},
         "S535 correct original born chords protected")
    c={"old_basicpitch_birth_pitches":[]}
    p=np.zeros(49);p[12]=.9
    need(choose(c,p,.7)=={52},
         "S535 independent physical pitch candidate generated without BP")
    print("S535 SELFTEST PASS audio fundamental/neighbor feature bank no MIDI leakage")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","s471-report","audio","audio-report",
              "trained","trained-report","protect-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","s471_report","audio","audio_report",
                  "trained","trained_report","protect_report","out"):
            need(getattr(a,z)is not None,"S535 missing "+z)
        run(a)
