"""S550: prospective S545 five-song MIDI-blind NoteOn freeze.

Predeclared six policies: BP, original source-trained S528 3-forest mean
raw>=.7 and protected dedup80, corrected source/co-composer held S546
8-forest mean raw>=.7 and protected dedup80 / dedup100+harmonic .8.

All probabilities come from genuinely fitted OLD piano checkpoint weights;
no S545 WAV/MIDI were used for source-model training. The only differences
are the additional seven/eight-source trained note expert vs S528 and the
fixed same-note duplicate controls, never opening MIDI in this script.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import proposed_notes,sha,need
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit as dedup
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup100

POLICIES=("BP_birth","S528_raw","S528_dedup80","S546_raw","S546_dedup80","S546_dedup100_harm080")
def readlines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def predict_models(X,report,directory,kind):
    import joblib
    if kind=="S528":
        chosen=sorted((x for x in report["models"] if x["feature_policy"]=="raw_only"),
                      key=lambda z:z["held_original_recording_SHA"])
        need(len(chosen)==3,"S550 requires original exact 3 raw trained S528 source models")
        features=X[:,:,[0,1,2,3,4,5,10]]
    else:
        chosen=sorted((x for x in report["models"] if x["architecture"]=="positive7"),
                      key=lambda z:z["held_composer_sha"])
        need(len(chosen)==8 and
             all(x["trained_on_other_source_recordings_only_with_disjoint_composer_components"]
                 for x in chosen) and
             report["shared_composer_collaboration_leakage_is_EXCLUDED"],
             "S550 requires 8 S546 checkpoint folds independently excluding shared co-composers")
        features=X[:,:,[0,1,2,3,4,5,10]]
    allprob=[]
    checkpoints=[]
    for fold,item in enumerate(chosen):
        if kind=="S528":
            name=f"S528_raw_only_held{fold}_ACTUAL_PIANO_AUDIO_NOTE_HEAD.joblib"
            expected=item["trained_model_SHA256"]
        else:
            name=f"S546_positive7_held{fold}_ACTUAL_NEURAL_STYLE_NOTE_DECISION_FOREST.joblib"
            expected=item["genuine_checkpoint_SHA"]
        path=directory/name
        need(path.is_file() and sha(path)==expected,
             f"S550 old fitted {kind} original pitch model SHA missing or changed: "+name)
        fitted=joblib.load(path)
        if kind=="S528":
            need(fitted["feature_columns"]==[0,1,2,3,4,5,10] and
                 fitted["held_sha"]==item["held_original_recording_SHA"],
                 "S550 S528 source held & feature column ABI changed")
            model=fitted["model"]
        else:model=fitted
        labels=list(model.classes_)
        need(1 in labels or True in labels,
             "S550 source-weighted model lacks actual true-note class")
        key=labels.index(1) if 1 in labels else labels.index(True)
        p=model.predict_proba(features.reshape(-1,features.shape[-1]))[:,key].reshape(X.shape[0],49)
        allprob.append(np.asarray(p,np.float32))
        checkpoints.append(dict(architecture=kind,filename=name,sha=expected))
    probs=np.mean(np.stack(allprob),axis=0)
    need(np.isfinite(probs).all() and
         probs.shape==(X.shape[0],49) and np.all((probs>=0)&(probs<=1)),
         "S550 new five source notes must have 49 real pitch probabilities")
    return probs,checkpoints
def emit_one(events,policy,head):
    if policy=="BP_birth" or policy.endswith("_raw"):
        source="BP_birth" if policy=="BP_birth" else "raw_only:0.7"
        out=[]
        for r,p in events:
            pitches=proposed_notes(r,p,source)
            if pitches:out.append(dict(time=float(r["original_V88_candidate_start_s"]),notes=sorted(pitches)))
        return out
    if policy.endswith("_dedup80"):
        result,_=dedup(events,dict(dedup_ms=80,harmonic_ratio=0))
        return result
    if policy=="S546_dedup100_harm080":
        result,_=dedup100(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
        return result
    raise ValueError("S550 unknown policy "+policy)
def run(a):
    need(not a.out.exists(),"S550 prelabel inference immutable, never rewrite")
    meta=json.loads(a.manifest.read_text())
    source=json.loads(a.v88_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    feat=json.loads(a.feature_report.read_text())
    r528=json.loads(a.old_model_report.read_text())
    r546=json.loads(a.new_model_report.read_text())
    need(meta["status"]=="S545_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and meta["five_next_distinct_composers"]==5
         and meta["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"]
         and meta["source_MIDI_files_not_read"]
         and source["status"]=="S547_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and source["prelabel_predictions_frozen"]
         and not source["paired_MIDI_annotation_files_OPENED"]
         and source["preregistered_S545_SHA"]==sha(a.manifest)
         and bp["status"]=="S548_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and not bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"]
         and feat["status"]=="S549_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and feat["frozen_feature_ledger_SHA256"]==sha(a.features)
         and feat["no_original_true_MIDI_annotation_read"]
         and r528["status"]=="S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES"
         and r546["status"]=="S546_8_COMPOSER_REAL_NOTE_IDENTITY_EXPERT_TRAINING_AND_RETRO_LOO"
         and r546["shared_composer_collaboration_leakage_is_EXCLUDED"]
         and r546["NOT_new_prospective_blind_generalization"],
         "S550 new source prelabel WAV and fitted S528/S546 provenance must all be valid")
    rows=readlines(a.features)
    ids={x["filename_SHA256"] for x in meta["chosen_metadata_only"]}
    need(len(ids)==5 and len(rows)==feat["original_V88_onset_candidates"]
         and {r["source_recording_sha"] for r in rows}==ids
         and len({r["V88_original_event_id"] for r in rows})==len(rows),
         "S550 must freeze every original V88 candidate for each five S545 source")
    X=np.asarray([r["per_midi_features_49x11"] for r in rows],np.float32)
    need(X.shape==(len(rows),49,11),"S550 unmodified 49x11 per pitch audio feature ABI required")
    p528,weights528=predict_models(X,r528,a.old_models,"S528")
    p546,weights546=predict_models(X,r546,a.new_models,"S546")
    bysource=collections.defaultdict(list)
    for i,r in enumerate(rows):bysource[r["source_recording_sha"]].append(i)
    allnotes=[]
    summary={}
    for policy in POLICIES:
        source_name="S528" if policy.startswith("S528") else "S546"
        pp=p528 if source_name=="S528" else p546
        per=[]
        for ident in sorted(bysource):
            idsrow=bysource[ident]
            events=[(rows[i],dict(raw_only_49_pitch_prob=pp[i].tolist())) for i in idsrow]
            predicted=emit_one(events,policy,source_name)
            predicted.sort(key=lambda z:z["time"])
            for z in predicted:
                note=[int(x) for x in z["notes"]]
                need(note and all(40<=x<=88 for x in note)
                     and 0<=float(z["time"])<12,
                     "S550 only genuine MIDI40..88 named notes at WAV audio event times")
                allnotes.append(dict(policy=policy,original_recording_filename_sha=ident,
                  onset_s=round(float(z["time"]),6),pitch_MIDI=sorted(note),
                  MIDI_ANNOTATIONS_NEVER_OPENED=True))
            per.append(dict(source_sha=ident,emitted_attack_groups=len(predicted),
                  emitted_notes=sum(len(z["notes"]) for z in predicted)))
        summary[policy]=dict(per_source=per,attack_groups=sum(q["emitted_attack_groups"] for q in per),
                             emitted_pitch_events=sum(q["emitted_notes"] for q in per))
        print("S550 FIVE NEW COMPOSER NOTE_ON POLICY BEFORE MIDI",
              json.dumps(dict(policy=policy,events=summary[policy]["attack_groups"],
                              pitches=summary[policy]["emitted_pitch_events"])),flush=True)
    a.out.mkdir(parents=True)
    filename=a.out/"S550_NEXT_FIVE_REAL_WAV_NOTEON_PREDICTIONS_PREMIDI.jsonl"
    filename.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in allnotes))
    probfile=a.out/"S550_S528_AND_S546_NEXT_FIVE_SOURCE_HELD_RAW49_PROB_PREMIDI.jsonl"
    probfile.write_text("".join(json.dumps(dict(
         original_v88_id=r["V88_original_event_id"],
         original_rec_sha=r["source_recording_sha"],
         s528_prob=p528[i].round(6).tolist(),
         s546_prob=p546[i].round(6).tolist()))
         +"\n" for i,r in enumerate(rows)))
    report=dict(status="S550_FIVE_COMPOSER_S545_PITCH_ONSET_SOURCE_FROZEN_BEFORE_ANY_MIDI",
      true_new_MAESTRO_original_split="train",
      original_five_songs_unseen_in_S528_and_S546_training=True,
      original_MIDI_NOT_opened_for_predictions=True,
      real_frozen_source_original_V88_candidate_count=len(rows),
      all_six_policies_declared_before_labels=list(POLICIES),
      threshold_shared_no_held_tuning=.7,
      duplicate_window_ms=[80,100],
      harmonic_veto=.8,
      original_BP_born_notes_always_protected=True,
      SHAs_of_three_original_S528_weight_checkpoints=weights528,
      SHAs_of_eight_S546_co_composer_disjoint_weight_checkpoints=weights546,
      frozen_named_note_prediction_sha=sha(filename),
      frozen_49pitch_probabilities_sha=sha(probfile),
      source_sha=dict(S545=sha(a.manifest),S547=sha(a.v88_report),
                      S548=sha(a.bp_report),S549=sha(a.features),
                      S528=sha(a.old_model_report),S546=sha(a.new_model_report)),
      predictions_by_policy=summary,
      original_S80_ExactK_Live_unchanged=True,no_promotion=True)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S550 SEALED",json.dumps(dict(note_sha=sha(filename),
         five_songs=len(ids),model_count=11,policies=list(POLICIES))),flush=True)
def selftest():
    need(len(POLICIES)==6 and len(set(POLICIES))==6
         and POLICIES[:3]==("BP_birth","S528_raw","S528_dedup80"),
         "S550 6 policies must be fixed without MIDI")
    r=dict(old_basicpitch_birth_pitches=[52,64],original_V88_candidate_start_s=.1)
    h=dict(raw_only_49_pitch_prob=np.zeros(49).tolist())
    for s in ["BP_birth","S546_raw"]:
        need(emit_one([(r,h)],s,"S546")[0]["notes"]==[52,64],
             "S550 cannot suppress existing BP polyphonic chord")
    print("S550 SELFTEST PASS MIDI blind six candidate policy ABI and protected BP chords")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","v88-report","bp-report","features","feature-report",
              "old-model-report","new-model-report","old-models","new-models","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","v88_report","bp_report","features","feature_report",
                  "old_model_report","new_model_report","old_models","new_models","out"):
            need(getattr(a,z) is not None,"S550 missing "+z)
        run(a)
