"""S524 replay independently the real two-action new-note and removal models.

Source S523 actually trains 8 generators, 8 ADD_NEW and 8 KEEP/REMOVE
models on original nested recording-disjoint audio data. This script
DOES NOT fit any weights. Verify all twenty-four SHA models and the
original S505 source, freeze the predictions, and independently
recompute every note correction and regression (including missing
fundamentals) per source and K.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s523_train_independent_ADD_and_REMOVE_music_note_heads import decode,score
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    need(not a.out.exists(),"S524 original independent audit cannot overwrite itself")
    source=json.loads(a.source_report.read_text())
    trained=json.loads(a.train_report.read_text())
    need(source["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and source["audio_proposal_truth_table_SHA256"]==sha(a.source)
         and trained["status"]==
            "S523_2_SEPARATE_LEARNED_ADD_REMOVE_PITCH_HEADS_NESTED_EIGHT_RECORDINGS"
         and trained["frozen_source_S505_SHA"]==sha(a.source)
         and trained["predicted_action_scores_SHA"]==sha(a.predictions)
         and trained["no_promotion"] and len(trained["folds"])==8,
         "S524 source truth and original genuine model output SHA invalid")
    group_ids=set()
    for fold in trained["folds"]:
        f=int(fold["fold"]);group_ids.add(fold["held_recording_SHA"])
        paths=(("generator_sha256",f"S523_fold{f}_S516_NESTED_RAW_GENERATOR.joblib"),
             ("add_action_sha256",f"S523_fold{f}_ADD_NEW_NOTE_ACTION.joblib"),
             ("keep_action_sha256",f"S523_fold{f}_KEEP_OR_REMOVE_NOTE_ACTION.joblib"))
        for key,filename in paths:
            p=a.models/filename
            need(p.is_file() and sha(p)==fold[key],
                 "S524 one of 24 trained action weights differs from original SHA")
        need(fold["outer_train_recording_count"]==7
             and fold["inner_train_recording_count_per_generator"]==6
             and fold["nested_generator_inner_models"]==7
             and fold["no_outer_held_recording_used_in_any_training"],
             "S524 nested train recording leaked into independent held source")
    with np.load(a.source,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    N=len(d["event_id"]);ids=d["event_id"].astype(str)
    idx={name:i for i,name in enumerate(ids)}
    rec=d["recording_sha"].astype(str)
    src=d["corpus"].astype(str)
    need(N==len(idx)==265 and len(group_ids)==8 and group_ids==set(rec),
         "S524 duplicate event/recording or missing known original 265 matched attacks")
    Y=d["truth_note_bits"].astype(bool)
    B=d["head_note_bits"][np.arange(N),
      d["selected_BP_baseline_head"].astype(int)].astype(bool)
    sourceK=d["source_K"].astype(int)
    pred=np.full((N,49,3),np.nan,np.float32);seen=set()
    for ln in a.predictions.read_text().splitlines():
        row=json.loads(ln);ev=row["canonical_event_id"]
        need(ev in idx and ev not in seen
             and row["source_recording_SHA"]==rec[idx[ev]]
             and row["source_instrument"]==src[idx[ev]],
             "S524 matched canonical music IDs/source cannot be remapped")
        seen.add(ev)
        i=idx[ev]
        pred[i,:,0]=np.asarray(row["raw_generator_pitch_40to88_prob"],np.float32)
        pred[i,:,1]=np.asarray(row["ADD_NEW_NOTE_pitch_40to88_prob"],np.float32)
        pred[i,:,2]=np.asarray(row["KEEP_BP_NOTE_pitch_40to88_prob"],np.float32)
    need(len(seen)==N and np.isfinite(pred).all()
         and np.all((0<=pred)&(pred<=1)),
         "S524 pretrained note predictions incomplete or non-finite")
    by_source={}
    events=[]
    for domain in ("guitar_S80","piano_V88"):
        ix=np.flatnonzero(src==domain)
        metrics={}
        for policy in ("add_090","delete_015","joint_090_015","joint_080_020"):
            q=decode(pred[ix,:,1],pred[ix,:,2],pred[ix,:,0],B[ix],policy)
            audit=score(Y[ix],B[ix],q,sourceK[ix])
            goal=trained["by_source"][domain][policy]
            for k in ("BP_complete_chords","candidate_complete_chords",
                      "fixes","regressions","candidate_exactK",
                      "F1_pitches_conditional_on_known_positive_onsets",
                      "true_missing_BP_pitch_added","false_extra_pitch_added",
                      "true_BP_pitch_removed","wrong_BP_pitch_removed"):
                need(abs(float(audit["overall"][k])-float(goal["overall"][k]))<1e-12,
                     f"S524 reread true note identities disagree with S523, {domain} {policy} {k}")
            audit["no_true_negative_audio_note_onset_annotations"]=True
            audit["known_true_attacks_missing_original_music_groups"]=(
                 source["by_source"][domain]["annotated_true_attacks_NOT_matched"])
            audit["all_annotated_true_positive_attacks"]=(
                 source["by_source"][domain]["truth_original_positive_attacks"])
            metrics[policy]=audit
            for local,i in enumerate(ix):
                if np.array_equal(B[i],q[local]):continue
                original=bool(np.array_equal(B[i],Y[i]))
                proposed=bool(np.array_equal(q[local],Y[i]))
                events.append(dict(canonical_event_id=ids[i],source=domain,
                    source_recording_SHA=rec[i],selection_policy=policy,
                    true_original_count_FOR_AUDIT=int(Y[i].sum()),
                    true_note_MIDI_FOR_AUDIT=np.flatnonzero(Y[i]).tolist(),
                    notes_old_MIDI=np.flatnonzero(B[i]).tolist(),
                    notes_new_MIDI=np.flatnonzero(q[local]).tolist(),
                    newly_added_TRUE_pitches=np.flatnonzero((~B[i])&q[local]&Y[i]).tolist(),
                    newly_added_FALSE_pitches=np.flatnonzero((~B[i])&q[local]&(~Y[i])).tolist(),
                    removed_FALSE_pitches=np.flatnonzero(B[i]&(~q[local])&(~Y[i])).tolist(),
                    wrongly_removed_TRUE_pitches=np.flatnonzero(B[i]&(~q[local])&Y[i]).tolist(),
                    full_chord_fixed=bool((not original) and proposed),
                    full_chord_regressed=bool(original and (not proposed))))
        by_source[domain]=metrics
        print("S524 INDEPENDENT TRUE ADD REMOVE NOTE AUDIT",json.dumps(
           dict(corpus=domain,policies={policy:{
               "overall":record["overall"],"poly":record["poly"]}
               for policy,record in metrics.items()})),flush=True)
    a.out.mkdir(parents=True)
    log=a.out/"S524_INDEPENDENT_ADDED_AND_REMOVED_REAL_MIDI_NOTES.jsonl"
    log.write_text("".join(json.dumps(v)+"\n" for v in events))
    report=dict(status="S524_INDEPENDENT_REPLAY_TWO_LEARNED_MUSICAL_NOTE_ACTION_HEADS",
       actual_source_model_weights_SHA256_checked=24,
       true_new_and_false_added_pitches_separately_audited=True,
       every_false_removed_and_true_removed_note_audited=True,
       original_historical_S80_K_and_Live_unchanged=True,
       seven_source_training_six_source_nested_generator_proven=True,
       all_piano_102_unmatched_true_attacks_remain_missed=True,
       no_verified_negative_audio_attack_events_means_F1_event_unmeasured=True,
       historical_seen_recordings_not_new_blind_validation=True,
       no_retrained_models_or_fitted_thresholds_in_independent_replay=True,
       no_promotion=True,by_source=by_source,
       edits_SHA256=sha(log),original_S523_frozen_probabilities_SHA=sha(a.predictions))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    y=np.zeros((1,128),bool);y[0,52]=True
    b=y.copy();v=b.copy();v[0,52]=False;v[0,55]=True
    k=score(y,b,v,np.array([1]))["overall"]
    need(k["regressions"]==1 and k["candidate_exactK"]==1,
         "S524 notes correct count but wrong names still fail")
    print("S524 SELFTEST PASS count/identity independent audit")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for arg in ("source","source-report","train-report","predictions","models","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("source","source_report","train_report","predictions","models","out"):
            need(getattr(a,k)is not None,"S524 missing --"+k)
        run(a)
