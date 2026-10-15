"""S582 SHA freeze the middle-register specialist vs general28 vs source-trained S569.

Four predeclared policies, fitted on previously labelled 23/28 recordings:
1 old S569 full23 attack specialist reference,
2 NEW28 global 52D full pitch prediction,
3 NEW28 with specialist predicting ONLY MIDI60..71,
4 NEW28 with half specialist/global blend ONLY MIDI60..71.
All original V88 K0 sites, actual BP births protected, fixed p>=0.7,
duplicate 100ms and harmonic80 old emitted policies. No S577 true MIDI.
Every model weight SHA must match the original training proof.
"""
from __future__ import annotations
import argparse,json,collections
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import temporal52
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup
from scripts.s578_train_28_source_middle12_specialist_vs_global52_composerblocked import policies
POLICIES=("S569_old23_control","S578_general28","S578_mid_replace","S578_mid_blend50")
def records(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def predict(rows,a,s569,s578):
    import joblib
    feats=temporal52(rows)
    prior=[x for x in s569["checkpoints_SHA"] if x["architecture"]=="attack_positive_sites"]
    need(len(prior)==1 and len(prior[0]["training_sources"])==23,
         "S582 23-source S569 bona fide causal previous model must be frozen")
    oldpath=a.old_weights/prior[0]["checkpoint_filename"]
    need(oldpath.is_file() and sha(oldpath)==prior[0]["checkpoint_SHA"],
         "S582 true old S569 model weights SHA mismatch")
    old=joblib.load(oldpath)
    oldprob=old.predict_proba(feats.reshape(-1,52))[:,list(old.classes_).index(1)].reshape(len(rows),49)
    trained={z["architecture"]:z for z in s578["weights_before_S577_exposure"]}
    need(set(trained)=={"general28","mid12"} and
         all(x["source_count"]==28 for x in trained.values()),
         "S582 both models fully fitted on same 28 original supervised old recordings")
    gg=a.new_weights/trained["general28"]["filename"]
    mm=a.new_weights/trained["mid12"]["filename"]
    need(gg.is_file() and mm.is_file() and
         sha(gg)==trained["general28"]["sha"] and sha(mm)==trained["mid12"]["sha"],
         "S582 28-song global or middle original checkpoint SHA changed")
    model=joblib.load(gg)
    mid=joblib.load(mm)
    gp=model.predict_proba(feats.reshape(-1,52))[:,list(model.classes_).index(1)].reshape(len(rows),49)
    mp=mid.predict_proba(feats[:,20:32,:].reshape(-1,52))[:,list(mid.classes_).index(1)].reshape(len(rows),12)
    heads=policies(gp,mp)
    out=dict(zip(POLICIES,(oldprob,heads["global28"],heads["specialist_mid_replace"],heads["specialist_mid_blend50"])))
    need(all(v.shape==(len(rows),49) and np.isfinite(v).all()
             and np.all((v>=0)&(v<=1)) for v in out.values()),
         "S582 real 49 MIDI pitch probabilities incomplete")
    return out,[(str(oldpath.name),sha(oldpath)),(str(gg.name),sha(gg)),(str(mm.name),sha(mm))]
def run(a):
    need(not a.out.exists(),"S582 pre-MIDI note policy freeze immutability mandatory")
    m=json.loads(a.manifest.read_text())
    v=json.loads(a.v88_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    fr=json.loads(a.feature_report.read_text())
    old=json.loads(a.old_model_report.read_text())
    new=json.loads(a.new_model_report.read_text())
    checks={
       "S577 registered original train split":m["status"]=="S577_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
          and m["MAESTRO_original_split_used"]=="train",
       "new composers excluded from previous28":m["original_previous_S469_3_S537_5_S545_5_S554_5_S561_5_S570_5_components_excluded"],
       "5 disjoint music composers":m["five_next_distinct_composers"]==5
          and m["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"],
       "no source WAV or MIDI opened before selection":m["source_WAV_files_not_read"] and m["source_MIDI_files_not_read"],
       "S579 original WAV V88":v["status"]=="S579_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
          and v["preregistered_S577_SHA"]==sha(a.manifest)
          and not v["paired_MIDI_annotation_files_OPENED"],
       "S580 Basic Pitch WAV only":bp["status"]=="S580_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
          and not bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"],
       "S581 original physical WAV note features":fr["status"]=="S581_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
          and fr["frozen_feature_ledger_SHA256"]==sha(a.features)
          and fr["no_original_true_MIDI_annotation_read"],
       "S569 genuine learned 23 music":old["status"]=="S569_TWENTYTHREE_SOURCE_RESTRIKE_HARMONIC_GHOST_TEMPORAL_HEADS_REAL_MIDI_TRAINED",
       "S578 fully trained 28 song global+middle":new["status"]=="S578_TWENTYEIGHT_TRUE_PIANO_SOURCE_MIDDLE_MIDI_60_71_EXPERT_VS_GLOBAL_NOTEON"
          and new["no_future_S577_MIDI_or_WAV_read_for_training"],
    }
    if not all(checks.values()):
        print("S582 MUST REFUSE MIDITRUTH OPEN IF PRELABEL SCHEMA CHANGED",
              json.dumps({k:v for k,v in checks.items() if not v}),flush=True)
    need(all(checks.values()),"S582 holdout prelabel provenance mandatory")
    rows=records(a.features)
    identities={x["filename_SHA256"] for x in m["chosen_metadata_only"]}
    need(len(identities)==5 and {x["source_recording_sha"] for x in rows}==identities
         and len(rows)==fr["original_V88_onset_candidates"],
         "S582 five sealed source original V88 pitch feature sets incomplete")
    pp,weights=predict(rows,a,old,new)
    bysrc=collections.defaultdict(list)
    for i,r in enumerate(rows):bysrc[r["source_recording_sha"]].append(i)
    output=[];summary={}
    for policy,prob in pp.items():
        per=[]
        for rec,ii in sorted(bysrc.items()):
            events=[(rows[i],dict(raw_only_49_pitch_prob=prob[i].tolist())) for i in ii]
            emitted,_=dedup(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
            emitted.sort(key=lambda z:z["time"])
            for event in emitted:
                notes=sorted(set(map(int,event["notes"])))
                need(notes and 0<=float(event["time"])<12
                     and all(40<=x<=88 for x in notes),
                     "S582 only genuine pre-MIDI audio timing/pitch output")
                output.append(dict(policy=policy,original_recording_filename_sha=rec,
                  onset_s=round(float(event["time"]),6),
                  pitch_MIDI=notes,MIDI_ANNOTATIONS_NEVER_OPENED=True))
            per.append(dict(source_sha=rec,emitted_attack_groups=len(emitted),
                            emitted_pitch_events=sum(len(z["notes"]) for z in emitted)))
        summary[policy]=dict(per_source=per)
        print("S582 FOUR POLICIES SOURCE-HELD WAV NOTEONS",
              json.dumps(dict(policy=policy,notes=sum(x["emitted_pitch_events"] for x in per))),flush=True)
    a.out.mkdir(parents=True)
    path=a.out/"S582_NEW_FIFTH_COMPOSERS_FOUR_NAMED_NOTES_PREMIDI.jsonl"
    path.write_text("".join(json.dumps(e,sort_keys=True)+"\n" for e in output))
    report=dict(status="S582_FIFTH_COMPOSER_NEW_MIDDLE_HEAD_FOUR_POLICIES_FROZEN_BEFORE_MIDI",
       original_metadata_five_composers_unseen_by_23_and_28_source_train=True,
       source_MIDI_NOT_read_before_frozen_named_note_event_sha=True,
       real_audio_BasicPitch_original_chords_preserved=True,
       true_note_onsets_and_all_pitch_events_fixed_before_MIDI=True,
       four_predeclared_old_vs_new_same_song_policies=list(POLICIES),
       fixed_threshold_original_070_dedup100_harmonic080=True,
       only_MIDI60_71_routed_to_specialist_all_other_unchanged=True,
       no_promotion=True,original_S80_Live_ExactK_GuitarSet_unchanged=True,
       source_shas=dict(S577=sha(a.manifest),S579=sha(a.v88_report),
           S580=sha(a.bp_report),S581=sha(a.features),S569=sha(a.old_model_report),
           S578=sha(a.new_model_report)),
       trained_models_sha=weights,
       frozen_named_noteon_SHA=sha(path),
       policy_counts=summary)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    need(len(POLICIES)==4 and len(set(POLICIES))==4,
         "S582 4 policies must be established before unseen MIDI")
    x=np.ones((1,49),np.float32)*.8;y=np.ones((1,12),np.float32)*.2
    q=policies(x,y)
    need(np.allclose(q["specialist_mid_replace"][:,20:32],.2)
         and np.allclose(q["specialist_mid_blend50"][:,20:32],.5)
         and np.allclose(q["specialist_mid_replace"][:,:20],.8),
         "S582 specialist cannot affect graves or highs")
    print("S582 SELFTEST PASS future MIDI unseen, only 60-71 middle pitch routing")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for key in ("manifest","v88-report","bp-report","features","feature-report",
      "old-model-report","new-model-report","old-weights","new-weights","out"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("manifest","v88_report","bp_report","features","feature_report",
          "old_model_report","new_model_report","old_weights","new_weights","out"):
            need(getattr(a,key) is not None,"S582 missing "+key)
        run(a)
