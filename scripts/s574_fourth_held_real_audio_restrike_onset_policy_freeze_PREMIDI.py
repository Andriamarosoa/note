"""S574 real five-source fourth-cohort S570 MIDI-BLIND note-on freeze.

All genuine source NoteOn predictors were FIT on previous 8,18 or 23
musical recordings. SHA-check every trained file, verify S570 distinct
composer metadata, and freeze seven declared MIDI note+time policies
before reading a single S570 paired MIDI. 49 pitches MIDI 40..88, always
preserve original BP-born chords and use original thresholds 0.70;
dedup80 and dedup100/harmonic 0.8 are frozen old policies. No fitting
or threshold tuning on S570. Original S80 Exact-K/GuitarSet/Live untouched.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,proposed_notes
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit as dedup80
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup100
from scripts.s562_train_18_composer_true_pitch_spectral_fundamental_heads import acoustic
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import temporal52
POLICIES=("BP_birth","S546_old8_100","S562_physical34_100",
          "S569_attack_positive_sites_80","S569_attack_positive_sites_100",
          "S569_attack_and_ghost_events_80","S569_attack_and_ghost_events_100")
ARCHS=("attack_positive_sites","attack_and_ghost_events")
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def models_predict(raw,a,old,new,extra):
    import joblib
    X=np.asarray([x["per_midi_features_49x11"] for x in raw],np.float32)
    need(X.shape==(len(raw),49,11) and np.isfinite(X).all(),
         "S574 original MIDI-blind source 49x11 physical note features must be complete")
    x7=acoustic(X,"raw7");x34=acoustic(X,"competing_fundamental34")
    x52=temporal52(raw)
    need(x52.shape==(len(raw),49,52),"S574 genuine source-separated audio temporal 52D")
    proof=[]
    mean=[]
    eligible=sorted([r for r in old["models"] if r["architecture"]=="positive7"],
                    key=lambda z:z["held_composer_sha"])
    need(len(eligible)==8 and old["shared_composer_collaboration_leakage_is_EXCLUDED"],
         "S574 genuine old 8 composer-held note experts required for same-song baseline")
    for i,m in enumerate(eligible):
        f=a.s546_weights/f"S546_positive7_held{i}_ACTUAL_NEURAL_STYLE_NOTE_DECISION_FOREST.joblib"
        need(f.is_file() and sha(f)==m["genuine_checkpoint_SHA"],
             "S574 old 8 strict composer held model weight SHA mismatch")
        z=joblib.load(f);cl=list(z.classes_).index(1)
        mean.append(z.predict_proba(x7.reshape(-1,7))[:,cl].reshape(len(raw),49))
        proof.append(dict(source="S546",file=f.name,sha=sha(f)))
    pred={"S546":np.mean(np.stack(mean),axis=0).astype(np.float32)}
    choose=[r for r in new["all_fitted_checkpoints_SHA256"]
            if r["architecture"]=="competing_fundamental34"]
    need(len(choose)==1 and choose[0]["number"]==18,
         "S574 old 18 source physical34 real-trained model must exist for control")
    m=choose[0]
    f=a.s562_weights/f"S562_FULL18_competing_fundamental34_REAL_COMPOSER_DISJOINT_HELD_SOURCE_FIT.joblib"
    need(f.is_file() and sha(f)==m["source_sha_model"],
         "S574 original 18 source full fit weights changed")
    z=joblib.load(f);cl=list(z.classes_).index(1)
    pred["S562"]=z.predict_proba(x34.reshape(-1,34))[:,cl].reshape(len(raw),49)
    proof.append(dict(source="S562",file=f.name,sha=sha(f)))
    need(extra["status"]=="S569_TWENTYTHREE_SOURCE_RESTRIKE_HARMONIC_GHOST_TEMPORAL_HEADS_REAL_MIDI_TRAINED"
         and extra["real_23_music_source_train_and_source_composer_held_folds"]
         and extra["temporal_52_acoustic_features_from_prior_actual_WAV_events_ONLY"]
         and extra["future_new_composers_NEVER_consulted_for_training"]
         and len(extra["checkpoints_SHA"])==2,
         "S574 23 source old fit cannot have new S570 MIDI or audio")
    for m in extra["checkpoints_SHA"]:
        arch=m["architecture"]
        need(arch in ARCHS and
             len(m["training_sources"])==23 and m["no_future_MIDI_used_in_checkpoint_training"],
             "S574 real 23 original source and no new holdout training")
        f=a.s569_weights/m["checkpoint_filename"]
        need(f.is_file() and sha(f)==m["checkpoint_SHA"],
             "S574 S569 learned reattack head model SHA mismatch")
        z=joblib.load(f);cl=list(z.classes_).index(1)
        pred["S569_"+arch]=z.predict_proba(x52.reshape(-1,52))[:,cl].reshape(len(raw),49)
        proof.append(dict(source="S569_"+arch,file=f.name,sha=sha(f)))
    need(len(proof)==11 and set(pred)=={"S546","S562","S569_attack_positive_sites","S569_attack_and_ghost_events"}
         and all(np.isfinite(p).all() and p.shape==(len(raw),49) and np.all((p>=0)&(p<=1))
                 for p in pred.values()),"S574 genuine trained 11 old weights four source pitch predictions")
    return pred,proof

def emitter(events,policy):
    if policy=="BP_birth":
        result=[]
        for r,p in events:
            notes=proposed_notes(r,p,"BP_birth")
            if notes:result.append(dict(time=float(r["original_V88_candidate_start_s"]),notes=sorted(notes)))
        return result
    need(policy in POLICIES,"S574 only original seven policies allowed")
    if policy.endswith("_80"):
        out,_=dedup80(events,dict(dedup_ms=80,harmonic_ratio=0))
    else:
        out,_=dedup100(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
    return out

def run(a):
    need(not a.out.exists(),"S574 new source MIDI-blind prediction artifact immutable")
    meta=json.loads(a.manifest.read_text())
    v=json.loads(a.v88_report.read_text())
    bp=json.loads(a.bp_report.read_text())
    feat=json.loads(a.feature_report.read_text())
    old=json.loads(a.s546_report.read_text())
    new=json.loads(a.s562_report.read_text())
    extra=json.loads(a.s569_report.read_text())
    checks={
      "S570 exact metadata":meta["status"]=="S570_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI",
      "S570 no old 23 composer":meta["original_previous_S469_3_S537_5_S545_5_S554_5_S561_5_component_excluded"],
      "S570 5 distinct":meta["five_next_distinct_composers"]==5 and meta["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"],
      "S570 prereg before wave":meta["source_WAV_files_not_read"] and meta["source_MIDI_files_not_read"],
      "S571 original V88":v["status"]=="S571_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI",
      "S571 names SHA":v["preregistered_S570_SHA"]==sha(a.manifest),
      "S571 labels unopened":not v["paired_MIDI_annotation_files_OPENED"] and v["prelabel_predictions_frozen"],
      "S572 basicpitch":bp["status"]=="S572_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI",
      "S572 MIDI unopened":not bp["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"],
      "S573 features":feat["status"]=="S573_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES",
      "S573 features SHA":feat["frozen_feature_ledger_SHA256"]==sha(a.features),
      "S573 MIDI unopened":feat["no_original_true_MIDI_annotation_read"],
      "S546 weights provenance":old["status"]=="S546_8_COMPOSER_REAL_NOTE_IDENTITY_EXPERT_TRAINING_AND_RETRO_LOO",
      "S562 weights provenance":new["status"]=="S562_REAL_EIGHTEEN_EXPOSED_COMPOSER_SOURCE_AUDIO_TRAINED_NOTE_PITCH_TWO_HEADS",
      "S569 actual learned weights provenance":extra["status"]=="S569_TWENTYTHREE_SOURCE_RESTRIKE_HARMONIC_GHOST_TEMPORAL_HEADS_REAL_MIDI_TRAINED",
    }
    if not all(checks.values()):print("S574 NO-MIDI PROVENANCE FAILURE",json.dumps({k:x for k,x in checks.items() if not x}),flush=True)
    need(all(checks.values()),"S574 refuse to open test MIDI or freeze from changed source")
    raw=rows(a.features)
    recs={r["filename_SHA256"] for r in meta["chosen_metadata_only"]}
    need(len(recs)==5 and {r["source_recording_sha"] for r in raw}==recs
         and len(raw)==feat["original_V88_onset_candidates"],
         "S574 every true old V88 K0 note candidate must be included")
    pp,models=models_predict(raw,a,old,new,extra)
    by=collections.defaultdict(list)
    for i,r in enumerate(raw):by[r["source_recording_sha"]].append(i)
    out=[];summary={}
    for policy in POLICIES:
        arch=("S546" if policy.startswith("S546") else "S562" if policy.startswith("S562")
              else "S569_attack_positive_sites" if "attack_positive_sites" in policy
              else "S569_attack_and_ghost_events")
        p=pp[arch]
        cnt=[]
        for src,indices in sorted(by.items()):
            events=emitter([(raw[i],dict(raw_only_49_pitch_prob=p[i].tolist())) for i in indices],policy)
            events.sort(key=lambda z:z["time"])
            for event in events:
                ps=sorted(set(map(int,event["notes"])))
                need(ps and all(40<=z<=88 for z in ps) and 0<=float(event["time"])<12,
                     "S574 emitted MIDI named pitch and event time must be supported")
                out.append(dict(policy=policy,original_recording_filename_sha=src,
                    onset_s=round(float(event["time"]),6),pitch_MIDI=ps,
                    MIDI_ANNOTATIONS_NEVER_OPENED=True))
            cnt.append(dict(source_sha=src,emitted_attack_groups=len(events),
                            emitted_pitch_events=sum(len(z["notes"]) for z in events)))
        summary[policy]=dict(per_source=cnt,
          emitted_pitch_events=sum(z["emitted_pitch_events"] for z in cnt))
        print("S574 SEVEN FULL NEW SONG MIDI BLIND NOTE POLICIES",
             json.dumps(dict(policy=policy,pitches=summary[policy]["emitted_pitch_events"])),flush=True)
    a.out.mkdir(parents=True)
    predictions=a.out/"S574_FIVE_FULL_REAL_AUDIO_NAMED_NOTEON_7_POLICIES_PREMIDI.jsonl"
    predictions.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in out))
    rep=dict(status="S574_FOURTH_SOURCE_FIVE_COMPOSER_REAL_NOTEON_SEVEN_POLICY_MIDI_BLIND_FROZEN",
      all_seven_policies_preregistered_before_true_MIDI=list(POLICIES),
      original_S570_MIDI_NOT_read_for_inference=True,
      no_S570_ground_truth_used_for_training_model_or_policy=True,
      original_BP_birth_notes_protected=True,
      true_MAESTRO_split="train",NOT_official_MAESTRO_test=True,
      original_S80_ExactK_GuitarSet_Live_unmodified=True,
      no_promotion=True,real_original_source_model_SHA_checkpoints=models,
      full_named_note_on_predictions_sha=sha(predictions),
      old_V88_pitch_site_count=len(raw),
      source_sha=dict(S570=sha(a.manifest),S571=sha(a.v88_report),S572=sha(a.bp_report),
          S573=sha(a.features),S546=sha(a.s546_report),S562=sha(a.s562_report),S569=sha(a.s569_report)),
      per_policy=summary)
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    need(len(POLICIES)==7 and len(set(POLICIES))==7,"S574 frozen seven policies")
    row=dict(old_basicpitch_birth_pitches=[52,64],original_V88_candidate_start_s=.1)
    prob=dict(raw_only_49_pitch_prob=np.zeros(49).tolist())
    need(emitter([(row,prob)],"BP_birth")==[dict(time=.1,notes=[52,64])],
         "S574 cannot erase old BP born chord")
    print("S574 SELFTEST PASS exact source old 7 policies, BP chord protected, no target MIDI")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true")
    for k in ("manifest","v88-report","bp-report","features","feature-report","s546-report",
              "s562-report","s569-report","s546-weights","s562-weights","s569-weights","out"):
        parser.add_argument("--"+k,type=Path)
    a=parser.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("manifest","v88_report","bp_report","features","feature_report","s546_report",
                  "s562_report","s569_report","s546_weights","s562_weights","s569_weights","out"):
            need(getattr(a,k) is not None,"S574 missing "+k)
        run(a)
