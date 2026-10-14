"""S530 full-interval MIDI NoteOn event evaluation of original S529 policies.

Evaluate ALL MIDI note-on groups on the same three 12-s original MAESTRO
piano excerpts: unmatched TRUE onsets retained, every predicted pitch
at all frozen 575 V88 times counted. Score pitch+onset matching with
one-to-one per-pitch Hungarian <=80 ms, and grouped event matching.

This is a retrospective fully annotated MIDI evaluation, NOT a new
blind split, NOT standalone trained YourMT3, and NOT an official
S80 K replay. MIDI pedal/audio ambiguity, onset alignment, and
the historical selected excerpts remain limitations. No ref promotion.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times,SR)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes

LIMIT=.080
NAMES=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(x).hexdigest()
def read_rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def name(p):return NAMES[int(p)%12]+str(int(p)//12-1)

def assignment(pred,true,limit=LIMIT):
    """Max-cardinality, min-distance matching with independent dummy columns."""
    n=len(pred);m=len(true)
    if not n or not m:return []
    d=np.abs(np.asarray(pred,float)[:,None]-np.asarray(true,float)[None,:])
    cost=np.concatenate((np.where(d<=limit,d/max(limit,1e-8)*.25,100000.),
                         np.ones((n,n),dtype=float)),axis=1)
    ri,ci=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(ri,ci) if j<m and d[i,j]<=limit]

def score_event_notes(p_events,t_events):
    """No MIDI labels used to select a model event; match on note name and time."""
    pt=collections.defaultdict(list);tt=collections.defaultdict(list)
    for event in p_events:
        for p in event["notes"]:pt[int(p)].append(float(event["time"]))
    for event in t_events:
        for p in event["notes"]:tt[int(p)].append(float(event["time"]))
    TP=0;paired={}
    for p in sorted(set(pt)|set(tt)):
        matches=assignment(pt[p],tt[p])
        TP+=len(matches)
        paired[str(name(p))]=len(matches)
    P=sum(map(len,pt.values()));T=sum(map(len,tt.values()))
    need(TP<=P and TP<=T,"S530 note onset TP cannot exceed either source")
    return dict(TP=TP,FP=P-TP,FN=T-TP,
                precision=TP/max(P,1),recall=TP/max(T,1),
                F1=2*TP/max(P+T,1),
                predicted_pitch_onsets=P,true_pitch_onsets=T,
                correct_pitch_names_by_identity=paired)

def score_event_birth(p_events,t_events):
    pair=assignment([x["time"] for x in p_events],
                    [x["time"] for x in t_events])
    tp=len(pair)
    full=sum(set(p_events[i]["notes"])==set(t_events[j]["notes"]) for i,j in pair)
    K=sum(len(p_events[i]["notes"])==len(t_events[j]["notes"]) for i,j in pair)
    return dict(physical_true_onset_TP=tp,
                physical_predicted_onset_FP=len(p_events)-tp,
                physical_true_onset_FN=len(t_events)-tp,
                true_onset_recall=tp/max(len(t_events),1),
                onset_precision=tp/max(len(p_events),1),
                onset_F1=2*tp/max(len(p_events)+len(t_events),1),
                matched_exact_full_note_chords=full,
                matched_exactK=K,
                note_groups_predicted=len(p_events),
                note_groups_true=len(t_events))

def load_sources(a):
    s469=json.loads(a.manifest.read_text())
    s471=json.loads(a.s471_report.read_text())
    s527=json.loads(a.audio_report.read_text())
    s528=json.loads(a.trained_report.read_text())
    s529=json.loads(a.protect_report.read_text())
    need(s469["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and not s469["MIDI_note_contents_read"]
         and s471["status"]=="S471_EXTERNAL_THIRD_REAL_MAESTRO_PIANO_V88_MIDI_POSTFREEZE"
         and s471["SHA256"]["S469_preregistration"]==sha(a.manifest)
         and s527["status"]=="S527_PRELABEL_ALL_575_V88_PIANO_ONSET_CANDIDATES_49_RAW_PITCH_FEATURES"
         and s527["frozen_feature_ledger_SHA256"]==sha(a.audio)
         and s528["status"]=="S528_TRAINED_THREE_RECORDING_HELD_PIANO_NOTE_HEADS_ALL_V88_K0_SITES"
         and s528["predictions_sha256"]==sha(a.trained)
         and s529["status"]=="S529_ORIGINAL_V88_ONSET_K0_PROTECTED_NOTE_PITCH_RESCUE_NO_RETRAIN",
         "S530 source identity/original MIDI provenance invalid")
    recordings={r["piano_filename_SHA"]:r for r in s471["per_recording"]}
    need(len(recordings)==3 and s471["cohort_sums"]["midi_groups"]==330
         and s471["cohort_sums"]["matched"]==322,
         "S530 original full 330-group MAESTRO MIDI annotations changed")
    raw=read_rows(a.audio); trained=read_rows(a.trained)
    need(len(raw)==len(trained)==575 and all(
        v["V88_original_event_id"]==p["V88_original_event_id"]
        for v,p in zip(raw,trained)), "S530 acoustic evidence and held pitch probs misaligned")
    source={q["filename_SHA256"]:q for q in s469["selected_sources"]}
    need(set(source)==set(recordings)==set(x["source_recording_sha"] for x in raw),
         "S530 three distinct piano source fingerprints drifted")
    for p in trained:
        for f in ("raw_only_49_pitch_prob","with_BP_49_pitch_prob"):
            need(len(p[f])==49 and all(0<=float(v)<=1 for v in p[f]),
                 "S530 must use real frozen trained probabilities for every MIDI pitch")
    return s469,s471,s527,s528,s529,source,recordings,raw,trained

def proposed_notes(original,learned,policy):
    born=set(int(p) for p in original["old_basicpitch_birth_pitches"]
             if 40<=int(p)<=88)
    if policy=="BP_birth":return born
    kind,threshold=policy.split(":")
    threshold=float(threshold)
    field="raw_only_49_pitch_prob" if kind=="raw_only" else "with_BP_49_pitch_prob"
    if born:return born
    return set(40+j for j,v in enumerate(learned[field]) if float(v)>=threshold)

def run(a):
    need(not a.out.exists(),"S530 immutable NoteOn MIDI full-stream audit already exists")
    spec,s471,s527,s528,s529,source,proof,raw,trained=load_sources(a)
    original=collections.defaultdict(list)
    for x,p in zip(raw,trained):
        original[x["source_recording_sha"]].append((x,p))
    source_true={}
    for ident,item in sorted(source.items()):
        blob=fetch_midi(spec["audio_mirror_directory"],
                        item["MIDI_filename_path_ONLY_not_opened"])
        need(digest(blob)==proof[ident]["MIDI_bytes_SHA256"],
             "S530 no original MIDI labels may be silently switched")
        notes=midi_events(blob)
        offset=float(item["audio_excerpt_start_seconds"])
        duration=float(item["audio_excerpt_duration_seconds"])
        groups=onset_groups(notes,offset,duration)
        truth=[dict(time=float(z["onset"]),
            notes=new_notes(notes,offset,float(z["onset"]))) for z in groups]
        need(len(truth)==proof[ident]["MIDI_attack_groups_in_excerpt"],
             "S530 MIDI grouping differs from immutable S471")
        need(all(len(x["notes"])>0 for x in truth),
             "S530 original MIDI onset note identities unexpectedly empty")
        candidates=[dict(time=x["original_V88_candidate_start_s"],
                         notes=[int(z) for z in x["old_basicpitch_birth_pitches"]])
                    for x,_ in original[ident]]
        original_pairs=match_times(
            np.asarray([v["time"] for v in candidates],float),
            np.asarray([v["time"] for v in truth],float))
        need(len(original_pairs)==proof[ident]["matched_onsets_independent_80ms"],
             "S530 original V88 322 total one-to-one candidates timing changed")
        source_true[ident]=truth
        print("S530 SOURCE COMPLETE ANNOTATED MIDI ONSETS",
              json.dumps(dict(rec=ident,
                MIDI_true_note_on_groups=len(truth),
                MIDI_true_note_pitches=sum(len(x["notes"]) for x in truth),
                V88_frozen_candidate_groups=len(candidates),
                V88_any_candidate_matched=len(original_pairs))),flush=True)
    results={}
    for policy in ("BP_birth","raw_only:0.5","raw_only:0.7",
                   "with_BP:0.5","with_BP:0.7"):
        totals=collections.Counter();per_source={}
        for ident in sorted(source):
            pred=[]
            for x,p in original[ident]:
                pitches=proposed_notes(x,p,policy)
                if pitches:
                    pred.append(dict(time=float(x["original_V88_candidate_start_s"]),
                                     notes=sorted(pitches)))
            truth=source_true[ident]
            note=score_event_notes(pred,truth)
            birth=score_event_birth(pred,truth)
            per_source[ident]=dict(noteon=note,onsets=birth)
            totals.update(note_tp=note["TP"],note_fp=note["FP"],note_fn=note["FN"],
                          groups_pred=birth["note_groups_predicted"],
                          groups_true=birth["note_groups_true"],
                          onset_tp=birth["physical_true_onset_TP"],
                          onset_fp=birth["physical_predicted_onset_FP"],
                          onset_fn=birth["physical_true_onset_FN"],
                          full_chords=birth["matched_exact_full_note_chords"],
                          exactK=birth["matched_exactK"])
        summary=dict(totals)
        summary["note_on_pitch_precision"]=summary["note_tp"]/max(1,summary["note_tp"]+summary["note_fp"])
        summary["note_on_pitch_recall"]=summary["note_tp"]/max(1,summary["note_tp"]+summary["note_fn"])
        summary["note_on_pitch_F1"]=2*summary["note_tp"]/max(1,2*summary["note_tp"]+summary["note_fp"]+summary["note_fn"])
        summary["physical_onset_F1"]=2*summary["onset_tp"]/max(1,2*summary["onset_tp"]+summary["onset_fp"]+summary["onset_fn"])
        results[policy]=dict(total=summary,per_recording=per_source)
        print("S530 TRUE FULL-INTERVAL NOTE_ON VERDICT",json.dumps(
              dict(policy=policy,summary=summary)),flush=True)
    need(results["BP_birth"]["total"]["groups_true"]==330,
         "S530 can't drop true MIDI onsets just because native V88 did not detect")
    a.out.mkdir(parents=True)
    report=dict(status="S530_FULL_36SECOND_MAESTRO_NOTEON_PITCH_AND_ATTACK_RETROSPECTIVE",
      real_complete_MAESTRO_MIDI_note_onsets_reopened_and_SHA_verified=True,
      evaluated_frozen_piano_excerpts=3,
      original_true_MIDI_onset_groups_including_8_without_V88=330,
      original_frozen_V88_candidate_groups=575,
      pitch_matching_at_most_80ms_one_to_one_exact_name=True,
      original_S529_4_policies_no_retraining_or_MIDI_decision_features=True,
      full_noteon_F1_for_three_HISTORICALLY_seen_piano_excerpts_ONLY=True,
      not_a_new_blind_independent_corpus=True,
      not_a_full_S80_or_GuitarSet_score=True,
      no_oracle_label_based_threshold_selection=True,
      original_S80_K_and_Live_unchanged=True,no_promotion=True,
      scoring=results,
      source_sha256=dict(S469=sha(a.manifest),S471=sha(a.s471_report),
                         S527=sha(a.audio),S528=sha(a.trained),S529=sha(a.protect_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    p=[dict(time=.1,notes=[52,59]),dict(time=.3,notes=[55]),
       dict(time=.31,notes=[52])]
    t=[dict(time=.11,notes=[52,55]),dict(time=.3,notes=[55])]
    z=score_event_notes(p,t)
    need(z["TP"]==2 and z["FP"]==2 and z["FN"]==1,
         "S530 a same-timestamp wrong named pitch cannot count as correct event")
    b=score_event_birth(p,t)
    need(b["physical_true_onset_TP"]==2 and
         b["physical_predicted_onset_FP"]==1,
         "S530 one physical true attack can match only one candidate")
    print("S530 SELFTEST PASS full NoteOn pitch/timing matching and duplicate false-onset penalty")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for x in ("manifest","s471-report","audio","audio-report",
              "trained","trained-report","protect-report","out"):
        p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("manifest","s471_report","audio","audio_report","trained",
                  "trained_report","protect_report","out"):
            need(getattr(a,x) is not None,"S530 missing "+x)
        run(a)
