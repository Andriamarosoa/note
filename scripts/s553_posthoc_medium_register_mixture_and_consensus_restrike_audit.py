"""S553: POST-HOC register-aware gate / real restrike protection research.

S551 notes already exposed; therefore NOTHING from this exploratory loop may
be called a new blind test, nor promoted. Use only SHA-verifiable S550 six
frozen MIDI-BLIND forecasts. Without fitting or opening fresh WAVs, construct
six candidate expert routing policies to identify the medium-range false
pitch mechanism and cost of true repeated attacks.

All candidate corrections/regressions must be logged separately by real
MIDI register, recording and source note pitch, not just their delta F1.
One-to-one pitch and timing matching +/-30/50/80ms; ORIGINAL FULL TRUE
MIDI40..88 ONSET count including every missed true note.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi,midi_events,onset_groups
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import sha,need
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
from scripts.s552_audit_new_S545_register_note_pitch_failure_after_independent_validation import RANGES,part

MID_LOW,MID_HIGH=60,71
POLICIES=("S546_dedup100_harm080","S528_mid80_S546_rest100",
          "mid_union_S52880_S546100","mid_intersection_S52880_S546100",
          "mid_S54680_S546_rest100","consensus_restrike_S546raw_S528raw",
          "mid_S52880_with_consensus_restrike")
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def eventset(ledger,ident,policy):
    notes=collections.defaultdict(set)
    for r in ledger:
        if r["original_recording_filename_sha"]==ident and r["policy"]==policy:
            t=round(float(r["onset_s"]),6)
            for note in r["pitch_MIDI"]:notes[t].add(int(note))
    return {(t,int(p)) for t,n in notes.items() for p in n}
def merged(ledger,ident,policy):
    base=eventset(ledger,ident,"S546_dedup100_harm080")
    old=eventset(ledger,ident,"S528_dedup80")
    new80=eventset(ledger,ident,"S546_dedup80")
    if policy=="S546_dedup100_harm080":out=base
    elif policy=="S528_mid80_S546_rest100":
        out={(t,p) for t,p in base if not MID_LOW<=p<=MID_HIGH}
        out|={(t,p) for t,p in old if MID_LOW<=p<=MID_HIGH}
    elif policy=="mid_union_S52880_S546100":
        out=base|{(t,p) for t,p in old if MID_LOW<=p<=MID_HIGH}
    elif policy=="mid_intersection_S52880_S546100":
        out={(t,p) for t,p in base if not MID_LOW<=p<=MID_HIGH or
             (t,p) in old}
    elif policy=="mid_S54680_S546_rest100":
        out={(t,p) for t,p in base if not MID_LOW<=p<=MID_HIGH}
        out|={(t,p) for t,p in new80 if MID_LOW<=p<=MID_HIGH}
    elif policy in ("consensus_restrike_S546raw_S528raw",
                  "mid_S52880_with_consensus_restrike"):
        if policy.startswith("consensus"):
            out=base.copy()
        else:
            current=merged(ledger,ident,"S528_mid80_S546_rest100")
            out={(round(float(z["time"]),6),int(p)) for z in current for p in z["notes"]}
        newraw=eventset(ledger,ident,"S546_raw")
        oldraw=eventset(ledger,ident,"S528_raw")
        before=len(out)
        # Only reinsert a suppressed potential SAME-PITCH re-strike if
        # independently source-trained OLD and NEW models agree at that WAV event.
        # A preceding accepted same pitch in 0 < dt<=0.100 is essential:
        # otherwise this is NOT a duplicate correction but arbitrary addition.
        for t,p in sorted((newraw&oldraw)-out):
            if any(pp==p and 0<round(t-tt,6)<=.100 for tt,pp in out):
                out.add((t,p))
    else:raise RuntimeError("S553 unknown policy")
    grouped=collections.defaultdict(list)
    for t,p in sorted(out):grouped[t].append(p)
    return [dict(time=t,notes=sorted(pp)) for t,pp in sorted(grouped.items())]

def score(guess,truth):
    return {str(t):note_metrics(guess,truth,t/1000.) for t in (30,50,80)}
def run(a):
    need(not a.out.exists(),"S553 post-hoc expert combinations must not overwrite")
    m=json.loads(a.manifest.read_text())
    frozen=json.loads(a.freeze_report.read_text())
    scored=json.loads(a.score_report.read_text())
    need(m["status"]=="S545_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and frozen["status"]=="S550_FIVE_COMPOSER_S545_PITCH_ONSET_SOURCE_FROZEN_BEFORE_ANY_MIDI"
         and frozen["frozen_named_note_prediction_sha"]==sha(a.predictions)
         and scored["status"]=="S551_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1"
         and scored["SHA_frozen_S550_prediction"]==sha(a.predictions)
         and scored["no_model_training_or_threshold_selection_using_S545_MIDI"],
         "S553 valid independent frozen S551 history required before exploratory posthoc")
    ledger=rows(a.predictions)
    source={s["filename_SHA256"]:s for s in m["chosen_metadata_only"]}
    oldproof={s["original_audio_filename_sha"]:s for s in scored["true_MIDI_provenance"]}
    need(len(source)==5 and set(source)==set(oldproof),"S553 all five real S545 sources required")
    reportpol={}
    persource={}
    for ident,item in sorted(source.items()):
        blob=fetch_midi(m["source_audio_mirror"],item["MIDI_filename_path_ONLY_not_opened"])
        need(hashlib.sha256(blob).hexdigest()==oldproof[ident]["original_MIDI_sha256"],
             "S553 independent true MIDI SHA drifted")
        midi=midi_events(blob)
        offset=float(item["audio_excerpt_start_seconds"])
        truth=[dict(time=float(g["onset"]),
                notes=[int(p) for p in new_notes(midi,offset,float(g["onset"]))
                       if 40<=int(p)<=88]) for g in onset_groups(
                           midi,offset,float(item["audio_excerpt_duration_seconds"]))]
        truth=[x for x in truth if x["notes"]]
        persource[ident]=dict(composer=item["composer"],
            true_note_count=sum(len(x["notes"]) for x in truth),results={})
        for policy in POLICIES:
            guess=merged(ledger,ident,policy)
            met=score(guess,truth)
            regs={}
            for name,(lo,hi) in RANGES.items():
                regs[name]=score(part(guess,lo,hi),part(truth,lo,hi))
            persource[ident]["results"][policy]=dict(
                scores=met,registers=regs,predicted_notes=sum(len(z["notes"]) for z in guess))
            if policy=="S546_dedup100_harm080":
                q=scored["policies"][policy]["per_new_composer"][ident]["pitch_F1_by_tolerance_ms"]["80"]
                need(tuple(met["80"][x] for x in ("TP","FP","FN"))==
                     tuple(q[x] for x in ("TP","FP","FN")),
                     "S553 original S551 score not reproducible for baseline")
    totals={}
    for policy in POLICIES:
        agg={str(t):collections.Counter() for t in (30,50,80)}
        regs={name:collections.Counter() for name in RANGES}
        for ident,record in persource.items():
            pr=record["results"][policy]
            for t,z in pr["scores"].items():
                agg[t].update({q:z[q] for q in ("TP","FP","FN")})
            for name,z in pr["registers"].items():
                regs[name].update({q:z["80"][q] for q in ("TP","FP","FN")})
        scored={}
        for tol,c in agg.items():
            x=dict(c)
            x["F1"]=2*x["TP"]/max(1,2*x["TP"]+x["FP"]+x["FN"])
            scored[tol]=x
        rr={}
        for name,c in regs.items():
            x=dict(c);x["F1"]=2*x["TP"]/max(1,2*x["TP"]+x["FP"]+x["FN"])
            rr[name]=x
        total80=scored["80"]
        need(tuple(sum(rr[name][q] for name in RANGES) for q in ("TP","FP","FN"))==
             tuple(total80[q] for q in ("TP","FP","FN")),
             "S553 all full NoteOn true events must sum exactly from three registers")
        totals[policy]=dict(full_noteon=scored,by_register=rr)
        print("S553 POSTHOC COMPOSER-EXPOSED REGISTER-GATE LOOP",
              json.dumps(dict(policy=policy,score80=scored["80"],registers=rr)),flush=True)
    base=totals["S546_dedup100_harm080"]["full_noteon"]["80"]
    verdict={}
    for policy in POLICIES[1:]:
        got=totals[policy]["full_noteon"]["80"]
        verdict[policy]=dict(deltaTP=got["TP"]-base["TP"],
            deltaFP=got["FP"]-base["FP"],deltaFN=got["FN"]-base["FN"],
            delta_F1_percentage_points=100*(got["F1"]-base["F1"]),
            per_new_song_delta_F1_percentage_points={
                r["composer"]:100*(r["results"][policy]["scores"]["80"]["F1"]-
                     r["results"]["S546_dedup100_harm080"]["scores"]["80"]["F1"])
                for r in persource.values()})
    a.out.mkdir(parents=True)
    report=dict(status="S553_POST_HOC_MIDDLE_PITCH_REGISTER_AND_REAL_RESTRIKE_EXPERT_ROUTING",
      original_frozen_S550_SHA=sha(a.predictions),
      original_independent_S551_score_SHA=sha(a.score_report),
      source_MIDI_previously_OPENED_in_S551=True,
      retrospectively_seen_development_SET_NOT_NEW_TEST=True,
      use_of_S545_MIDI_for_architecture_select_must_NOT_be_counted_as_generalization=True,
      midi40to88_and_tolerance_30_50_80_ms=True,
      original_K_S80_live_and_freeze_local_combo_unchanged=True,no_promotion=True,
      tested_policies=list(POLICIES),scoring=totals,
      delta_from_prior_independently_frozen_S546_baseline=verdict,
      per_composer=persource)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S553 AUDIT POSTHOC POLICY DELTAS",json.dumps(verdict),flush=True)
def selftest():
    ledger=[]
    for pol,pitches in [("S546_dedup100_harm080",[52,65,84]),
                        ("S528_dedup80",[52,66,84]),
                        ("S546_dedup80",[52,65,84]),
                        ("S546_raw",[52,65,84]),
                        ("S528_raw",[52,66,84])]:
        ledger.append(dict(policy=pol,original_recording_filename_sha="fake",
                           onset_s=.25,pitch_MIDI=pitches))
    replaced=merged(ledger,"fake","S528_mid80_S546_rest100")
    need(replaced==[dict(time=.25,notes=[52,66,84])],
         "S553 routing S528 mid and S546 bass/treble incorrectly selected")
    need(merged(ledger,"fake","mid_intersection_S52880_S546100")==
         [dict(time=.25,notes=[52,84])],"S553 intersection should remove contested midi65 only")
    print("S553 SELFTEST PASS deterministic MIDI-blind routing and 49-note bass/mid/high filters")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("manifest","freeze-report","score-report","predictions","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("manifest","freeze_report","score_report","predictions","out"):
            need(getattr(a,k) is not None,"S553 missing "+k)
        run(a)
