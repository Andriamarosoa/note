"""S533 repeated frozen-policy loops: harmonic rivalry & same-pitch attack dedup.

Starting point: historical S530 full-interval MAESTRO raw-only .70 NoteOn
(TP213 FP206 FN203; F1 51.02%). All source predictions and original
MAESTRO 330 MIDI onset groups are fixed. No retraining or truth-assisted
decisions, and original Basic Pitch note births stay untouched.

Independent audit of hypotheses (not a blind improvement search):
- same-note reentry within 40/80 ms: skip only newly generated notes;
- near-octave/fifth harmonics 12/19/24 semitones: suppress only an
  unprotected higher note if an accepted lower fundamental has stronger
  novel/burst acoustic support and sufficient S528 probability;
- alternatives and combinations, with per-piece TP/FP/FN and full chords.

Historical data was seen in earlier rounds; never promote from these loops.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    load_sources,score_event_notes,score_event_birth,proposed_notes,
    need,sha,digest)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes

def emit(candidates, policy):
    """Only modify notes newly born at originally empty BP events."""
    bytime=sorted(candidates,key=lambda pair:float(pair[0]["original_V88_candidate_start_s"]))
    history={}
    groups=[]
    rawcount=removed_duplicate=removed_harmonic=0
    for r,head in bytime:
        t=float(r["original_V88_candidate_start_s"])
        birth=proposed_notes(r,head,"BP_birth")
        if birth:
            for p in birth:history[p]=t
            groups.append(dict(time=t,notes=sorted(birth)))
            continue
        v=np.asarray(head["raw_only_49_pitch_prob"],float)
        indices=set(map(int,np.flatnonzero(v>=.70)))
        rawcount+=len(indices)
        candidates_new=set(indices)
        if policy["dedup_ms"]:
            previous=history
            for j in list(candidates_new):
                midi=j+40
                if midi in previous and (t-previous[midi])*1000<=policy["dedup_ms"]+1e-5:
                    candidates_new.remove(j)
                    removed_duplicate+=1
        if policy["harmonic_ratio"]>0:
            feat=np.asarray(r["per_midi_features_49x11"],float)
            # Process from low to high frequency so retained lower true
            # fundamental must be present in the CURRENT emitted chord.
            keep=set()
            for j in sorted(candidates_new):
                found=False
                for d in (12,19,24):
                    lower=j-d
                    if lower not in keep or v[lower]<.70:continue
                    burst_low=max(.001,feat[lower,3])
                    novelty_low=max(.001,feat[lower,5])
                    burst_high=max(0.,feat[j,3])
                    novelty_high=max(0.,feat[j,5])
                    weak=(burst_high < policy["harmonic_ratio"]*burst_low and
                          novelty_high < policy["harmonic_ratio"]*novelty_low)
                    # Only a plausible harmonic, never blanket octave removal.
                    if weak:
                        found=True;break
                if found:removed_harmonic+=1
                else:keep.add(j)
            candidates_new=keep
        if candidates_new:
            emitted={40+j for j in candidates_new}
            for midi in emitted:history[midi]=t
            groups.append(dict(time=t,notes=sorted(emitted)))
    return groups,dict(original_new_pitch_candidates=rawcount,
      suppressed_repeated_new_pitch=removed_duplicate,
      suppressed_probable_harmonic_new_pitch=removed_harmonic)

def run(a):
    need(not a.out.exists(),"S533 posthoc audit must remain immutable")
    meta,s471,s527,s528,s529,source,proof,raw,trained=load_sources(a)
    per=collections.defaultdict(list)
    for r,t in zip(raw,trained):per[r["source_recording_sha"]].append((r,t))
    truth={}
    for ident,info in sorted(source.items()):
        blob=fetch_midi(meta["audio_mirror_directory"],
            info["MIDI_filename_path_ONLY_not_opened"])
        need(digest(blob)==proof[ident]["MIDI_bytes_SHA256"],
             "S533 original full MIDI labels changed")
        midi=midi_events(blob)
        start=float(info["audio_excerpt_start_seconds"])
        groups=onset_groups(midi,start,float(info["audio_excerpt_duration_seconds"]))
        truth[ident]=[dict(time=float(g["onset"]),
           notes=new_notes(midi,start,float(g["onset"]))) for g in groups]
        need(len(truth[ident])==proof[ident]["MIDI_attack_groups_in_excerpt"],
             "S533 source original 330 onset group count changed")
    policies=[dict(name="baseline",dedup_ms=0,harmonic_ratio=0),
        dict(name="dedup40",dedup_ms=40,harmonic_ratio=0),
        dict(name="dedup80",dedup_ms=80,harmonic_ratio=0),
        dict(name="harmonic050",dedup_ms=0,harmonic_ratio=.50),
        dict(name="harmonic080",dedup_ms=0,harmonic_ratio=.80),
        dict(name="dedup40_harmonic050",dedup_ms=40,harmonic_ratio=.50),
        dict(name="dedup80_harmonic050",dedup_ms=80,harmonic_ratio=.50),
        dict(name="dedup40_harmonic080",dedup_ms=40,harmonic_ratio=.80),
        dict(name="dedup80_harmonic080",dedup_ms=80,harmonic_ratio=.80)]
    scores={}
    for policy in policies:
        tally=collections.Counter()
        source_scores={}
        for ident in sorted(per):
            events,proposals=emit(per[ident],policy)
            truth_events=truth[ident]
            scored=score_event_notes(events,truth_events)
            event=score_event_birth(events,truth_events)
            tally.update(note_tp=scored["TP"],note_fp=scored["FP"],note_fn=scored["FN"],
                 emitted_groups=event["note_groups_predicted"],
                 original_true_groups=event["note_groups_true"],
                 true_onset_tp=event["physical_true_onset_TP"],
                 onset_fp=event["physical_predicted_onset_FP"],
                 onset_fn=event["physical_true_onset_FN"],
                 full_chords=event["matched_exact_full_note_chords"],
                 exactK=event["matched_exactK"],**proposals)
            source_scores[ident]=dict(noteon=scored,event=event,proposals=proposals)
        result=dict(tally)
        result["F1_noteon"]=2*result["note_tp"]/max(1,2*result["note_tp"]+result["note_fp"]+result["note_fn"])
        result["precision_noteon"]=result["note_tp"]/max(1,result["note_tp"]+result["note_fp"])
        result["recall_noteon"]=result["note_tp"]/max(1,result["note_tp"]+result["note_fn"])
        scores[policy["name"]]=dict(policy=policy,summary=result,per_recording=source_scores)
        print("S533 FUNDAMENTAL VS HARMONIC AND DUPLICATE LOOP",
            json.dumps(dict(name=policy["name"],summary=result)),flush=True)
    baseline=scores["baseline"]["summary"]
    s530=json.loads(a.full_report.read_text())
    want=s530["scoring"]["raw_only:0.7"]["total"]
    need((baseline["note_tp"],baseline["note_fp"],baseline["note_fn"])
         ==(want["note_tp"],want["note_fp"],want["note_fn"])
         and baseline["original_true_groups"]==330,
         "S533 S530 independent replay must be exactly unchanged")
    a.out.mkdir(parents=True)
    report=dict(status="S533_NINE_FROZEN_HARMONIC_DEDUP_LOOPS_MAESTRO_NOTEON",
       original_330_true_MIDI_onsets_all_evaluated=True,
       nine_predeclared_policies_including_baseline=True,
       independent_original_S530_score_replayed=True,
       original_raw_WAV_and_pitch_models_frozen=True,
       no_MIDI_truth_in_inference=True,
       source_V88_K0_never_hard_note_cap=True,
       original_BP_birth_notes_not_deleted=True,
       same_three_historically_seen_piano_sources_NOT_blind=True,
       no_promotion=True,reference_S80_ExactK_Live_untouched=True,
       tests=scores,
       pinned_SHA=dict(S527=sha(a.audio),S528=sha(a.trained),
                       S530=sha(a.full_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    x={"old_basicpitch_birth_pitches":[],
       "original_V88_candidate_start_s":.1,
       "per_midi_features_49x11":np.zeros((49,11)).tolist()}
    probs=np.zeros(49);probs[12]=.98;probs[24]=.91
    feat=np.asarray(x["per_midi_features_49x11"])
    feat[12,3]=feat[12,5]=1
    feat[24,3]=feat[24,5]=.1
    x["per_midi_features_49x11"]=feat.tolist()
    y={"raw_only_49_pitch_prob":probs.tolist()}
    pol=dict(dedup_ms=0,harmonic_ratio=.5)
    groups,c=emit([(x,y)],pol)
    need(len(groups)==1 and groups[0]["notes"]==[52]
         and c["suppressed_probable_harmonic_new_pitch"]==1,
         "S533 octave has to be removed only when weaker than lower acoustic fundamental")
    x2=dict(x,old_basicpitch_birth_pitches=[52,64])
    groups,c=emit([(x2,y)],pol)
    need(groups[0]["notes"]==[52,64] and
         c["suppressed_probable_harmonic_new_pitch"]==0,
         "S533 cannot change existing BP true polyphonic chord")
    print("S533 SELFTEST PASS suppress only acoustically weak new harmonics, protect BP chords")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","s471-report","audio","audio-report",
              "trained","trained-report","protect-report","full-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","s471_report","audio","audio_report",
                  "trained","trained_report","protect_report","full_report","out"):
            need(getattr(a,z)is not None,"S533 missing "+z)
        run(a)
