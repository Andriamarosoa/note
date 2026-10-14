"""S532 ablation: fixed cardinality caps on NEW note pitches for V88 K0 sites.

Restore actual S527 raw features + frozen S528 train-held pitch probabilities.
Preserve original BP-note-births exactly; for blank BP birth events, take
trained S528 raw-only p>=0.70 notes, optionally keep only top1/top2/top3.
No source K as hard cap (V88 K0 can be true K1/K2), no MIDI in decision.
Score every true original MAESTRO MIDI pitch+attack across complete 3x12s.

This test is a post-hoc retrospective exploration, not blind validation.
Potential increase in F1 cannot be promoted on same already inspected data.
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

def topcap(audio,learned,cap):
    old=proposed_notes(audio,learned,"BP_birth")
    if old:return old
    v=np.asarray(learned["raw_only_49_pitch_prob"],float)
    idx=np.flatnonzero(v>=.7)
    if cap and len(idx)>cap:
        idx=sorted(idx,key=lambda j:(-v[j],int(j)))[:cap]
    return {40+int(i) for i in idx}

def run(a):
    need(not a.out.exists(),"S532 exploration should remain immutable")
    source,orig_report,audio_report,trained_report,protect_report,items,proof,rows,pred=load_sources(a)
    bysource=collections.defaultdict(list)
    for x,p in zip(rows,pred):
        bysource[x["source_recording_sha"]].append((x,p))
    tr={}
    for ident,meta in sorted(items.items()):
        blob=fetch_midi(source["audio_mirror_directory"],meta["MIDI_filename_path_ONLY_not_opened"])
        need(digest(blob)==proof[ident]["MIDI_bytes_SHA256"],
             "S532 original complete MIDI groundtruth fingerprint changed")
        events=midi_events(blob)
        g=onset_groups(events,float(meta["audio_excerpt_start_seconds"]),
                         float(meta["audio_excerpt_duration_seconds"]))
        tr[ident]=[dict(time=float(x["onset"]),
              notes=new_notes(events,float(meta["audio_excerpt_start_seconds"]),
                              float(x["onset"]))) for x in g]
        need(len(tr[ident])==proof[ident]["MIDI_attack_groups_in_excerpt"],
             "S532 no annotated attack from S471 may be dropped")
    by_cap={}
    for cap in (0,1,2,3):
        totals=collections.Counter()
        rec={}
        for ident in sorted(items):
            detected=[]
            for old,p in bysource[ident]:
                proposed=topcap(old,p,cap)
                if proposed:
                    detected.append(dict(time=float(old["original_V88_candidate_start_s"]),
                                         notes=sorted(proposed)))
            target=tr[ident]
            note=score_event_notes(detected,target)
            onset=score_event_birth(detected,target)
            rec[ident]=dict(noteon=note,onsets=onset)
            totals.update(note_tp=note["TP"],note_fp=note["FP"],note_fn=note["FN"],
                 predicted_groups=onset["note_groups_predicted"],
                 true_groups=onset["note_groups_true"],
                 onset_tp=onset["physical_true_onset_TP"],
                 onset_fp=onset["physical_predicted_onset_FP"],
                 onset_fn=onset["physical_true_onset_FN"],
                 full_chords=onset["matched_exact_full_note_chords"],
                 exactK=onset["matched_exactK"])
        result=dict(totals)
        result["NoteOn_F1"]=2*result["note_tp"]/max(1,2*result["note_tp"]+result["note_fp"]+result["note_fn"])
        result["NoteOn_precision"]=result["note_tp"]/max(1,result["note_tp"]+result["note_fp"])
        result["NoteOn_recall"]=result["note_tp"]/max(1,result["note_tp"]+result["note_fn"])
        by_cap["no_cap" if cap==0 else "max_"+str(cap)]=dict(overall=result,per_recording=rec)
        print("S532 FIXED PITCH-CARDINALITY POSTFILTER MIDI VERDICT",
              json.dumps(dict(policy=cap,metrics=result)),flush=True)
    # load_sources returns the original S471 report in tuple position 1,
    # so fetch the independent S530 report explicitly for this replay.
    baseline=json.loads(a.full_report.read_text())
    need(baseline["status"]=="S530_FULL_36SECOND_MAESTRO_NOTEON_PITCH_AND_ATTACK_RETROSPECTIVE",
         "S532 needs original successful independent complete MIDI S530 source")
    b=baseline["scoring"]["raw_only:0.7"]["total"]
    a0=by_cap["no_cap"]["overall"]
    for k,g in (("note_tp","note_tp"),("note_fp","note_fp"),
                ("note_fn","note_fn"),("full_chords","full_chords")):
        need(a0[k]==b[g],"S532 original S530 no-cap true NoteOn F1 independently drifted")
    need(a0["true_groups"]==330,"S532 full MIDI original 330 onset cohort incomplete")
    a.out.mkdir(parents=True)
    report=dict(status="S532_INDEPENDENT_FROZEN_PITCH_CARDINALITY_ABLATION_FULL_MIDI",
        original_S530_no_cap_note_predictions_independently_reproduced=True,
        three_predeclared_max_pitch_caps=[1,2,3],
        original_all_S80_V88_predictions_and_Live_unchanged=True,
        no_training_only_raw_S528_held_note_probabilities_reused=True,
        original_BP_note_births_never_deleted_or_modified=True,
        no_hard_K0_cardinality_constraint=True,
        complete_original_piano_MIDI_attacks=330,
        only_historically_exposed_source_retrospective_not_blind=True,
        no_threshold_selected_using_held_MIDI_for_promotion=True,
        no_promotion=True,policies=by_cap,
        source_sha256=dict(S527=sha(a.audio),S528=sha(a.trained),
             S530=sha(a.full_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    original={"old_basicpitch_birth_pitches":[]}
    p={"raw_only_49_pitch_prob":[.95,.82,.71,.61]+[0]*45}
    need(topcap(original,p,1)=={40}
         and topcap(original,p,2)=={40,41}
         and topcap(original,p,3)=={40,41,42}
         and topcap(original,p,0)=={40,41,42},
         "S532 fixed top cap should never use true test note labels")
    original["old_basicpitch_birth_pitches"]=[52,55]
    need(topcap(original,p,1)=={52,55},
         "S532 preserve original born polyphonic chord")
    print("S532 SELFTEST PASS cap only brand-new pitches never existing exact BP chord")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for f in ("manifest","s471-report","audio","audio-report",
              "trained","trained-report","protect-report","full-report","out"):
        p.add_argument("--"+f,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for f in ("manifest","s471_report","audio","audio_report","trained",
                  "trained_report","protect_report","full_report","out"):
            need(getattr(a,f)is not None,"S532 missing "+f)
        run(a)
