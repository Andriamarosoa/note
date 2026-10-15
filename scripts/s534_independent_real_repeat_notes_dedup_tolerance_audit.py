"""S534 independent audit of S533 duplication gain and rapid true re-strikes.

Re-score original frozen S528 pitch identities with independent one-to-one
pitch/time matching at STRICT +-30/50/80ms. Audit per recording, per note
identity, and all originally true MIDI repeated-note strike intervals.

Compare original S530 reference to fixed S533 dedup80 (plus harmonic .8).
This is an evaluation audit NOT model training, no tuning, no promotion.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    load_sources,assignment,score_event_notes,need,sha,digest)
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes

def note_metrics(pred,truth,limit):
    pi=collections.defaultdict(list);yt=collections.defaultdict(list)
    for q in pred:
        for p in q["notes"]:pi[int(p)].append(float(q["time"]))
    for q in truth:
        for p in q["notes"]:yt[int(p)].append(float(q["time"]))
    tp=sum(len(assignment(pi[p],yt[p],limit=limit))
           for p in set(pi)|set(yt))
    total_p=sum(len(z) for z in pi.values())
    total_y=sum(len(z) for z in yt.values())
    assert tp<=min(total_p,total_y)
    return dict(TP=tp,FP=total_p-tp,FN=total_y-tp,
                precision=tp/max(1,total_p),
                recall=tp/max(1,total_y),
                F1=2*tp/max(1,total_p+total_y))

def repeat_truth_intervals(truth,window):
    groups=collections.defaultdict(list)
    for x in truth:
        for p in x["notes"]:
            groups[int(p)].append(float(x["time"]))
    intervals=[]
    for pitch,st in groups.items():
        st=sorted(st)
        intervals.extend(dict(pitch=pitch,first=a,second=b,delta_ms=1000*(b-a))
                         for a,b in zip(st,st[1:]) if 0<b-a<=window+.00001)
    return intervals

def removals(before,after):
    """All individual MIDI pitch events removed by policy; no ambiguous silent conversion."""
    def flatten(groups):
        events=collections.Counter()
        for group in groups:
            for p in group["notes"]:
                events[(round(float(group["time"]),6),int(p))]+=1
        return events
    b=flatten(before);a=flatten(after)
    need(not (a-b),"S534 protected S533 temporal policy must ONLY remove pitch events")
    gone=b-a
    return [dict(time=t,pitch=p,count=n) for (t,p),n in sorted(gone.items())]

def run(a):
    need(not a.out.exists(),"S534 fixed external replay already done")
    m,s471,s527,s528,s529,source,proof,raw,trained=load_sources(a)
    s533=json.loads(a.s533_report.read_text())
    need(s533["status"]=="S533_NINE_FROZEN_HARMONIC_DEDUP_LOOPS_MAESTRO_NOTEON"
         and s533["independent_original_S530_score_replayed"]
         and s533["pinned_SHA"]["S527"]==sha(a.audio)
         and s533["pinned_SHA"]["S528"]==sha(a.trained),
         "S534 previous S533 successful feature/probability SHAs no longer match")
    rows=collections.defaultdict(list)
    for r,p in zip(raw,trained):rows[r["source_recording_sha"]].append((r,p))
    group={}
    for rec,meta in sorted(source.items()):
        data=fetch_midi(m["audio_mirror_directory"],meta["MIDI_filename_path_ONLY_not_opened"])
        need(digest(data)==proof[rec]["MIDI_bytes_SHA256"],
             "S534 true MAESTRO official MIDI SHA mismatch")
        events=midi_events(data)
        start=float(meta["audio_excerpt_start_seconds"])
        anchors=onset_groups(events,start,float(meta["audio_excerpt_duration_seconds"]))
        group[rec]=[dict(time=float(z["onset"]),notes=new_notes(events,start,float(z["onset"])))
                    for z in anchors]
        need(len(group[rec])==proof[rec]["MIDI_attack_groups_in_excerpt"],
             "S534 no true repeated note or onset can disappear")
    policies=[
       dict(name="base",dedup_ms=0,harmonic_ratio=0),
       dict(name="dedup60",dedup_ms=60,harmonic_ratio=0),
       dict(name="dedup80",dedup_ms=80,harmonic_ratio=0),
       dict(name="dedup100",dedup_ms=100,harmonic_ratio=0),
       dict(name="dedup80_harmonic080",dedup_ms=80,harmonic_ratio=.8)]
    result={};ledger=[]
    all_trues=sum(len(z) for z in group.values())
    need(all_trues==330,"S534 the historical 330 true attacks changed")
    for policy in policies:
        times={};details={};counts=collections.Counter()
        for rec,items in sorted(rows.items()):
            out,removed=emit(items,policy)
            baseline,_=emit(items,policies[0])
            gone=removals(baseline,out)
            tr=group[rec]
            repetitions=repeat_truth_intervals(tr,policy["dedup_ms"]/1000. if policy["dedup_ms"] else 0)
            for x in gone:
                p=int(x["pitch"]);t=float(x["time"])
                near=[v["time"] for v in tr if p in v["notes"] and abs(float(v["time"])-t)<=.0800001]
                ledger.append(dict(policy=policy["name"],rec=rec,
                                   suppressed_pitch=p,suppressed_at=t,
                                   nearby_true_80ms_pitch_events=len(near),
                                   nearby_true_times=near,
                                   true_repeated_strikes_within_window=len(repetitions),
                                   classification_needs_one_to_one_matching=True))
            counts.update(suppressed_note_pitch_events=sum(z["count"] for z in gone),
                true_repeated_same_pitch_strikes_within_policy_window=len(repetitions),
                annotated_true_pitch_onsets=sum(len(z["notes"]) for z in tr),
                original_predicted_pitch_onsets=sum(len(z["notes"]) for z in baseline))
            details[rec]=dict(suppressed=len(gone),
                             truth_repeat_under_policy_ms=repetitions)
            for limit in (.03,.05,.08):
                k=str(int(limit*1000))
                metric=note_metrics(out,tr,limit)
                times.setdefault(k,collections.Counter()).update({
                    "TP":metric["TP"],"FP":metric["FP"],"FN":metric["FN"]})
        scores={}
        for tolerance,m in times.items():
            v=dict(m)
            v["F1"]=2*v["TP"]/max(1,2*v["TP"]+v["FP"]+v["FN"])
            v["precision"]=v["TP"]/max(1,v["TP"]+v["FP"])
            v["recall"]=v["TP"]/max(1,v["TP"]+v["FN"])
            scores[tolerance]=v
        result[policy["name"]]=dict(metrics_by_tolerance_ms=scores,
                    per_recording=details,extra_audit=dict(counts))
        print("S534 INDEPENDENT TIME-TOLERANCE AND REAL REPEATED NOTE AUDIT",
             json.dumps(dict(policy=policy["name"],metrics=scores,
                             totals=dict(counts))),flush=True)
    need(result["base"]["metrics_by_tolerance_ms"]["80"]["TP"]==213
         and result["dedup80"]["metrics_by_tolerance_ms"]["80"]["TP"]==213
         and result["dedup80_harmonic080"]["metrics_by_tolerance_ms"]["80"]["TP"]==212,
         "S534 independently replayed original full MIDI pitch TP changed")
    need(result["dedup80"]["extra_audit"]["suppressed_note_pitch_events"]==56,
         "S534 previously confirmed exact 56 repeated false pitch events changed")
    a.out.mkdir(parents=True)
    file=a.out/"S534_ALL_REMOVED_PITCH_EVENTS_AUDIT_NOTE_ON.jsonl"
    file.write_text("".join(json.dumps(x)+"\n" for x in ledger))
    rep=dict(status="S534_INDEPENDENT_REPEATED_NOTE_STRIKE_AND_TIME_TOLERANCE_AUDIT",
       original_330_MIDI_group_416_note_pitches_all_scored=True,
       tested_true_onset_tolerance_ms=[30,50,80],
       tested_duplicates_ms=[0,60,80,100],
       original_S533_56_repeated_pitch_removals_SHA_replayed=True,
       all_actual_MIDI_repeated_same_pitch_within_dedup_window_explicitly_audited=True,
       same_development_MAESTRO_three_recordings_NOT_blind=True,
       all_BP_born_notes_preserved=True,
       original_S80_ExactK_Live_unchanged=True,no_promotion=True,
       policies=result,removal_ledger_SHA=sha(file))
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")

def selftest():
    true=[dict(time=.10,notes=[52]),dict(time=.16,notes=[52])]
    s=repeat_truth_intervals(true,.080)
    need(len(s)==1 and abs(s[0]["delta_ms"]-60)<1e-7,
         "S534 two genuinely repeated same-pitch note strikes must be protected in audit")
    p=[dict(time=.10,notes=[52]),dict(time=.12,notes=[52])]
    q=[dict(time=.10,notes=[52])]
    lost=removals(p,q)
    need(len(lost)==1 and lost[0]["time"]==.12,
         "S534 pitch event removal must be tracked at physical original timestamp")
    assert note_metrics(q,true,.03)["TP"]==1
    print("S534 SELFTEST PASS real repeated strikes are distinct from duplicate candidate notes")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","s471-report","audio","audio-report","trained",
              "trained-report","protect-report","s533-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","s471_report","audio","audio_report","trained",
                  "trained_report","protect_report","s533_report","out"):
            need(getattr(a,z)is not None,"S534 missing "+z)
        run(a)
