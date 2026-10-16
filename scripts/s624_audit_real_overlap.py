"""S624 audited census of ACTUAL note overlap in S613 TRAIN/DEV annotations.

Counts original labeled note events; NEVER confuses selected/predicted Exact-K
with the density and difficulty of true training polyphony. No new model,
no model predictions, no fabricated synthetic mix. No test sources opened.

Only original pinned GuitarSet annotation ZIP and URMP individual Notes_*.txt
are accessed, not the huge WAV archives. SHA-256 verified where advertised.

Distinct metrics:
- active simultaneous notes: different MIDI pitches overlapping in time,
  integrated duration from intervals. Not onset K and not Exact-K accuracy.
- independent onset clusters: multiple pitch onsets separated <=30ms
  grouped like S615's declared heuristic.
- note pairs with >10ms overlap: counts each (unordered) note-instance
  pair, including separate pairs from a dense chord; NOT number of chords.
- near-simultaneous starts: distinct-pitch onset pairs within 10/30/50/100ms.
- close pitches: semitone interval 1 or 2 among concurrently held notes.
- harmonic-confusable intervals: 12/19/24 semitones (2x/3x/4x ratios),
  and 7 semitones as an approximate 3:2 fifth relation; explicitly NOT
  acoustic proof of partial overlap, which requires spectral evidence.

All time intervals clip to [0,source.seconds) and pitch to NOTE's
40..88 inclusive for model-relevant statistics, and report excluded notes.
The labels cannot establish *audible* intensity or whether separate
harmonic sources can actually be distinguished acoustically.

Use:
  python -m scripts.s624_audit_real_overlap --out results/S624_note_overlap.json
"""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

from scripts.s613_prepare_strict_live import labels, download
from scripts.s615_exactk_audit import group_onsets


PROTOCOL=Path("analysis/evidence/v273-s613-strict-live/protocol.json")
MAX_DURATION=24.0
SUPPORTED_MIN=40
SUPPORTED_MAX=88
THRESH=0.010
NEAR_THRESHOLDS=(.010,.030,.050,.100)
INTERVAL_GROUPS=(1,2,7,12,19,24)


def clipped_notes(rows, duration):
    active=[]
    outside=invalid=0
    for row in rows:
        try:
            pitch=int(row["pitch"])
            start=float(row["time"])
            end=start+float(row["duration"])
            if not all(map(lambda x:x==x and abs(x)!=float("inf"),(start,end))):
                invalid+=1;continue
        except (ValueError,TypeError,KeyError):
            invalid+=1;continue
        if not SUPPORTED_MIN<=pitch<=SUPPORTED_MAX:
            outside+=1
            continue
        a=max(0.,start)
        b=min(float(duration),end)
        if b <= a:
            invalid+=1
            continue
        active.append(dict(pitch=pitch,start=a,end=b,
                           track=str(row.get("track","unknown"))))
    return active,dict(original_annotated=len(rows),excluded_outside_midi_range=outside,
                       excluded_invalid_or_outside_duration=invalid)


def active_distribution(notes, duration):
    events=[]
    for n in notes:
        events.append((n["start"],1,n["pitch"]))
        events.append((n["end"],0,n["pitch"]))
    events.sort(key=lambda x:(x[0],x[1]))  # end BEFORE start on ties
    active=Counter()
    last=0.
    seconds=Counter()
    maxima=0
    for t,edge,p in events:
        if t>last:
            k=len(active)
            seconds[str(k)]+=t-last
            last=t
        if edge==1:
            active[p]+=1
        else:
            active[p]-=1
            if active[p]==0:del active[p]
            if active[p]<0:raise ValueError("Negative note active duration")
        maxima=max(maxima,len(active))
    seconds[str(len(active))]+=float(duration)-last
    if active:
        raise AssertionError("Unclosed source note")
    total=sum(seconds.values())
    if abs(total-duration)>1e-6:
        raise AssertionError((total,duration))
    return {k:round(v,6) for k,v in seconds.items()},maxima


def analyze_source(spec,raw):
    duration=float(spec["seconds"])
    note_events,excluded=clipped_notes(raw,duration)
    spans,maximum=active_distribution(note_events,duration)
    events=sorted(note_events,key=lambda x:x["start"])
    concurrent=Counter()
    starts=Counter()
    disjoint=Counter()
    start_pairs=Counter()
    total_pairs=0
    truly_overlapping_pairs=0
    attacked_while_other_sounding=0
    note_instances_overlapping=set()
    common_pairs=Counter()
    # Compare only simultaneously active historical intervals.
    ongoing=[]
    for i,note in enumerate(events):
        ongoing=[(j,x) for j,x in ongoing if x["end"]>note["start"]]
        other_active_pitch={o["pitch"] for j,o in ongoing if o["pitch"]!=note["pitch"]}
        if other_active_pitch:
            attacked_while_other_sounding+=1
        for j,other in ongoing:
            if other["pitch"]==note["pitch"]:
                disjoint["same_pitch_different_instance"]+=1
                continue
            overlap=min(other["end"],note["end"])-note["start"]
            if overlap <= 0:
                continue
            pair=abs(note["pitch"]-other["pitch"])
            total_pairs+=1
            if overlap >= THRESH:
                truly_overlapping_pairs+=1
                note_instances_overlapping.update((i,j))
                common_pairs[str(pair)]+=1
                if pair <= 2:
                    concurrent["close_1_or_2_semitones"]+=1
                if pair==1:
                    concurrent["semitone_1"]+=1
                if pair==2:
                    concurrent["whole_tone_2"]+=1
                if pair in (12,19,24):
                    concurrent["harmonic_2x_3x_4x_intervals"]+=1
                if pair==7:
                    concurrent["perfect_fifth_3_to_2_interval"]+=1
                if pair==12:
                    concurrent["octave_12"]+=1
                if pair==19:
                    concurrent["twelfth_19"]+=1
                if pair==24:
                    concurrent["two_octaves_24"]+=1
            delta=note["start"]-other["start"]
            for thresh in NEAR_THRESHOLDS:
                if delta<=thresh+1e-10:
                    start_pairs[str(round(thresh*1000))]+=1
        ongoing.append((i,note))
    # Independent onset K labels are NOT concurrent K when notes overlap.
    clusters=group_onsets([dict(time=n["start"],pitch=n["pitch"])
                           for n in events],window_s=.030)
    onset_k=Counter(str(len(group["pitches"])) for group in clusters)
    note_k_from_clusters=sum(len(g["pitches"]) for g in clusters)
    ktime={str(k):float(v) for k,v in spans.items()}
    polytime=sum(v for k,v in ktime.items() if int(k)>=2)
    note_audible_time=sum(v for k,v in ktime.items() if int(k)>=1)
    return dict(
        source=spec["name"],split=spec["split"],corpus=spec["corpus"],
        composition=spec["composition"],
        duration_s=duration,notes_in_pitch_range=len(events),
        **excluded,
        onset_groups_30ms=dict(onset_k),
        onset_groups_total=len(clusters),
        onset_distinct_pitch_slots=note_k_from_clusters,
        duration_by_active_distinct_pitch_K_s=spans,
        duration_active_mono_or_poly_s=round(note_audible_time,6),
        duration_active_poly_K_ge2_s=round(polytime,6),
        peak_simultaneously_active_distinct_pitches=maximum,
        note_attacks_while_a_different_note_still_sustains=attacked_while_other_sounding,
        overlapping_distinct_pitch_note_pairs_any_positive_time=total_pairs,
        overlapping_distinct_pitch_note_pairs_at_least_10ms=truly_overlapping_pairs,
        note_instances_in_at_least_one_overlap_pair_at_least_10ms=len(note_instances_overlapping),
        overlapping_pair_interval_semitones=dict(common_pairs),
        overlapping_pair_categories=dict(concurrent),
        onset_pairs_while_first_still_active_under_ms=dict(start_pairs),
        same_pitch_overlap_pairs=disjoint["same_pitch_different_instance"],
    )


def summarize(sources):
    total=Counter()
    active=Counter()
    onset=Counter()
    intervals=Counter()
    paircats=Counter()
    near=Counter()
    peak=0
    for s in sources:
        total.update({
            "sources":1,"duration_s":s["duration_s"],
            "notes_in_pitch_range":s["notes_in_pitch_range"],
            "original_annotated":s["original_annotated"],
            "excluded_outside_midi_range":s["excluded_outside_midi_range"],
            "excluded_invalid_or_outside_duration":s["excluded_invalid_or_outside_duration"],
            "onset_groups_total":s["onset_groups_total"],
            "onset_distinct_pitch_slots":s["onset_distinct_pitch_slots"],
            "duration_active_mono_or_poly_s":s["duration_active_mono_or_poly_s"],
            "duration_active_poly_K_ge2_s":s["duration_active_poly_K_ge2_s"],
            "note_attacks_while_a_different_note_still_sustains":
                s["note_attacks_while_a_different_note_still_sustains"],
            "overlapping_distinct_pitch_note_pairs_any_positive_time":
                s["overlapping_distinct_pitch_note_pairs_any_positive_time"],
            "overlapping_distinct_pitch_note_pairs_at_least_10ms":
                s["overlapping_distinct_pitch_note_pairs_at_least_10ms"],
            "note_instances_in_at_least_one_overlap_pair_at_least_10ms":
                s["note_instances_in_at_least_one_overlap_pair_at_least_10ms"],
            "same_pitch_overlap_pairs":s["same_pitch_overlap_pairs"],
        })
        active.update(s["duration_by_active_distinct_pitch_K_s"])
        onset.update(s["onset_groups_30ms"])
        intervals.update(s["overlapping_pair_interval_semitones"])
        paircats.update(s["overlapping_pair_categories"])
        near.update(s["onset_pairs_while_first_still_active_under_ms"])
        peak=max(peak,s["peak_simultaneously_active_distinct_pitches"])
    poly=float(total["duration_active_poly_K_ge2_s"])
    sounding=float(total["duration_active_mono_or_poly_s"])
    seconds=float(total["duration_s"])
    return dict(
        raw=dict(total),
        onset_groups_30ms_by_unique_pitch_K=dict(onset),
        duration_by_active_distinct_pitch_K_s={
            k:round(v,3) for k,v in sorted(active.items(),key=lambda x:int(x[0]))},
        duration_active_poly_fraction_of_all_audio=(round(poly/seconds,5) if seconds else None),
        duration_active_poly_fraction_of_sounding_audio=(round(poly/sounding,5) if sounding else None),
        maximum_simultaneously_active_distinct_pitches=peak,
        overlapping_pair_categories=dict(paircats),
        overlapping_pair_interval_semitones=dict(intervals),
        onset_pairs_while_first_still_active_under_ms=dict(near),
        source_counts_with_at_least_one_polyphonic_time=sum(
            s["duration_active_poly_K_ge2_s"]>0 for s in sources),
        source_counts_with_at_least_one_close_pitch_pair=sum(
            s["overlapping_pair_categories"].get("close_1_or_2_semitones",0)>0 for s in sources),
        source_counts_with_at_least_one_harmonic_interval_pair=sum(
            s["overlapping_pair_categories"].get("harmonic_2x_3x_4x_intervals",0)>0 for s in sources),
    )


def census(protocol,root):
    roots=Path(root)
    (roots/"data").mkdir(parents=True,exist_ok=True)
    archives=protocol["guitar_archives"]
    annotated_guitar=[s for s in protocol["sources"]
                      if s["split"] in ("train","dev") and s["corpus"]=="GuitarSet"]
    if annotated_guitar:
        meta=archives["GuitarSet_annotations"]
        download(meta["url"],roots/"data/GuitarSet_annotations.zip",
                 meta["size"],meta["sha256"])
    # Deliberately do NOT load previously exposed test music here.
    specs=[s for s in protocol["sources"] if s["split"] in ("train","dev")]
    if len(specs)!=70:
        raise AssertionError(f"Expected 70 real train+dev sources, got {len(specs)}")
    rows=[]
    for i,s in enumerate(specs,1):
        truth=labels(s,roots)
        if not truth["notes"]:
            raise ValueError("Unexpected empty annotations: "+s["name"])
        row=analyze_source(s,truth["notes"])
        row["annotation_sha256"]=truth["sha256"]
        rows.append(row)
        print(json.dumps(dict(index=i,total=len(specs),split=s["split"],
            corpus=s["corpus"],source=s["name"],
            annotations=row["original_annotated"],usable_notes=row["notes_in_pitch_range"],
            poly_seconds=row["duration_active_poly_K_ge2_s"],
            active_peak=row["peak_simultaneously_active_distinct_pitches"])),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("data/S624_note_annotation_audit"))
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Do not overwrite previous census report")
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows=census(protocol,args.root)
    summary={}
    for split in ("train","dev"):
        own=[r for r in rows if r["split"]==split]
        summary[split]=summarize(own)
        for corpus in ("GuitarSet","URMP"):
            summary[f"{split}_{corpus}"]=summarize([
                r for r in own if r["corpus"]==corpus])
    report=dict(status="S624_REAL_ANNOTATIONS_OVERLAP_CENSUS_TRAIN_DEV",
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        protocol_schema=protocol["schema"],
        number_of_original_sources=len(rows),
        train_sources=sum(r["split"]=="train" for r in rows),
        dev_sources=sum(r["split"]=="dev" for r in rows),
        test_sources_read=0,
        duration_scope="first 24 seconds per file only; 44100Hz native NOTE40..88",
        definitions={
            "active_K":"number of DIFFERENT pitches simultaneously sustaining; time integral",
            "onset_K":"distinct pitches with starts clustered <=30ms (NOT historical S80 Exact-K)",
            "pair_overlap":"two distinct pitches co-active for at least 10ms (one chord makes several pairs)",
            "near_start":"two different pitches with onsets separated by up to 10/30/50/100ms while the earlier is still active",
            "harmonic_pairs":"interval in MIDI semitones 12/19/24; NOT proof of acoustically merged harmonics",
            "no_audibility":"annotation-only count does not establish actual captured loudness",
        },
        summaries=summary,per_source=rows,
    )
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for group,data in summary.items():
        t=data["raw"]
        print("AUDITED_SUMMARY "+json.dumps(dict(
            group=group,source_count=t["sources"],notes=t["notes_in_pitch_range"],
            duration_s=t["duration_s"],
            active_poly_s=t["duration_active_poly_K_ge2_s"],
            fraction_all=data["duration_active_poly_fraction_of_all_audio"],
            fraction_sounding=data["duration_active_poly_fraction_of_sounding_audio"],
            onset_K=data["onset_groups_30ms_by_unique_pitch_K"],
            active_K_seconds=data["duration_by_active_distinct_pitch_K_s"],
            overlap_pairs_10ms=t["overlapping_distinct_pitch_note_pairs_at_least_10ms"],
            pairs_close_1_2=data["overlapping_pair_categories"].get("close_1_or_2_semitones",0),
            pairs_harmonic_12_19_24=data["overlapping_pair_categories"].get("harmonic_2x_3x_4x_intervals",0),
            near_onset_30ms_pairs=data["onset_pairs_while_first_still_active_under_ms"].get("30",0),
        ),sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    main()
