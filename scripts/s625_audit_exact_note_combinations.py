"""S625: exact-pitch and pitch-class combination census of original NOTE labels.

S624 counted only active-K duration and *interval classes* (octaves, 1/2
semitones). S625 now counts exact combinations including E2+E3 and
simultaneous chord pitch-class unions like F major + E minor.

S613 PINNED train/dev ONLY: 53 train/17 dev GuitarSet+URMP source clips,
first 24 seconds, MIDI40..88. Exact original JAMS/URMP note labels, not
model predictions and NOT synthetic augmented audio. No test labels/audio.

Important:
* "E2+E3" is exact pitches MIDI 40+52, not generic octave 12.
* "F+Em" can mean F major CHORD + E minor CHORD, vs F SINGLE NOTE
  + Em chord. We report BOTH separately as pitch-class inclusion, NOT
  claim that two separately performed harmonic sources exist in the WAV.
* True simultaneous activity = duration with both notes held (>0ms).
  An onset cluster with <=30ms spacing is a different statistic.
* Co-active pitch pairs count EACH pair of note instances; 6 distinct
  simultaneous notes create 15 pairs. They are not 15 independent WAVs.
* Exact ACTIVE pitch sets are time-weighted after processing all
  simultaneous starts/ends. A segment is not an independent recording.
* Different voicings, inversions and octaves collapse only in separate
  pitch-class reports, NEVER in MIDI-pitch results.

Labels are read with pinned S613 loaders and their GuitarSet SHA checks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json

from scripts.s613_prepare_strict_live import labels, download
from scripts.s624_audit_real_overlap import PROTOCOL, clipped_notes
from scripts.s615_exactk_audit import group_onsets


PITCH_NAMES=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
FOCI={
    "E2_plus_E3": ("pitches", (40,52)),
    "E3_plus_F3": ("pitches", (52,53)),
    "E3_plus_Fsharp3": ("pitches", (52,54)),
    "E2_plus_E3_plus_G3": ("pitches", (40,52,55)),
    "F_note_plus_Em_chord": ("pitch_classes", (4,5,7,11)), # E F G B
    "F_major_chord_plus_E_minor_chord": ("pitch_classes", (0,4,5,7,9,11)), # C E F G A B
    "F3_A3_C4_plus_E3_G3_B3_exact_voicing":
        ("pitches",(52,53,55,57,59,60)),
}


def pitch_name(p):
    return PITCH_NAMES[p%12]+str(p//12-1)


def key_string(seq):
    return ",".join(str(int(p)) for p in seq)


def decoded(key, *, pcs=False):
    notes=[int(x) for x in key.split(",")]
    if pcs:
        return "+".join(PITCH_NAMES[x] for x in notes)
    return "+".join(pitch_name(x) for x in notes)


def top_entries(counts, sources, *, limit=25, pcs=False,
                duration=False):
    items=sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:limit]
    return [
        dict(pitches=[int(x) for x in key.split(",")],
             combination=decoded(key,pcs=pcs),
             duration_s=round(float(value),5) if duration else None,
             count=None if duration else int(value),
             sources=int(len(sources.get(key,set()))))
        for key,value in items
    ]


def summarize_source(spec, truth):
    duration=float(spec["seconds"])
    notes,excluded=clipped_notes(truth,duration)
    notes.sort(key=lambda n:(n["start"],n["pitch"]))
    keyset=(spec["corpus"],spec["composition"])
    onset_groups=group_onsets([
        dict(time=n["start"],pitch=n["pitch"]) for n in notes
    ],window_s=.030)
    onset=Counter()
    onset_by_k=Counter()
    onset_start_foci=Counter()
    for g in onset_groups:
        ps=tuple(sorted(g["pitches"]))
        if len(ps)<2:
            continue
        k=key_string(ps)
        onset[k]+=1
        onset_by_k[str(len(ps))]+=1
        for label,(kind,subset) in FOCI.items():
            if kind=="pitches" and set(subset).issubset(ps):
                onset_start_foci[label]+=1
            elif kind=="pitch_classes" and set(subset).issubset(
                    {p%12 for p in ps}):
                onset_start_foci[label]+=1

    active_edges=defaultdict(list)
    for n in notes:
        active_edges[n["start"]].append((1,n["pitch"]))
        active_edges[n["end"]].append((-1,n["pitch"]))
    active=Counter()
    active_durations=Counter()
    active_segments=Counter()
    pc_durations=Counter()
    pc_segments=Counter()
    focus_s=Counter()
    focus_segments=Counter()
    latest_set=None
    latest_pc=None
    focus_prev={k:False for k in FOCI}
    prev=0.
    for t in sorted(active_edges):
        if t>prev:
            activep=tuple(sorted(active))
            pitchclasses=tuple(sorted({p%12 for p in activep}))
            if len(activep)>=2:
                pitchkey=key_string(activep)
                pckey=key_string(pitchclasses)
                active_durations[pitchkey]+=t-prev
                pc_durations[pckey]+=t-prev
                if latest_set!=activep:
                    active_segments[pitchkey]+=1
                if latest_pc!=pitchclasses:
                    pc_segments[pckey]+=1
            for label,(kind,subset) in FOCI.items():
                current=(set(subset).issubset(
                     set(activep) if kind=="pitches" else set(pitchclasses)))
                if current:
                    focus_s[label]+=t-prev
                    if not focus_prev[label]:
                        focus_segments[label]+=1
                focus_prev[label]=current
            latest_set=activep
            latest_pc=pitchclasses
            prev=t
        # End before start at the SAME timestamp.
        for edge,pitch in sorted(active_edges[t],key=lambda v:v[0]):
            active[pitch]+=edge
            if active[pitch]<0:
                raise AssertionError("Negative overlapping MIDI pitch count")
            if active[pitch]==0:
                del active[pitch]
    if active:
        raise AssertionError("Unfinished MIDI notes after last note offset")
    # Close final active state to exactly source.seconds; should be silence,
    # since clipped notes guarantee their offset <= duration.
    assert prev<=duration

    overlap_pairs=Counter()
    overlap_sources=set()
    onset_pairs_30ms=Counter()
    onset_pairs_100ms=Counter()
    pair_lag_bins=Counter()
    overlaps_at_least_10ms=0
    current=[]
    for n in notes:
        current=[old for old in current if old["end"]>n["start"]]
        for prior in current:
            if prior["pitch"]==n["pitch"]:
                continue
            duration_overlap=min(prior["end"],n["end"])-n["start"]
            if duration_overlap<.010:
                continue
            overlap_key=key_string(tuple(sorted((prior["pitch"],n["pitch"]))))
            overlap_pairs[overlap_key]+=1
            overlaps_at_least_10ms+=1
            delta=n["start"]-prior["start"]
            if delta<=.030+1e-10:
                onset_pairs_30ms[overlap_key]+=1
            if delta<=.100+1e-10:
                onset_pairs_100ms[overlap_key]+=1
            lag_label=(
                "0_to_10ms" if delta<=.010
                else "10_to_30ms" if delta<=.030
                else "30_to_100ms" if delta<=.100
                else "100_to_250ms" if delta<=.250
                else "after_250ms")
            pair_lag_bins[lag_label]+=1
        current.append(n)
    exact_e2e3=overlap_pairs[key_string((40,52))]
    return dict(
        id=spec["id"],source=spec["name"],split=spec["split"],
        corpus=spec["corpus"],composition=spec["composition"],
        notes=len(notes),original_annotations=len(truth),exclusions=excluded,
        onset_exact_combinations=dict(onset),
        onset_K2plus_by_K=dict(onset_by_k),
        active_pitch_set_duration_s=dict(active_durations),
        active_pitch_set_segments=dict(active_segments),
        active_pitch_class_duration_s=dict(pc_durations),
        active_pitch_class_segments=dict(pc_segments),
        exact_pair_overlap_at_least_10ms=dict(overlap_pairs),
        near_onset_pair_30ms=dict(onset_pairs_30ms),
        near_onset_pair_100ms=dict(onset_pairs_100ms),
        overlapping_pair_start_lag_bins=dict(pair_lag_bins),
        focus_activity_seconds={name:round(float(focus_s[name]),6) for name in FOCI},
        focus_active_segments={name:int(focus_segments[name]) for name in FOCI},
        focus_onset_clusters={name:int(onset_start_foci[name]) for name in FOCI},
        pair_counts_e2_e3=int(exact_e2e3),
        overlapping_pair_total=int(overlaps_at_least_10ms),
    )


def aggregate(rows):
    combined={name:Counter() for name in (
        "onset_exact_combinations","onset_K2plus_by_K",
        "active_pitch_set_duration_s","active_pitch_set_segments",
        "active_pitch_class_duration_s","active_pitch_class_segments",
        "exact_pair_overlap_at_least_10ms","near_onset_pair_30ms",
        "near_onset_pair_100ms","overlapping_pair_start_lag_bins",
        "focus_activity_seconds","focus_active_segments","focus_onset_clusters")}
    compositions=defaultdict(set)
    sourceids=defaultdict(set)
    for s in rows:
        for field in combined:
            combined[field].update(s[field])
        for field in ("onset_exact_combinations","active_pitch_set_duration_s",
                      "active_pitch_class_duration_s",
                      "exact_pair_overlap_at_least_10ms"):
            for key,val in s[field].items():
                if val:
                    sourceids[field,key].add(s["id"])
                    compositions[field,key].add((s["corpus"],s["composition"]))
    def tops(field,limit=25,pcs=False,dur=False):
        sourcecounts={key:v for (fld,key),v in sourceids.items() if fld==field}
        items=top_entries(combined[field],sourcecounts,
                          limit=limit,pcs=pcs,duration=dur)
        for ent in items:
            key=key_string(ent["pitches"])
            ent["independent_composition_groups"]=len(compositions.get((field,key),()))
        return items
    def focal(label):
        kind,subset=FOCI[label]
        if kind=="pitches" and len(subset)==2:
            pair_key=key_string(subset)
            pair_count=int(combined["exact_pair_overlap_at_least_10ms"][pair_key])
            pair_sources=len(sourceids.get(("exact_pair_overlap_at_least_10ms",pair_key),()))
            pair_compositions=len(compositions.get(
                 ("exact_pair_overlap_at_least_10ms",pair_key),()))
        else:
            pair_count=pair_sources=pair_compositions=None
        return dict(
            definition="exact MIDI pitches" if kind=="pitches" else "pitch-class inclusion (not separable chord performance)",
            included_pitch_numbers=list(subset),
            pitches="+".join(pitch_name(p) for p in subset) if kind=="pitches"
                     else "+".join(PITCH_NAMES[pc] for pc in subset),
            active_simultaneously_seconds=round(float(combined["focus_activity_seconds"][label]),5),
            activity_spans=int(combined["focus_active_segments"][label]),
            onset_clusters_30ms_with_all_notes=int(combined["focus_onset_clusters"][label]),
            overlapping_note_instance_pairs_ge10ms=pair_count,
            source_clips_with_overlapping_pair=pair_sources,
            independent_composition_groups_for_pair=pair_compositions,
        )
    return dict(
        sources=len(rows),
        total_usable_note_events=sum(x["notes"] for x in rows),
        original_annotation_rows=sum(x["original_annotations"] for x in rows),
        total_exact_pitch_pairs_overlapping_ge10ms=sum(
            x["overlapping_pair_total"] for x in rows),
        number_of_distinct_exact_midi_pitch_pairs=len(combined["exact_pair_overlap_at_least_10ms"]),
        number_of_distinct_multi_note_onset_pitch_sets=len(combined["onset_exact_combinations"]),
        number_of_distinct_active_pitch_sets=len(combined["active_pitch_set_duration_s"]),
        number_of_distinct_active_pitch_class_sets=len(combined["active_pitch_class_duration_s"]),
        onset_K2plus_by_K=dict(combined["onset_K2plus_by_K"]),
        overlap_onset_pair_lag_bins=dict(combined["overlapping_pair_start_lag_bins"]),
        focal_cases={name:focal(name) for name in FOCI},
        most_frequent_exact_midi_pairs=tops(
            "exact_pair_overlap_at_least_10ms",50),
        most_frequent_near_onset_exact_midi_pairs_30ms=
            top_entries(combined["near_onset_pair_30ms"],{},limit=30),
        most_frequent_exact_multi_note_onset_sets=tops(
            "onset_exact_combinations",35),
        most_dominant_active_exact_midi_sets_by_seconds=tops(
            "active_pitch_set_duration_s",40,dur=True),
        most_dominant_active_pitch_class_sets_by_seconds=tops(
            "active_pitch_class_duration_s",35,pcs=True,dur=True),
    )


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("data/S625_original_annotations_only"))
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sources=[s for s in protocol["sources"] if s["split"] in ("train","dev")]
    assert len(sources)==70
    archive=protocol["guitar_archives"]["GuitarSet_annotations"]
    (args.root/"data").mkdir(parents=True,exist_ok=True)
    download(archive["url"],args.root/"data/GuitarSet_annotations.zip",
             archive["size"],archive["sha256"])
    rows=[]
    for i,s in enumerate(sources,1):
        real=labels(s,args.root)
        row=summarize_source(s,real["notes"])
        row["annotation_sha256"]=real["sha256"]
        rows.append(row)
        print(json.dumps(dict(index=i,source=s["name"],split=s["split"],
            onset_groups=sum(row["onset_K2plus_by_K"].values()),
            e2_e3=row["pair_counts_e2_e3"])),flush=True)
    report=dict(
        status="S625_TRUE_EXACT_PITCH_COMBINATIONS_TRAIN_DEV_ONLY",
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        sources=len(rows),train_sources=sum(s["split"]=="train" for s in sources),
        dev_sources=sum(s["split"]=="dev" for s in sources),test_sources_read=0,
        scope="First 24s/source, NOTE MIDI pitches 40..88; original annotated real recordings only",
        definitions={
            "active":"distinct pitches simultaneously active at the same time; time weighted",
            "onset_group":"distinct pitches with starts within 30ms of FIRST group onset",
            "note_pair":"unordered exact MIDI note-instance pair co-active >=10ms",
            "F_plus_Em":"F major chord [F A C] plus E minor [E G B] pitch-class union",
            "F_note_plus_Em":"single F note with E minor triad [E G B], pitch-class union",
            "not_a_separate_chord_verification":True,
            "musical_note_not_acoustic_harmonic":True,
        },
        audit_by_split={
            split:aggregate([r for r in rows if r["split"]==split])
            for split in ("train","dev")},
        audit_by_corpus_split={
            f"{split}_{corpus}":aggregate([
                r for r in rows if r["split"]==split and r["corpus"]==corpus])
            for split in ("train","dev") for corpus in ("GuitarSet","URMP")},
        per_source=rows,
        warning="Exact annotation combinations measured; absence or repetition does NOT alone prove added training examples improve NOTE. No held-out test and no recording-level acoustic separation inferred."
    )
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for name,rec in report["audit_by_split"].items():
        print("AUDITED_EXACT_COMBINATIONS "+json.dumps(dict(
            split=name,sources=rec["sources"],notes=rec["total_usable_note_events"],
            total_pairs=rec["total_exact_pitch_pairs_overlapping_ge10ms"],
            unique_exact_pairs=rec["number_of_distinct_exact_midi_pitch_pairs"],
            unique_pitch_sets_onset=rec["number_of_distinct_multi_note_onset_pitch_sets"],
            onset_K=rec["onset_K2plus_by_K"],
            foci=rec["focal_cases"],
            top_pairs=rec["most_frequent_exact_midi_pairs"][:15],
            top_onset_pitch_sets=rec["most_frequent_exact_multi_note_onset_sets"][:12],
            top_active_pitch_sets=rec["most_dominant_active_exact_midi_sets_by_seconds"][:12],
            top_pitch_class_sets=rec["most_dominant_active_pitch_class_sets_by_seconds"][:10],
        ),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
