"""S714: audit frozen S708 emitted events, by true K, texture, harmonic coincidences.

Consumes ONLY pinned S708 held/dev event artifacts. No checkpoint, annotations
for training, DEV tuning or novel threshold. Frozen metrics remain unchanged.
"""
import argparse
from pathlib import Path
import zipfile
import json

import numpy as np

SR = 44100
HOP = 256
VARIANTS = ("native", "native_s675_causal", "candidate")
HARMONICS = (12, 19, 24)


def events_to_frames(events, ends):
    pred = np.zeros((len(ends), 49), bool)
    active = {}
    for e in events:
        p = int(e["pitch"]) - 40
        a, d = int(e["available_sample"]), int(e["decision_sample"])
        if not 0 <= p < 49:
            raise ValueError("Invalid pitch")
        if a < d or a < 0:
            raise ValueError("Backdated event")
        if abs(float(e["time"]) - a / SR) > 1e-10:
            raise ValueError("Event clock does not match availability")
        frame = int(np.searchsorted(ends, a))
        if e["type"] == "note_on":
            if p in active:
                raise ValueError("Duplicate simultaneous pitch")
            active[p] = frame
        elif e["type"] == "note_off":
            if p not in active:
                raise ValueError("NoteOff without NoteOn")
            pred[active.pop(p):frame,p] = True
        else:
            raise ValueError("Unknown MIDI event")
    for p, begin in active.items():
        pred[begin:,p] = True
    return pred


def annotation_frames(notes, ends):
    times = ends / SR
    unique = np.zeros((len(ends),49),bool)
    count = np.zeros(len(ends),np.int16)
    for n in notes:
        p = int(n["pitch"]) - 40
        if not 0 <= p < 49:
            raise ValueError("Truth MIDI pitch outside 40..88")
        on = (times >= n["time"]) & (times < n["time"]+n["duration"])
        unique[on,p] = True
        count[on] += 1
    return unique, count


def one_item(item):
    ends = np.arange(HOP,int(item["samples"])+1,HOP)
    truth, count = annotation_frames(item["truth"],ends)
    unique = truth.sum(axis=1)
    summary = dict(name=item["name"],frames=len(ends),
                   simultaneous_same_pitch_frames=int(np.count_nonzero(unique!=count)),
                   variants={})
    for variant in VARIANTS:
        predicted = events_to_frames(item["events"][variant],ends)
        pc = predicted.sum(axis=1)
        diff = np.minimum(pc,6)-np.minimum(count,6)
        unique_diff = np.minimum(pc,6)-np.minimum(unique,6)
        miss = truth & ~predicted
        fp = predicted & ~truth
        harmonic = np.zeros_like(fp)
        for interval in HARMONICS:
            harmonic[:,interval:] |= truth[:,:-interval]
        rows = {}
        for k in (1,2,3,4,5,6):
            mask = count==k if k<6 else count>=6
            rows[str(k)] = dict(frames=int(mask.sum()),
                correct=int(np.count_nonzero(mask&(diff==0))),
                under=int(np.count_nonzero(mask&(diff<0))),
                over=int(np.count_nonzero(mask&(diff>0))))
        by_register = {}
        for name,start,stop in (("low_40_52",0,13),("mid_53_64",13,25),("high_65_88",25,49)):
            by_register[name]=dict(truth_note_frames=int(truth[:,start:stop].sum()),
              missed_note_frames=int(miss[:,start:stop].sum()),
              false_note_frames=int(fp[:,start:stop].sum()))
        summary["variants"][variant]=dict(frames=len(ends),correct=int(np.count_nonzero(diff==0)),
           active_frames=int(np.count_nonzero(count)),
           active_correct=int(np.count_nonzero((count>0)&(diff==0))),
           under=int(np.count_nonzero(diff<0)), over=int(np.count_nonzero(diff>0)),
           unique_pitch_k_correct=int(np.count_nonzero(unique_diff==0)),
           missed_note_frames=int(miss.sum()), false_note_frames=int(fp.sum()),
           harmonic_coincidence_fp=int((fp&harmonic).sum()),
           by_true_k=rows,by_register=by_register)
    return summary


def aggregate(rows):
    result={}
    for variant in VARIANTS:
        raw=[row["variants"][variant] for row in rows]
        keys=("frames","correct","active_frames","active_correct","under","over",
              "unique_pitch_k_correct",
              "missed_note_frames","false_note_frames","harmonic_coincidence_fp")
        r={k:sum(x[k] for x in raw) for k in keys}
        r["exact_k"]=r["correct"]/max(1,r["frames"])
        r["unique_pitch_exact_k_DIAGNOSTIC_ONLY"]=r["unique_pitch_k_correct"]/max(1,r["frames"])
        r["active_exact_k"]=r["active_correct"]/max(1,r["active_frames"])
        r["under_rate"]=r["under"]/max(1,r["frames"])
        r["over_rate"]=r["over"]/max(1,r["frames"])
        r["by_true_k"]={str(k):{stat:sum(x["by_true_k"][str(k)][stat] for x in raw)
            for stat in ("frames","correct","under","over")} for k in (1,2,3,4,5,6)}
        r["by_register"]={k:{stat:sum(x["by_register"][k][stat] for x in raw)
            for stat in ("truth_note_frames","missed_note_frames","false_note_frames")}
            for k in ("low_40_52","mid_53_64","high_65_88")}
        result[variant]=r
    return result


def audit_zip(artifact):
    with zipfile.ZipFile(artifact) as z:
        report=json.loads(z.read("report.json"))
        output=dict(schema="S714_S708_CAUSAL_EVENT_K_FORENSICS",source_run=39094682419,
                    no_training=True,dev_not_used_for_selection=True,
                    harmonic_coincidence_is_not_proof_of_harmonic_causation=True,splits={})
        for split in ("HELD","DEV"):
            rows=[one_item(json.loads(x)) for x in z.read(split.lower()+"_events.jsonl").splitlines()]
            stat=aggregate(rows)
            for variant in VARIANTS:
                for key in ("exact_k","active_exact_k"):
                    observed=stat[variant][key]
                    original=report[split][variant][key]
                    if abs(observed-original)>1e-10:
                        raise AssertionError(f"{split} {variant} {key} replay mismatch: {observed} != {original}")
            output["splits"][split]=dict(tracks=len(rows),
                ambiguous_simultaneous_pitch_frames=sum(r["simultaneous_same_pitch_frames"] for r in rows),
                summary=stat,by_texture={
                    group:aggregate([r for r in rows if r["name"].endswith("_"+group)])
                    for group in ("comp","solo")},raw=rows)
        return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Refuse overwriting an audit result")
    result=audit_zip(args.artifact)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    for split,data in result["splits"].items():
        for variant in ("native_s675_causal","candidate"):
            r=data["summary"][variant]
            print("S714 AUDIT",json.dumps(dict(split=split,variant=variant,
                  tracks=data["tracks"],exact_k=r["exact_k"],
                  active_exact_k=r["active_exact_k"],under=r["under"],over=r["over"],
                  harmonic_coincidence_fp=r["harmonic_coincidence_fp"],
                  unison_frames=data["ambiguous_simultaneous_pitch_frames"])),
                  flush=True)


if __name__=="__main__":
    main()
