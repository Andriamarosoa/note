"""S726: how many *real* independent pitched attacks overlap ongoing notes?

Read audio-verified GuitarSet source inventory but count note annotations only.
This is a DATASET COVERAGE audit, not a detector score or a new model.
Avoid claiming E2+E3 generalization based on the only CAL pair.
FIT/CAL/HELD/DEV compositional separation fixed by S708/S699.
TEST excluded. Dataset labels never enter S708/S725 inference.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import tempfile
import zipfile
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,HELD_COMPS
from scripts.s613_train_strict_live import real_notes
from causal_note.strict_live import SR

PAIRS=(12,19,3,4,7)
def annotate_overlap(notes,min_prior=.060,range_lo=40,range_hi=88):
    """Only independent *onsets*, after another lower pitch already rings."""
    hist=Counter();cases=[]
    ordered=sorted(notes,key=lambda n:n["time"])
    for n in ordered:
        higher=int(n["pitch"]);on=float(n["time"])
        if higher<range_lo or higher>range_hi:continue
        lows=[prev for prev in ordered if prev["pitch"]<higher
              and prev["pitch"]>=range_lo and prev["time"]+min_prior<=on
              and prev["time"]+prev["duration"]>on]
        intervals={int(higher-prev["pitch"]) for prev in lows}
        if lows:
            hist["any_lower_sustained"]+=1
            hist["one_other_active" if len(lows)==1 else "multiple_other_active"]+=1
        for d in PAIRS:
            if d in intervals:
                hist[f"interval_{d}"]+=1
                if d==12 and higher==52:
                    hist["E2_then_E3"]+=1
                cases.append(dict(onset_pitch=higher,parent_pitch=higher-d,
                                  interval=d,time=round(on,6)))
    hist["total_attacks"]=len(ordered)
    hist["notes_eligible_range"]=sum(range_lo<=x["pitch"]<=range_hi for x in ordered)
    return hist,cases


def main(args):
    if args.out.exists():raise FileExistsError("S726 immutable corpus audit path exists")
    args.out.mkdir(parents=True)
    report=dict(schema="S726_REAL_GUITARSET_POLYPHONIC_OVERLAP_PREVALENCE",
       tested_S708=False,train_only_analysis=True,TEST_opened=False,
       promoted=False,annotations_used_only_to_count_dataset_events=True,
       interval_semitones=PAIRS,min_parent_sustain_ms=60,splits={},sources=[])
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,_,_,_=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=protocol["guitar_archives"][key]
            download(meta["url"],root/"data"/name,meta["size"],meta["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as a,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as b:
            expanded=expanded_train_specs(train,a,b)
            splits={"FIT":[x for x in expanded if x["composition"] in FIT_COMPS],
                    "CAL":[x for x in expanded if x["composition"] in CAL_COMPS],
                    "HELD":[x for x in expanded if x["composition"] in HELD_COMPS],
                    "DEV":dev}
            assert [len(splits[x]) for x in ("FIT","CAL","HELD","DEV")]==[80,32,32,12]
            for phase,specs in splits.items():
                counts=Counter()
                tracks=[]
                for i,spec in enumerate(specs):
                    notes=real_notes(labels(spec,root)["notes"],float(spec["seconds"]))
                    row,events=annotate_overlap(notes)
                    counts.update(row)
                    cases=[x for x in events if x["interval"] in (12,19)]
                    tracks.append(dict(name=spec["name"],counts=dict(row), cases=cases[:90]))
                    report["sources"].append(dict(split=phase,name=spec["name"],
                                composition=spec["composition"],audio_filename=spec["audio"]))
                report["splits"][phase]=dict(n_tracks=len(specs),
                    counts=dict(counts),tracks=tracks)
                print("S726 CORPUS COVERAGE",phase,json.dumps(dict(counts)),flush=True)
                save()
        fit=report["splits"]["FIT"]["counts"]
        report["dataset_suitable_for_E2_E3_real_training"]=fit.get("E2_then_E3",0)>=30
        report["dataset_suitable_for_octave_overlap_training"]=fit.get("interval_12",0)>=100
        save()
    print("S726 COVERAGE FINAL",json.dumps(dict(
          totals={n:{"octave":v["counts"].get("interval_12",0),
                       "E2_then_E3":v["counts"].get("E2_then_E3",0),
                       "fifth":v["counts"].get("interval_19",0)}
                  for n,v in report["splits"].items()},
          fit_sufficient=report["dataset_suitable_for_E2_E3_real_training"],
          promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    main(ap.parse_args())
