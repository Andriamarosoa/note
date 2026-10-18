"""S725 E2->E3 independent real-acoustic onset audit on unchanged GuitarSet CAL.

Does not tune or promote; this diagnostic may report zero E2/E3 overlaps if
the dataset contains none. Predictions are finalized before opening JAMS.
Read already trained frozen checkpoints only. Reuses strict S708/S707 decoder
and learned Exact-K guided selector for all models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from causal_note.strict_live import SR
from causal_note.s725_source_memory import augmented_track,disabled_memory_track,BASE
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import setup,record,load_model
from scripts.s725_source_memory_exactk_training import score_events

NAMES=("S708_frozen","no_memory_epoch2","with_memory_epoch2")


def actual_e3_with_e2_sounding(notes,minimum_prior=.060):
    """Real E3 attacks on top of an already active E2, NOT synthetic label."""
    e2=[n for n in notes if n["pitch"]==40]
    candidates=[]
    for i,n in enumerate(notes):
        if n["pitch"]!=52:continue
        t=float(n["time"])
        if any(float(low["time"])+minimum_prior<=t and
               t<float(low["time"])+float(low["duration"]) for low in e2):
            candidates.append(i)
    return candidates


def characterize_predicted_onset(events,truth):
    """Unique same-pitch +/-80ms onset matching, before label selection."""
    onset=[e for e in events if e["type"]=="note_on"]
    _,pairs=event_score(events,truth,tolerance=.080)
    true_ids=set(actual_e3_with_e2_sounding(truth))
    hit=sum(t in true_ids for p,t in pairs)
    matched={p for p,t in pairs}
    e2=[n for n in truth if n["pitch"]==40]
    false_while_e2=sum(
        e["pitch"]==52 and j not in matched and any(
          low["time"]+.060<=e["time"]<
             low["time"]+low["duration"]
          for low in e2)
        for j,e in enumerate(onset)
    )
    return dict(E3_on_E2_total=len(true_ids),E3_on_E2_matched=hit,
                E3_on_E2_missed=len(true_ids)-hit,
                false_E3_while_true_E2_sustains=false_while_e2)


def main(args):
    if args.out.exists():raise FileExistsError("S725 E2->E3 diagnostic refuses overwrite")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    trained=json.loads((args.s725_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=original["checkpoint_sha256"] or trained["s708_sha256"]!=sha:
        raise AssertionError("S725 only signed S708-derived checkpoints allowed")
    policy=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    models={}
    for name in NAMES:
        if name=="S708_frozen":path=args.s708_dir/"s708_selected.pt"
        else:
            m=trained["train"][name]
            path=args.s725_dir/m["path"]
            if digest(path.read_bytes())!=m["sha256"]:
                raise AssertionError("S725 trained model checkpoint SHA mismatch")
        models[name]=load_model(path)[0]
    report=dict(schema="S725_ACTUAL_E2_THEN_E3_WHILE_E2_SOUNDING_CAL32",
        promoted=False,TEST_opened=False,CAL_only=True,
        comparisons=list(NAMES),
        note="E2=40, E3=52; E2 started >=60ms before E3 onset and lasts into E3, no future labels in inference",
        original_s708_sha256=sha,tracks=[],totals={name:{
            "E3_on_E2_total":0,"E3_on_E2_matched":0,
            "E3_on_E2_missed":0,"false_E3_while_true_E2_sustains":0} for name in NAMES})
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=proto["guitar_archives"][key]
            download(meta["url"],root/"data"/file,meta["size"],meta["sha256"])
        original_specs=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(original_specs,wav,ann)
            cal=[s for s in expanded if s["composition"] in original["cal_compositions"]]
            assert len(cal)==32
            for i,spec in enumerate(cal):
                item=record(spec,wav,heads,selection,offset)
                raw=item["x"]
                actual=augmented_track(raw)
                emitted={}
                for name,model in models.items():
                    if name=="S708_frozen":x=raw
                    elif name.startswith("no_memory"):x=disabled_memory_track(raw,model.mean[BASE:].numpy())
                    else:x=actual
                    p,k=predict_track(model,x)
                    emitted[name]=score_events(item,p,k,policy,offset,release)
                # Ground truth opened only after each candidate emitted causal
                # NoteOns/Offs; labels never enter the source-memory state.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                rows={name:characterize_predicted_onset(e,truth) for name,e in emitted.items()}
                for name,row in rows.items():
                    for key,value in row.items():
                        report["totals"][name][key]+=value
                report["tracks"].append(dict(name=spec["name"],sha256=item["sha256"],
                     E2_E3_truth_indices=actual_e3_with_e2_sounding(truth),
                     diagnostics=rows))
                save()
                print("S725 REAL E2/E3",i+1,len(cal),spec["name"],flush=True)
    if any(row["E3_on_E2_total"] for row in report["totals"].values()):
        report["scope_warning"]="Counts on exposed CAL only; not a new independent test"
    else:
        report["scope_warning"]="Zero actual E2->E3 overlaps identified; no sensitivity conclusion possible"
    save()
    print("S725 E2 E3 FINAL",json.dumps(dict(totals=report["totals"],
              warning=report["scope_warning"],promoted=False)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--s725-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):main(p.parse_args())
