"""S728: real 151+ octave overlapped onsets, separately verify S727 variants.

S726 identified 151 octave-overlap ground-truth attacks on GuitarSet CAL,
and 791 on FIT. Unlike the one-example E2/E3 CAL shadow, this audit measures
all REAL independent octave + 19-semitone attacks, baseline matched recall and
false *upper* notes created while an actual lower pitch is sustaining.

Predicted event streams and source memory are built from audio WITHOUT labels;
only then do we open annotations for one-to-one onset matching.
No held/DEV/TEST tuning or promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from causal_note.strict_live import SR
from causal_note.s725_source_memory import augmented_track
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import event_score,real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import record,load_model
from scripts.s705_train_temporal_loops import setup
from scripts.s727_train_harmonic_negative_loops import score_events

SOURCE="s708_selected.pt"
NAMES=("S708_frozen","harmonic_1p0_epoch1","harmonic_1p8_epoch1",
       "harmonic_1p8_epoch2","harmonic_2p6_epoch1","harmonic_2p6_epoch2")
INTERVALS=(12,19)


def has_earlier_parent(notes,pitch,t,d,min_prior=.060):
    parent=pitch-d
    return any(n["pitch"]==parent and n["time"]+min_prior<=t
           and t<n["time"]+n["duration"] for n in notes)


def overlap_results(events,truth,d):
    predictions=[e for e in events if e["type"]=="note_on"]
    matched,links=event_score(events,truth,tolerance=.080)
    pred_matched={p for p,t in links}
    truth_matched={t for p,t in links}
    target=[i for i,n in enumerate(truth) if has_earlier_parent(
        truth,n["pitch"],n["time"],d)]
    false=[j for j,e in enumerate(predictions) if j not in pred_matched
           and has_earlier_parent(truth,e["pitch"],e["time"],d)]
    true_hit=[i for i in target if i in truth_matched]
    result=dict(
        observed_real_targets=len(target),correct_real_onsets=len(true_hit),
        missed_real_onsets=len(target)-len(true_hit),
        spurious_harmonic_child_onsets=len(false),
        observed_by_register={
           "low_52_64":sum(52<=truth[i]["pitch"]<=64 for i in target),
           "mid_65_76":sum(65<=truth[i]["pitch"]<=76 for i in target),
           "high_77_88":sum(77<=truth[i]["pitch"]<=88 for i in target)},
        matched_pitch_counts=dict(Counter(str(truth[i]["pitch"]) for i in true_hit)),
        matched_delta_ms=[round((predictions[p]["time"]-truth[t]["time"])*1000,3)
                       for p,t in links if t in target]
    )
    return result


def main(args):
    if args.out.exists():raise FileExistsError("S728 refuses mutable overwrite")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    trained=json.loads((args.s727_dir/"report.json").read_text())
    sha=digest((args.s708_dir/SOURCE).read_bytes())
    if frozen["checkpoint_sha256"]!=sha or trained["s708_sha256"]!=sha:
        raise AssertionError("S728 model origin SHA mismatch")
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    models={}
    for name in NAMES:
        if name=="S708_frozen":path=args.s708_dir/SOURCE
        else:
            config=trained["train"][name]
            path=args.s727_dir/config["path"]
            if digest(path.read_bytes())!=config["sha256"]:
                raise AssertionError("S728 checkpoint hash differs from S727 signed training")
        models[name]=load_model(path)[0]
    report=dict(schema="S728_REAL_GUITARSET_CAL_OCTAVE_ONSET_TARGETED_FORENSIC",
        promoted=False,TEST_opened=False,CAL_only=True,
        s708_sha256=sha,intervals=INTERVALS,models=list(models),
        no_retuning_on_true_onset_errors=True,
        minimum_parent_preexisting_ms=60,onset_tolerance_ms=80,tracks=[],
        sums={name:{str(d):dict(observed_real_targets=0,correct_real_onsets=0,
                               missed_real_onsets=0,spurious_harmonic_child_onsets=0)
                   for d in INTERVALS} for name in models})
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=proto["guitar_archives"][key]
            download(meta["url"],root/"data"/file,meta["size"],meta["sha256"])
        originals=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(originals,audio,ann)
            specs=[s for s in expanded if s["composition"] in frozen["cal_compositions"]]
            assert len(specs)==32
            for i,spec in enumerate(specs):
                item=record(spec,audio,heads,selection,offset)
                native=item["x"];item["native_features"]=native[:,:,:126].copy()
                extended=augmented_track(native)
                emitted={}
                for name,model in models.items():
                    x=native if name=="S708_frozen" else extended
                    p,k=predict_track(model,x)
                    emitted[name]=score_events(item,p,k,policy,offset,release)
                # Decode ALL event streams before reading any real annotations.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                rows={name:{str(d):overlap_results(events,truth,d)
                    for d in INTERVALS} for name,events in emitted.items()}
                for name,entry in rows.items():
                    for d,values in entry.items():
                        for key in report["sums"][name][d]:
                            report["sums"][name][d][key]+=values[key]
                report["tracks"].append(dict(name=spec["name"],
                    audio_sha256=item["sha256"],counts=rows))
                print("S728 REAL OCTAVE",i+1,len(specs),spec["name"],flush=True)
                save()
    for d in INTERVALS:
        for name in models:
            c=report["sums"][name][str(d)]
            assert c["correct_real_onsets"]+c["missed_real_onsets"]==c["observed_real_targets"]
        assert report["sums"]["S708_frozen"][str(d)]["observed_real_targets"]>=20
    report["interpretation"]=(
        "Paired REAL CAL event diagnostic, not new out-of-sample proof."
        " Avoid selecting penalty based on these errors after CAL selection."
        " Real E2->E3 CAL contains just one event; use all true octave onsets.")
    save()
    print("S728 OCTAVE FINAL",json.dumps(dict(totals=report["sums"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--s727-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):main(ap.parse_args())
