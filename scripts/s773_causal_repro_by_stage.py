"""S773: isolate causal S708 replay drift by stage on FIT/CAL WAV clips.

No labels, no model fitting, no CAL-tuned decisions, no artifact promotion.
Compare repeated extraction/prediction/MIDI outputs within the same process.
Matrix jobs with 1/2 PyTorch threads test cross-worker determinism.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch

from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,digest
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input

def fingerprint(array):
    x=np.asarray(array)
    h=hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return dict(sha256=h.hexdigest(),shape=list(x.shape),
                dtype=str(x.dtype),finite=bool(np.isfinite(x).all()))

def event_fingerprint(events):
    # JSON serialises only already-observed events. Never uses truth, scores or next audio.
    raw=json.dumps(events,sort_keys=True,separators=(",",":"),default=str).encode()
    return dict(sha256=hashlib.sha256(raw).hexdigest(),count=len(events),
                noteons=sum(e["type"]=="note_on" for e in events),
                noteoffs=sum(e["type"]=="note_off" for e in events))

def stable_stages(rows):
    keys=rows[0]["stages"]
    diffs={}
    for key in keys:
        stamps=[row["stages"][key]["sha256"] for row in rows]
        diffs[key]=dict(same=len(set(stamps))==1,sha256=stamps)
    return diffs

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S773 output must be immutable")
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(773)
    np.random.seed(773)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    assert sha==signed["checkpoint_sha256"]
    policy=GuidedPolicy(**signed["frozen_candidate"]["choice"]["policy"])
    release=signed["frozen_candidate"]["choice"]["release"]
    model=load_model(args.s708_dir/"s708_selected.pt")[0].eval()
    result=dict(schema="S773_CAUSAL_STAGE_DETERMINISM",
        complete=False,reference_sha=sha,promoted=False,
        LIVE_changed=False,no_labels_read=True,TEST_opened=False,
        HELD_opened=False,torch_threads=args.threads,
        experiments=[],identical_within_worker=None)
    def save():
        (args.out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    save()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,archive in (("GuitarSet_audio","GuitarSet_audio.zip"),
                            ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/archive,remote["size"],remote["sha256"])
        train=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(train,wav,ann)
            examples=[
                next(s for s in specs if s["composition"]==FIT_COMPS[0] and "_comp" in s["name"]),
                next(s for s in specs if s["composition"]==FIT_COMPS[0] and "_solo" in s["name"]),
                next(s for s in specs if s["composition"]==CAL_COMPS[0] and "_comp" in s["name"]),
                next(s for s in specs if s["composition"]==CAL_COMPS[0] and "_solo" in s["name"])]
            for spec in examples:
                trials=[]
                for rep in range(args.repeats):
                    item,post,k,events,acoustic,ctx,eligible=baseline_input(
                        spec,wav,root,heads,selection,offset,model,policy,release)
                    p_check,k_check=predict_track(model,item["x"])
                    stages=dict(
                        audio_feature=fingerprint(item["x"]),
                        native_post=fingerprint(item["native_post"]),
                        network_onset=fingerprint(p_check),
                        network_k=fingerprint(k_check),
                        blended_post=fingerprint(post),
                        emitted_midi=event_fingerprint(events),
                        acoustic_candidates=fingerprint(acoustic),
                        scene_context=fingerprint(ctx),
                        eligible_mask=fingerprint(eligible))
                    trials.append(dict(repetition=rep,stages=stages))
                differences=stable_stages(trials)
                result["experiments"].append(dict(name=spec["name"],
                    composition=spec["composition"],repeats=args.repeats,
                    stages=trials,within_worker= differences,
                    all_stages_reproducible=all(v["same"] for v in differences.values())))
                save()
                print("S773_TRACK",spec["name"],"deterministic",
                      result["experiments"][-1]["all_stages_reproducible"],flush=True)
    result["complete"]=True
    result["identical_within_worker"]=all(x["all_stages_reproducible"]
                                          for x in result["experiments"])
    save()
    print("S773_DONE",json.dumps(dict(threads=args.threads,
        within_worker=result["identical_within_worker"],
        stages=[{x["name"]:{k:v["same"] for k,v in x["within_worker"].items()}}
                for x in result["experiments"]])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,choices=(1,2),required=True)
    p.add_argument("--repeats",type=int,default=3)
    evaluate(p.parse_args())
