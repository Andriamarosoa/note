"""S669: frozen native release-head activation gating in strict causal NOTE.

Ablate learned release posterior (.10-.95) without changing any model
weights, K, acoustic ABI or use of future audio. PRECOMMIT release policies
before looking at DEV, calibrate using five TRAIN composition families and
four held TRAIN families, then show DEV as exploratory. Scores strict Offset
±50 ms conditional on native per-policy correct NoteOn, with simulated
S630 batch4 emission rounding separately. TEST unopened, no promotion.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR,HOP,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes,summary
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s668_causal_live_noteoff_hazard import (
    raw_guitar_audio,note_instances,native_posteriors,
    GUITARSET_PINNED_CONTROL
)

GATES=(.15,.30,.45,.60,.75,.90)
POLICIES=(("original",None,.25,5),)+tuple(
    (f"gate_{gate:.2f}_sustain_{sustain:.2f}_persist_{persist}",
     gate,sustain,persist)
    for gate in GATES
    for sustain in (.15,.25)
    for persist in (3,5)
)


def score(events,truth,ends,batch_hops=1):
    onset,pairs=event_score(events,truth)
    instances=note_instances(events,ends)
    if len(instances)!=onset["TP"]+onset["FP"]:
        raise AssertionError("S669 NoteOn sequence not paired")
    correct=eligible=forced=long_correct=long_eligible=0
    for i,j in pairs:
        n=truth[j]
        finish=float(n["time"]+n["duration"])
        if finish>=ends[-1]/SR:continue
        eligible+=1
        note=instances[i]
        if n["duration"]>=1.:long_eligible+=1
        off=note["native_end"]
        if off is None:
            forced+=1
            continue
        # Quantize delivery to fully received four-hop ONNX microbatch.
        available=((int(off)+1+batch_hops-1)//batch_hops)*batch_hops
        time=min(len(ends),available)*HOP/SR
        success=int(abs(time-finish)<=.05)
        correct+=success
        if n["duration"]>=1.:long_correct+=success
    return dict(onset=onset,offset=dict(
        strict50=correct,eligible=eligible,
        accuracy50=correct/max(1,eligible),
        forced=forced,long_correct=long_correct,
        long_eligible=long_eligible))


def one(spec,root,archive,heads,original):
    wav,sha=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(wav)/SR)
    ends,x=feature_track(wav,block_size=4096)
    post,k=native_posteriors(x,heads)
    out={}
    for name,threshold,sustain,persist in POLICIES:
        if name=="original":
            events=decode(ends,post,k,original,len(wav))
        else:
            pp=post.copy()
            # Reinterpret ONLY the frozen model's own offset-posterior:
            # 0.95 above gate, 0 below, same per-hop evidence and K.
            pp[:,:,2]=np.where(post[:,:,2]>=threshold,.95,.05)
            selector=replace(original,active_off=sustain,
                             release_frames=persist)
            events=decode(ends,pp,k,selector,len(wav))
        native=score(events,gt,ends,batch_hops=1)
        batch4=score(events,gt,ends,batch_hops=4)
        out[name]=dict(native=native,batch4=batch4)
    return dict(name=spec["name"],composition=spec["composition"],
                pcm_sha256=sha,results=out)


def aggregate(records,mode):
    summary_stats={}
    for name,*_ in POLICIES:
        rows=[r["results"][name][mode] for r in records]
        counts=Counter()
        for row in rows:
            for col in ("strict50","eligible","forced","long_correct","long_eligible"):
                counts[col]+=row["offset"][col]
        summary_stats[name]=dict(
            noteon=summary([r["onset"] for r in rows]),
            strict50=int(counts["strict50"]),
            eligible=int(counts["eligible"]),
            accuracy50=counts["strict50"]/max(1,counts["eligible"]),
            long_correct=int(counts["long_correct"]),
            long_eligible=int(counts["long_eligible"]),
        )
    return summary_stats


def run(out):
    if out.exists():raise FileExistsError("No evidence overwrite")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(train)==36 and len(dev)==12
    comps=sorted({s["composition"] for s in train})
    assert len(comps)==9 and not set(comps)&{s["composition"] for s in dev}
    calibration=set(comps[:5])
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/name,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            tr=[]
            for j,s in enumerate(train,1):
                tr.append(one(s,root,archive,heads,config))
                print("S669 TRAIN",j,"/36",s["name"],flush=True)
            cal=[r for r in tr if r["composition"] in calibration]
            held=[r for r in tr if r["composition"] not in calibration]
            ca=aggregate(cal,"batch4")
            h=aggregate(held,"batch4")
            base="original"
            acceptable=[]
            for name,*_ in POLICIES[1:]:
                if ca[name]["noteon"]["F1"]<ca[base]["noteon"]["F1"]-.002:
                    continue
                if ca[name]["strict50"]<=ca[base]["strict50"]+4:
                    continue
                comps_improved=sum(
                    aggregate([r for r in held if r["composition"]==comp],"batch4")[name]["strict50"] >
                    aggregate([r for r in held if r["composition"]==comp],"batch4")[base]["strict50"]
                    for comp in sorted({r["composition"] for r in held})
                )
                if h[name]["strict50"]>=h[base]["strict50"]+6 and comps_improved>=2 and h[name]["noteon"]["F1"]>=h[base]["noteon"]["F1"]-.003:
                    acceptable.append((name,h[name]["strict50"],comps_improved))
            chosen=max(acceptable,key=lambda x:x[1])[0] if acceptable else "original"
            print("S669 FROZEN TRAIN SELECTION",chosen,flush=True)
            dv=[]
            for j,s in enumerate(dev,1):
                dv.append(one(s,root,archive,heads,config))
                print("S669 DEV",j,"/12",s["name"],flush=True)
    scores={m:aggregate(dv,m) for m in ("native","batch4")}
    assert scores["native"]["original"]["noteon"]["TP"]==GUITARSET_PINNED_CONTROL["TP"]
    for m in scores:
        for name,v in scores[m].items():
            print("S669 SCORE",m,name,"onset_F1",round(v["noteon"]["F1"],6),
                  "correct",v["strict50"],"eligible",v["eligible"],
                  "offset_accuracy",round(v["accuracy50"],6),
                  flush=True)
    report=dict(status="S669_FROZEN_RELEASE_HEAD_THRESHOLD_SEARCH",
       original_heads_original_Exact_K_unchanged=True,
       original_LIVE_program_untouched=True,
       native_note_on_changed_if_active_selection_changed=True,
       PCM_past_only=True,
       TEST_unopened=True,
       DEV_previously_exposed=True,
       preselected_by_TRAIN_only=chosen,
       TRAIN_calibration=ca,TRAIN_held=h,
       DEV_score=scores,DEV_by_recording=dv,
       target_80_met=bool(scores["batch4"][chosen]["accuracy50"]>=.80),
       no_promotion=True)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S669 FINAL",json.dumps(dict(
        choice=chosen,native=scores["batch4"]["original"]["accuracy50"],
        selected=scores["batch4"][chosen]["accuracy50"],
        best_exploratory=max(scores["batch4"].items(),
                              key=lambda x:x[1]["accuracy50"])[0])),
        flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.out)
