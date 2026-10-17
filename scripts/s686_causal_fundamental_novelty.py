"""S686 causal spectral fundamental novelty for weak native Onset rescue.

Unlike S683's unspecific learned-onset amplification, rescue requires three
independent current/past observations: (1) original native onset posterior
above a preregistered FLOOR, (2) fundamental multi-resolution energy RISE
vs only earlier FFTs, and (3) original native sustain support. The unchanged
selector and original K posterior still decide admission and pitch.

Thresholds are frozen by five composition groups of TRAIN, vetoed by four
other TRAIN groups before an exploratory 12-track DEV report. Evaluation
includes frozen S675 NoteOff and full-note F1. No future PCM or DEV tuning.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    ROOT,MODEL,raw_guitar_audio,native_posteriors,
    note_instances,native_rhythm_clock
)
from scripts.s682_onset_four_lock_guard import (
    native_metrics,aggregate,full_aggregate,safe_full,score_release
)

FLOORS=(.20,.30,.40)
NOVELTY=(.12,.24,.40,.65)
SUPPORT=.45
POLICIES=("native",)+tuple(f"rescue_{floor:.2f}_rise_{rise:.2f}"
                           for floor in FLOORS for rise in NOVELTY)


def rescue(p,features,policy,native_onset):
    if policy=="native":return p
    _,floor,_,novelty=policy.split("_")
    floor=float(floor);novelty=float(novelty)
    # Causal physical scores: 54-56 current minus t-2; 66-68
    # current minus t-8. All FFT features are right-aligned at t.
    near=np.maximum(features[:,:,54:57],0).mean(axis=2)
    history=np.maximum(features[:,:,66:69],0).mean(axis=2)
    fundamental=np.maximum(features[:,:,[0,8,16]],0).mean(axis=2)
    delta=np.maximum(.65*near+.35*history,0)/(.2+fundamental)
    approved=((p[:,:,0]>=floor)&(p[:,:,0]<native_onset)&
              (p[:,:,1]>=SUPPORT)&(delta>=novelty))
    out=p.copy()
    out[:,:,0]=np.where(approved,native_onset+.025,out[:,:,0])
    if not np.array_equal(out[:,:,1:],p[:,:,1:]):
        raise AssertionError("S686 modified native sustain or NoteOff")
    return out


def record(spec,root,archive,heads,sel,policies,offmodel=None):
    wave,_=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,feat=feature_track(wave,block_size=4096)
    p,k=native_posteriors(feat,heads)
    original_sha=digest(p[:,:,1:].tobytes()+k.tobytes())
    scores={};release={}
    for name in policies:
        proposed=rescue(p,feat,name,sel.onset)
        emitted=decode(ends,proposed,k,sel,len(wave))
        scores[name]=native_metrics(emitted,gt,ends)
        if offmodel is not None:
            ons,pairs=event_score(emitted,gt,tolerance=.080)
            row=dict(truth=gt,ends=ends,feats=feat,post=p,k=k,events=emitted,
                     onset=ons,pairs=pairs,notes=note_instances(emitted,ends),
                     clock=native_rhythm_clock(emitted,ends))
            release[name]=score_release(row,offmodel)
        assert digest(proposed[:,:,1:].tobytes()+k.tobytes())==original_sha
    return dict(name=spec["name"],composition=spec["composition"],
                scores=scores,release=release)


def choose(cal):
    sums={v:aggregate([r["scores"] for r in cal],v) for v in POLICIES}
    baseline=sums["native"]
    groups=sorted({r["composition"] for r in cal})
    qualified=[]
    for name in POLICIES[1:]:
        val=sums[name]
        groupgood=sum(aggregate([r["scores"] for r in cal
                                 if r["composition"]==grp],name)["onset_F1"] >=
                      aggregate([r["scores"] for r in cal
                                 if r["composition"]==grp],"native")["onset_F1"]
                      for grp in groups)
        if (val["tp"]>=baseline["tp"] and
            val["K_frames_correct"]>=baseline["K_frames_correct"] and
            val["onset_F1"]>=baseline["onset_F1"]+.001 and groupgood>=3):
            qualified.append(name)
    selected=max(qualified,key=lambda n:sums[n]["onset_F1"]) if qualified else "native"
    return selected,sums,qualified


def run(out,checkpoint):
    if out.exists():raise FileExistsError("S686 immutable evidence path exists")
    t0=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    trains=[s for s in proto["sources"]
            if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({s["composition"] for s in trains})
    assert len(comps)==9 and len(trains)==36 and len(dev)==12
    calgroup=set(comps[:5])
    train_cal=[s for s in trains if s["composition"] in calgroup]
    held=[s for s in trains if s["composition"] not in calgroup]
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    sel=Selection(**heads.metadata["selection"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    release_head=S672TrialNoteOff.load(checkpoint,heads)
    report=dict(schema="S686_FUNDAMENTAL_NOVELTY_RESCUE",
        native_heads_immutable=True,original_K_head_unchanged=True,
        original_sustain_release_heads_unchanged=True,
        original_LIVE_untouched=True,experiment_strictly_past_audio=True,
        original_S675_release_head_unchanged=True,
        TEST_unopened=True,DEV_previously_exposed=True,
        train_cal_compositions=sorted(calgroup),
        train_held_compositions=sorted(set(comps)-calgroup),
        original_model_sha256=frozen["model_sha256"],
        S675_sha256=digest(checkpoint.read_bytes()))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/file,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            records=[]
            for i,spec in enumerate(train_cal,1):
                records.append(record(spec,root,archive,heads,sel,POLICIES))
                print("S686 CAL",i,"/",len(train_cal),spec["name"],flush=True)
            chosen,cal_scores,qualified=choose(records)
            report["TRAIN_cal_native_and_candidates"]=cal_scores
            report["TRAIN_cal_qualified"]=qualified
            report["TRAIN_frozen_choice"]=chosen
            print("S686 CAL FROZEN",chosen,json.dumps(cal_scores[chosen]),flush=True)
            variants=("native",) if chosen=="native" else ("native",chosen)
            heldrows=[]
            for i,spec in enumerate(held,1):
                heldrows.append(record(spec,root,archive,heads,sel,variants,
                                       release_head.model))
                print("S686 HELD",i,"/",len(held),spec["name"],flush=True)
            held_scores={n:aggregate([r["scores"] for r in heldrows],n) for n in variants}
            held_release={n:full_aggregate([r["release"] for r in heldrows],n)
                          for n in variants}
            report["TRAIN_held_onset_K"]=held_scores
            report["TRAIN_held_S675_full_NOTE"]=held_release
            heldsafe=(chosen!="native" and
                      held_scores[chosen]["tp"]>=held_scores["native"]["tp"] and
                      held_scores[chosen]["K_frames_correct"]>=held_scores["native"]["K_frames_correct"] and
                      held_scores[chosen]["onset_F1"]>held_scores["native"]["onset_F1"] and
                      safe_full(held_release["native"],held_release[chosen]))
            report["TRAIN_held_four_locks"]=bool(heldsafe)
            print("S686 HELD LOCK",heldsafe,flush=True)
            devrows=[]
            for i,spec in enumerate(dev,1):
                devrows.append(record(spec,root,archive,heads,sel,variants,
                                      release_head.model))
                print("S686 DEV",i,"/",len(dev),spec["name"],flush=True)
            dscores={n:aggregate([r["scores"] for r in devrows],n) for n in variants}
            drelease={n:full_aggregate([r["release"] for r in devrows],n) for n in variants}
            report["DEV_onset_K"]=dscores
            report["DEV_S675_full_NOTE"]=drelease
            report["DEV_by_recording"]=devrows
    safe=(chosen!="native" and heldsafe and
          dscores[chosen]["onset_F1"]>dscores["native"]["onset_F1"] and
          dscores[chosen]["K_frames_correct"]>=dscores["native"]["K_frames_correct"] and
          safe_full(drelease["native"],drelease[chosen]))
    report["four_locks_passed"]=bool(safe)
    report["promoted"]=False
    report["elapsed_s"]=round(time.perf_counter()-t0,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S686 FINAL",json.dumps(dict(selected=chosen,train_held_pass=heldsafe,
         DEV_base=dscores["native"],DEV_candidate=dscores[chosen],
         DEV_offset_base=drelease["native"],DEV_offset_candidate=drelease[chosen],
         safe=safe,reference_unchanged=True)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.out,args.checkpoint)
