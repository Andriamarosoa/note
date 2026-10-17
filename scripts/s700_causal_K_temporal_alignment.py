"""S700 quantify K temporal misalignment from ORIGINAL frozen causal NOTE.

For each of 48 authentic GuitarSet MIC recordings, compare predicted
cardinality to annotated active-note K on original source time stamps:
- native decoded MIDI K(t),
- frozen K-head argmax(t),
- causal-only DELAY d: K(t-d), d>=0,
- diagnostics only FUTURE oracle K(t+d) (never implementable live).
This audit tests whether the weak Exact-K 54/58 percent is a trivial
timebase alignment bug. No model or event writes. TRAIN composition
calibration freezes one causal shift, then held TRAIN and DEV report.
"""
from __future__ import annotations
import argparse,json,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,real_notes,event_score
from scripts.s672_causal_offset_timing_grid import ROOT,MODEL,raw_guitar_audio,native_posteriors
from scripts.s696_direct_K_causal_audit import native_count_timeline,gt_count_timeline,score

DELAYS=(0,1,2,3,4,6,8,12,16,24,32,48,64)
FUTURE_ORACLE=(0,4,8,16,32,64)

def causal_delay(pred,delay):
    pred=np.asarray(pred)
    if int(delay)<0:raise ValueError("Only nonnegative K time delay is causal")
    return pred[np.maximum(0,np.arange(len(pred))-int(delay))]

def oracle_future(pred,ahead):
    """For diagnosis ONLY; never valid as causal inference or new accuracy."""
    pred=np.asarray(pred)
    if int(ahead)<0:raise ValueError("Positive future index only")
    return pred[np.minimum(len(pred)-1,np.arange(len(pred))+int(ahead))]

def metrics(p,gt):
    return score(np.asarray(p),np.asarray(gt))

def record(spec,root,archive,heads,sel):
    audio,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,features=feature_track(audio,block_size=4096)
    p,k=native_posteriors(features,heads)
    orig=digest(p.tobytes()+k.tobytes())
    events=decode(ends,p,k,sel,len(audio))
    predicted=dict(native_MIDI=native_count_timeline(events,ends),
                   frozen_head=k.argmax(axis=1))
    gt=gt_count_timeline(truth,ends)
    all_scores={}
    for name,curve in predicted.items():
        for delta in DELAYS:
            all_scores[f"{name}_delay_{delta}"]=metrics(causal_delay(curve,delta),gt)
        for delta in FUTURE_ORACLE:
            all_scores[f"NONCAUSAL_ORACLE_{name}_future_{delta}"]=metrics(
                oracle_future(curve,delta),gt)
    assert digest(p.tobytes()+k.tobytes())==orig
    return dict(name=spec["name"],composition=spec["composition"],
                split=spec["split"],audio_sha256=sha,metrics=all_scores)

def aggregate(rows,name):
    parts=("correct","total","active_correct","active_total",
           "transition_correct","transitions")
    a={field:sum(row["metrics"][name][field] for row in rows) for field in parts}
    return dict(**a,
      accuracy=a["correct"]/max(1,a["total"]),
      active_accuracy=a["active_correct"]/max(1,a["active_total"]),
      transition_accuracy=a["transition_correct"]/max(1,a["transitions"]))

def choose(rows):
    policies=[f"{name}_delay_{delta}" for name in ("native_MIDI","frozen_head")
              for delta in DELAYS]
    totals={p:aggregate(rows,p) for p in policies}
    native=totals["native_MIDI_delay_0"]
    preferred=[p for p in policies if
         totals[p]["accuracy"]>native["accuracy"] and
         totals[p]["active_accuracy"]>=native["active_accuracy"] and
         totals[p]["transition_accuracy"]>=native["transition_accuracy"]]
    preferred.sort(key=lambda x:(totals[x]["active_accuracy"],totals[x]["accuracy"]),
                   reverse=True)
    return preferred[0] if preferred else "native_MIDI_delay_0",totals

def run(out):
    if out.exists():raise FileExistsError("S700 immutable evidence already exists")
    t0=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    groups=sorted({s["composition"] for s in train})
    assert len(train)==36 and len(dev)==12 and len(groups)==9
    CAL=set(groups[:5]);HELD=set(groups[5:])
    native=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**native.metadata["selection"])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/"data").mkdir()
        for key,n in (("GuitarSet_audio","GuitarSet_audio.zip"),
                      ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/n,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            allrows=[]
            for i,spec in enumerate(train+dev,1):
                allrows.append(record(spec,root,archive,native,selection))
                print("S700 AUDIO",i,"/48",spec["name"],flush=True)
    cal=[x for x in allrows if x["split"]=="train" and x["composition"] in CAL]
    held=[x for x in allrows if x["split"]=="train" and x["composition"] in HELD]
    dv=[x for x in allrows if x["split"]=="dev"]
    selected,train_all=choose(cal)
    held_native=aggregate(held,"native_MIDI_delay_0")
    held_selected=aggregate(held,selected)
    dev_native=aggregate(dv,"native_MIDI_delay_0")
    dev_selected=aggregate(dv,selected)
    dev_all={name:aggregate(dv,name)
             for name in allrows[0]["metrics"]}
    oracle_keys=[name for name in dev_all if name.startswith("NONCAUSAL_ORACLE_")]
    best_future=max(oracle_keys,key=lambda n:dev_all[n]["active_accuracy"])
    report=dict(schema="S700_K_TEMPORAL_CAUSAL_SHIFT_AND_ORACLE_UPPER_BOUND",
        K_head_original_frozen=True,native_NoteOn_pitch_NoteOff_unchanged=True,
        S675_unchanged=True,original_LIVE_unchanged=True,
        tested_causal_delays_hops=list(DELAYS),
        noncausal_future_keys_are_DIAGNOSTIC_ONLY=True,
        no_future_audio_in_recommended_policy=True,
        TEST_unopened=True,DEV_previously_exposed=True,
        TRAIN_frozen_choice=selected,
        CAL_candidate_scores=train_all,
        TRAIN_held_native=held_native,TRAIN_held_selected=held_selected,
        DEV_native=dev_native,DEV_selected=dev_selected,
        DEV_all=dev_all,NONCAUSAL_oracle_best=best_future,
        NONCAUSAL_oracle_score=dev_all[best_future],
        by_recording=allrows,promoted=False,
        elapsed_seconds=round(time.perf_counter()-t0,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S700 FINAL",json.dumps({
      "frozen_choice":selected,
      "DEV_native_K":dev_native,
      "DEV_selected_CAU":dev_selected,
      "NONCAUSAL_DIAGNOSTIC_BEST":best_future,
      "NONCAUSAL_ONLY":dev_all[best_future],
      "ready_for_live":False}),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--out",required=True,type=Path)
    args=a.parse_args()
    with threadpool_limits(limits=1):run(args.out)
