"""S696 strict causal K head-vs-MIDI K consistency and 80% feasibility audit.

Tests REAL GuitarSet MIC audio, frozen original 7-way K head:
(1) current MIDI-active count, (2) K argmax, (3) causal smoothed K,
(4) strictly past-voted last N K decisions, (5) optional K fusion with
the current number of MIDI active notes. Never edit native NOTE Onset,
pitch, offsets, frozen K head or S675 release head.

Calibration on first five TRAIN composition families, held-veto on four
separate TRAIN families; DEV explicitly retrospective. Reports all frames
AND genuinely active frames, avoiding a misleading 80% from silence.
No controls are installed into Live as a consequence of this audit.
"""
from __future__ import annotations
import argparse,json,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR,HOP,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,real_notes,event_score
from scripts.s672_causal_offset_timing_grid import ROOT,MODEL,raw_guitar_audio,native_posteriors
from scripts.s682_onset_four_lock_guard import native_metrics


POLICIES=("native_MIDI","K_argmax","K_ema_0.25","K_ema_0.50",
          "K_ema_0.75","K_vote_3","K_vote_7",
          "K_fuse_0.35","K_fuse_0.60")

def native_count_timeline(events,ends):
    active=set();diff=np.zeros(len(ends)+1,np.int32)
    # Native event times are required, not WAV-lookahead times.
    for e in events:
        f=int(np.searchsorted(ends,int(e["decision_sample"]),"left"))
        if f>=len(ends):continue
        pitch=e["pitch"]
        if e["type"]=="note_on" and pitch not in active:
            active.add(pitch);diff[f]+=1
        elif e["type"]=="note_off" and pitch in active:
            active.remove(pitch);diff[f]-=1
    return np.clip(np.cumsum(diff[:-1]),0,6)

def gt_count_timeline(truth,ends):
    d=np.zeros(len(ends)+1,np.int32)
    grid=np.asarray(ends,dtype=np.float64)/SR
    for note in truth:
        a=int(np.searchsorted(grid,float(note["time"]),"left"))
        b=int(np.searchsorted(grid,float(note["time"])+float(note["duration"]),"left"))
        if 0<=a<b and a<len(ends):
            d[a]+=1;d[min(len(ends),b)]-=1
    return np.clip(np.cumsum(d[:-1]),0,6)

def posterior_ema(k,alpha):
    scores=np.empty_like(k)
    last=k[0].copy()
    for i,frame in enumerate(k):
        last=alpha*frame+(1-alpha)*last
        scores[i]=last
    return scores

def causal_vote(k,window):
    choice=k.argmax(axis=1)
    result=np.zeros(len(k),np.int8)
    for t in range(len(k)):
        history=choice[max(0,t-window+1):t+1]
        result[t]=np.argmax(np.bincount(history,minlength=7))
    return result

def predictions(k,midi):
    k=np.asarray(k,np.float32)
    if k.ndim!=2 or k.shape[1]!=7 or not np.isfinite(k).all():
        raise ValueError("K head output invalid")
    if len(midi)!=len(k):raise ValueError("Native MIDI K timeline mismatch")
    result={"native_MIDI":midi,
            "K_argmax":k.argmax(axis=1)}
    for alpha in (.25,.50,.75):
        result[f"K_ema_{alpha:.2f}"]=posterior_ema(k,alpha).argmax(axis=1)
    for n in (3,7):
        result[f"K_vote_{n}"]=causal_vote(k,n)
    # Prior to all decisions, signal comes exclusively from current
    # cumulative MIDI state (generated from received frames only).
    onehot=np.eye(7,dtype=np.float32)[midi]
    for w in (.35,.60):
        result[f"K_fuse_{w:.2f}"]=(w*k+(1-w)*onehot).argmax(axis=1)
    return result

def score(p,t):
    assert len(p)==len(t)
    active=t>0
    nonzero=int(active.sum())
    changes=np.r_[True,t[1:]!=t[:-1]]
    return dict(correct=int((p==t).sum()),total=len(t),
                active_correct=int(((p==t)&active).sum()),
                active_total=nonzero,
                transition_correct=int(((p==t)&changes).sum()),
                transitions=int(changes.sum()),
                truth_K_hist=np.bincount(t,minlength=7).tolist(),
                pred_K_hist=np.bincount(p,minlength=7).tolist())

def aggregate(rows,policy):
    acc={x:sum(r["scores"][policy][x] for r in rows)
         for x in ("correct","total","active_correct","active_total",
                   "transition_correct","transitions")}
    acc.update(
        accuracy=acc["correct"]/max(1,acc["total"]),
        active_accuracy=acc["active_correct"]/max(1,acc["active_total"]),
        transition_accuracy=acc["transition_correct"]/max(1,acc["transitions"]))
    return acc

def record(spec,root,archive,heads,selector):
    waveform,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(waveform)/SR)
    ends,features=feature_track(waveform,block_size=4096)
    p,k=native_posteriors(features,heads)
    preserved=digest(k.tobytes()+p.tobytes())
    events=decode(ends,p,k,selector,len(waveform))
    midi=native_count_timeline(events,ends)
    true=gt_count_timeline(truth,ends)
    pred=predictions(k,midi)
    assert digest(k.tobytes()+p.tobytes())==preserved
    onset,_=event_score(events,truth,tolerance=.080)
    return dict(name=spec["name"],split=spec["split"],
                composition=spec["composition"],wave_sha256=sha,
                onset_source=dict(tp=onset["TP"],fp=onset["FP"],fn=onset["FN"]),
                scores={key:score(val,true) for key,val in pred.items()})

def choose(calrows):
    scores={p:aggregate(calrows,p) for p in POLICIES}
    baseline=scores["native_MIDI"]
    qualified=[]
    for p in POLICIES[1:]:
        v=scores[p]
        if (v["accuracy"]>baseline["accuracy"] and
            v["active_accuracy"]>=baseline["active_accuracy"] and
            v["transition_accuracy"]>=baseline["transition_accuracy"]):
            qualified.append(p)
    qualified.sort(key=lambda q:(scores[q]["active_accuracy"],
                                 scores[q]["accuracy"]),reverse=True)
    return qualified[0] if qualified else "native_MIDI",scores,qualified

def run(out):
    if out.exists():raise FileExistsError("Immutable evidence directory already exists")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted(set(r["composition"] for r in train))
    assert len(comps)==9 and len(train)==36 and len(dev)==12
    cal=set(comps[:5]);held=set(comps[5:])
    original=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    native_selection=Selection(**original.metadata["selection"])
    start=time.perf_counter()
    report=dict(schema="S696_EXACT_K_DIRECT_CAUSAL_FOUR_WAY_AUDIT",
        original_7class_k_head_frozen=True,
        native_noteon_pitch_noteoff_unchanged=True,
        original_S675_unchanged=True,original_live_unchanged=True,
        no_future_features=True,DEV_previously_exposed=True,
        TEST_unopened=True,train_calibration=sorted(cal),
        train_held=sorted(held),
        distinguishes_native_MIDI_active_count_from_direct_K_prediction=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            arc=proto["guitar_archives"][key]
            download(arc["url"],root/"data"/filename,arc["size"],arc["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zipaudio:
            collected=[]
            for i,spec in enumerate(train+dev,1):
                item=record(spec,root,zipaudio,original,native_selection)
                collected.append(item)
                print("S696 REAL",i,"/",len(train+dev),spec["name"],flush=True)
    calibration=[r for r in collected if r["split"]=="train" and r["composition"] in cal]
    heldout=[r for r in collected if r["split"]=="train" and r["composition"] in held]
    devrows=[r for r in collected if r["split"]=="dev"]
    winner,cal_metrics,qualified=choose(calibration)
    frozenheld={p:aggregate(heldout,p) for p in ("native_MIDI",winner)}
    dev_all={p:aggregate(devrows,p) for p in POLICIES}
    report["CAL_all"]=cal_metrics
    report["CAL_selected"]=winner
    report["CAL_passing"]=qualified
    report["HELD_native_vs_selected"]=frozenheld
    report["HELD_pass"]=(winner!="native_MIDI" and
                         frozenheld[winner]["accuracy"]>=frozenheld["native_MIDI"]["accuracy"] and
                         frozenheld[winner]["active_accuracy"]>=frozenheld["native_MIDI"]["active_accuracy"] and
                         frozenheld[winner]["transition_accuracy"]>=frozenheld["native_MIDI"]["transition_accuracy"])
    report["DEV_all_reported_for_diagnostics_only"]=dev_all
    report["DEV_selected"]=dev_all[winner]
    report["DEV_native"]=dev_all["native_MIDI"]
    report["by_recording"]=collected
    report["DEV_native_onset"]={
        key:sum(x["onset_source"][key] for x in devrows)
        for key in ("tp","fp","fn")}
    report["elapsed_s"]=round(time.perf_counter()-start,2)
    report["promoted"]=False
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S696 FINAL",json.dumps({
        "selected":winner,"held_pass":report["HELD_pass"],
        "DEV_native":dev_all["native_MIDI"],
        "DEV_selected":dev_all[winner],
        "DEV_direct_K":dev_all["K_argmax"],
        "DEV_onset":report["DEV_native_onset"],
        "not_a_new_note_detection_score":True,
        "original_live_unchanged":True}),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    with threadpool_limits(limits=1):run(args.out)
