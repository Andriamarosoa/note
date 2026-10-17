"""S670 -- actual-PCM, past-only offset hazard classifier on frozen NOTE S630 heads.

RESEARCH ONLY: original NoteOn/K/pitch streams, original checkpoints and
production Live untouched. Trains a new per-active-note hazard head on
GuitarSet TRAIN (five composition families); holds out four TRAIN
families for calibration; re-trains on nine TRAIN families only after
policies are frozen; assesses DEV after the freeze. Test never opened.

All features are from <= current source sample: acoustic spectral track,
current/prior causal NOTE head posteriors, current/native K and elapsed
time since the *predicted* NoteOn. No ground truth in inference.
S630 batch4 output time conservatively rounds hazard decisions UP to the
available callback batch boundary; no event is backdated.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import tempfile,time,zipfile
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,real_notes,event_score
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scipy.io import wavfile
from scipy.signal import resample_poly
import io
GUITARSET_PINNED_CONTROL=dict(TP=998,FP=585,FN=582)

def raw_guitar_audio(spec,archive):
    """Decode S613 pinned GuitarSet MIC WAV without any source future."""
    blob=archive.read(spec["audio"])
    rate,pcm=wavfile.read(io.BytesIO(blob))
    if pcm.dtype not in (np.dtype("int16"),np.dtype("int32")):
        raise ValueError("Unexpected native guitar PCM")
    samples=pcm.astype(np.float32)/(32768 if pcm.dtype==np.int16 else 2147483648)
    if samples.ndim==2:samples=samples.mean(axis=1)
    if rate!=SR:
        common=np.gcd(rate,SR)
        samples=resample_poly(samples,SR//common,rate//common).astype(np.float32)
    samples=samples[:int(spec["seconds"]*SR)].copy()
    if len(samples)<16384 or not np.isfinite(samples).all():
        raise ValueError("Invalid GuitarSet PCM")
    return samples,digest(blob)

# Precommitted variants, never retuned from DEV.
GATES=(.30,.50,.70,.85)
MIN_AGE_FRAMES=6
MAX_AGE_FRAMES=720
SELECTED_HORIZON=4
PROTO_OFFSET_TOL_MS=50.
TRAIN_CAP_PER_RECORD=48000


def native_posteriors(features,heads):
    frames=[heads.predict(features[i:i+128]) for i in range(0,len(features),128)]
    return (np.concatenate([x for x,_ in frames]),
            np.concatenate([k for _,k in frames]))


def note_instances(events,ends):
    """Native detected NoteOns are the ONLY onset candidates in research.
    Native NoteOff is retained as a baseline candidate, never a training label.
    """
    notes=[]
    active={}
    for e in events:
        pitch=int(e["pitch"])
        j=int(np.searchsorted(ends,int(e["decision_sample"]),side="left"))
        if e["type"]=="note_on":
            if pitch in active:
                notes[active.pop(pitch)]["native_end"]=j
            active[pitch]=len(notes)
            notes.append(dict(pitch=pitch,start=j,native_end=None))
        elif pitch in active:
            n=notes[active.pop(pitch)]
            if e["reason"] not in ("stream_end","shutdown","stream_error"):
                n["native_end"]=j
    last_by_pitch={}
    for i in range(len(notes)-1,-1,-1):
        n=notes[i]
        n["next_start"]=last_by_pitch.get(n["pitch"],len(ends))
        last_by_pitch[n["pitch"]]=n["start"]
    return notes


def frame_matrix(feat,post,k,ends,pitch,start,frames,clock=None):
    """Per-frame 48+ correlated acoustic/hazard features. All lags
    resolve from frames <= current, and source NOTE onset index <= frames.
    """
    frames=np.asarray(frames,dtype=np.int64)
    j=int(pitch)-40
    f=feat[frames,j,:]
    # Fixed offsets refer to spectra/competition of original native ABI.
    spectrum=f[:,:54]
    relatives=f[:,54:84]
    physical=f[:,-31:-25]
    recent=f[:,-24:-10]
    common=f[:,-6:-1]
    t=post[frames,j,:]
    prev8=post[np.maximum(frames-8,0),j,:]
    prev24=post[np.maximum(frames-24,0),j,:]
    prev64=post[np.maximum(frames-64,0),j,:]
    bass=post[frames,max(0,j-12),1:2]
    high=post[frames,min(48,j+12),1:2]
    relative_energy=spectrum[:,[0,1,2,8,9,16,17,18,24,25]]
    # Take 54 acoustic dims, including fundamental/upper partials; model
    # remains small enough for post-block hazard scoring.
    X=np.column_stack((
        relative_energy,
        relatives[:,[0,1,8,9,16,17]],
        physical,
        recent[:,[0,1,2,3,4,5,6,7]],
        common,
        t,prev8,prev24,prev64,
        t-prev8,t-prev24,
        bass,high,
        k[frames,:],
        np.full(len(frames),(j+40)/127.,np.float32),
        np.log1p(np.maximum(0,frames-start)*HOP*1000/SR),
        np.minimum(1.0,(frames-start)*HOP/(SR*2.0)),
    )).astype(np.float32)
    if clock is not None:
        period=np.maximum(4.,np.expm1(clock[frames,1]))
        age=np.maximum(0,frames-start)
        ratio=age/period
        X=np.column_stack((X,clock[frames],
            np.minimum(12.,ratio),
            np.cos(2*np.pi*ratio),
            np.cos(4*np.pi*ratio),
            np.cos(8*np.pi*ratio),
            np.minimum(1.,np.abs(ratio-np.round(ratio))),
            np.minimum(1.,np.abs(2*ratio-np.round(2*ratio))),
        )).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("S670 nonfinite causal hazard features")
    return X



def native_rhythm_clock(events,ends):
    """Compute inter-onset rhythm evidence available at or before each hop.
    No future/onset label: events are original native NOTE emitted NoteOns.
    """
    starts=np.asarray(sorted(set(
        int(np.searchsorted(ends,int(event["decision_sample"]),side="left"))
        for event in events if event["type"]=="note_on")),dtype=np.int64)
    n=len(ends)
    out=np.zeros((n,4),np.float32)
    if not len(starts):
        return out
    prev=np.searchsorted(starts,np.arange(n),side="right")-1
    ok=prev>=0
    cursor=np.maximum(prev,0)
    tempo=np.zeros(len(starts),np.float32)
    for i in range(len(starts)):
        recent=starts[max(0,i-8):i+1]
        gap=np.diff(recent)
        tempo[i]=float(np.median(gap)) if len(gap) else 50.
    last=starts[cursor]
    since=np.maximum(0,np.arange(n)-last)
    localtempo=np.maximum(4,tempo[cursor])
    out[:,0]=np.where(ok,np.log1p(since),0)
    out[:,1]=np.where(ok,np.log1p(localtempo),0)
    out[:,2]=np.where(ok,np.minimum(6.0,since/localtempo),0)
    out[:,3]=np.where(ok,np.cos(2*np.pi*since/localtempo),0)
    return out

def prepare(spec,root,archive,heads,selection):
    audio,hash_audio=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,feats=feature_track(audio,block_size=4096)
    post,k=native_posteriors(feats,heads)
    events=decode(ends,post,k,selection,len(audio))
    native_on,pairs=event_score(events,truth)
    instances=note_instances(events,ends)
    if len(instances)!=native_on["TP"]+native_on["FP"]:
        raise AssertionError("Native NoteOn event count differs")
    clock=native_rhythm_clock(events,ends)
    return dict(spec=spec,truth=truth,ends=ends,feats=feats,post=post,clock=clock,
                k=k,events=events,onset=native_on,pairs=pairs,
                notes=instances,audio_sha256=hash_audio)


def training_data(item):
    matched={a:b for a,b in item["pairs"]}
    Xs,ys=[],[]
    ends=item["ends"];notes=item["notes"]
    for i,n in enumerate(notes):
        if i not in matched:continue
        actual=item["truth"][matched[i]]
        stop=float(actual["time"]+actual["duration"])
        if stop>=ends[-1]/SR-0.010:continue
        stopframe=int(np.searchsorted(ends,round(stop*SR),"left"))
        start=n["start"]
        limit=min(n["next_start"],len(ends),start+MAX_AGE_FRAMES,
                  stopframe+11)
        if limit<=start+MIN_AGE_FRAMES or stopframe>=limit:continue
        # Balanced subsampling of sustained frames + exhaustive ±60ms end.
        early=np.arange(start+MIN_AGE_FRAMES,max(start+MIN_AGE_FRAMES,stopframe-14),7)
        close=np.arange(max(start+MIN_AGE_FRAMES,stopframe-14),limit)
        positions=np.unique(np.concatenate((early,close)))
        if not len(positions):continue
        y=(positions>=stopframe).astype(np.uint8)
        if not np.any(y):continue
        Xs.append(frame_matrix(item["feats"],item["post"],item["k"],
                               ends,n["pitch"],start,positions,item["clock"]))
        ys.append(y)
    if not Xs:
        raise ValueError("No positive GuitarSet offset hazard samples")
    X=np.vstack(Xs);y=np.concatenate(ys)
    if len(y)>TRAIN_CAP_PER_RECORD:
        rng=np.random.default_rng(668)
        pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
        chosen=np.r_[pos,rng.choice(neg,size=min(len(neg),max(0,TRAIN_CAP_PER_RECORD-len(pos))),replace=False)]
        X=X[chosen];y=y[chosen]
    return X,y


def train(examples):
    X=np.vstack([pair[0] for pair in examples])
    y=np.concatenate([pair[1] for pair in examples])
    if y.sum()<500 or (1-y).sum()<1000:raise AssertionError("Undertrained hazard")
    model=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=.075,max_iter=130,
        max_leaf_nodes=17,min_samples_leaf=70,l2_regularization=15,
        early_stopping=False,random_state=668)
    model.fit(X,y,sample_weight=np.where(y>0,2.0,1.0))
    return model,dict(rows=len(y),positive=int(y.sum()),negative=int((1-y).sum()),
                      feature_dims=int(X.shape[1]))


def hazard_predictions(item,model):
    """Predict once on all active-note trajectories; all rows are causal."""
    xs=[];record_index=[];candidate_frames=[]
    ends=item["ends"]
    for i,n in enumerate(item["notes"]):
        lo=n["start"]+MIN_AGE_FRAMES
        hi=min(n["next_start"],len(ends),n["start"]+MAX_AGE_FRAMES)
        if hi<=lo:
            candidate_frames.append(np.empty(0,np.int64));continue
        frames=np.arange(lo,hi,dtype=np.int64)
        candidate_frames.append(frames)
        xs.append(frame_matrix(item["feats"],item["post"],item["k"],ends,
                               n["pitch"],n["start"],frames,item["clock"]))
        record_index.append((i,len(frames)))
    output=[np.empty(0,np.float32) for _ in item["notes"]]
    if xs:
        probs=model.predict_proba(np.vstack(xs))[:,1]
        at=0
        for i,num in record_index:
            output[i]=probs[at:at+num].copy()
            at+=num
    return candidate_frames,output


def choose_offset(item,index,frames,prob,gate,mode,batch_hops=1):
    n=item["notes"][index]
    original=n["native_end"]
    if original is None:
        original=len(item["ends"]) # censored baseline, never success
    hazard=None
    if len(frames):
        agrees=np.convolve((prob>=gate).astype(int),np.ones(2,dtype=int),mode="full")[:len(frames)]
        hits=np.flatnonzero(agrees>=2)
        if len(hits):hazard=int(frames[hits[0]])
    if mode=="hazard" and hazard is not None:
        decided=hazard
    elif mode=="earlier" and hazard is not None:
        decided=min(hazard,original)
    elif mode=="later" and hazard is not None:
        decided=max(hazard,original)
    elif mode=="near" and hazard is not None and abs(hazard-original)<=20:
        decided=hazard
    else:
        decided=original
    decided=min(decided,n["next_start"])
    # No 3D model uses unobserved future audio; the batch may delay physical
    # emissions to the last captured hop. Preserve the causally emitted time.
    emission=min(len(item["ends"]),((decided+1+batch_hops-1)//batch_hops)*batch_hops)
    return float(emission*HOP/SR)


def score(item,proposals,gate,mode,batch_hops=1):
    frames,probabilities=proposals
    positions=[choose_offset(item,i,frames[i],probabilities[i],gate,mode,
                             batch_hops=batch_hops)
               for i in range(len(item["notes"]))]
    matched={a:b for a,b in item["pairs"]}
    eligible=correct=forced=long_notes=long_ok=0
    for i,j in matched.items():
        actual=item["truth"][j]
        off=actual["time"]+actual["duration"]
        if off>=item["ends"][-1]/SR:continue
        eligible+=1
        pred=positions[i]
        # Native forced EOF does not count, nor an artificially selected
        # terminal timestamp without a genuine inferred release.
        if pred>=item["ends"][-1]/SR-1e-7:
            forced+=1;continue
        good=abs(pred-off)<=.05
        correct+=int(good)
        if actual["duration"]>=1:
            long_notes+=1;long_ok+=int(good)
    return dict(correct=int(correct),eligible=int(eligible),
                accuracy=float(correct/max(1,eligible)),
                forced=int(forced),long_eligible=int(long_notes),
                long_correct=int(long_ok))


def run(out):
    if out.exists():raise FileExistsError("S670 immutable report exists")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train_specs=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev_specs=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(train_specs)==36 and len(dev_specs)==12
    comps=sorted({s["composition"] for s in train_specs})
    assert len(comps)==9
    cal_comps=set(comps[:5])
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**heads.metadata["selection"])
    ts=time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/path,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            # Keep compact supervised rows, not 36 full 126D track matrices:
            # each 24s wav yields ~100MB of features. All raw tracks are
            # released immediately, avoiding OOM and 3GB resident growth.
            full_data=[];cal=[]
            held_specs=[]
            for i,s in enumerate(train_specs,1):
                item=prepare(s,root,archive,heads,selection)
                examples=training_data(item)
                full_data.append(examples)
                if s["composition"] in cal_comps:
                    cal.append(examples)
                else:
                    held_specs.append(s)
                del item
                print("S670 PREPARE TRAIN",i,"/36",s["name"],flush=True)
            hazard,training=train(cal)
            print("S670 TRAIN CAL",json.dumps(training),flush=True)
            configs=[("native",0.,"native")]
            configs.extend((f"{mode}_{gate:.2f}",gate,mode)
                           for mode in ("hazard","earlier","later","near")
                           for gate in GATES)
            hold_scores=defaultdict(lambda:Counter())
            by_comp=defaultdict(lambda:defaultdict(lambda:Counter()))
            for i,s in enumerate(held_specs,1):
                item=prepare(s,root,archive,heads,selection)
                pro=hazard_predictions(item,hazard)
                comp=item["spec"]["composition"]
                for name,gate,mode in configs:
                    metrics=score(item,pro,gate,mode)
                    hold_scores[name].update({"correct":metrics["correct"],"eligible":metrics["eligible"]})
                    by_comp[comp][name].update({"correct":metrics["correct"],"eligible":metrics["eligible"]})
                print("S670 HELD",i,"/",len(held_specs),s["name"],flush=True)
                del item,pro
            baseline=hold_scores["native"]
            good=[]
            for name,gate,mode in configs[1:]:
                val=hold_scores[name]
                comp_improve=sum(by_comp[c][name]["correct"]>by_comp[c]["native"]["correct"] for c in by_comp)
                if val["correct"]>=baseline["correct"]+10 and comp_improve>=2:
                    good.append((name,gate,mode,val["correct"],comp_improve))
            choice=max(good,key=lambda t:t[3]) if good else ("native",0.,"native",baseline["correct"],0)
            print("S670 TRAIN-FROZEN CHOICE",json.dumps(choice),flush=True)
            del hazard
            model,full_proof=train(full_data)
            del full_data,cal
            dev=[]
            for i,s in enumerate(dev_specs,1):
                item=prepare(s,root,archive,heads,selection)
                pro=hazard_predictions(item,model)
                row=dict(name=s["name"],composition=s["composition"],
                         audio_sha256=item["audio_sha256"],
                         native_onset=item["onset"],
                         policies={name:score(item,pro,gate,mode,batch_hops=1)
                                   for name,gate,mode in configs},
                         batch4={name:score(item,pro,gate,mode,batch_hops=4)
                                 for name,gate,mode in configs})
                dev.append(row)
                print("S670 DEV",i,"/12",s["name"],flush=True)
                del item,pro
    totals={}
    for label in ("policies","batch4"):
        totals[label]={}
        for name,gate,mode in configs:
            c=sum(row[label][name]["correct"] for row in dev)
            e=sum(row[label][name]["eligible"] for row in dev)
            totals[label][name]=dict(correct=c,eligible=e,
                                     accuracy=c/max(1,e))
            print("S670 SCORE",label,name,c,e,
                  round(c/max(1,e),6),flush=True)
    dev_onset=sum(row["native_onset"]["TP"] for row in dev)
    assert dev_onset==GUITARSET_PINNED_CONTROL["TP"]
    report=dict(
       status="S670_CAUSAL_NOTE_OFF_HAZARD_RESEARCH_ONLY",
       native_NoteOn_and_Exact_K_frozen=True,
       original_LIVE_and_ONNX_files_unchanged=True,
       source_features_past_PCM_only=True,
       past_detected_NoteOn_rhythm_only=True,
       no_future_onset_information=True,
       no_future_audio_or_backdated_noteoffs=True,
       emits_first_threshold_in_real_batch_available_sample=True,
       native_194K_baseline_evaluated_same_notes=True,
       train_calibration_compositions=sorted(cal_comps),
       train_held_compositions=sorted(set(comps)-cal_comps),
       held_scores={k:dict(v) for k,v in hold_scores.items()},
       held_by_composition={comp:{name:dict(counter) for name,counter in rows.items()}
                            for comp,rows in by_comp.items()},
       TRAIN_selection=choice[0],
       TRAIN_selection_policy=dict(gate=choice[1],mode=choice[2]),
       TRAIN_proof=full_proof,
       no_DEV_choice=True,
       TEST_labels_never_opened=True,
       DEV_previously_exposed=True,
       no_promotion=True,
       benchmark_strict_offset50=totals,
       dev_records=dev,
       target_80_reached_by_train_selected_policy=bool(
           totals["batch4"][choice[0]]["accuracy"]>=.8),
       elapsed_seconds=round(time.perf_counter()-ts,2),
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S670 FINAL",json.dumps(dict(
        selected_on_train=choice[0],
        native_batch4=totals["batch4"]["native"],
        selected_batch4=totals["batch4"][choice[0]],
        best_exploratory=max(totals["batch4"].items(),
                             key=lambda z:z[1]["accuracy"]))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out)
