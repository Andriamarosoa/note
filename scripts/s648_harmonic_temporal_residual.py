"""S648 — harmonic-residual + attack-novelty research loops on NOTE GuitarSet.

Research hypotheses evaluated on frozen NOTE:
  A. S641 proven 41D native + transposed evidence control;
  B. add original spectral competitive f0/octave evidence (S641 65D);
  C. original f0/harmonic/neighbor *temporal novelty* before emission;
  D. combine harmonic competition and temporal novelty;
  E. a different nonlinear small tree with the same temporal facts.

CRUCIAL: Every candidate is a real event emitted by the ungated Bass
Rescue causal decoder. Supervision is ONLY whether that event correctly
rescues an annotation missed by frozen NOTE. No artificial chord labels
are used for model training. Nine TRAIN compositions control the model
and abstention threshold through GroupKFold; original twelve GuitarSet
DEV are scored only AFTER all training choices are frozen.
This workflow DOES NOT use guitarset/IDMT/URMP TEST sources.
"""
from __future__ import annotations

import argparse
from collections import Counter,deque
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR,Selection,CausalSelection,load_heads
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,real_notes,summary
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,GUITARSET_PINNED_CONTROL
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s640_learned_overlap_rescue import (
    iter_acoustic,get_candidates,relay,pcm_truth,FEATURE_NAMES)
from scripts.s641_fundamental_consistency import (
    original_fundamental_evidence,ADDITIONAL_NAMES)

DELAYS=(4,12)  # previous native NOTE frames: 23.22ms and 69.66ms, no future
THRESHOLDS=(.70,.80,.85,.90,.92,.95,.97,.99,1.01)
SCALE=2.0 # one true correction offsets up to two FP in OOF selection
PATTERNS=("control41","competitive65","temporal41","temporal65","tree_temporal65")
TEMPORAL_COLUMNS=tuple(
    f"{window}_{source}_{lag}"
    for window in ("short","medium","long")
    for source in ("fund","h2","h3","h4","octave","neighbor")
    for lag in ("prev4","prev12")
) + tuple(
    f"{window}_{feature}"
    for window in ("short","medium","long")
    for feature in ("fund_minus_oct_delta4","fund_minus_oct_delta12",
                    "fund_minus_h2_delta4","fund_minus_neighbor_delta4",
                    "harmonic_odd_even_change","fund_minus_h4_delta12")
) + ("pitch_onset_vs_octave", "pitch_sustain_vs_octave",
     "max_other_attack", "max_other_sustain",
     "fundamental_novelty_mean", "fundamental_novelty_min")
WIDTH=len(TEMPORAL_COLUMNS)


def novelty_features(x,old4,old12,native,i):
    """Physical right-edge harmonic trajectories, independent low-f0 evidence.

    Every input is the original native acoustic feature frame at the
    decision hop or at 4/12 earlier past hops; labels aren't read.
    Older notes' harmonics can explain candidate harmonics; compare
    fractional changes, not only a prominent single bin.
    """
    if x.shape!=(49,126) or old4.shape!=x.shape or old12.shape!=x.shape:
        raise ValueError("S648 126D NOTE acoustic feature ABI changed")
    if not 0<=i<24:raise ValueError("Only 40..63 bass candidates")
    ix=i+12
    vals=[]
    novelty=[]
    for j in range(3):
        off=8*j
        for src in ("fund","h2","h3","h4","octave","neighbor"):
            if src=="octave":
                t,old4v,old12v=(float(a[ix,off]) for a in (x,old4,old12))
            elif src=="neighbor":
                t,old4v,old12v=(
                    max(float(a[max(0,i-1),off]),float(a[min(48,i+1),off]))
                    for a in (x,old4,old12))
            else:
                h=dict(fund=0,h2=1,h3=2,h4=3)[src]
                t,old4v,old12v=(float(a[i,off+h]) for a in (x,old4,old12))
            vals.extend((t-old4v,t-old12v))
            if src=="fund":novelty.append(t-old4v)
        f4=float(x[i,off]-old4[i,off])
        f12=float(x[i,off]-old12[i,off])
        oct4=float(x[ix,off]-old4[ix,off])
        oct12=float(x[ix,off]-old12[ix,off])
        h2_4=float(x[i,off+1]-old4[i,off+1])
        neighbor4=max(
            float(x[max(0,i-1),off]-old4[max(0,i-1),off]),
            float(x[min(48,i+1),off]-old4[min(48,i+1),off]))
        h4_12=float(x[i,off+3]-old12[i,off+3])
        # For a single true low pitch, odd+even harmonic peaks can rise
        # together; an additional high octave has different ratios.
        old_odd_even=float(old4[i,off]+old4[i,off+2]-old4[i,off+1]-old4[i,off+3])
        new_odd_even=float(x[i,off]+x[i,off+2]-x[i,off+1]-x[i,off+3])
        vals.extend((f4-oct4,f12-oct12,f4-h2_4,
                     f4-neighbor4,new_odd_even-old_odd_even,f12-h4_12))
    on=native[:,0]
    sust=native[:,1]
    other=np.delete(on,i)
    vals.extend((float(on[i]-on[ix]),float(sust[i]-sust[ix]),
                 float(other.max()),float(np.delete(sust,i).max()),
                 float(np.mean(novelty)),float(np.min(novelty))))
    a=np.asarray(vals,np.float32)
    if a.shape!=(WIDTH,) or not np.isfinite(a).all():
        raise ValueError(f"S648 temporal feature corruption {a.shape} != {WIDTH}")
    return a


def build_model(name):
    if name=="tree_temporal65":
        return HistGradientBoostingClassifier(
            max_iter=95,learning_rate=.06,max_leaf_nodes=7,max_depth=3,
            min_samples_leaf=22,l2_regularization=8.,
            class_weight="balanced",random_state=648)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.35 if name=="control41" else .2,
                           class_weight="balanced",max_iter=1350,
                           random_state=648))


def matrices(feature,physical,novelty):
    return dict(
        control41=feature,
        competitive65=np.concatenate((feature,physical)),
        temporal41=np.concatenate((feature,novelty)),
        temporal65=np.concatenate((feature,physical,novelty)),
        tree_temporal65=np.concatenate((feature,physical,novelty)))


def historical(history,x):
    """Use ONLY earlier features, never same/future for the lag."""
    oldest=history[0] if history else np.zeros_like(x)
    def at(lag):
        return history[-lag] if len(history)>=lag else oldest
    return at(4),at(12)


def train_record(spec,root,archive,heads,selection):
    sound,truth,sha=pcm_truth(spec,root,archive)
    native_state=CausalSelection(selection)
    rescued_state=CausalSelection(selection)
    events0=[];eventsR=[]
    candidate_vectors=[]
    history=deque(maxlen=16)
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        old4,old12=historical(history,x)
        candidates=get_candidates(x,native,k,bass,valid,selection.onset)
        events0.extend(native_state.step(native,k,int(end),int(end)))
        altered,_=relay(native,bass,rescued_state,
                        {i:1. for i in candidates},0.)
        emitted=rescued_state.step(altered,k,int(end),int(end))
        for e in emitted:
            i=int(e["pitch"])-40
            if e["type"]!="note_on" or i not in candidates or native[i,0]>=selection.onset:
                continue
            f=candidates[i]
            phys=original_fundamental_evidence(x,i)
            dyn=novelty_features(x,old4,old12,native,i)
            candidate_vectors.append((int(end),int(e["pitch"]),
                                      matrices(f,phys,dyn)))
        eventsR.extend(emitted)
        history.append(x.copy())
    events0.extend(native_state.panic(len(sound)))
    eventsR.extend(rescued_state.panic(len(sound)))
    _,m0=event_score(events0,truth)
    _,mR=event_score(eventsR,truth)
    native_gt={j for _,j in m0}
    rescue_matched={i:j for i,j in mR}
    r_ons=[e for e in eventsR if e["type"]=="note_on"]
    index={(int(e["decision_sample"]),int(e["pitch"])):i for i,e in enumerate(r_ons)}
    Y=[]
    arrays={n:[] for n in PATTERNS}
    for sample,pitch,feat in candidate_vectors:
        truth_index=rescue_matched.get(index[(sample,pitch)])
        Y.append(int(truth_index is not None and truth_index not in native_gt))
        for name in PATTERNS:
            arrays[name].append(feat[name])
    return dict(name=spec["name"],composition=spec["composition"],
                sha256=sha,
                matrices={n:np.asarray(arrays[n],np.float32).reshape(-1,len(matrices(
                    np.zeros(len(FEATURE_NAMES)),np.zeros(len(ADDITIONAL_NAMES)),np.zeros(WIDTH))[n]))
                          for n in PATTERNS},
                y=np.asarray(Y,np.int8))


def select_on_train(records):
    fit={};cv={}
    ys=np.concatenate([r["y"] for r in records])
    groups=np.concatenate([np.repeat(r["composition"],len(r["y"])) for r in records])
    if len(ys)<400 or ys.sum()<25 or len(set(groups))!=9:
        raise ValueError(f"Invalid TRAIN candidate pool {len(ys)} positives {sum(ys)}")
    splits=list(GroupKFold(n_splits=5).split(np.zeros((len(ys),1)),ys,groups))
    for tr,va in splits:
        assert not set(groups[tr])&set(groups[va])
        if np.unique(ys[tr]).size!=2:raise ValueError("Fold without TRAIN positives")
    for name in PATTERNS:
        x=np.concatenate([r["matrices"][name] for r in records])
        oof=np.zeros(len(ys),np.float64)
        for tr,va in splits:
            model=build_model(name).fit(x[tr],ys[tr])
            oof[va]=model.predict_proba(x[va])[:,1]
        trials=[]
        for threshold in THRESHOLDS:
            yes=oof>=threshold
            tp=int(ys[yes].sum())
            fp=int(yes.sum())-tp
            trials.append(dict(threshold=threshold,TP=tp,FP=fp,
                               utility=tp-fp/SCALE))
        best=max(trials,key=lambda r:(r["utility"],-r["FP"],r["threshold"]))
        fit[name]=dict(model=build_model(name).fit(x,ys),threshold=best["threshold"])
        cv[name]=dict(TRAIN_candidates=len(ys),TRAIN_positives=int(ys.sum()),
                      selected=best,OOF=trials)
        print("S648 CV TRAIN",name,len(ys),int(ys.sum()),best,flush=True)
    winner=max(PATTERNS,key=lambda n:(
        cv[n]["selected"]["utility"],-cv[n]["selected"]["FP"],n))
    return fit,cv,winner


def validate(spec,root,archive,heads,config,gates):
    sound,truth,sha=pcm_truth(spec,root,archive)
    names=("S614","bass_raw",*PATTERNS)
    states={n:CausalSelection(config) for n in names}
    events={n:[] for n in names}
    counts=Counter()
    history=deque(maxlen=16)
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        old4,old12=historical(history,x)
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        predictions={n:{} for n in PATTERNS}
        if candidates:
            packed={n:[] for n in PATTERNS}
            for i,feature in candidates.items():
                phys=original_fundamental_evidence(x,i)
                dyn=novelty_features(x,old4,old12,native,i)
                features=matrices(feature,phys,dyn)
                for n in PATTERNS:packed[n].append(features[n])
            for n in PATTERNS:
                probs=gates[n]["model"].predict_proba(np.stack(packed[n]))[:,1]
                predictions[n]=dict(zip(candidates,probs))
        events["S614"].extend(states["S614"].step(native,k,int(end),int(end)))
        raw,amount=relay(native,bass,states["bass_raw"],
                         {i:1. for i in candidates},0.)
        counts["bass_raw"]+=amount
        events["bass_raw"].extend(states["bass_raw"].step(raw,k,int(end),int(end)))
        for n in PATTERNS:
            p,amount=relay(native,bass,states[n],predictions[n],
                            gates[n]["threshold"])
            counts[n]+=amount
            events[n].extend(states[n].step(p,k,int(end),int(end)))
        history.append(x.copy())
    truth_group=overlap_groups(truth)
    out={}
    baseline_truth=None
    for n in names:
        events[n].extend(states[n].panic(len(sound)))
        score,pairs=event_score(events[n],truth)
        matched={j for _,j in pairs}
        if n=="S614":baseline_truth=matched
        out[n]=dict(score=score,
                    poly_TP=len(matched&truth_group["onset_overlap"]),
                    K3plus_TP=len(matched&truth_group["onset_K3plus"]),
                    K2_TP=len(matched&truth_group["onset_K2"]),
                    matched=matched,admissions=int(counts[n]))
    for n in names:
        matches=out[n].pop("matched")
        out[n]["new_true"]=len(matches-baseline_truth)
        out[n]["lost_true"]=len(baseline_truth-matches)
    return dict(name=spec["name"],composition=spec["composition"],
                SHA256=sha,results=out)


def run(out):
    if out.exists():raise FileExistsError("S648 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in protocol["sources"]
           if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert len({s["composition"] for s in train})==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/filename,z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            records=[]
            for i,s in enumerate(train,1):
                r=train_record(s,root,archive,heads,config)
                records.append(r)
                print("S648 TRAIN",i,"/36",r["name"],
                      "candidates",len(r["y"]),"+",int(r["y"].sum()),flush=True)
            gates,cv,winner=select_on_train(records)
            print("S648 FROZEN CHOICE",winner,flush=True)
            devrows=[]
            for i,s in enumerate(dev,1):
                r=validate(s,root,archive,heads,config,gates)
                devrows.append(r)
                print("S648 DEV",i,"/12",r["name"],
                      "controlFP",r["results"]["S614"]["score"]["FP"],
                      "winnerFP",r["results"][winner]["score"]["FP"],flush=True)
    results={}
    control=None
    for n in devrows[0]["results"]:
        z=[r["results"][n] for r in devrows]
        score=summary([row["score"] for row in z])
        results[n]=dict(score=score,
                        overlap_TP=sum(row["poly_TP"] for row in z),
                        K2_TP=sum(row["K2_TP"] for row in z),
                        K3plus_TP=sum(row["K3plus_TP"] for row in z),
                        new_true=sum(row["new_true"] for row in z),
                        lost_true=sum(row["lost_true"] for row in z),
                        admitted=sum(row["admissions"] for row in z))
        if n=="S614":control=score
    assert all(control[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),control
    assert results["bass_raw"]["score"]["TP"]==1044
    assert results["bass_raw"]["score"]["FP"]==834
    assert results["control41"]["score"]["TP"]==1009,results["control41"]["score"]
    assert results["control41"]["score"]["FP"]==594,results["control41"]["score"]
    for n,v in results.items():
        s=v["score"]
        v["delta_TP"]=s["TP"]-control["TP"]
        v["delta_FP"]=s["FP"]-control["FP"]
        v["delta_F1_points"]=round(100*(s["F1"]-control["F1"]),7)
        print("S648 SCORE",n,"TP",s["TP"],"FP",s["FP"],
              "F1",s["F1"],"overlap",v["overlap_TP"],
              "K3+",v["K3plus_TP"],"deltaF1pt",v["delta_F1_points"],flush=True)
    report=dict(status="S648_HARMONIC_TEMPORAL_RESIDUAL_LOOPS_ON_REAL_GUITARSET",
                native_NOTE_checkpoint_SHA=frozen["model_sha256"],
                unchanged_S614_AND_Exact_K=True,
                all_features_past_only=True,
                added_future_lookahead_ms=0,
                experimental_source="36 TRAIN 12 DEV pinned real GuitarSet",
                TEST_not_read=True,DEV_previously_exposed=True,
                TRAIN_only_OOF_composition_group_cv=True,
                model_chosen_on_TRAIN=winner,
                full_train_sources=36,TRAIN_groups=9,
                temporal_feature_columns=list(TEMPORAL_COLUMNS),
                model_CV=cv,results=results,per_recording=devrows,
                elapsed_seconds=round(time.perf_counter()-start,2),
                promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S648 FINAL",json.dumps(dict(winner=winner,
          models={n:dict(TP=v["score"]["TP"],FP=v["score"]["FP"],
                         F1=v["score"]["F1"],K3=v["K3plus_TP"])
                  for n,v in results.items()})),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
