"""S653: learned causal Tone candidate-repair heads and realistic NoteOn scoring.

Unlike S652 global averaging, note proposals originate from the native
and two causal Tone views. A new NOTE pitch is admitted ONLY if a
TRAIN composition-disjoint learned specialist predicts that the
filtered-onset proposal repairs a real native S614 missed note.

Five GroupKFold splits on 9 GuitarSet TRAIN compositions select model
and threshold, all *before* GuitarSet DEV. Whole-frame K from original
S614 is never overwritten. No test data, no retraining original NOTE,
no lookahead, no edits to microphone production.
"""
from __future__ import annotations
from collections import Counter
import argparse
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

from causal_note.strict_live import SR, Selection,CausalSelection,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio,GUITARSET_PINNED_CONTROL
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s652_tone_filter_multiview import (
    causal_lowpass,predict_causal_view,probabilities,FILTERS
)

MODEL_NAMES=("linear_reg","linear_strong","trees","K_only")
THRESHOLDS=(.50,.60,.70,.75,.80,.85,.90,.93,.96,.98,1.01)
SOURCE_FILTER_FLOOR=.55
SOURCE_FILTER_SUSTAIN=.50
MIN_OOF_UTILITY=2.
# Training objective: 1 recovered onset compensates up to 2 False NoteOn.
COST_FP=.50
FEATURE_VERSION="S653_pitch_Tone_native_K_and_4hop_history_48D"


def mask_candidates(a,b,c,config):
    """Candidate evidence requires filtered pitch onset & sustain.

    This is a hypothesis to be adjudicated, not a generated ground-truth
    label. Gate never sees future samples or annotations.
    """
    assert a.shape==b.shape==c.shape==(49,3)
    eligible=((a[:,0]<config.onset)
              & (((b[:,0]>=SOURCE_FILTER_FLOOR)&(b[:,1]>=SOURCE_FILTER_SUSTAIN))
                 | ((c[:,0]>=SOURCE_FILTER_FLOOR)&(c[:,1]>=SOURCE_FILTER_SUSTAIN))))
    return np.flatnonzero(eligible)


def feature_for_pitch(a,b,c,k,ka,kb,old_a,old_b,old_c,i):
    """48 causal values: original/filtered pitch posterior and context."""
    competitors=np.delete(a[:,0],i)
    competitors2=np.delete(a[:,1],i)
    octave=i+12
    lower=i-12
    high=a[octave] if octave<49 else np.zeros(3,np.float32)
    low=a[lower] if lower>=0 else np.zeros(3,np.float32)
    vals=[
        *a[i],*b[i],*c[i],         # 9
        *(b[i]-a[i]),*(c[i]-a[i]), # 6 = 15
        *k,                        # 7 = 22
        float(i+40)/127.,         # 1 = 23
        float(competitors.max()),
        float(competitors2.max()),
        *high[:2],*low[:2],        # 6 = 29
        float(b[:,0].max()),float(c[:,0].max()),
        float(ka.argmax())/6.,float(kb.argmax())/6.,
        float(ka[1:].sum()),float(kb[1:].sum()), # 6 =35
        float(a[i,0]-old_a[i,0]),
        float(b[i,0]-old_b[i,0]),
        float(c[i,0]-old_c[i,0]),
        float(a[i,1]-old_a[i,1]),
        float(b[i,1]-old_b[i,1]),
        float(c[i,1]-old_c[i,1]),
        float(b[i,0]-c[i,0]),
        float(b[i,1]-c[i,1]),
        float(a[i,0]-high[0]),float(a[i,0]-low[0]),
        float(k.argmax())/6.,float(k.max()),
        float(k[0]),float(k[1]), # 14 = 49
    ]
    v=np.asarray(vals,np.float32)
    if not np.isfinite(v).all():
        raise ValueError("Invalid Tone multiview candidate features")
    if v.shape!=(49,):
        raise AssertionError(f"Feature ABI {v.shape} rather than 49D")
    return v


def candidate_pitch_prob(a,b,c,i,config):
    """Modify this candidate only, without introducing false high octaves."""
    p=a.copy()
    p[i,0]=max(float(config.onset),float(b[i,0]),float(c[i,0]))
    p[i,1]=max(float(config.active_on),float(b[i,1]),float(c[i,1]))
    return p


def propose(a,b,c,config,candidate_indices,accepted,state):
    out=a.copy()
    admitted=0
    for idx,yes in zip(candidate_indices,accepted):
        idx=int(idx)
        if not yes or state.active[idx] or not state.armed[idx]:
            continue
        out[idx,0]=max(float(config.onset),float(b[idx,0]),float(c[idx,0]))
        out[idx,1]=max(float(config.active_on),float(b[idx,1]),float(c[idx,1]))
        admitted+=1
    return out,admitted


def models():
    return dict(
        linear_reg=make_pipeline(StandardScaler(),
                 LogisticRegression(C=.25,random_state=653,
                                    class_weight="balanced",max_iter=1800)),
        linear_strong=make_pipeline(StandardScaler(),
                 LogisticRegression(C=1.0,random_state=653,
                                    class_weight="balanced",max_iter=1800)),
        trees=HistGradientBoostingClassifier(
                 max_iter=90,max_leaf_nodes=6,max_depth=3,min_samples_leaf=15,
                 class_weight="balanced",l2_regularization=8.,
                 learning_rate=.06,random_state=653),
        K_only=make_pipeline(StandardScaler(),
                 LogisticRegression(C=.3,random_state=653,
                                    class_weight="balanced",max_iter=1800)),
    )


def project(x,name):
    # K-only diagnosis: candidate pitch+K, no raw Tone differences.
    if name=="K_only":
        return np.asarray(x)[:,[22,*range(15,22),*range(31,35),*range(45,49)]]
    return x


def precomputed(spec,root,archive,heads):
    pcm,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(pcm)/SR)
    ends,p,k=predict_causal_view(pcm,heads)
    views={}
    for name,fc in FILTERS.items():
        tone=causal_lowpass(pcm,fc)
        e,pp,kk=predict_causal_view(tone,heads)
        if not np.array_equal(ends,e):raise ValueError("Tone view timestamp mismatch")
        views[name]=(pp,kk)
    return dict(name=spec["name"],composition=spec["composition"],source_sha256=sha,
                pcm_samples=len(pcm),ends=ends,truth=truth,native=p,native_k=k,
                mild=views["tone_mild_2800"][0],dark=views["tone_dark_1100"][0],
                mild_k=views["tone_mild_2800"][1],dark_k=views["tone_dark_1100"][1])


def training_candidates(r,config):
    """Generate samples from genuine NOTE selector decisions on real audio."""
    original=CausalSelection(config)
    unlimited=CausalSelection(config)
    native_events=[];union_events=[]
    feats={}
    accepted=Counter()
    ends=r["ends"]
    a=r["native"];b=r["mild"];c=r["dark"];k=r["native_k"]
    for j,end in enumerate(ends):
        now_a=a[j];now_b=b[j];now_c=c[j]
        indices=mask_candidates(now_a,now_b,now_c,config)
        native_events.extend(original.step(now_a,k[j],int(end),int(end)))
        out,_=propose(now_a,now_b,now_c,config,indices,
                      np.ones(len(indices),bool),unlimited)
        emitted=unlimited.step(out,k[j],int(end),int(end))
        present=set(map(int,indices))
        old_idx=max(0,j-4)
        for event in emitted:
            if event["type"]!="note_on":continue
            i=int(event["pitch"])-40
            if i not in present or now_a[i,0]>=config.onset:
                continue
            f=feature_for_pitch(now_a,now_b,now_c,k[j],r["mild_k"][j],
                    r["dark_k"][j],a[old_idx],b[old_idx],c[old_idx],i)
            identity=(int(end),int(event["pitch"]))
            if identity in feats:
                raise AssertionError("Duplicated same-hop NOTE candidate")
            feats[identity]=f
            accepted["tone_candidate_noteon"]+=1
        union_events.extend(emitted)
    native_events.extend(original.panic(r["pcm_samples"]))
    union_events.extend(unlimited.panic(r["pcm_samples"]))
    native_score,native_pairs=event_score(native_events,r["truth"])
    union_score,union_pairs=event_score(union_events,r["truth"])
    native_truth={j for _,j in native_pairs}
    union_map={i:j for i,j in union_pairs}
    on=[e for e in union_events if e["type"]=="note_on"]
    arr=[];lbl=[];events=[]
    for idx,e in enumerate(on):
        key=(int(e["decision_sample"]),int(e["pitch"]))
        if key not in feats:continue
        matched=union_map.get(idx)
        answer=int(matched is not None and matched not in native_truth)
        arr.append(feats[key])
        lbl.append(answer)
        events.append(dict(time=float(e["time"]),pitch=int(e["pitch"]),
                           newly_correct=answer))
    return dict(name=r["name"],composition=r["composition"],
                x=np.asarray(arr,np.float32).reshape(-1,49),
                y=np.asarray(lbl,np.int8),
                native_score=native_score,raw_union_score=union_score,
                candidate_samples=events,
                pos=int(sum(lbl)),n=len(lbl))


def fit_training(records):
    X=np.concatenate([r["x"] for r in records])
    y=np.concatenate([r["y"] for r in records])
    groups=np.concatenate([np.repeat(r["composition"],len(r["y"])) for r in records])
    if len(X)<50 or y.sum()<5 or len(set(groups))!=9:
        raise ValueError(f"Too few genuine new Tone corrections: {len(X)}, {sum(y)}")
    folds=list(GroupKFold(n_splits=5).split(X,y,groups))
    winner=None
    scores={}
    fitted={}
    for name,proto in models().items():
        probs=np.zeros(len(X),np.float64)
        for tr,va in folds:
            assert not set(groups[tr])&set(groups[va])
            if len(np.unique(y[tr]))<2:
                raise ValueError("No repair class in CV train fold")
            trained=clone(proto).fit(project(X[tr],name),y[tr])
            probs[va]=trained.predict_proba(project(X[va],name))[:,1]
        trials=[]
        for floor in THRESHOLDS:
            accept=probs>=floor
            tp=int(y[accept].sum())
            fp=int(accept.sum())-tp
            utility=tp-COST_FP*fp
            trials.append(dict(threshold=floor,TP=tp,FP=fp,FN=int(y.sum())-tp,
                               utility=utility))
        best=max(trials,key=lambda d:(d["utility"],-d["FP"],d["threshold"]))
        scores[name]=dict(best=best,all_thresholds=trials)
        print("S653 OOF TRAIN",name,"positives",int(y.sum()),
              "candidate",len(y),best,flush=True)
        fitted[name]=clone(proto).fit(project(X,name),y)
        if winner is None or (best["utility"],-best["FP"],name)>(
            scores[winner]["best"]["utility"],
            -scores[winner]["best"]["FP"],winner):
            winner=name
    selected=winner if scores[winner]["best"]["utility"]>=MIN_OOF_UTILITY else "S614_original"
    return fitted,scores,winner,selected,dict(
        n=len(y),positive=int(y.sum()),
        grouped_train_compositions=sorted(set(groups)))


def validate_one(r,config,models_fit,model_cv):
    outputs=("S614_original","ungated_Tone_union",
             "Tone_mild_single","Tone_dark_single",*MODEL_NAMES)
    states={name:CausalSelection(config) for name in outputs}
    emitted={name:[] for name in outputs}
    proposals=Counter()
    a=r["native"];b=r["mild"];c=r["dark"];k=r["native_k"]
    ends=r["ends"]
    for j,end in enumerate(ends):
        idxs=mask_candidates(a[j],b[j],c[j],config)
        old=max(0,j-4)
        if len(idxs):
            rows=np.stack([feature_for_pitch(a[j],b[j],c[j],k[j],
                r["mild_k"][j],r["dark_k"][j],a[old],b[old],c[old],int(i))
                for i in idxs])
            selected={}
            for name in MODEL_NAMES:
                prob=models_fit[name].predict_proba(project(rows,name))[:,1]
                selected[name]=prob>=model_cv[name]["best"]["threshold"]
            selected["ungated_Tone_union"]=np.ones(len(idxs),bool)
            selected["Tone_mild_single"]=(b[j,idxs,0]>=SOURCE_FILTER_FLOOR)&(
                                       b[j,idxs,1]>=SOURCE_FILTER_SUSTAIN)
            selected["Tone_dark_single"]=(c[j,idxs,0]>=SOURCE_FILTER_FLOOR)&(
                                       c[j,idxs,1]>=SOURCE_FILTER_SUSTAIN)
        else:
            selected={name:[] for name in outputs if name!="S614_original"}
        emitted["S614_original"].extend(states["S614_original"].step(a[j],k[j],int(end),int(end)))
        for name,mask in selected.items():
            p,count=propose(a[j],b[j],c[j],config,idxs,mask,states[name])
            proposals[name]+=count
            emitted[name].extend(states[name].step(p,k[j],int(end),int(end)))
    groups=overlap_groups(r["truth"])
    m={}
    native_truth=None
    for name,state in states.items():
        emitted[name].extend(state.panic(r["pcm_samples"]))
        sc,pairs=event_score(emitted[name],r["truth"])
        correct={j for _,j in pairs}
        if name=="S614_original":native_truth=correct
        m[name]=dict(score=sc,
              overlap_TP=len(correct & groups["onset_overlap"]),
              K2_TP=len(correct & groups["onset_K2"]),
              K3plus_TP=len(correct & groups["onset_K3plus"]),
              E2_TP=sum(r["truth"][q]["pitch"]==40 for q in correct),
              E3_TP=sum(r["truth"][q]["pitch"]==52 for q in correct),
              annotations=correct,proposed=int(proposals[name]))
    for row in m.values():
        matches=row.pop("annotations")
        row["new_vs_original"]=len(matches-native_truth)
        row["lost_vs_original"]=len(native_truth-matches)
    return dict(name=r["name"],composition=r["composition"],stats=m)


def aggregate(rows):
    keys=rows[0]["stats"]
    final={}
    for name in keys:
        parts=[r["stats"][name] for r in rows]
        final[name]=dict(
            score=summary([p["score"] for p in parts]),
            K2_TP=sum(p["K2_TP"] for p in parts),
            K3plus_TP=sum(p["K3plus_TP"] for p in parts),
            overlap_TP=sum(p["overlap_TP"] for p in parts),
            E2_TP=sum(p["E2_TP"] for p in parts),
            E3_TP=sum(p["E3_TP"] for p in parts),
            new_gt=sum(p["new_vs_original"] for p in parts),
            lost_gt=sum(p["lost_vs_original"] for p in parts),
            proposals=sum(p["proposed"] for p in parts))
    return final


def run(out):
    if out.exists():raise FileExistsError("S653 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    now=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            a=proto["guitar_archives"][key]
            download(a["url"],root/"data"/filename,a["size"],a["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            records=[]
            for i,spec in enumerate(train,1):
                acoustic=precomputed(spec,root,arc,heads)
                r=training_candidates(acoustic,cfg)
                records.append(r)
                print("S653 REAL TRAIN",i,"/36",r["name"],
                      "Tone_eligible_noteons",r["n"],
                      "new_true_notes",r["pos"],flush=True)
                del acoustic
            trained,cv,winner,selected,inventory=fit_training(records)
            print("S653 TRAIN CHOICE",json.dumps(dict(
                selected=selected,winner=winner,train=inventory,
                trials={n:v["best"] for n,v in cv.items()})),flush=True)
            evaluated=[]
            for i,spec in enumerate(dev,1):
                acoustic=precomputed(spec,root,arc,heads)
                r=validate_one(acoustic,cfg,trained,cv)
                evaluated.append(r)
                print("S653 REAL DEV",i,"/12",r["name"],
                      "S614_TP",r["stats"]["S614_original"]["score"]["TP"],
                      "ungated_TP",r["stats"]["ungated_Tone_union"]["score"]["TP"],
                      flush=True)
                del acoustic
    m=aggregate(evaluated)
    original=m["S614_original"]["score"]
    assert all(original[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    for name,row in m.items():
        sc=row["score"]
        row["delta_TP"]=sc["TP"]-original["TP"]
        row["delta_FP"]=sc["FP"]-original["FP"]
        row["delta_F1_points"]=round(100*(sc["F1"]-original["F1"]),7)
        print("S653 SCORE",name,"TP",sc["TP"],"FP",sc["FP"],
              "F1",round(sc["F1"],8),"K3",row["K3plus_TP"],
              "E2",row["E2_TP"],"delta",row["delta_F1_points"],flush=True)
    report=dict(
        status="S653_TONE_REPAIR_HEAD_TRAIN_ONLY_GROUP_OOF",
        selected_without_DEV_labels=selected,
        winner_from_TRAIN_only=winner,
        train_inventory=inventory,
        train_CV=cv,
        S614_model_SHA256=frozen["model_sha256"],
        no_NOTE_pitch_or_count_head_retraining=True,
        original_K_used_for_all_note_selectors=True,
        original_live_untouched=True,
        sample_lookahead_after_candidate_ms=0,
        signal_tone_proxy_not_real_Stracocaster=True,
        GuitarSet_TRAIN=36,GuitarSet_DEV=12,GuitarSet_TEST_unopened=True,
        DEV_previously_exposed=True,
        independent_NoteOn_validation=m,
        per_recording=evaluated,
        time_seconds=round(time.perf_counter()-now,2),
        promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S653 FINAL",json.dumps({
        "selected":selected,
        "candidate_pool":inventory,
        "DEV_F1":{n:v["score"]["F1"] for n,v in m.items()}
    }),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(a.out)
