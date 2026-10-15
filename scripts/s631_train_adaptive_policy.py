"""S631 real-audio training of a per-note adaptive WAIT/EMIT/REJECT head.

TRAIN: 17 URMP + 18 GuitarSet sources, with composition-disjoint DEV.
Original S614 and S80 weights, and the 22-source published test, untouched.
Every policy sees only right-aligned PCM-derived native S614 features.
Training-only oracle labels are built from actual annotated attacks;
the learned action head never reads truth or future frames at inference.

IMPORTANT: the offline supervision labels a *good decision time* using
annotations and deadline costs. It is not reinforcement learning, and the
observed F1 must be measured from streaming emissions, not from teacher labels.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR, Selection,load_heads
from causal_note.s631_adaptive_latency import (
    CandidateTracker,DecisionPolicy,WAIT,EMIT,REJECT,
    FEATURE_DIM,eval_stream,MAX_FRAMES)
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import decode,event_score,offset_audit,summary

CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}
AGE_SAMPLES=(0,1,2,3,5,8,12,16,20,25)
GRID=(
    ("S631_emit040_reject060_28",DecisionPolicy(.40,.60,28)),
    ("S631_emit055_reject060_28",DecisionPolicy(.55,.60,28)),
    ("S631_emit070_reject060_28",DecisionPolicy(.70,.60,28)),
    ("S631_emit040_reject075_28",DecisionPolicy(.40,.75,28)),
    ("S631_emit055_reject075_28",DecisionPolicy(.55,.75,28)),
    ("S631_emit070_reject075_28",DecisionPolicy(.70,.75,28)),
    ("S631_emit055_reject060_14",DecisionPolicy(.55,.60,14)),
    ("S631_emit055_reject075_14",DecisionPolicy(.55,.75,14)),
)


def training_examples(row,rng):
    """Create causal candidate histories once; supervision may inspect truth."""
    tr=CandidateTracker()
    episodes=defaultdict(list)
    for i,(end,feature,prob,k) in enumerate(zip(row["ends"],row["x"],row["p"],row["k"])):
        for pitch,age,vec,birth in tr.step(prob,feature,k):
            if age in AGE_SAMPLES:
                episodes[(pitch,birth)].append((i,age,vec))
    # At most one track may claim each actual pitch onset.
    truth_by_pitch=defaultdict(list)
    for i,note in enumerate(row["truth"]):
        truth_by_pitch[int(note["pitch"])-40].append((i,float(note["time"])))
    reserved=set()
    x,y=[],[]
    episode_counts=Counter()
    for (pitch,birth),sequence in sorted(episodes.items(),key=lambda item:(item[0][1],item[0][0])):
        born=(int(row["ends"][birth])/SR)
        nearest=[(abs(t-born),index,t) for index,t in truth_by_pitch.get(pitch,[])
                 if -.10 <= t-born <= .085 and (pitch,index) not in reserved]
        positive=bool(nearest)
        if positive:
            _,ix,truth_time=min(nearest)
            reserved.add((pitch,ix))
            episode_counts["true_episode"]+=1
        else:
            truth_time=float("inf")
            episode_counts["false_episode"]+=1
        for i,age,feature in sequence:
            now=float(row["ends"][i])/SR
            if positive:
                # Optimal target under monotonic delay loss: output at the
                # first available frame after truth onset (not retroactively);
                # don't train the model to stall by arbitrary fixed milliseconds.
                action=WAIT if now+1e-6<truth_time else EMIT
                # A too-late emission cannot count as a correct ±80ms onset.
                if now-truth_time>.08:action=REJECT
            else:
                # Ambiguous candidates get a short chance for evidence;
                # false candidates should learn to reject before max budget.
                action=WAIT if age<=2 else REJECT
            x.append(feature)
            y.append(action)
    if not x:return (np.empty((0,FEATURE_DIM),np.float32),
                     np.empty((0,),np.int8),dict(episode_counts))
    matrix=np.stack(x).astype(np.float32)
    target=np.asarray(y,np.int8)
    # Label counts help audit the training time/false-positive tradeoff.
    return matrix,target,dict(episode_counts,WAIT=int((target==WAIT).sum()),
        EMIT=int((target==EMIT).sum()),REJECT=int((target==REJECT).sum()),
        feature_rows=len(target),original_truth_notes=len(row["truth"]))


class CachedClassifier:
    """Reuse *identical* acoustic states across DEV threshold ablations."""
    def __init__(self,model):
        self.model=model
        self.classes_=model.classes_
        self.cache={}
    def predict_proba(self,features):
        keys=[x.tobytes() for x in features]
        missing=[(i,k) for i,k in enumerate(keys) if k not in self.cache]
        if missing:
            vectors=np.stack([features[i] for i,k in missing])
            scores=self.model.predict_proba(vectors)
            for (i,key),q in zip(missing,scores):self.cache[key]=q
        return np.asarray([self.cache[k] for k in keys],np.float32)


def measure(row,classifier,config):
    baseline=decode(row["ends"],row["p"],row["k"],config,row["n"])
    original,pairs=event_score(baseline,row["truth"])
    result={"S614_original":dict(score=original,
             offset=offset_audit(baseline,row["truth"],pairs,row["n"]/SR),
             elapsed_ms=[], decisions=[0,0,0])}
    cached=CachedClassifier(classifier)
    for name,policy in GRID:
        emitted,selector=eval_stream(row["ends"],row["x"],row["p"],row["k"],
                           cached,config,policy,row["n"])
        score,pairs=event_score(emitted,row["truth"])
        offsets=offset_audit(emitted,row["truth"],pairs,row["n"]/SR)
        result[name]=dict(score=score,offset=offsets,
             elapsed_ms=selector.latencies,decisions=selector.decision_counts.tolist())
    return result


def run(out):
    if out.exists():raise FileExistsError("Immutable experimental evidence")
    out.mkdir(parents=True)
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    protocol=json.loads((ROOT/"protocol.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    training_u=[s for s in protocol["sources"] if s["split"]=="train" and s["corpus"]=="URMP"]
    training_g=pick_original_guitar_training(protocol)
    dev_u=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="URMP"]
    dev_g=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert (len(training_u),len(training_g),len(dev_u),len(dev_g))==(17,18,5,12)
    group=lambda x:(x["corpus"],x["composition"])
    assert not {group(s) for s in training_u+training_g}&{group(s) for s in dev_u+dev_g}
    assert all(s["split"]!="test" for s in training_u+training_g+dev_u+dev_g)
    heads=load_heads(MODEL,expected_sha256=sha)
    config=Selection(**heads.metadata["selection"])
    clock=time.perf_counter()
    xs,ys,proof=[],[],[]
    rows=[]
    rng=np.random.default_rng(631)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for s in training_u:
            data=get_source(s,root,heads)
            x,y,audit=training_examples(data,rng)
            xs.append(x);ys.append(y)
            proof.append(dict(name=s["name"],corpus=s["corpus"],**audit))
            print("S631 TRAIN URMP",s["name"],audit,flush=True)
        aud=protocol["guitar_archives"]["GuitarSet_audio"]
        ann=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(aud["url"],root/"data/GuitarSet_audio.zip",aud["size"],aud["sha256"])
        download(ann["url"],root/"data/GuitarSet_annotations.zip",ann["size"],ann["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as z:
            for s in training_g:
                data=get_source(s,root,heads,z)
                x,y,audit=training_examples(data,rng)
                xs.append(x);ys.append(y)
                proof.append(dict(name=s["name"],corpus=s["corpus"],**audit))
                print("S631 TRAIN GUITAR",s["name"],audit,flush=True)
            a=np.concatenate(xs);b=np.concatenate(ys)
            assert a.shape[1]==FEATURE_DIM and len(set(b.tolist()))==3 and len(b)>1000
            # Three actual learned decisions. Offline labels are never used
            # inside CandidateTracker or inference.
            model=HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=22,
                max_depth=6,min_samples_leaf=45,learning_rate=.065,
                l2_regularization=3.,early_stopping=False,random_state=631)
            weights=np.choose(b,np.asarray([1.0,2.5,2.0]))
            model.fit(a,b,sample_weight=weights)
            checkpoint=out/"S631_learned_wait_emit_reject_train_only.joblib"
            joblib.dump(model,checkpoint,compress=3)
            model_sha=digest(checkpoint.read_bytes())
            print("S631 TRAINED HEAD",model_sha,"counts",dict(Counter(b.tolist())),
                  "samples",len(b),flush=True)
            del a,b,xs,ys
            for s in dev_u:
                data=get_source(s,root,heads)
                predictions=measure(data,model,config)
                rows.append(dict(source=s["name"],corpus="URMP",policies=predictions))
                print("S631 DEV URMP",s["name"],flush=True)
            for s in dev_g:
                data=get_source(s,root,heads,z)
                predictions=measure(data,model,config)
                rows.append(dict(source=s["name"],corpus="GuitarSet",policies=predictions))
                print("S631 DEV GUITAR",s["name"],flush=True)
    result=[]
    for name,policy in [("S614_original",None),*GRID]:
        corpus={}
        for key,expected in CONTROL.items():
            group=[v for v in rows if v["corpus"]==key]
            aggregate=summary([v["policies"][name]["score"] for v in group])
            if name=="S614_original":
                assert all(aggregate[k]==val for k,val in expected.items()),(key,aggregate)
            baseline_f1=2*expected["TP"]/(2*expected["TP"]+expected["FP"]+expected["FN"])
            latencies=[d for v in group for d in v["policies"][name]["elapsed_ms"]]
            decisions=np.sum([v["policies"][name]["decisions"] for v in group],axis=0)
            eligible=sum(v["policies"][name]["offset"]["matched_noteons_with_observed_reference_offset"] for v in group)
            offs=sum(v["policies"][name]["offset"]["offsets_within_max_50ms_or_20pct_duration"] for v in group)
            corpus[key]=dict(TP=aggregate["TP"],FP=aggregate["FP"],
                FN=aggregate["FN"],F1=aggregate["F1"],
                delta_F1_points=100*(aggregate["F1"]-baseline_f1),
                changed_wins=sum(v["policies"][name]["score"]["F1"]>
                                 v["policies"]["S614_original"]["score"]["F1"] for v in group),
                changed_losses=sum(v["policies"][name]["score"]["F1"]<
                                 v["policies"]["S614_original"]["score"]["F1"] for v in group),
                policy_wait_emit_reject_counts=decisions.tolist(),
                adaptive_post_candidate_latency_ms={
                    "p50":float(np.percentile(latencies,50)),
                    "p95":float(np.percentile(latencies,95)),
                    "max":float(max(latencies))} if latencies else None,
                valid_NoteOff_conditional_accuracy=offs/max(1,eligible),
                matched_emission_delay_ms=aggregate.get("matched_emission_delay_ms"))
        result.append(dict(name=name,policy=asdict(policy) if policy else None,by_corpus=corpus))
        print("S631 DEV SCORE",name,"G",round(corpus["GuitarSet"]["F1"],5),
              "U",round(corpus["URMP"]["F1"],5),flush=True)
    winners=[r["name"] for r in result if r["name"]!="S614_original" and
             all(r["by_corpus"][c]["delta_F1_points"]>0 for c in CONTROL)]
    report=dict(status="S631_REAL_TRAINED_ADAPTIVE_PER_NOTE_LATENCY_DEV_ONLY",
        original_S80_S614_unchanged=True,source_checkpoint_sha256=sha,
        action_checkpoint_sha256=model_sha,
        label_semantics="offline oracle first eligible true onset vs false candidate; no truth in inference",
        train_source_count=len(proof),training_sources=proof,
        dev_source_count=len(rows),
        candidates_age_limit_frames=MAX_FRAMES,
        decision_action_names=["WAIT","EMIT","REJECT"],
        comparison=result,simultaneously_positive_dev=winners,
        test_22_source_audio_and_annotations_not_read=True,
        developer_exposed_dev_no_independent_claim=True,
        no_official_promotion=True,elapsed_seconds=round(time.perf_counter()-clock,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S631 SUMMARY",json.dumps(dict(checkpoint=model_sha,
             source_count=len(proof),positive_both=winners,
             totals=[dict(name=v["name"],
                 G=v["by_corpus"]["GuitarSet"]["F1"],
                 U=v["by_corpus"]["URMP"]["F1"]) for v in result]),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
