"""Train a REAL neural pre-decision error predictor on native PCM256 Onset events.

For each OUTER composition, train a separate 63D MLP on 3 other compositions
and choose its fixed cut solely on a fourth CAL composition. Original signed
FIT80 truth is used only for labels/evaluation AFTER independently generated
causal PCM decisions. This is OOF for the NEW neural risk head, but NOT an
independent test for the frozen S613/S701 teachers (known FIT80 train overlap).
"""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from onset256.predecision_risk import SCHEMA,DIM,event_vector
from onset256.acoustics import HOP,SR
from scripts.replay_native256 import match_onsets

COMPOSITIONS=("BN1","Funk2","Jazz1","Rock3","SS1")
CUTS=(0.,.1,.2,.3,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9)
SEED=926
REFERENCE=dict(tp=8528,fp=3489,fn=2739)
S613_EXPECTED=dict(tp=8099,fp=4600,fn=3168)

def metrics(tp,fp,fn):
    denom=2*tp+fp+fn
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),
                f1=2*tp/denom if denom else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.)

def signed_truth(traces,manifest_path):
    m=json.loads(Path(manifest_path).read_text())
    if m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not m.get("complete"):
        raise ValueError("original signed corpus identity missing")
    files=m.get("fit_files",[])
    if len(files)!=80:raise ValueError("80 signed tracks required")
    truths={};comps={};samples={}
    for item in files:
        filename=Path(traces)/f"fit_{item['index']:03d}.json"
        actual=hashlib.sha256(filename.read_bytes()).hexdigest()
        if actual!=item["json_sha256"]:
            raise ValueError("signed per-track truth digest mismatch")
        meta=json.loads(filename.read_text())
        if meta["name"]!=item["name"]:
            raise ValueError("signed track/manifest mismatch")
        truths[meta["name"]]=meta["fit_truth"]
        comps[meta["name"]]=item["composition"]
        samples[meta["name"]]=meta["samples"]
    if len(truths)!=80:raise ValueError("duplicate track names")
    if any(sum(x==c for x in comps.values())!=16 for c in COMPOSITIONS):
        raise ValueError("wrong composition-disjoint corpus")
    return truths,comps,samples

def read_examples(event_file,truths,comps,samples):
    records=[]
    for line in Path(event_file).open():
        if not line.strip():continue
        item=json.loads(line)
        if item.get("policy")!="native" or item.get("label") not in ("TP","FP"):
            continue
        name=item.get("track")
        if name not in truths or item.get("composition")!=comps[name]:
            raise ValueError("unsigned or wrong-track predecision feature")
        if (type(item.get("sample")) is not int or item["sample"]%HOP or
            item["sample"]>samples[name] or
            not 40<=int(item["pitch"])<=88):
            raise ValueError("not a causal PCM256 proposal")
        v=event_vector(item)
        records.append(dict(name=name,comp=comps[name],sample=int(item["sample"]),
                           pitch=int(item["pitch"]),X=v,y=int(item["label"]=="TP")))
    if not records:raise ValueError("no genuine native onset proposals")
    groups=defaultdict(list)
    for i,row in enumerate(records):groups[row["name"]].append(i)
    if set(groups)!=set(truths):
        raise ValueError("every signed track must have native onset candidates")
    all_events=[SimpleNamespace(pitch=r["pitch"],sample=r["sample"]) for r in records]
    for name,idx in groups.items():
        expected={k:sum(int(records[i]["y"]==(k=="tp")) for i in idx)
                  for k in ("tp","fp")}
        baseline=len(match_onsets([all_events[i] for i in idx],truths[name]))
        if baseline!=expected["tp"]:
            raise AssertionError("signed matched TP provenance disagrees with live matching")
    return records,groups,all_events

def score_subset(records,groups,events,truths,allowed,confidence,cut):
    tp=fp=fn=0;track={}
    for name in allowed:
        indices=[i for i in groups[name] if confidence[i]>=cut]
        predicted=[events[i] for i in indices]
        n=len(match_onsets(predicted,truths[name]))
        count=metrics(n,len(indices)-n,len(truths[name])-n)
        track[name]=count
        tp+=count["tp"];fp+=count["fp"];fn+=count["fn"]
    return metrics(tp,fp,fn),track

def score_auc(rows,indices,prob):
    labels=np.array([rows[i]["y"] for i in indices],np.int8)
    preds=np.array([prob[i] for i in indices],np.float64)
    return dict(roc_auc=roc_auc_score(labels,preds) if len(np.unique(labels))==2 else None,
                brier=brier_score_loss(labels,preds))

def train_one(rows,train_indices,seed):
    x=np.stack([rows[i]["X"] for i in train_indices])
    y=np.array([rows[i]["y"] for i in train_indices],np.int8)
    if x.shape[1]!=DIM or set(np.unique(y))!={0,1}:
        raise ValueError("requires genuinely labeled both TP and FP at PCM256")
    net=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",solver="adam",
        alpha=.01,batch_size=256,learning_rate_init=.0015,max_iter=65,
        tol=.0001,n_iter_no_change=12,early_stopping=True,validation_fraction=.12,
        random_state=seed)
    pipe=make_pipeline(StandardScaler(),net)
    pipe.fit(x,y)
    if getattr(pipe,"n_features_in_",None)!=DIM:
        raise AssertionError("trained native256 MLP incompatible with 63D")
    return pipe

def run(events,traces,manifest,out):
    if out.exists():raise FileExistsError("do not overwrite immutable training experiment")
    truth,comps,samples=signed_truth(traces,manifest)
    rows,bytrack,event_objects=read_examples(events,truth,comps,samples)
    names={c:sorted([name for name in truth if comps[name]==c]) for c in COMPOSITIONS}
    total_base,per_track=score_subset(rows,bytrack,event_objects,truth,list(truth),
                                     np.ones(len(rows),np.float32),0)
    if any(total_base[k]!=v for k,v in S613_EXPECTED.items()):
        raise AssertionError("native reference 8099 TP / 4600 FP / 3168 FN not reproduced")
    out.mkdir(parents=True)
    checkpoints=out/"checkpoints";checkpoints.mkdir()
    folds=[];oof={}
    for outer_i,outer in enumerate(COMPOSITIONS):
        cal=COMPOSITIONS[(outer_i+1)%len(COMPOSITIONS)]
        trains=tuple(c for c in COMPOSITIONS if c not in (outer,cal))
        train_indices=[i for i,r in enumerate(rows) if r["comp"] in trains]
        cal_indices=[i for i,r in enumerate(rows) if r["comp"]==cal]
        outer_indices=[i for i,r in enumerate(rows) if r["comp"]==outer]
        net=train_one(rows,train_indices,SEED+outer_i)
        probabilities=np.asarray(net.predict_proba(
            np.stack([r["X"] for r in rows]))[:,1],np.float32)
        if probabilities.shape!=(len(rows),) or not np.isfinite(probabilities).all():
            raise ValueError("neural risk output invalid")
        cal_base,_=score_subset(rows,bytrack,event_objects,truth,names[cal],
                                probabilities,0)
        stats={}
        eligible=[]
        for cut in CUTS:
            res,per=score_subset(rows,bytrack,event_objects,truth,
                                names[cal],probabilities,cut)
            lost=cal_base["tp"]-res["tp"]
            removed=cal_base["fp"]-res["fp"]
            regressed=sum(per[n]["f1"]+1e-12<per_track[n]["f1"] for n in names[cal])
            stats[str(cut)]=dict(**res,tp_lost=lost,fp_removed=removed,
                                 regressed_tracks=regressed)
            # Protect attacks: no policy that loses >=2% of native CAL TP.
            if cut and res["f1"]>cal_base["f1"]+0.0001 and lost<=.02*cal_base["tp"] and removed>=3:
                eligible.append(cut)
        cut=max(eligible,key=lambda c:(stats[str(c)]["f1"],-c),default=0.)
        outer_score,by_outer=score_subset(rows,bytrack,event_objects,truth,
                                         names[outer],probabilities,cut)
        outer_base,outer_base_tracks=score_subset(rows,bytrack,event_objects,truth,
                                                 names[outer],probabilities,0)
        x=checkpoints/f"onset_risk_outer_{outer}.joblib"
        joblib.dump(net,x,compress=3)
        sha=hashlib.sha256(x.read_bytes()).hexdigest()
        metadata=dict(schema=SCHEMA,trained_hop_samples=256,decision_hop_samples=256,
                      dim=DIM,model_type="StandardScaler+MLPClassifier_64_32",
                      checkpoint_sha256=sha,calibrated_cut=float(cut),
                      train_compositions=trains,cal_composition=cal,
                      outer_composition=outer,selector="CAL_ONLY_NO_OUTER_SELECTION",
                      training_rows=len(train_indices))
        (checkpoints/f"onset_risk_outer_{outer}.json").write_text(
            json.dumps(metadata,indent=2)+"\n")
        folds.append(dict(outer_composition=outer,cal_composition=cal,
                          train_compositions=trains,training_events=len(train_indices),
                          cal_events=len(cal_indices),outer_events=len(outer_indices),
                          calibrated_cut=cut,cal_baseline=cal_base,
                          cal_policy_metrics=stats,outer_baseline=outer_base,
                          outer_shadow_filtered=outer_score,
                          outer_roc_auc=score_auc(rows,outer_indices,probabilities),
                          outer_track_changes={name:dict(
                              original=outer_base_tracks[name],filtered=by_outer[name])
                              for name in names[outer]}))
        oof[outer]=dict(cut=cut,source_file=x.name,sha256=sha)
        print("ONSET_PREDECISION_256_TRAINED_FOLD",
              json.dumps(dict(outer=outer,cal=cal,cut=cut,
                              outer_baseline_f1=outer_base["f1"],
                              outer_shadow_f1=outer_score["f1"],
                              outer_auc=folds[-1]["outer_roc_auc"]["roc_auc"])),flush=True)
    tp=sum(f["outer_shadow_filtered"]["tp"] for f in folds)
    fp=sum(f["outer_shadow_filtered"]["fp"] for f in folds)
    fn=sum(f["outer_shadow_filtered"]["fn"] for f in folds)
    measured=metrics(tp,fp,fn)
    if sum(f["outer_baseline"]["tp"] for f in folds)!=8099:
        raise AssertionError("outer native numerator corrupted")
    # Save a separate pooled experimental checkpoint; never treat FIT80 as
    # independent validation for this pooled model or earlier teachers.
    full=train_one(rows,list(range(len(rows))),SEED+100)
    frozen=float(np.median([x["calibrated_cut"] for x in folds]))
    deployed=checkpoints/"onset_risk_pooled_experimental.joblib"
    joblib.dump(full,deployed,compress=3)
    sha=hashlib.sha256(deployed.read_bytes()).hexdigest()
    pooled_meta=dict(schema=SCHEMA,trained_hop_samples=256,dim=DIM,
        calibrated_cut=frozen,checkpoint_sha256=sha,
        trained_on_fit80=True,external_test_completed=False,
        private_axis="onset",no_s708_1024_head=True,not_promoted=True,
        model_type="StandardScaler+MLPClassifier_64_32")
    (checkpoints/"onset_risk_pooled_experimental.json").write_text(
        json.dumps(pooled_meta,indent=2)+"\n")
    result=dict(schema=SCHEMA,complete=True,axis="onset",
       weights_are_actually_trained=True,training_hop_samples=256,
       forward_decision_hop_samples=256,model="MLP_63_64_32_1",
       five_fold_composition_oof_risk_head=True,
       frozen_teachers_pretrained_on_outer_fit80=True,
       independently_generalized_full_pipeline=False,
       score_kind="POSTHOC_CANDIDATE_FILTER_PROXY_NOT_LIVE_GATED",
       historical_s708_1024_f1=2*REFERENCE["tp"]/(2*REFERENCE["tp"]+REFERENCE["fp"]+REFERENCE["fn"]),
       original_s613_native=total_base,predecision_risk_oof_shadow=measured,
       folds=folds,pooled_experimental_model=pooled_meta,
       no_midi_production_change=True,other_axes_unchanged=True,
       promoted=False)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("ONSET_PREDECISION_NEURAL256_TRAINED_OOF_FINAL",
          json.dumps(dict(native=total_base,shadow=measured,
                          roc_auc_by_fold={x["outer_composition"]:x["outer_roc_auc"]["roc_auc"]
                                           for x in folds},pooled_checkpoint_sha256=sha)),
          flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for flag in ("events","traces","manifest","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    args=p.parse_args()
    run(args.events,args.traces,args.manifest,args.out)
if __name__=="__main__":
    main()
