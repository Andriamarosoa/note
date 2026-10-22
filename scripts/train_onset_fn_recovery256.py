"""TRAIN independent fourth FNRecoveryHead native PCM256 on genuine missed attacks.

Each fold: three compositions TRAIN, fourth CAL (policy + threshold), fifth
OUTER untouched until CAL is frozen. Three genuinely trained 69D neural
candidate specialists, one per fixed low-threshold proposal stream. Labels
are GT-FN matches assigned AFTER complete causal original WAV inference.

The frozen third ErrorPrediction MLP69 model is reused and evaluated as the
true MAIN accepted stream on each composition. This is a candidate-event
OFFLINE PROXY only: scientific promotion requires a second full WAV replay
where both learned heads predict every 256 new PCM samples.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
from types import SimpleNamespace
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from onset256.fn_recovery import (
    SCHEMA,DIM,CANDIDATE_POLICIES,FNRecoverySelector)
from onset256.error_evidence69 import load_hard_model
from scripts.replay_native256 import match_onsets
from scripts.train_onset_predecision256 import (
    COMPOSITIONS,SEED,signed_truth,metrics)

CUTS=(.15,.25,.35,.45,.55,.65,.75,.85,.90,.95,.98,.99)
HISTORIC_S613=dict(tp=8099,fp=4600,fn=3168)
HISTORIC_MLP69=dict(tp=8054,fp=4411,fn=3213)
EXPECTED_ORACLE={"sensitive":1923,"moderate":1668,"cautious":1367}

def evidence_records(path,truth,comps,samples):
    candidates=defaultdict(list)
    stats=Counter()
    keyset=set()
    with Path(path).open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            track=row.get("track");policy=row.get("policy")
            if (track not in truth or row.get("composition")!=comps[track] or
                policy not in CANDIDATE_POLICIES):
                raise ValueError("unsigned or unknown recovery candidate")
            sample=row["sample"];pitch=row["pitch"]
            x=np.asarray(row["hard_predecision_vector"],np.float32)
            if (type(sample) is not int or not 0<sample<=samples[track] or sample%256 or
                type(pitch) is not int or not 40<=pitch<=88 or
                x.shape!=(DIM,) or not np.isfinite(x).all()):
                raise ValueError("candidate is not a complete causal trained256 frame")
            label=row.get("label")
            if label not in ("recoverable_fn","not_fn"):
                raise ValueError("FN target label missing")
            key=(track,policy,sample,pitch)
            if key in keyset:
                raise ValueError("duplicate candidate in signed FN audit")
            keyset.add(key)
            y=int(label=="recoverable_fn")
            candidates[policy].append(dict(track=track,comp=comps[track],
                pitch=pitch,sample=sample,X=x,y=y))
            stats[(policy,"n")]+=1
            stats[(policy,"positive")]+=y
    for policy,expected in EXPECTED_ORACLE.items():
        if stats[(policy,"positive")]!=expected:
            raise AssertionError("signed FN oracle coverage mismatch: "+policy)
    return candidates,{p:{"events":stats[(p,"n")],
                           "positive":stats[(p,"positive")]}
                       for p in CANDIDATE_POLICIES}

def main_candidates(path,truth,comps):
    records=defaultdict(list);seen=set()
    for line in Path(path).open():
        if not line.strip():continue
        x=json.loads(line)
        track=x["track"]
        if track not in truth or x["composition"]!=comps[track]:
            raise ValueError("main native event not signed")
        key=(track,int(x["sample"]),int(x["pitch"]))
        if key in seen:raise ValueError("duplicate original NoteOn")
        seen.add(key)
        records[track].append(SimpleNamespace(
            sample=int(x["sample"]),pitch=int(x["pitch"])))
    if len(seen)!=12699 or set(records)!=set(truth):
        raise AssertionError("did not reproduce S613 12,699 NoteOns")
    for x in records.values():
        x.sort(key=lambda a:a.sample)
    return records

def risk_filtered_main(main,prior_events,prior_models,comps):
    """Reproduce EXACT earlier CAL-frozen 69D ErrorPrediction choices.

    Inputs from the signed earlier run are read-only and used solely to form
    a frozen baseline, not a new risk training or new risk threshold.
    """
    models={}
    for comp in COMPOSITIONS:
        base=Path(prior_models)/"checkpoints"
        path=base/f"onset_error_outer_{comp}.joblib"
        meta_path=base/f"onset_error_outer_{comp}.json"
        metadata=json.loads(meta_path.read_text())
        if (metadata["outer_composition"]!=comp or comp in
            metadata["train_compositions"] or comp==metadata["cal_composition"]):
            raise ValueError("old MLP69 weights are not composition-heldout")
        models[comp]=load_hard_model(path,meta_path,
                      expected_sha256=metadata["checkpoint_sha256"])
    grouped=defaultdict(list);seen=set()
    for line in Path(prior_events).open():
        if not line.strip():continue
        row=json.loads(line)
        if row.get("policy")!="native" or row.get("label") not in ("TP","FP"):
            continue
        track=row["track"];comp=row["composition"]
        if track not in main or comps[track]!=comp:
            raise ValueError("cross-file frozen baseline identity mismatch")
        key=(track,int(row["sample"]),int(row["pitch"]))
        if key in seen:
            raise AssertionError("multiple old MLP69 samples for same Main")
        seen.add(key)
        feature=np.asarray(row["hard_predecision_vector"],np.float32)
        if feature.shape!=(69,):
            raise ValueError("risk baseline missing 69D genuinely-native checkpoint evidence")
        grouped[comp].append((key,feature))
    if len(seen)!=12699 or seen!={(t,e.sample,e.pitch)
                                for t,notes in main.items() for e in notes}:
        raise AssertionError("MLP69 baseline event identity drift")
    selected=defaultdict(list)
    for comp,pairs in grouped.items():
        x=np.stack([feature for _,feature in pairs])
        confidence=models[comp].model.predict_proba(x)[:,1]
        if not np.isfinite(confidence).all():raise ValueError("NaN frozen risk probability")
        for (track,sample,pitch),q in (
            (pair[0],float(p)) for pair,p in zip(pairs,confidence)):
            if q>=models[comp].cut:
                selected[track].append(SimpleNamespace(pitch=pitch,sample=sample))
    for record in selected.values():
        record.sort(key=lambda e:e.sample)
    return selected,models

def train_one(rows,train_indices,seed):
    """Actual independently learned neural 69D model, not a rule."""
    rng=np.random.default_rng(seed)
    pos=[i for i in train_indices if rows[i]["y"]==1]
    neg=[i for i in train_indices if rows[i]["y"]==0]
    if len(pos)<30 or len(neg)<30:
        raise ValueError("insufficient real signed FN candidate evidence")
    negatives=rng.choice(neg,size=min(len(neg),3*len(pos)),replace=False).tolist()
    selected=pos+negatives
    rng.shuffle(selected)
    x=np.stack([rows[i]["X"] for i in selected])
    y=np.asarray([rows[i]["y"] for i in selected],np.int8)
    net=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",
        solver="adam",alpha=.012,batch_size=256,learning_rate_init=.001,
        max_iter=65,tol=.0001,n_iter_no_change=12,early_stopping=True,
        validation_fraction=.12,random_state=seed)
    pipe=make_pipeline(StandardScaler(),net)
    pipe.fit(x,y)
    if pipe.n_features_in_!=DIM:
        raise ValueError("neural FNRecovery head trained on wrong ABI")
    return pipe,dict(tp_candidates=len(pos),negative_candidates=len(neg),
                     training_rows=len(selected))

def score_selected(tracks,truth,main_raw,main_accepted,
                   candidate_rows,confidence,cut,enabled):
    """Exactly same causal FNRecoverySelector logic as LIVE, no GT in selection."""
    n_total=Counter();bytrack={}
    for track in tracks:
        raw=main_raw[track]
        accepted=main_accepted[track]
        proposals=[(r,float(confidence[i]))
                   for i,r in enumerate(candidate_rows)
                   if r["track"]==track]
        grouped=defaultdict(list)
        for r,q in proposals:
            grouped[r["sample"]].append((r,q))
        raws=defaultdict(list);accepted_by_sample=defaultdict(list)
        for e in raw:raws[e.sample].append(e)
        for e in accepted:accepted_by_sample[e.sample].append(e)
        sel=FNRecoverySelector(cut,enabled)
        produced=[]
        # Main was already selected with frozen MLP69 threshold. Never edit it.
        for sample in range(256,max(int(raw[-1].sample) if raw else 0,
                                    int(accepted[-1].sample) if accepted else 0,
                                    max(grouped,default=0),
                                    256)+256,256):
            p=np.zeros(49,np.float32)
            events=[]
            for row,q in grouped.get(sample,()):
                p[row["pitch"]-40]=q
                events.append(dict(pitch=row["pitch"],decision_sample=sample))
            extra=sel.select(end_sample=sample,
                main_raw=raws.get(sample,()),
                main_accepted=accepted_by_sample.get(sample,()),
                proposals=events,recovery_probability=p)
            produced.extend(SimpleNamespace(pitch=e["pitch"],sample=e["decision_sample"])
                            for e in extra)
        merged=sorted([*accepted,*produced],key=lambda e:(e.sample,e.pitch))
        pairs=match_onsets(merged,truth[track])
        sc=metrics(len(pairs),len(merged)-len(pairs),len(truth[track])-len(pairs))
        bytrack[track]=dict(**sc,added=len(produced))
        for k in ("tp","fp","fn","added"):n_total[k]+=bytrack[track][k]
    aggregate=metrics(n_total["tp"],n_total["fp"],n_total["fn"])
    aggregate["added"]=int(n_total["added"])
    return aggregate,bytrack

def run(candidate_file,main_file,old_event_file,old_models,traces,manifest,out):
    if out.exists():raise FileExistsError("immutable experimental weights/report")
    truth,comps,samples=signed_truth(traces,manifest)
    candidates,stat=evidence_records(candidate_file,truth,comps,samples)
    main=main_candidates(main_file,truth,comps)
    risk_accepted,risk_models=risk_filtered_main(main,old_event_file,old_models,comps)
    tracks_by_comp={c:tuple(sorted(t for t in truth if comps[t]==c))
                    for c in COMPOSITIONS}
    baseline,baseline_tracks=score_selected(
        tuple(truth),truth,main,risk_accepted,[],np.empty(0),1.,False)
    if any(baseline[k]!=v for k,v in HISTORIC_MLP69.items()):
        raise AssertionError("69D frozen head must reproduce TP8054 FP4411 FN3213: "+str(baseline))
    out.mkdir(parents=True)
    checkpoints=out/"checkpoints";checkpoints.mkdir()
    folds=[]
    for fold_i,outer in enumerate(COMPOSITIONS):
        cal=COMPOSITIONS[(fold_i+1)%len(COMPOSITIONS)]
        train=tuple(c for c in COMPOSITIONS if c not in (outer,cal))
        choices=[]
        all_experts={}
        cal_baseline,_=score_selected(tracks_by_comp[cal],truth,main,risk_accepted,
                                      [],np.empty(0),1.,False)
        for pi,policy in enumerate(CANDIDATE_POLICIES):
            rows=candidates[policy]
            train_indices=[i for i,r in enumerate(rows) if r["comp"] in train]
            cal_indices=[i for i,r in enumerate(rows) if r["comp"]==cal]
            outer_indices=[i for i,r in enumerate(rows) if r["comp"]==outer]
            model,stats=train_one(rows,train_indices,SEED+8000+fold_i+pi*100)
            all_experts[policy]=model
            probabilities=np.zeros(len(rows),np.float32)
            for indices in (cal_indices,outer_indices):
                if indices:
                    probabilities[indices]=model.predict_proba(
                        np.stack([rows[i]["X"] for i in indices]))[:,1]
            if not np.isfinite(probabilities).all():
                raise ValueError("invalid true FNRecovery neural output")
            for cut in CUTS:
                cal_scores,_=score_selected(
                    tracks_by_comp[cal],truth,main,risk_accepted,rows,probabilities,
                    cut,True)
                added_tp=cal_scores["tp"]-cal_baseline["tp"]
                added_fp=cal_scores["fp"]-cal_baseline["fp"]
                if (cal_scores["f1"]>cal_baseline["f1"]+1e-4 and
                    added_tp>=3 and
                    added_fp<=max(6,2*added_tp)):
                    choices.append(dict(policy=policy,cut=cut,score=cal_scores,
                                       delta_tp=added_tp,delta_fp=added_fp))
            print("ONSET_FN_RECOVERY_MLP_TRAINED",json.dumps(
                dict(outer=outer,cal=cal,policy=policy,**stats,
                     cal_candidates=len(cal_indices),outer_candidates=len(outer_indices))),
                  flush=True)
        # Only the CAL subset chooses a policy; never inspect OUTER scores
        # before freezing the threshold and own neural checkpoint.
        if choices:
            chosen=max(choices,key=lambda x:(x["score"]["f1"],
                                      x["delta_tp"],-x["delta_fp"]))
            policy,cut,enabled=chosen["policy"],chosen["cut"],True
        else:
            policy,cut,enabled="moderate",.99,False
            chosen=None
        model=all_experts[policy]
        path=checkpoints/f"fn_recovery_outer_{outer}.joblib"
        joblib.dump(model,path,compress=3)
        hashv=hashlib.sha256(path.read_bytes()).hexdigest()
        info=dict(schema=SCHEMA,checkpoint_sha256=hashv,trained_hop_samples=256,
                  dim=69,frozen_before_outer=True,calibrated_only_on_cal=True,
                  train_compositions=train,cal_composition=cal,
                  outer_composition=outer,candidate_policy=policy,
                  calibrated_cut=float(cut),enabled=enabled,
                  trained_neural_network="StandardScaler+MLPClassifier_64_32",
                  fn_labels_attached_post_inference=True)
        (checkpoints/f"fn_recovery_outer_{outer}.json").write_text(
            json.dumps(info,indent=2)+"\n")
        outer_rows=candidates[policy]
        outer_indices=[i for i,r in enumerate(outer_rows) if r["comp"]==outer]
        confidence=np.zeros(len(outer_rows),np.float32)
        if outer_indices:
            confidence[outer_indices]=model.predict_proba(
                np.stack([outer_rows[i]["X"] for i in outer_indices]))[:,1]
        outer_scores,changes=score_selected(
            tracks_by_comp[outer],truth,main,risk_accepted,outer_rows,confidence,
            cut,enabled)
        outer_baseline,_=score_selected(
            tracks_by_comp[outer],truth,main,risk_accepted,
            [],np.empty(0),1.,False)
        folds.append(dict(outer=outer,cal=cal,train=train,
            selected_policy=policy,cal_cut=cut,enabled=enabled,
            cal_baseline=cal_baseline,cal_chosen=chosen,
            outer_baseline=outer_baseline,outer_proxy_score=outer_scores,
            sha256=hashv,per_track=changes))
        print("ONSET_FN_RECOVERY_CAL_FROZEN_OUTER_PROXY",json.dumps(dict(
            outer=outer,cal=cal,policy=policy,cut=cut,enabled=enabled,
            cal_f1=chosen["score"]["f1"] if chosen else cal_baseline["f1"],
            outer_baseline=outer_baseline,outer_proxy=outer_scores)),flush=True)
    oof=metrics(*[sum(f["outer_proxy_score"][k] for f in folds)
                   for k in ("tp","fp","fn")])
    report=dict(schema=SCHEMA,complete=True,
        genuinely_trained_independent_fourth_head=True,
        neural_models_trained=15,
        frozen_calibration_compositions=True,
        trained_hop_samples=256,decision_hop_samples=256,
        candidate_oracle_coverage=stat,
        original_s613=HISTORIC_S613,prior_three_head_mlp69=baseline,
        FNRecovery_event_only_proxy=oof,folds=folds,
        true_causal_audio_replay_completed=False,
        teacher_train_overlap_known=True,
        full_external_generalization=False,
        no_midi_official_change=True,other_axes_unchanged=True,promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_FN_RECOVERY_4TH_HEAD_5FOLD_PROXY_FINAL",
          json.dumps(dict(baseline=baseline,proxy=oof,
                 selections=[dict(outer=f["outer"],policy=f["selected_policy"],
                        cut=f["cal_cut"],enabled=f["enabled"]) for f in folds])),
          flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for name in ("candidates","main-events","old-risk-events","old-models",
                 "traces","manifest","out"):
        p.add_argument("--"+name,type=Path,required=True)
    x=p.parse_args()
    return run(x.candidates,x.main_events,x.old_risk_events,x.old_models,
               x.traces,x.manifest,x.out)
if __name__=="__main__":main()
