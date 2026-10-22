"""Train Onset ERROR-PREDICTION head on genuine 256-sample hard negatives.

Three independent MLP hypotheses compete on CAL ONLY:
  balanced     -- learn better global FP/TP separation
  harmonic     -- over-sample confirmed FP with overtone competition
  restrike     -- over-sample confirmed FP from active/rearmed notes
Each 5-fold outer composition is never seen in risk training nor calibration.
The frozen S613 and S701 source heads do overlap historical FIT80; this run
must NEVER be advertised as independent generalization of the full pipeline.

No fake offline 'improvement': final policy goes through full causal 256 PCM
replay with the Onset ErrorPredictionHead wired ahead of private selector.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from onset256.error_evidence69 import DIM,SCHEMA,hard_event_vector
from scripts.train_onset_predecision256 import (
    COMPOSITIONS,SEED,S613_EXPECTED,CUTS,
    signed_truth,metrics,score_subset)

VARIANTS=("balanced","harmonic","restrike")
# Selection uses only held-in-CAL composition, never tested on outer results.
CAL_CUTS=(0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8)
FEATURE_RETRIGGER_ACTIVE=63
FEATURE_PARENT_STRENGTH=1
FEATURE_OWN_STRENGTH=0
FEATURE_REFRACTORY=68

def records_from_preselection(events,truth,comps,samples):
    records=[];groups=defaultdict(list)
    for line in Path(events).open():
        if not line.strip():continue
        item=json.loads(line)
        if item.get("policy")!="native" or item.get("label") not in ("TP","FP"):
            continue
        name=item.get("track")
        if name not in truth or item.get("composition")!=comps[name]:
            raise ValueError("wrong signed training origin for hard negatives")
        pitch=item.get("pitch");end=item.get("sample")
        if (type(end) is not int or end%256 or not 0<end<=samples[name]
            or type(pitch) is not int or not 40<=pitch<=88):
            raise ValueError("not a genuine frozen native256 Onset")
        x=hard_event_vector(item)
        if (item.get("predecision_vector") is None or
            not np.array_equal(np.asarray(item["predecision_vector"],np.float32),x[:63])):
            raise ValueError("69D/63D ABI mismatch; labels must be after forward pass")
        i=len(records)
        records.append(dict(name=name,comp=comps[name],sample=end,pitch=pitch,
                            X=x,y=int(item["label"]=="TP"),
                            texture=item["texture"],
                            hard_harmonic=bool(item["context"]["strongest_lower_fundamental"] >
                                               item["context"]["own_fundamental"])))
        groups[name].append(i)
    if set(groups)!=set(truth):
        raise ValueError("must cover all 80 signed recordings")
    counts=dict(tp=sum(r["y"] for r in records),
                fp=sum(1-r["y"] for r in records))
    if counts!={"tp":8099,"fp":4600}:
        raise AssertionError("original S613 labels have changed: "+str(counts))
    events=[SimpleNamespace(pitch=r["pitch"],sample=r["sample"]) for r in records]
    return records,groups,events

def weighted_training_indices(rows,indices,kind,seed):
    """Deterministic hard-negative oversampling, never touches CAL/OUTER labels."""
    if kind not in VARIANTS:raise ValueError("unknown pretrained specialist")
    out=[]
    categories=Counter()
    for i in indices:
        row=rows[i];x=row["X"];y=row["y"]
        harmonic=bool(row["hard_harmonic"]) and row["pitch"]>=60
        retrigger=bool(x[FEATURE_RETRIGGER_ACTIVE]>.5)
        # Class weight is expressed as resampling (sklearn 1.4 MLP.fit lacks
        # sample_weight). Extra duplicates ONLY if the label really is FP.
        repeats=1 if y else 2
        if kind=="harmonic" and not y and harmonic and row["texture"]=="comp":
            repeats=5
            categories["harmonic_fp"]+=1
        elif kind=="restrike" and not y and retrigger:
            repeats=5
            categories["active_restrike_fp"]+=1
        elif kind=="restrike" and not y and x[FEATURE_REFRACTORY]<1.:
            repeats=3
            categories["near_refractory_fp"]+=1
        out.extend([i]*repeats)
    rng=np.random.default_rng(seed)
    rng.shuffle(out)
    return out,dict(categories)

def train_expert(rows,indices,kind,seed):
    selected,specialists=weighted_training_indices(rows,indices,kind,seed)
    x=np.asarray([rows[i]["X"] for i in selected],np.float32)
    y=np.asarray([rows[i]["y"] for i in selected],np.int8)
    if x.ndim!=2 or x.shape[1]!=DIM or set(y.tolist())!={0,1}:
        raise ValueError("invalid 69D hardnegative training matrix")
    nn=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",
        solver="adam",alpha=.012,batch_size=256,learning_rate_init=.001,
        max_iter=65,tol=.0001,n_iter_no_change=12,
        early_stopping=True,validation_fraction=.12,
        random_state=seed)
    pipeline=make_pipeline(StandardScaler(),nn)
    pipeline.fit(x,y)
    if pipeline.n_features_in_!=DIM:
        raise AssertionError("trained neural head dimensions drift")
    return pipeline,dict(training_rows=len(indices),oversampled_rows=len(selected),
                         positives=int(np.count_nonzero(y)),negatives=int(len(y)-np.count_nonzero(y)),
                         specialists=specialists)

def train_report(events,traces,manifest,out):
    if out.exists():raise FileExistsError("scientific reports immutable")
    truth,comps,samples=signed_truth(traces,manifest)
    rows,groups,real_events=records_from_preselection(events,truth,comps,samples)
    all_track_names=tuple(sorted(truth))
    expected,per_track=score_subset(rows,groups,real_events,truth,all_track_names,
                                     np.ones(len(rows),np.float32),0.)
    if any(expected[k]!=v for k,v in S613_EXPECTED.items()):
        raise AssertionError("lost frozen S613 native256 reference")
    out.mkdir(parents=True)
    save=out/"checkpoints";save.mkdir()
    name_by_comp={c:tuple(sorted(k for k in truth if comps[k]==c))
                  for c in COMPOSITIONS}
    fold_data=[]
    for outer_i,outer in enumerate(COMPOSITIONS):
        cal=COMPOSITIONS[(outer_i+1)%len(COMPOSITIONS)]
        train_comps=tuple(c for c in COMPOSITIONS if c not in (outer,cal))
        indexes=[i for i,r in enumerate(rows) if r["comp"] in train_comps]
        outer_indexes=[i for i,r in enumerate(rows) if r["comp"]==outer]
        if (len(set(rows[i]["name"] for i in indexes))!=48
            or len(name_by_comp[cal])!=16 or len(name_by_comp[outer])!=16):
            raise AssertionError("risk training outer composition leakage")
        cal_base,_=score_subset(rows,groups,real_events,truth,name_by_comp[cal],
                                np.ones(len(rows),np.float32),0.)
        outer_base,_=score_subset(rows,groups,real_events,truth,name_by_comp[outer],
                                  np.ones(len(rows),np.float32),0.)
        models={};candidates=[]
        for kind in VARIANTS:
            model,stat=train_expert(rows,indexes,kind,SEED+outer_i+1000*VARIANTS.index(kind))
            models[kind]=model
            q=np.asarray(model.predict_proba(
                np.stack([r["X"] for r in rows if r["comp"] in (cal,outer)]))[:,1],
                np.float32)
            relevant=[i for i,r in enumerate(rows) if r["comp"] in (cal,outer)]
            correct=np.ones(len(rows),np.float32)
            correct[relevant]=q
            if not np.isfinite(correct).all() or np.any(correct<0) or np.any(correct>1):
                raise ValueError("invalid learned error risk score")
            y_outer=np.asarray([rows[i]["y"] for i in outer_indexes],np.int8)
            score_outer=correct[outer_indexes]
            auc=float(roc_auc_score(y_outer,score_outer))
            brier=float(brier_score_loss(y_outer,score_outer))
            for cut in CAL_CUTS[1:]:
                r,tracks=score_subset(rows,groups,real_events,truth,name_by_comp[cal],
                                      correct,cut)
                loss=cal_base["tp"]-r["tp"]
                gain=cal_base["fp"]-r["fp"]
                # CAL-only guard: 1.5% maximum TP loss, at least 5 rescued FP.
                if (r["f1"]>cal_base["f1"]+1e-4 and
                    loss<=.015*cal_base["tp"] and gain>=5):
                    candidates.append(dict(kind=kind,cut=cut,cal=r,
                        cal_lost_tp=loss,cal_removed_fp=gain,
                        cal_worse_tracks=sum(tracks[n]["f1"]<per_track[n]["f1"]-1e-12
                                             for n in name_by_comp[cal]),
                        outer_auc=auc,outer_brier=brier))
            print("ONSET_HARDNEGATIVE_NEURAL_TRAINED",json.dumps(dict(
                outer=outer,cal=cal,expert=kind,**stat,outer_auc=auc,
                outer_brier=brier)),flush=True)
        if candidates:
            selected=max(candidates,key=lambda row:(row["cal"]["f1"],
                                                    -row["cal_lost_tp"],
                                                    -row["cal_worse_tracks"],
                                                    -VARIANTS.index(row["kind"])))
            kind,cut=selected["kind"],selected["cut"]
        else:
            # Still save a genuinely fitted learned head; cut 0 means its
            # opinion is reported but its veto is not admitted by CAL.
            kind,cut="balanced",0.
            selected=None
        chosen=models[kind]
        confidence=np.ones(len(rows),np.float32)
        confidence[outer_indexes]=chosen.predict_proba(
            np.stack([rows[i]["X"] for i in outer_indexes]))[:,1]
        scored,tracks=score_subset(rows,groups,real_events,truth,name_by_comp[outer],
                                   confidence,cut)
        chk=save/f"onset_error_outer_{outer}.joblib"
        joblib.dump(chosen,chk,compress=3)
        hashv=hashlib.sha256(chk.read_bytes()).hexdigest()
        meta=dict(schema=SCHEMA,training_hop_samples=256,
            decision_hop_samples=256,feature_dim=DIM,
            checkpoint_sha256=hashv,
            frozen_before_outer=True,
            train_compositions=train_comps,cal_composition=cal,
            outer_composition=outer,
            selected_expert=kind,calibrated_cut=float(cut),
            calibrated_on_only="CAL_FROZEN_BEFORE_OUTER",
            external_outer_for_frozen_S613_teacher=False,
            trained_on_labelled_native256_notes=True)
        (save/f"onset_error_outer_{outer}.json").write_text(json.dumps(meta,indent=2)+"\n")
        fold_data.append(dict(outer=outer,cal=cal,train=train_comps,
            selected_expert=kind,calibrated_cut=cut,
            cal_selection=selected,cal_baseline=cal_base,outer_baseline=outer_base,
            outer_policy=scored,
            outer_events=len(outer_indexes),
            changed_tracks=[dict(name=name,f1=float(value["f1"]),
                         reference=float(per_track[name]["f1"]))
                         for name,value in tracks.items()
                         if abs(value["f1"]-per_track[name]["f1"])>1e-12],
            signed_sha256=hashv))
        print("ONSET_HARDNEGATIVE_SELECTION_CAL_FROZEN",json.dumps(dict(
            outer=outer,cal=cal,kind=kind,cut=cut,
            baseline_f1=outer_base["f1"],filtered_f1=scored["f1"])),flush=True)
    oof=metrics(*(sum(f["outer_policy"][k] for f in fold_data) for k in ("tp","fp","fn")))
    result=dict(schema="ONSET_ERROR_HEAD_HARDNEGATIVE_5FOLD_PCM256_V1",
        complete=True,trained_neural_weights=True,private_axis="Onset",
        training_hop_samples=256,forward_hop_samples=256,
        five_composition_disjoint_new_head=True,
        old_s613_s701_teacher_train_overlap=True,independent_full_pipeline_test=False,
        source_baseline=expected,oof_event_filter_proxy=oof,
        folds=fold_data,checkpoint_names=sorted(p.name for p in save.iterdir()),
        official_midi_changed=False,other_axes_changed=False,promoted=False,
        proxy_is_not_live_result=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("ONSET_HARDNEGATIVE_MLP69_OOF_PROXY_FINAL",json.dumps({
        "native":expected,"proxy":oof,
        "selected":[dict(outer=f["outer"],kind=f["selected_expert"],
                         cut=f["calibrated_cut"]) for f in fold_data],
        "promoted":False}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ("events","traces","manifest","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    train_report(a.events,a.traces,a.manifest,a.out)
if __name__=="__main__":main()
