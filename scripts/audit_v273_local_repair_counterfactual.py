"""Counterfactual repair audit for the pre-HPR Exact-K local-cardinality defect.

No model is trained. The frozen frames-31-uniform model is replayed. Two
label-oracle counterfactuals are diagnostic upper bounds only:
  1) oracle_v88_local: replace candidate-local V8.8 cardinality/router outputs
     by their true +/-20 ms targets and recompute dependent cluster stats.
  2) oracle_group_interface: provide exact group K through the existing
     local-count statistics to test whether the frozen Exact-K head can use a
     non-capped group-count signal.

Neither counterfactual is an inference-time algorithm.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from causal_note.guitarset import load_boundary_slots
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM, LOCAL_CLUSTER_SAMPLES
from scripts.train_v90_structured_cluster_cardinality import FROZEN_CANDIDATE_DIM
from scripts.train_v260_count_weighting import build_model

def find_one(root: Path, pattern: str) -> Path:
    hits=list(Path(root).rglob(pattern))
    if len(hits)!=1:
        raise RuntimeError(f"expected one {pattern} under {root}, found {len(hits)}")
    return hits[0]

def load_outer(root):
    parts=[]
    for path in sorted(Path(root).rglob("v100-spectral-shard-*.npz")):
        with np.load(path,allow_pickle=False) as z:
            parts.append({k:np.asarray(z[k]) for k in z.files})
    if not parts:
        raise RuntimeError("no outer spectral shards")
    keys=("sequence","mask","stats","spectral","exact","members","cluster_start_samples","candidate_samples")
    out={k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}
    out["members"]=out["members"].astype(str)
    return out

def align_outer(outer, pred):
    outer_keys=[(str(m),int(s)) for m,s in zip(outer["members"],outer["cluster_start_samples"])]
    pred_keys=[(str(m),int(s)) for m,s in zip(pred["member"].astype(str),pred["cluster_start_samples"])]
    if len(set(outer_keys))!=len(outer_keys) or len(set(pred_keys))!=len(pred_keys):
        raise RuntimeError("nonunique row key")
    lut={key:i for i,key in enumerate(outer_keys)}
    if set(lut)!=set(pred_keys):
        raise RuntimeError("outer and prediction populations differ")
    order=np.asarray([lut[k] for k in pred_keys],np.int64)
    return {k:v[order] for k,v in outer.items()}

def annotations(annotation_zip,members):
    result={}
    for member in sorted(set(map(str,members))):
        refs=[b.onset_sample for slot in load_boundary_slots(annotation_zip,member) for b in slot]
        result[member]=np.asarray(sorted(refs),np.int64)
    return result

def local_targets(outer, refs):
    target=np.full(outer["mask"].shape,-1,np.int8)
    for row,(member,mask,samples) in enumerate(zip(outer["members"],outer["mask"],outer["candidate_samples"])):
        keep=np.asarray(mask,bool)
        rr=refs[str(member)]
        for slot,sample in zip(np.flatnonzero(keep),np.asarray(samples)[keep]):
            target[row,slot]=min(3,int(np.sum(np.abs(rr-int(sample))<=LOCAL_CLUSTER_SAMPLES)))
    return target

def inputs(outer, sequence=None, stats=None):
    return {
        "candidate_set":np.asarray(outer["sequence"] if sequence is None else sequence,np.float32),
        "candidate_mask":np.asarray(outer["mask"],np.float32),
        "cluster_stats":np.asarray(outer["stats"] if stats is None else stats,np.float32),
        "spectral_map":np.asarray(outer["spectral"][:,:31],np.float32),
    }

def predict(model,x,batch=128):
    return np.asarray(model.predict(x,batch_size=batch,verbose=0),np.float64)

def oracle_v88_local_inputs(outer,true_local):
    seq=np.asarray(outer["sequence"],np.float32).copy()
    stats=np.asarray(outer["stats"],np.float32).copy()
    mask=np.asarray(outer["mask"],bool)
    if FROZEN_CANDIDATE_DIM != V88_FEATURE_DIM+8:
        raise RuntimeError("unexpected V8.8 candidate layout")
    router_i=V88_FEATURE_DIM
    card0=V88_FEATURE_DIM+1
    fused_i=V88_FEATURE_DIM+7
    for row in range(len(seq)):
        slots=np.flatnonzero(mask[row])
        if not len(slots):
            continue
        t=true_local[row,slots].astype(int)
        if np.any(t<0):
            raise RuntimeError("missing local target")
        seq[row,slots,router_i]=(t>=2).astype(np.float32)
        seq[row,slots,card0:card0+4]=0.
        seq[row,slots,card0+t]=1.
        fused=np.maximum(seq[row,slots,fused_i].astype(np.float64),1e-6)
        stats[row,4]=float(np.mean(t>=2))
        stats[row,5]=float(np.mean(t)/3.)
        stats[row,6]=float(np.sum(t*fused)/np.sum(fused)/3.)
        stats[row,7]=float(np.max(t>0))
    return seq,stats

def oracle_group_interface_stats(outer):
    stats=np.asarray(outer["stats"],np.float32).copy()
    k=np.minimum(np.asarray(outer["exact"],np.float32),6)
    stats[:,4]=(k>=2).astype(np.float32)
    stats[:,5]=k/3.
    stats[:,6]=k/3.
    stats[:,7]=(k>0).astype(np.float32)
    return stats

def metrics(k,pred,select):
    idx=np.flatnonzero(select)
    y=k[idx];p=pred[idx]
    return {
        "rows":int(len(idx)),
        "exact":float(np.mean(p==y)) if len(idx) else None,
        "under":int(np.sum(p<y)),
        "over":int(np.sum(p>y)),
        "mean_error":float(np.mean(p-y)) if len(idx) else None,
        "confusion":np.bincount(y*7+p,minlength=49).reshape(7,7).tolist(),
    }

def transitions(k,a,b,select):
    idx=np.flatnonzero(select)
    y=k[idx];aa=a[idx];bb=b[idx]
    return {
        "rows":int(len(idx)),
        "corrected":int(np.sum((aa!=y)&(bb==y))),
        "regressed":int(np.sum((aa==y)&(bb!=y))),
        "under_to_correct":int(np.sum((aa<y)&(bb==y))),
        "over_to_correct":int(np.sum((aa>y)&(bb==y))),
        "correct_to_under":int(np.sum((aa==y)&(bb<y))),
        "correct_to_over":int(np.sum((aa==y)&(bb>y))),
        "net_correct":int(np.sum(bb==y)-np.sum(aa==y)),
    }

def load_cluster_a(path):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0})
        d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-root",type=Path,required=True)
    ap.add_argument("--outer",type=Path,required=True)
    ap.add_argument("--annotations",type=Path,required=True)
    ap.add_argument("--cluster-rows",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)

    reports=list(Path(a.model_root).rglob("report.json"))
    roots=[p.parent for p in reports if (p.parent/"final"/"predictions.npz").exists() and (p.parent/"final"/"latest.weights.h5").exists()]
    if len(roots)!=1:
        raise RuntimeError(f"expected one model root, found {len(roots)}")
    model_root=roots[0]
    pred_path=model_root/"final"/"predictions.npz"
    weights=model_root/"final"/"latest.weights.h5"
    report=model_root/"report.json"
    training=model_root/"final"/"training.json"
    with np.load(pred_path,allow_pickle=False) as z:
        saved={k:np.asarray(z[k]) for k in z.files}
    train=json.loads(training.read_text())
    rep=json.loads(report.read_text())

    outer=align_outer(load_outer(a.outer),saved)
    np.testing.assert_array_equal(np.minimum(outer["exact"].astype(np.int32),6),saved["k"].astype(np.int32))
    refs=annotations(a.annotations,outer["members"])
    true_local=local_targets(outer,refs)

    model=build_model("uniform",int(train["seed"]),time_frames=31)
    model.load_weights(weights)
    baseline_prob=predict(model,inputs(outer))
    saved_prob=np.asarray(saved["probability"],np.float64)
    max_abs=float(np.max(np.abs(baseline_prob-saved_prob)))
    if not np.array_equal(baseline_prob.argmax(1),saved_prob.argmax(1)) or max_abs>1e-5:
        raise RuntimeError(f"baseline replay mismatch max_abs={max_abs}")
    baseline=baseline_prob.argmax(1).astype(np.int32)

    seq1,stats1=oracle_v88_local_inputs(outer,true_local)
    local_prob=predict(model,inputs(outer,seq1,stats1))
    local_pred=local_prob.argmax(1).astype(np.int32)

    stats2=oracle_group_interface_stats(outer)
    group_prob=predict(model,inputs(outer,None,stats2))
    group_pred=group_prob.argmax(1).astype(np.int32)

    k=saved["k"].astype(np.int32)
    global_index=saved["global_index"].astype(np.int64)
    cid,A_ids=load_cluster_a(a.cluster_rows)
    A=np.isin(global_index,list(A_ids))
    k234=np.isin(k,(2,3,4))

    variants={"baseline":baseline,"oracle_v88_local":local_pred,"oracle_group_interface":group_pred}
    result={
        "status":"completed",
        "training":False,
        "model_promotion":False,
        "label_oracle_diagnostic_only":True,
        "source_model_report_status":rep.get("status"),
        "baseline_replay_max_abs_probability_error":max_abs,
        "cluster_A_id":int(cid),
        "cluster_A_rows":int(A.sum()),
        "variants":{},
        "transitions":{},
        "limitations":[
            "Both treatments use annotation labels and are impossible at inference; they are diagnostic upper bounds only.",
            "The frozen Exact-K head was not trained on oracle-replaced V8.8 outputs.",
            "oracle_group_interface may expose local-count stats above their historical training range for K>=4.",
            "No claim of deployable accuracy improvement is made.",
        ],
    }
    for name,p in variants.items():
        result["variants"][name]={
            "all":metrics(k,p,np.ones(len(k),bool)),
            "k234":metrics(k,p,k234),
            "cluster_A":metrics(k,p,A),
            "by_k":{str(v):metrics(k,p,k==v) for v in (2,3,4)},
        }
    for name,p in (("oracle_v88_local",local_pred),("oracle_group_interface",group_pred)):
        result["transitions"][name]={
            "k234":transitions(k,baseline,p,k234),
            "cluster_A":transitions(k,baseline,p,A),
            "by_k":{str(v):transitions(k,baseline,p,k==v) for v in (2,3,4)},
        }

    np.savez_compressed(a.output/"v273-local-repair-counterfactual-predictions.npz",
        global_index=global_index,k=k,baseline=baseline,oracle_v88_local=local_pred,
        oracle_group_interface=group_pred,baseline_probability=baseline_prob,
        oracle_v88_local_probability=local_prob,oracle_group_interface_probability=group_prob)
    (a.output/"v273-local-repair-counterfactual.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Counterfactual repair audit — Exact-K pre-HPR",
        "",
        f"Baseline replay max |ΔP| : **{max_abs:.3g}**.",
        f"Cluster A : **{int(A.sum())}** lignes.",
        "",
        "Les deux traitements utilisent les annotations : **diagnostic uniquement, aucun correcteur deployable**.",
        "",
        "| Variante | A exact | A under | A over | K2 exact | K3 exact | K4 exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        v=result["variants"][name]
        lines.append(
            f"| {name} | {100*v['cluster_A']['exact']:.2f}% | {v['cluster_A']['under']} | {v['cluster_A']['over']} | "
            f"{100*v['by_k']['2']['exact']:.2f}% | {100*v['by_k']['3']['exact']:.2f}% | {100*v['by_k']['4']['exact']:.2f}% |"
        )
    lines += ["","## Corrections/regressions dans A",""]
    for name in ("oracle_v88_local","oracle_group_interface"):
        t=result["transitions"][name]["cluster_A"]
        lines.append(f"- **{name}** : corriges {t['corrected']}, regressions {t['regressed']}, net {t['net_correct']:+d}.")
    (a.output/"v273-local-repair-counterfactual.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
