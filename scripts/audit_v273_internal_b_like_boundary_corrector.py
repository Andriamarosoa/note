"""Internal-only B-like boundary corrector above robust hidden1 group [42,52,61,64].

For each validation fold v in {0,1,2,4}:
  * load the matched checkpoint trained on the other three internal folds;
  * replay freeze_local_combo and transplant robust hidden1 neurons [42,52,61,64];
  * build structural, label-free features from model inputs;
  * fit a 2-cluster KMeans on FIT K2/K3/K4 failures only;
  * name A-like as the cluster with the highest under-count rate on FIT,
    and B-like as the complementary cluster;
  * train a small multinomial logistic corrector on FIT rows assigned B-like
    with true K in {2,3,4};
  * apply it on validation rows assigned B-like whose current prediction is
    in {2,3,4}. Therefore correct K0/K1 predictions are never touched.

Outer fold 3 is never loaded for selection or evaluation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

SEED=v260.SEED+1003
FOLDS=(0,1,2,4)
NEURONS=(42,52,61,64)
C_GRID=(0.01,0.03,0.1,0.3,1.0,3.0)
WEIGHTS=(None,"balanced")

def discover_reports(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        f=r.get("protocol",{}).get("validation_fold")
        if f in FOLDS and "single_neuron" in r:
            require(f not in out,f"duplicate fold {f}")
            out[f]=(p,r)
    require(set(out)==set(FOLDS),f"missing fold reports {out.keys()}")
    return out

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,"nested base mismatch")
    return xs[0]

def predict(model,cache,ids):
    p=np.asarray(model.predict(batches(cache,ids),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(p.shape==(len(ids),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),"bad probabilities")
    return p,p.argmax(1).astype(np.int32)

def fold_ids(cache,cfg,val_fold):
    names=np.asarray(cache["members"]).astype(str)
    f=np.asarray([cfg["member_folds"][m] for m in names],np.int32)
    val=np.flatnonzero(f==val_fold)
    fit=np.flatnonzero(np.isin(f,[x for x in FOLDS if x!=val_fold]))
    outer=np.flatnonzero(f==3)
    require(len(val)>0 and len(fit)>0 and not np.intersect1d(np.r_[fit,val],outer).size,"outer leak")
    return fit,val

def structural_features(cache,ids,prob):
    ids=np.asarray(ids,np.int64)
    stats=np.asarray(cache["stats"][ids],np.float32)
    mask=np.asarray(cache["mask"][ids],np.float32)
    seq=np.asarray(cache["sequence"][ids],np.float32)
    spec=np.asarray(cache["spectral"][ids,:31],np.float32)

    count=mask.sum(1,keepdims=True)
    den=np.maximum(count,1.0)
    # Last eight candidate features contain the V8.8 pseudo-count interface + right-side context.
    tail=seq[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+8]
    m=mask[:,:,None]
    tail_mean=(tail*m).sum(1)/den
    tail_max=np.max(np.where(m>0,tail,-1e9),axis=1)
    tail_max=np.where(np.isfinite(tail_max)&(tail_max>-1e8),tail_max,0.0)

    flat=spec.reshape(len(ids),-1)
    spec_summary=np.column_stack([
        flat.mean(1),flat.std(1),flat.max(1),
        np.percentile(flat,75,axis=1),np.percentile(flat,90,axis=1)
    ]).astype(np.float32)

    # Probabilities are inference-time features and contain no labels.
    return np.column_stack([stats,count,tail_mean,tail_max,spec_summary,prob]).astype(np.float64)

def exact_counts(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {
      "global":int(np.sum(y==p)),
      "low":int(np.sum((y==p)&(y<=1))),
      "poly":int(np.sum((y==p)&(y>=2))),
      "under":int(np.sum(p<y)),
      "over":int(np.sum(p>y)),
    }

def deltas(y,base,new):
    a=exact_counts(y,base);b=exact_counts(y,new)
    return {
      "global_net":b["global"]-a["global"],
      "low_net":b["low"]-a["low"],
      "poly_net":b["poly"]-a["poly"],
      "under_delta":b["under"]-a["under"],
      "over_delta":b["over"]-a["over"],
      "corrections":int(np.sum((base!=y)&(new==y))),
      "regressions":int(np.sum((base==y)&(new!=y))),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"invalid val fold")
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    reports=discover_reports(a.fold_root)
    fold_dir=reports[a.val_fold][0].parent
    uw=fold_dir/"uniform.weights.h5";fw=fold_dir/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),"missing checkpoints")

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold)
    y_all=np.minimum(cache["exact"].astype(np.int32),6)
    y_fit=y_all[fit];y_val=y_all[val]

    mu=build_model("learned_gate",SEED);mu.load_weights(uw)
    mg=build_model("learned_gate",SEED);mg.load_weights(fw)
    bu=nested_base(mu);bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    idx=np.asarray(NEURONS,np.int64)
    kg[:,idx]=ku[:,idx];bg_bias[idx]=bu_bias[idx]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])

    Pf,Gf=predict(mg,cache,fit)
    Pv,Gv=predict(mg,cache,val)
    Xf=structural_features(cache,fit,Pf)
    Xv=structural_features(cache,val,Pv)

    # Clustering uses only structural/inference features; labels only define the FIT failure population.
    fail_fit=np.isin(y_fit,(2,3,4))&(Gf!=y_fit)
    require(int(fail_fit.sum())>=100,"too few fit K234 failures")
    cl_pipe=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+a.val_fold))
    ])
    cl_pipe.fit(Xf[fail_fit])
    cf=cl_pipe.predict(Xf)
    cv=cl_pipe.predict(Xv)

    # Name A-like/B-like on FIT only.
    stats={}
    for c in (0,1):
        m=fail_fit&(cf==c)
        require(int(m.sum())>20,"small cluster")
        under=int(np.sum(Gf[m]<y_fit[m]));over=int(np.sum(Gf[m]>y_fit[m]))
        stats[c]={"rows":int(m.sum()),"under":under,"over":over,
                  "under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(stats[c]["under_rate"],stats[c]["rows"]))
    b_like=1-a_like

    train_mask=np.isin(y_fit,(2,3,4))&(cf==b_like)
    require(int(train_mask.sum())>=150,"too few B-like fit rows")
    # Apply only to rows already predicted 2/3/4; correct K0/K1 can never regress.
    apply_val=(cv==b_like)&np.isin(Gv,(2,3,4))

    configs=[]
    for cw in WEIGHTS:
      for C in C_GRID:
        pipe=Pipeline([
          ("scale",StandardScaler()),
          ("lr",LogisticRegression(
              C=C,max_iter=3000,solver="lbfgs",
              class_weight=cw,random_state=27380+a.val_fold
          ))
        ])
        pipe.fit(Xf[train_mask],y_fit[train_mask])
        q=pipe.predict(Xv[apply_val]).astype(np.int32)
        new=Gv.copy();new[apply_val]=q
        rec={
          "config_id":f"C={C:g}|cw={cw or 'none'}",
          "C":float(C),"class_weight":cw or "none",
          "fit_B_like_rows":int(train_mask.sum()),
          "val_B_like_applied_rows":int(apply_val.sum()),
          **deltas(y_val,Gv,new),
          "metrics":metrics(y_val,new),
          "by_k_net":{str(k):int(np.sum((new==y_val)&(y_val==k))-np.sum((Gv==y_val)&(y_val==k))) for k in range(7)}
        }
        configs.append(rec)

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_like_boundary_corrector",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "base":"freeze_local_combo + hidden1 [42,52,61,64]",
        "cluster_features":"stats + candidate pseudo-count tail aggregates + spectral summaries + current probabilities",
        "cluster_count":2,
        "B_like_definition":"complement of highest-undercount failure cluster, defined on fit only",
        "corrector":"multinomial logistic regression K2/K3/K4",
        "apply_condition":"B-like assignment and current prediction in {2,3,4}",
        "automatic_promotion":False
      },
      "fit_failure_clusters":{str(k):v for k,v in stats.items()},
      "A_like_cluster":int(a_like),"B_like_cluster":int(b_like),
      "reference":{"group_metrics":metrics(y_val,Gv),"counts":exact_counts(y_val,Gv)},
      "configs":configs
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(configs,key=lambda x:(x["global_net"],x["poly_net"],-x["regressions"]),reverse=True)
    lines=[
      f"# Internal B-like boundary corrector — fold {a.val_fold}","",
      f"FIT failure clusters: {stats}; A-like={a_like}, B-like={b_like}.",
      f"Validation B-like rows touched: **{int(apply_val.sum())}**.","",
      "| config | global net | poly net | corrections | regressions | under Δ | over Δ |",
      "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for x in top:
      lines.append(f"| {x['config_id']} | {x['global_net']:+d} | {x['poly_net']:+d} | {x['corrections']} | {x['regressions']} | {x['under_delta']:+d} | {x['over_delta']:+d} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
