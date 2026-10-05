"""Internal K2-vs-not-K2 discriminator inside B_low when base predicts K3.

Goal: keep only the useful 3->2 branch of the B_low corrector and decide,
from internal data only, which B_low/base-K3 rows should actually move to K2.

For each rotating internal validation fold:
  * replay robust hidden1 group [42,52,61,64];
  * reconstruct coarse B-like and B_low using fit-only clustering;
  * train a binary logistic discriminator on FIT rows satisfying
      B_low assignment AND base prediction == K3,
    target = (true K == 2);
  * evaluate a predefined family of regularization/class-weight/probability
    thresholds on the held-out fold;
  * action is ONLY 3->2. No other base prediction can change.

Outer fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS, NEURONS, SEED, discover_reports, nested_base, predict, fold_ids,
    structural_features, exact_counts, deltas,
)
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

C_GRID=(0.01,0.03,0.1,0.3,1.0,3.0)
WEIGHTS=(None,"balanced")
THRESHOLDS=(0.25,0.35,0.45,0.50,0.55,0.65,0.75)

def fit_lr(X,y,C,cw,seed):
    p=Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(
        C=C,max_iter=3000,solver="lbfgs",
        class_weight=cw,random_state=seed
      ))
    ])
    p.fit(X,y)
    return p

def build_blow(Xf,Xv,yf,Gf,val_G,val_fold):
    fail=np.isin(yf,(2,3,4))&(Gf!=yf)
    coarse=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+val_fold))
    ])
    coarse.fit(Xf[fail])
    cf=coarse.predict(Xf);cv=coarse.predict(Xv)
    coarse_stats={}
    for c in (0,1):
        m=fail&(cf==c); require(int(m.sum())>20,"small coarse cluster")
        under=int(np.sum(Gf[m]<yf[m]))
        coarse_stats[c]={"rows":int(m.sum()),"under":under,
                         "over":int(np.sum(Gf[m]>yf[m])),
                         "under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(coarse_stats[c]["under_rate"],coarse_stats[c]["rows"]))
    b_like=1-a_like

    b_fail=fail&(cf==b_like)
    sub=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420+val_fold))
    ])
    sub.fit(Xf[b_fail])
    sf=sub.predict(Xf);sv=sub.predict(Xv)
    sub_stats={}
    for c in (0,1):
        m=b_fail&(sf==c); require(int(m.sum())>20,"small subcluster")
        under=int(np.sum(Gf[m]<yf[m]))
        sub_stats[c]={"rows":int(m.sum()),"under":under,
                      "over":int(np.sum(Gf[m]>yf[m])),
                      "under_rate":float(under/max(1,int(m.sum())))}
    low_id=max((0,1),key=lambda c:(sub_stats[c]["under_rate"],sub_stats[c]["rows"]))
    fit_blow=(cf==b_like)&(sf==low_id)
    val_blow=(cv==b_like)&(sv==low_id)
    return fit_blow,val_blow,coarse_stats,sub_stats,b_like,low_id

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fold-root",type=Path,required=True)
    ap.add_argument("--val-fold",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.val_fold in FOLDS,"invalid fold")
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
    yf=y_all[fit];yv=y_all[val]

    mu=build_model("learned_gate",SEED);mu.load_weights(uw)
    mg=build_model("learned_gate",SEED);mg.load_weights(fw)
    bu=nested_base(mu);bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ids=np.asarray(NEURONS,np.int64)
    kg[:,ids]=ku[:,ids];bg_bias[ids]=bu_bias[ids]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])

    Pf,Gf=predict(mg,cache,fit)
    Pv,Gv=predict(mg,cache,val)
    Xf=structural_features(cache,fit,Pf)
    Xv=structural_features(cache,val,Pv)

    fit_blow,val_blow,coarse_stats,sub_stats,b_like,low_id=build_blow(
        Xf,Xv,yf,Gf,Gv,a.val_fold)

    train=fit_blow&(Gf==3)
    test=val_blow&(Gv==3)
    require(int(train.sum())>=80,"too few B_low/baseK3 fit rows")
    require(int(test.sum())>=10,"too few B_low/baseK3 val rows")
    ty=(yf[train]==2).astype(np.int32)
    vy=(yv[test]==2).astype(np.int32)
    require(len(np.unique(ty))==2,"binary target collapsed")

    configs=[]
    for cw in WEIGHTS:
      for C in C_GRID:
        clf=fit_lr(Xf[train],ty,C,cw,27650+a.val_fold)
        prob=clf.predict_proba(Xv[test])[:,1]
        auc=float(roc_auc_score(vy,prob)) if len(np.unique(vy))==2 else None
        for thr in THRESHOLDS:
            take_local=prob>=thr
            ids_val=np.flatnonzero(test)
            chosen=ids_val[take_local]
            new=Gv.copy();new[chosen]=2
            configs.append({
              "config_id":f"C={C:g}|cw={cw or 'none'}|thr={thr:.2f}",
              "C":float(C),"class_weight":cw or "none","threshold":float(thr),
              "fit_rows":int(train.sum()),"fit_true_k2":int(ty.sum()),
              "val_candidate_rows":int(test.sum()),"val_true_k2":int(vy.sum()),
              "val_auc_k2_vs_rest":auc,
              "applied_rows":int(len(chosen)),
              **deltas(yv,Gv,new),
              "metrics":metrics(yv,new),
              "by_k_net":{str(k):int(np.sum((new==yv)&(yv==k))-np.sum((Gv==yv)&(yv==k))) for k in range(7)}
            })

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_baseK3_K2_discriminator",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "base":"freeze_local_combo + hidden1 [42,52,61,64]",
        "population":"B_low assignment and base prediction K3",
        "target":"true K2 vs not-K2",
        "action":"only 3->2 when predicted P(K2) >= threshold",
        "C_grid":list(C_GRID),
        "class_weights":["none","balanced"],
        "thresholds":list(THRESHOLDS),
        "automatic_promotion":False
      },
      "coarse_failure_clusters":{str(k):v for k,v in coarse_stats.items()},
      "B_like_cluster":int(b_like),
      "B_subclusters":{str(k):v for k,v in sub_stats.items()},
      "B_low_cluster":int(low_id),
      "reference":{"group_metrics":metrics(yv,Gv),"counts":exact_counts(yv,Gv)},
      "configs":configs
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(configs,key=lambda x:(x["global_net"],x["poly_net"],-x["regressions"]),reverse=True)[:24]
    lines=[
      f"# Internal B_low base-K3 discriminator — fold {a.val_fold}","",
      f"Fit B_low/base-K3 rows: **{int(train.sum())}**; true K2: **{int(ty.sum())}**.",
      f"Validation B_low/base-K3 rows: **{int(test.sum())}**; true K2: **{int(vy.sum())}**.",
      f"Configurations: **{len(configs)}**.","",
      "| config | AUC | applied | global net | K2 net | K3 net | K4 net | corrections | regressions |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        auc="n/a" if x["val_auc_k2_vs_rest"] is None else f"{x['val_auc_k2_vs_rest']:.3f}"
        lines.append(
          f"| {x['config_id']} | {auc} | {x['applied_rows']} | {x['global_net']:+d} | "
          f"{x['by_k_net']['2']:+d} | {x['by_k_net']['3']:+d} | {x['by_k_net']['4']:+d} | "
          f"{x['corrections']} | {x['regressions']} |"
        )
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
