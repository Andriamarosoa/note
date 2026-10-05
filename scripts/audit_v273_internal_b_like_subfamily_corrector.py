"""Internal-only B-like subfamily correction audit.

Builds on scripts/audit_v273_internal_b_like_boundary_corrector.py.
For each rotating validation fold:
  1) replay robust group [42,52,61,64];
  2) identify coarse B-like on FIT only;
  3) split FIT B-like K2/K3/K4 failures into two unsupervised subfamilies;
  4) canonically name them:
       B_low  = higher under-count rate on FIT failures
       B_high = the complementary subfamily
     (naming only, clustering never receives labels);
  5) train separate K2/K3/K4 logistic correctors per subfamily;
  6) evaluate B_low only, B_high only, and both on the held-out internal fold.

Outer fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS, NEURONS, C_GRID, WEIGHTS, SEED,
    discover_reports, nested_base, predict, fold_ids,
    structural_features, exact_counts, deltas,
)
from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

SUB_NAMES=("B_low","B_high")
APPLY_MODES=("B_low","B_high","both")

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
    uw=fold_dir/"uniform.weights.h5"; fw=fold_dir/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),"missing checkpoints")

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold)
    y_all=np.minimum(cache["exact"].astype(np.int32),6)
    y_fit=y_all[fit]; y_val=y_all[val]

    mu=build_model("learned_gate",SEED); mu.load_weights(uw)
    mg=build_model("learned_gate",SEED); mg.load_weights(fw)
    bu=nested_base(mu); bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ids=np.asarray(NEURONS,np.int64)
    kg[:,ids]=ku[:,ids]; bg_bias[ids]=bu_bias[ids]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])

    Pf,Gf=predict(mg,cache,fit)
    Pv,Gv=predict(mg,cache,val)
    Xf=structural_features(cache,fit,Pf)
    Xv=structural_features(cache,val,Pv)

    # Coarse B-like, exactly as the previous audit.
    fail_fit=np.isin(y_fit,(2,3,4))&(Gf!=y_fit)
    coarse=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+a.val_fold))
    ])
    coarse.fit(Xf[fail_fit])
    cf=coarse.predict(Xf); cv=coarse.predict(Xv)
    coarse_stats={}
    for c in (0,1):
        m=fail_fit&(cf==c)
        require(int(m.sum())>20,"small coarse cluster")
        under=int(np.sum(Gf[m]<y_fit[m]))
        coarse_stats[c]={
          "rows":int(m.sum()),
          "under":under,
          "over":int(np.sum(Gf[m]>y_fit[m])),
          "under_rate":float(under/max(1,int(m.sum())))
        }
    a_like=max((0,1),key=lambda c:(coarse_stats[c]["under_rate"],coarse_stats[c]["rows"]))
    b_like=1-a_like

    # Subcluster B-like failures only, label-free.
    b_fail=fail_fit&(cf==b_like)
    require(int(b_fail.sum())>=100,"too few B-like failures")
    sub=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420+a.val_fold))
    ])
    sub.fit(Xf[b_fail])
    sf=sub.predict(Xf); sv=sub.predict(Xv)

    sub_stats={}
    for c in (0,1):
        m=b_fail&(sf==c)
        require(int(m.sum())>20,"small B subcluster")
        under=int(np.sum(Gf[m]<y_fit[m]))
        sub_stats[c]={
          "rows":int(m.sum()),
          "under":under,
          "over":int(np.sum(Gf[m]>y_fit[m])),
          "under_rate":float(under/max(1,int(m.sum()))),
          "true_k":{str(k):int(np.sum(y_fit[m]==k)) for k in (2,3,4)},
          "pred_k":{str(k):int(np.sum(Gf[m]==k)) for k in (2,3,4)}
        }
    low_id=max((0,1),key=lambda c:(sub_stats[c]["under_rate"],sub_stats[c]["rows"]))
    high_id=1-low_id
    name_to_id={"B_low":low_id,"B_high":high_id}

    rows=[]
    for cw in WEIGHTS:
      for C in C_GRID:
        models={}
        for name,cid in name_to_id.items():
            train=np.isin(y_fit,(2,3,4))&(cf==b_like)&(sf==cid)
            require(int(train.sum())>=80,f"too few train rows {name}")
            models[name]=fit_lr(Xf[train],y_fit[train],C,cw,27500+a.val_fold+cid)
        for mode in APPLY_MODES:
            new=Gv.copy()
            touched=np.zeros(len(val),dtype=bool)
            use=SUB_NAMES if mode=="both" else (mode,)
            by_sub={}
            for name in use:
                cid=name_to_id[name]
                m=(cv==b_like)&(sv==cid)&np.isin(Gv,(2,3,4))
                q=models[name].predict(Xv[m]).astype(np.int32)
                new[m]=q; touched|=m
                by_sub[name]=int(m.sum())
            rec={
              "config_id":f"{mode}|C={C:g}|cw={cw or 'none'}",
              "mode":mode,"C":float(C),"class_weight":cw or "none",
              "touched":int(touched.sum()),
              "touched_by_subfamily":by_sub,
              **deltas(y_val,Gv,new),
              "metrics":metrics(y_val,new),
              "by_k_net":{str(k):int(np.sum((new==y_val)&(y_val==k))-np.sum((Gv==y_val)&(y_val==k))) for k in range(7)}
            }
            rows.append(rec)

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_like_subfamily_corrector",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "base":"freeze_local_combo + hidden1 [42,52,61,64]",
        "coarse_B_like":"same unsupervised 2-cluster definition as prior B-like audit",
        "subclustering":"2-cluster KMeans on FIT B-like K234 failures only",
        "subfamily_naming":"B_low = higher undercount-rate subcluster on FIT; B_high = complement",
        "corrector":"separate multinomial K2/K3/K4 logistic regressions per subfamily",
        "application":"B_low only, B_high only, or both; only when current prediction in {2,3,4}",
        "automatic_promotion":False
      },
      "coarse_failure_clusters":{str(k):v for k,v in coarse_stats.items()},
      "A_like_cluster":int(a_like),"B_like_cluster":int(b_like),
      "B_subclusters":{str(k):v for k,v in sub_stats.items()},
      "B_low_cluster":int(low_id),"B_high_cluster":int(high_id),
      "reference":{"group_metrics":metrics(y_val,Gv),"counts":exact_counts(y_val,Gv)},
      "configs":rows
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(rows,key=lambda x:(x["global_net"],x["poly_net"],-x["regressions"]),reverse=True)[:20]
    lines=[
      f"# Internal B-like subfamily audit — fold {a.val_fold}","",
      f"Coarse clusters: {coarse_stats}; B-like={b_like}.",
      f"B subclusters: {sub_stats}; B_low={low_id}, B_high={high_id}.","",
      "| config | touched | global net | poly net | corrections | regressions |",
      "|---|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        lines.append(f"| {x['config_id']} | {x['touched']} | {x['global_net']:+d} | {x['poly_net']:+d} | {x['corrections']} | {x['regressions']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
