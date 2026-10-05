"""Internal exhaustive transition-guard audit for the selected B_low corrector.

The B_low corrector itself is frozen from internal run 37297010313:
  C=0.01, class_weight=None.

For each held-out internal fold, the proposed corrector changes are partitioned
by all six directed transitions among K2/K3/K4:
  2->3, 2->4, 3->2, 3->4, 4->2, 4->3.

All 63 non-empty subsets of these transition types are evaluated. This is an
exhaustive predefined family; outer fold 3 is never loaded.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_v273_internal_b_like_boundary_corrector import (
    FOLDS, NEURONS, SEED, discover_reports, nested_base, predict, fold_ids,
    structural_features, exact_counts, deltas,
)
from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require

SELECTED_C=0.01
TRANSITIONS=((2,3),(2,4),(3,2),(3,4),(4,2),(4,3))

def fit_lr(X,y,seed):
    p=Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(
        C=SELECTED_C,max_iter=3000,solver="lbfgs",
        class_weight=None,random_state=seed
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
    uw=fold_dir/"uniform.weights.h5";fw=fold_dir/"freeze_local_combo.weights.h5"
    require(uw.exists() and fw.exists(),"missing checkpoints")

    cfg=load_config(a.config)
    cache,_,_=load_bundle(a.bundle,a.config)
    fit,val=fold_ids(cache,cfg,a.val_fold)
    y_all=np.minimum(cache["exact"].astype(np.int32),6)
    y_fit=y_all[fit];y_val=y_all[val]

    # Replay robust hidden1 group.
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

    # Coarse B-like fit-only partition.
    fail_fit=np.isin(y_fit,(2,3,4))&(Gf!=y_fit)
    coarse=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370+a.val_fold))
    ])
    coarse.fit(Xf[fail_fit])
    cf=coarse.predict(Xf);cv=coarse.predict(Xv)
    coarse_stats={}
    for c in (0,1):
        m=fail_fit&(cf==c)
        require(int(m.sum())>20,"small coarse cluster")
        under=int(np.sum(Gf[m]<y_fit[m]))
        coarse_stats[c]={"rows":int(m.sum()),"under":under,
          "over":int(np.sum(Gf[m]>y_fit[m]),
          ),"under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(coarse_stats[c]["under_rate"],coarse_stats[c]["rows"]))
    b_like=1-a_like

    # B_low subfamily fit-only partition.
    b_fail=fail_fit&(cf==b_like)
    sub=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420+a.val_fold))
    ])
    sub.fit(Xf[b_fail])
    sf=sub.predict(Xf);sv=sub.predict(Xv)
    sub_stats={}
    for c in (0,1):
        m=b_fail&(sf==c)
        require(int(m.sum())>20,"small subcluster")
        under=int(np.sum(Gf[m]<y_fit[m]))
        sub_stats[c]={"rows":int(m.sum()),"under":under,
          "over":int(np.sum(Gf[m]>y_fit[m])),
          "under_rate":float(under/max(1,int(m.sum())))}
    low_id=max((0,1),key=lambda c:(sub_stats[c]["under_rate"],sub_stats[c]["rows"]))

    train=np.isin(y_fit,(2,3,4))&(cf==b_like)&(sf==low_id)
    require(int(train.sum())>=80,"too few B_low training rows")
    clf=fit_lr(Xf[train],y_fit[train],27500+a.val_fold+low_id)

    eligible=(cv==b_like)&(sv==low_id)&np.isin(Gv,(2,3,4))
    proposed=Gv.copy()
    proposed[eligible]=clf.predict(Xv[eligible]).astype(np.int32)

    # Enumerate all non-empty subsets of the six directed K234 transitions.
    rows=[]
    for r in range(1,len(TRANSITIONS)+1):
      for combo in itertools.combinations(TRANSITIONS,r):
        allow=np.zeros(len(val),dtype=bool)
        for src,dst in combo:
            allow |= eligible&(Gv==src)&(proposed==dst)
        new=Gv.copy();new[allow]=proposed[allow]
        label="+".join(f"{s}->{d}" for s,d in combo)
        rec={
          "guard_id":label,
          "transitions":[[int(s),int(d)] for s,d in combo],
          "applied_rows":int(allow.sum()),
          **deltas(y_val,Gv,new),
          "metrics":metrics(y_val,new),
          "by_k_net":{str(k):int(np.sum((new==y_val)&(y_val==k))-np.sum((Gv==y_val)&(y_val==k))) for k in range(7)}
        }
        rows.append(rec)

    report={
      "status":"completed","training":False,
      "protocol":{
        "experiment":"v273_internal_B_low_transition_guard",
        "validation_fold":a.val_fold,
        "fit_folds":[x for x in FOLDS if x!=a.val_fold],
        "outer_fold_3_used":False,
        "base":"freeze_local_combo + hidden1 [42,52,61,64]",
        "corrector":"B_low C=0.01 class_weight=None",
        "transition_family":"all 63 non-empty subsets of directed transitions among K2/K3/K4",
        "automatic_promotion":False
      },
      "coarse_failure_clusters":{str(k):v for k,v in coarse_stats.items()},
      "B_like_cluster":int(b_like),
      "B_subclusters":{str(k):v for k,v in sub_stats.items()},
      "B_low_cluster":int(low_id),
      "eligible_rows":int(eligible.sum()),
      "proposed_changed_rows":int(np.sum(eligible&(proposed!=Gv))),
      "reference":{"group_metrics":metrics(y_val,Gv),"counts":exact_counts(y_val,Gv)},
      "guards":rows
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    top=sorted(rows,key=lambda x:(x["global_net"],x["poly_net"],-x["regressions"]),reverse=True)[:20]
    lines=[
      f"# Internal B_low transition guards — fold {a.val_fold}","",
      f"Eligible B_low rows: **{int(eligible.sum())}**; proposed changed rows: **{int(np.sum(eligible&(proposed!=Gv)))}**.",
      f"Guards evaluated: **{len(rows)}**.","",
      "| guard | applied | global net | poly net | corrections | regressions |",
      "|---|---:|---:|---:|---:|---:|"
    ]
    for x in top:
        lines.append(f"| {x['guard_id']} | {x['applied_rows']} | {x['global_net']:+d} | {x['poly_net']:+d} | {x['corrections']} | {x['regressions']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
