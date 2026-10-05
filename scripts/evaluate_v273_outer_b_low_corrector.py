"""One-shot outer evaluation of the internally selected B_low corrector.

Selection is frozen from internal run 37297010313:
  mode=B_low, C=0.01, class_weight=none.

No outer search, threshold tuning, or model selection occurs here.
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
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    nested_base,predict,structural_features,exact_counts,deltas,FOLDS,NEURONS,SEED
)

SELECTED_C=0.01
SELECTED_MODE="B_low"

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def fit_lr(X,y):
    p=Pipeline([
      ("scale",StandardScaler()),
      ("lr",LogisticRegression(
        C=SELECTED_C,max_iter=3000,solver="lbfgs",
        class_weight=None,random_state=27610
      ))
    ])
    p.fit(X,y)
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--uniform-weights",type=Path,required=True)
    ap.add_argument("--freeze-weights",type=Path,required=True)
    ap.add_argument("--outer-group-predictions",type=Path,required=True)
    ap.add_argument("--selection-report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    sel=json.loads(a.selection_report.read_text())
    require(sel["outer_fold_3_used"] is False,"selection leaked outer")
    s=sel["selected_config"]
    require(s is not None and s["strict_robust"] is True,"no robust internal selection")
    require(s["config_id"]=="B_low|C=0.01|cw=none","selection drift")
    require(s["positive_global_folds"]==3 and s["min_global_net"]==0,"selection stats drift")
    require(s["total_global_net"]==19 and s["total_poly_net"]==19,"selection totals drift")

    cfg=load_config(a.config)
    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    y=np.minimum(cache["exact"].astype(np.int32),6)
    y_fit=y[fit]; y_out=y[outer]

    # Build final base and apply the already validated hidden1 group.
    mu=build_model("learned_gate",SEED);mu.load_weights(a.uniform_weights)
    mg=build_model("learned_gate",SEED);mg.load_weights(a.freeze_weights)
    bu=nested_base(mu);bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ids=np.asarray(NEURONS,np.int64)
    kg[:,ids]=ku[:,ids];bg_bias[ids]=bu_bias[ids]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])

    Pf,Gf=predict(mg,cache,fit)
    Po,Go=predict(mg,cache,outer)

    frozen=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(frozen["global_index"],outer)
    np.testing.assert_array_equal(frozen["k"],y_out)
    np.testing.assert_array_equal(frozen["predicted"],Go)
    np.testing.assert_allclose(frozen["probability"],Po,rtol=0,atol=1e-6)

    Xf=structural_features(cache,fit,Pf)
    Xo=structural_features(cache,outer,Po)

    # Coarse B-like on all internal final-fit failures only.
    fail_fit=np.isin(y_fit,(2,3,4))&(Gf!=y_fit)
    require(int(fail_fit.sum())>=100,"too few internal failures")
    coarse=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=50,random_state=27370))
    ])
    coarse.fit(Xf[fail_fit])
    cf=coarse.predict(Xf);co=coarse.predict(Xo)
    coarse_stats={}
    for c in (0,1):
        m=fail_fit&(cf==c)
        under=int(np.sum(Gf[m]<y_fit[m]))
        coarse_stats[c]={"rows":int(m.sum()),"under":under,"over":int(np.sum(Gf[m]>y_fit[m])),
                         "under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(coarse_stats[c]["under_rate"],coarse_stats[c]["rows"]))
    b_like=1-a_like

    # Split B-like failures and define B_low on internal labels only.
    b_fail=fail_fit&(cf==b_like)
    require(int(b_fail.sum())>=100,"too few B-like failures")
    sub=Pipeline([
      ("scale",StandardScaler()),
      ("km",KMeans(n_clusters=2,n_init=80,random_state=27420))
    ])
    sub.fit(Xf[b_fail])
    sf=sub.predict(Xf);so=sub.predict(Xo)
    sub_stats={}
    for c in (0,1):
        m=b_fail&(sf==c)
        under=int(np.sum(Gf[m]<y_fit[m]))
        sub_stats[c]={"rows":int(m.sum()),"under":under,"over":int(np.sum(Gf[m]>y_fit[m])),
                      "under_rate":float(under/max(1,int(m.sum())))}
    low_id=max((0,1),key=lambda c:(sub_stats[c]["under_rate"],sub_stats[c]["rows"]))

    train=np.isin(y_fit,(2,3,4))&(cf==b_like)&(sf==low_id)
    require(int(train.sum())>=100,"too few B_low train rows")
    clf=fit_lr(Xf[train],y_fit[train])

    apply=(co==b_like)&(so==low_id)&np.isin(Go,(2,3,4))
    corrected=Go.copy()
    corrected[apply]=clf.predict(Xo[apply]).astype(np.int32)

    base_counts=exact_counts(y_out,Go)
    new_counts=exact_counts(y_out,corrected)
    paired=deltas(y_out,Go,corrected)
    by_k={str(k):int(np.sum((corrected==y_out)&(y_out==k))-np.sum((Go==y_out)&(y_out==k))) for k in range(7)}

    report={
      "status":"completed",
      "protocol":{
        "experiment":"v273_outer_B_low_one_shot",
        "outer_fold":3,
        "outer_selection_or_tuning":False,
        "internal_selection_run":37297010313,
        "selected_config":"B_low|C=0.01|cw=none",
        "base":"freeze_local_combo + candidate_hidden1 [42,52,61,64]",
        "fit_population":"all final internal folds 0/1/2/4",
        "automatic_promotion":False
      },
      "internal_fit":{
        "coarse_failure_clusters":{str(k):v for k,v in coarse_stats.items()},
        "A_like_cluster":int(a_like),
        "B_like_cluster":int(b_like),
        "B_subclusters":{str(k):v for k,v in sub_stats.items()},
        "B_low_cluster":int(low_id),
        "B_low_training_rows":int(train.sum())
      },
      "outer":{
        "rows":int(len(outer)),
        "B_low_applied_rows":int(apply.sum()),
        "base_metrics":metrics(y_out,Go),
        "corrected_metrics":metrics(y_out,corrected),
        "base_counts":base_counts,
        "corrected_counts":new_counts,
        "paired_vs_base":paired,
        "by_k_net":by_k
      }
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(
      a.output/"predictions.npz",
      global_index=outer,k=y_out,
      base_probability=Po,
      base_predicted=Go,
      corrected_predicted=corrected,
      B_low_mask=apply.astype(np.uint8),
      coarse_cluster=co.astype(np.int8),
      sub_cluster=so.astype(np.int8),
    )

    bm=report["outer"]["base_metrics"];cm=report["outer"]["corrected_metrics"]
    lines=[
      "# One-shot outer evaluation — B_low corrector","",
      "Selection fixed internally from run 37297010313. No outer tuning.","",
      "| Measure | Base [42,52,61,64] | + B_low corrector | Delta |",
      "|---|---:|---:|---:|",
      f"| exact global | {100*bm['exact']:.3f}% | {100*cm['exact']:.3f}% | {100*(cm['exact']-bm['exact']):+.3f} pt |",
      f"| exact poly | {100*bm['poly_exact']:.3f}% | {100*cm['poly_exact']:.3f}% | {100*(cm['poly_exact']-bm['poly_exact']):+.3f} pt |",
      f"| under | {bm['under']} | {cm['under']} | {cm['under']-bm['under']:+d} |",
      f"| over | {bm['over']} | {cm['over']} | {cm['over']-bm['over']:+d} |",
    ]
    for k in range(6):
        x=bm["by_k"][str(k)]["exact"];z=cm["by_k"][str(k)]["exact"]
        lines.append(f"| K{k} exact | {100*x:.3f}% | {100*z:.3f}% | {100*(z-x):+.3f} pt |")
    lines += [
      "",
      f"B_low rows touched on outer: **{int(apply.sum())}**.",
      f"Paired corrections/regressions: **{paired['corrections']}/{paired['regressions']}**.",
      f"Net exact vs base: **{paired['global_net']:+d}**.",
      "By-K net exact: "+", ".join(f"K{k} {by_k[str(k)]:+d}" for k in range(6))+".",
      "",
      "No automatic promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
