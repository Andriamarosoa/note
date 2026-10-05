"""Re-cluster current residual K2/K3/K4 Exact-K failures after the robust hidden1 group.

Diagnostic only. Uses current outer predictions from the one-shot validated
configuration:
  freeze_local_combo + candidate_hidden1 [42,52,61,64]

Clustering features are the same frozen acoustic/context audit features used in
historical V27.3 failure clustering. Current true/predicted K and historical
cluster labels are excluded from clustering and used only for interpretation
and overlap analysis.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.cluster_v273_failures import feature_matrix, load_npz

RANDOM_STATE=27361

def load_current(path:Path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_historical(path:Path):
    out={}
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["global_index"])]=int(r["cluster"])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--low-k",type=Path,required=True)
    ap.add_argument("--acoustic",type=Path,required=True)
    ap.add_argument("--current-predictions",type=Path,required=True)
    ap.add_argument("--historical-rows",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)

    # Locate frozen feature sources.
    low=list(a.low_k.rglob("row-context.npz"))
    ac=list(a.acoustic.rglob("rows.npz"))
    if len(low)!=1 or len(ac)!=1:
        raise RuntimeError(f"feature sources mismatch low={len(low)} ac={len(ac)}")
    context=load_npz(low[0]); acoustic=load_npz(ac[0])
    current=load_current(a.current_predictions)
    hist=load_historical(a.historical_rows)

    gid=np.asarray(acoustic["global_index"],np.int64)
    true=np.asarray(acoustic["k"],np.int32)
    np.testing.assert_array_equal(gid,np.asarray(context["global_index"],np.int64))

    cur_gid=np.asarray(current["global_index"],np.int64)
    cur_true=np.asarray(current["k"],np.int32)
    cur_pred=np.asarray(current["predicted"],np.int32)
    pos={int(g):i for i,g in enumerate(cur_gid)}
    if not all(int(g) in pos for g in gid):
        raise RuntimeError("current predictions missing feature rows")
    order=np.asarray([pos[int(g)] for g in gid],np.int64)
    np.testing.assert_array_equal(cur_true[order],true)
    pred=cur_pred[order]

    X,names=feature_matrix(acoustic,context)
    target=np.isin(true,(2,3,4))
    failures=target&(pred!=true)
    rows=np.flatnonzero(failures)
    if len(rows)<100:
        raise RuntimeError(f"too few current residual failures: {len(rows)}")

    scaler=StandardScaler().fit(X[target])
    Z_all=scaler.transform(X);Z=Z_all[rows]

    candidates=[];best=None
    max_k=min(8,max(2,len(rows)//80))
    for k in range(2,max_k+1):
        model=KMeans(n_clusters=k,n_init=50,random_state=RANDOM_STATE)
        labels=model.fit_predict(Z)
        counts=np.bincount(labels,minlength=k)
        sil=float(silhouette_score(Z,labels))
        candidates.append({"k":k,"silhouette":sil,"min_cluster_rows":int(counts.min()),"max_cluster_rows":int(counts.max())})
        admissible=counts.min()>=max(15,int(0.02*len(rows)))
        if admissible and (best is None or sil>best[0]):
            best=(sil,model,labels)
    if best is None:
        k=max(candidates,key=lambda x:x["silhouette"])["k"]
        model=KMeans(n_clusters=k,n_init=50,random_state=RANDOM_STATE)
        labels=model.fit_predict(Z);best=(float(silhouette_score(Z,labels)),model,labels)
    sil,model,labels=best

    pca=PCA(n_components=2,random_state=RANDOM_STATE)
    xy=pca.fit_transform(Z)

    clusters=[]
    for cid in range(model.n_clusters):
        local=np.flatnonzero(labels==cid);rr=rows[local]
        under=pred[rr]<true[rr];over=pred[rr]>true[rr]
        trans=Counter(f"{int(true[r])}→{int(pred[r])}" for r in rr)
        hist_counts=Counter(hist.get(int(gid[r]),-1) for r in rr)
        center=Z_all[rr].mean(0);orderf=np.argsort(-np.abs(center))
        clusters.append({
          "cluster":int(cid),
          "rows":int(len(rr)),
          "share_of_current_failures":float(len(rr)/len(rows)),
          "true_k":{str(k):int(np.sum(true[rr]==k)) for k in (2,3,4)},
          "undercounts":int(under.sum()),
          "overcounts":int(over.sum()),
          "undercount_rate":float(under.mean()),
          "transitions":[{"transition":t,"rows":int(n)} for t,n in trans.most_common(10)],
          "historical_overlap":{
            "historical_B_cluster0":int(hist_counts.get(0,0)),
            "historical_A_cluster1":int(hist_counts.get(1,0)),
            "not_historical_failure":int(hist_counts.get(-1,0))
          },
          "top_feature_signature":[
            {"feature":names[j],"mean_z":float(center[j])} for j in orderf[:10]
          ]
        })

    # Stable naming only for interpretation: A' = highest undercount-rate cluster;
    # B' = largest complementary cluster. Does not affect clustering.
    aid=max(range(len(clusters)),key=lambda i:(clusters[i]["undercount_rate"],clusters[i]["rows"]))
    others=[i for i in range(len(clusters)) if i!=aid]
    bid=max(others,key=lambda i:clusters[i]["rows"]) if others else None

    # Historical B survival under current model.
    hist_b=[i for i,g in enumerate(gid) if hist.get(int(g),-1)==0]
    hb=np.asarray(hist_b,np.int64)
    historical_b_survival={
      "historical_rows":int(len(hb)),
      "current_correct":int(np.sum(pred[hb]==true[hb])),
      "current_errors":int(np.sum(pred[hb]!=true[hb])),
      "current_under":int(np.sum(pred[hb]<true[hb])),
      "current_over":int(np.sum(pred[hb]>true[hb])),
    }

    result={
      "status":"completed",
      "scope":"current residual fold-3 K2/K3/K4 failures after robust hidden1 group [42,52,61,64]",
      "clustering_inputs_exclude_labels":True,
      "current_source_run":37290794473,
      "rows_total":int(len(true)),
      "target_rows_k234":int(target.sum()),
      "current_failure_rows_k234":int(failures.sum()),
      "candidate_cluster_counts":candidates,
      "selected_clusters":int(model.n_clusters),
      "selected_silhouette":float(sil),
      "cluster_A_prime":int(aid),
      "cluster_B_prime":None if bid is None else int(bid),
      "historical_B_survival":historical_b_survival,
      "clusters":clusters,
      "limitations":[
        "Diagnostic on outer fold 3 only; do not tune a correction on these labels.",
        "Historical feature extraction is reused; only predictions/error membership are updated.",
        "Any intervention inspired by B' must be selected on internal folds 0/1/2/4."
      ]
    }
    (a.output/"v273-current-residual-clustering.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    with (a.output/"v273-current-residual-clustering-rows.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["global_index","true_k","predicted_k","error_delta","cluster","pca1","pca2","historical_cluster"])
        for j,r in enumerate(rows):
            w.writerow([int(gid[r]),int(true[r]),int(pred[r]),int(pred[r]-true[r]),int(labels[j]),float(xy[j,0]),float(xy[j,1]),hist.get(int(gid[r]),-1)])

    lines=[
      "# Re-clustering des erreurs résiduelles après [42,52,61,64]","",
      f"Erreurs K2/K3/K4 actuelles : **{len(rows)}**.",
      f"Clusters sélectionnés : **{model.n_clusters}** (silhouette **{sil:.4f}**).","",
      f"Ancien B : {historical_b_survival['historical_rows']} cas -> **{historical_b_survival['current_errors']} erreurs restantes**, "
      f"**{historical_b_survival['current_correct']} désormais corrects**.",
      f"Ancien B restant : {historical_b_survival['current_under']} under / {historical_b_survival['current_over']} over.","",
      f"A' = cluster {aid}; B' = cluster {bid}.",""
    ]
    for c in clusters:
        lines += [
          f"## Cluster {c['cluster']} — {c['rows']} cas ({100*c['share_of_current_failures']:.1f}%)",
          f"- under / over : {c['undercounts']} / {c['overcounts']}",
          f"- vrais K : {c['true_k']}",
          "- transitions : "+", ".join(f"{x['transition']} ({x['rows']})" for x in c["transitions"][:6]),
          f"- overlap ancien B/A/nouveau : {c['historical_overlap']['historical_B_cluster0']} / "
          f"{c['historical_overlap']['historical_A_cluster1']} / {c['historical_overlap']['not_historical_failure']}",
          ""
        ]
    (a.output/"v273-current-residual-clustering.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
