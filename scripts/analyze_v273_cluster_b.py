"""Deep diagnostic of first-stage failure Cluster B.

Cluster B is defined as the complementary/larger first-stage unsupervised
K2/K3/K4 failure cluster after Cluster A (highest undercount-rate cluster)
has been identified. This is diagnostic only: no prediction is changed.

Unlike Cluster A, B mixes under- and over-counting, so the audit reports
both directions explicitly and subclusters B without using true/predicted K.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.analyze_v273_cluster_a import (
    FEATURES, find_one, load_npz, load_assignments, choose_cluster_a,
    matrix, expected_k, compare_to_correct,
)

SEED=27352

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--low-k",type=Path,required=True)
    ap.add_argument("--acoustic",type=Path,required=True)
    ap.add_argument("--clustering",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)

    context=load_npz(find_one(a.low_k,"row-context.npz"))
    acoustic=load_npz(find_one(a.acoustic,"rows.npz"))
    probes=load_npz(find_one(a.low_k,"frames-31-uniform-probes.npz"))
    assignments=load_assignments(a.clustering/"v273-failure-clustering-rows.csv")

    np.testing.assert_array_equal(context["global_index"],acoustic["global_index"])
    true_k=acoustic["k"].astype(int)
    pred=acoustic["predicted"].astype(int)
    gid=acoustic["global_index"].astype(np.int64)
    gid_to_row={int(v):i for i,v in enumerate(gid)}

    cluster_a,stats=choose_cluster_a(assignments)
    candidates=[c for c in stats if c!=cluster_a]
    if not candidates:
        raise RuntimeError("no complementary first-stage cluster")
    cluster_b=max(candidates,key=lambda c:stats[c]["n"])
    rows=np.array([
        gid_to_row[int(r["global_index"])]
        for r in assignments if int(r["cluster"])==cluster_b
    ],dtype=int)

    X,names=matrix(acoustic,context)
    scaler=StandardScaler().fit(X[rows])
    Z=scaler.transform(X[rows])

    cand=[];best=None
    for k in range(2,7):
        model=KMeans(n_clusters=k,n_init=50,random_state=SEED)
        labels=model.fit_predict(Z)
        counts=np.bincount(labels,minlength=k)
        sil=float(silhouette_score(Z,labels))
        cand.append({"k":k,"silhouette":sil,"min_rows":int(counts.min()),"max_rows":int(counts.max())})
        admissible=counts.min()>=20
        if admissible and (best is None or sil>best[0]):
            best=(sil,model,labels)
    if best is None:
        k=max(cand,key=lambda x:x["silhouette"])["k"]
        model=KMeans(n_clusters=k,n_init=50,random_state=SEED)
        labels=model.fit_predict(Z);best=(float(silhouette_score(Z,labels)),model,labels)
    sil,model,labels=best

    full=np.asarray(context["full_candidates"],int)
    retained=np.asarray(context["retained_candidates"],int)
    lb=np.asarray(context["local_birth_max"],float)
    lc=np.asarray(context["local_count_weighted_norm"],float)
    carried=np.asarray(context["carried_notes_at_start"],int)
    offsets=np.asarray(context["offsets_31"],int)

    med_birth={k:float(np.median(lb[(true_k==k)&(pred==k)])) for k in (2,3,4)}
    med_count={k:float(np.median(lc[(true_k==k)&(pred==k)])) for k in (2,3,4)}

    flags={}
    flags["full_candidate_shortage"]=full[rows] < true_k[rows]
    flags["retention_shortage"]=(full[rows]>=true_k[rows])&(retained[rows]<true_k[rows])
    flags["enough_retained_candidates"]=retained[rows]>=true_k[rows]
    flags["weak_birth_vs_sameK_correct_median"]=np.array([lb[r]<med_birth[int(true_k[r])] for r in rows])
    flags["weak_count_vs_sameK_correct_median"]=np.array([lc[r]<med_count[int(true_k[r])] for r in rows])
    flags["carried_note_present"]=carried[rows]>0
    flags["offset_present"]=offsets[rows]>0

    baseline=np.asarray(probes["baseline"],float)
    zero_c=np.asarray(probes["zero_candidate_context"],float)
    zero_s=np.asarray(probes["zero_spectral_context"],float)
    tail_zero=np.asarray(probes["tail_zero"],float)
    base_e=expected_k(baseline)
    candidate_support=base_e-expected_k(zero_c)
    spectral_support=base_e-expected_k(zero_s)
    tail_support=base_e-expected_k(tail_zero)

    summary_flags={n:{"rows":int(v.sum()),"rate":float(v.mean())} for n,v in flags.items()}

    sub=[]
    for cid in range(model.n_clusters):
        loc=np.flatnonzero(labels==cid);rr=rows[loc]
        center=Z[loc].mean(0);order=np.argsort(-np.abs(center))
        trans=Counter(f"{true_k[r]}→{pred[r]}" for r in rr)
        under=pred[rr]<true_k[rr];over=pred[rr]>true_k[rr]
        sub.append({
          "subcluster":int(cid),
          "rows":int(len(rr)),
          "share_of_B":float(len(rr)/len(rows)),
          "true_k":{str(k):int(np.sum(true_k[rr]==k)) for k in (2,3,4)},
          "undercounts":int(under.sum()),
          "overcounts":int(over.sum()),
          "undercount_rate":float(under.mean()),
          "overcount_rate":float(over.mean()),
          "transitions":[{"transition":t,"rows":int(n)} for t,n in trans.most_common(10)],
          "flags":{n:{"rows":int(np.sum(v[loc])),"rate":float(np.mean(v[loc]))} for n,v in flags.items()},
          "support_expectedK":{
            "candidate_context_mean":float(np.mean(candidate_support[rr])),
            "spectral_context_mean":float(np.mean(spectral_support[rr])),
            "tail_frames_mean":float(np.mean(tail_support[rr])),
          },
          "top_standardized_features_within_B":[
            {"feature":names[j],"mean_z":float(center[j])} for j in order[:10]
          ],
        })

    # Direction-specific descriptive signatures inside B.
    direction={}
    for label,mask in (
        ("under",pred[rows]<true_k[rows]),
        ("over",pred[rows]>true_k[rows]),
    ):
        rr=rows[mask]
        center=scaler.transform(X[rr]).mean(0)
        order=np.argsort(-np.abs(center))
        direction[label]={
          "rows":int(len(rr)),
          "true_k":{str(k):int(np.sum(true_k[rr]==k)) for k in (2,3,4)},
          "top_standardized_features_within_B_scale":[
            {"feature":names[j],"mean_z":float(center[j])} for j in order[:10]
          ],
          "support_expectedK":{
            "candidate_context_mean":float(np.mean(candidate_support[rr])),
            "spectral_context_mean":float(np.mean(spectral_support[rr])),
            "tail_frames_mean":float(np.mean(tail_support[rr])),
          }
        }

    result={
      "status":"completed",
      "scope":"deep audit of pre-HPR first-stage failure Cluster B",
      "cluster_a_id":int(cluster_a),
      "cluster_b_id":int(cluster_b),
      "cluster_b_selection":"largest complementary first-stage cluster after A=highest undercount-rate cluster",
      "first_stage_cluster_stats":stats,
      "rows":int(len(rows)),
      "undercounts":int(np.sum(pred[rows]<true_k[rows])),
      "overcounts":int(np.sum(pred[rows]>true_k[rows])),
      "true_k":{str(k):int(np.sum(true_k[rows]==k)) for k in (2,3,4)},
      "transitions":[
        {"transition":t,"rows":int(n)}
        for t,n in Counter(f"{true_k[r]}→{pred[r]}" for r in rows).most_common(12)
      ],
      "flags":summary_flags,
      "support_expectedK":{
        "candidate_context_mean":float(np.mean(candidate_support[rows])),
        "spectral_context_mean":float(np.mean(spectral_support[rows])),
        "tail_frames_mean":float(np.mean(tail_support[rows])),
      },
      "direction_split":direction,
      "subcluster_candidates":cand,
      "selected_subclusters":int(model.n_clusters),
      "selected_silhouette":float(sil),
      "subclusters":sub,
      "feature_comparison_to_sameK_correct":compare_to_correct({**acoustic,**context},true_k,pred,rows),
      "limitations":[
        "Descriptive audit on the frozen historical fold-3 failure clustering.",
        "True/predicted K are excluded from subclustering and used only for interpretation.",
        "Zeroing probes are sensitivity tests, not causal correction rules.",
        "Any correction inspired by this audit must be developed and selected on internal folds only."
      ]
    }
    (a.output/"v273-cluster-b-analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Décomposition du Cluster B","",
      f"Cluster B : **{len(rows)} erreurs**, dont **{result['undercounts']} sous-comptages** et **{result['overcounts']} surcomptages**.",
      f"Vrais K : {result['true_k']}.","",
      "Transitions dominantes : "+", ".join(f"{x['transition']} ({x['rows']})" for x in result["transitions"][:8]),"",
      "## Indices structurels","",
      "| Test | Cas | Part de B |","|---|---:|---:|"
    ]
    for n in ("full_candidate_shortage","retention_shortage","enough_retained_candidates",
              "weak_birth_vs_sameK_correct_median","weak_count_vs_sameK_correct_median",
              "carried_note_present","offset_present"):
        x=summary_flags[n]
        lines.append(f"| {n} | {x['rows']} | {100*x['rate']:.1f}% |")
    lines += ["","## Sensibilité moyenne","",
      f"- candidate context: {result['support_expectedK']['candidate_context_mean']:+.3f} E[K]",
      f"- spectral context: {result['support_expectedK']['spectral_context_mean']:+.3f} E[K]",
      f"- tail frames: {result['support_expectedK']['tail_frames_mean']:+.3f} E[K]","",
      f"Sous-clustering B retenu : **{model.n_clusters}** familles (silhouette **{sil:.4f}**).",""
    ]
    for s in sub:
        lines += [
          f"### B{s['subcluster']} — {s['rows']} cas ({100*s['share_of_B']:.1f}%)",
          f"- under / over : {s['undercounts']} / {s['overcounts']}",
          "- transitions : "+", ".join(f"{x['transition']} ({x['rows']})" for x in s["transitions"][:6]),
          "- signature : "+", ".join(f"{x['feature']} {x['mean_z']:+.2f}σ" for x in s["top_standardized_features_within_B"][:6]),
          ""
        ]
    lines += [
      "## Portée","",
      "Cet audit décrit B. Il ne sélectionne aucun correcteur sur le fold 3. La prochaine intervention devra être construite et validée sur folds internes 0/1/2/4."
    ]
    (a.output/"v273-cluster-b-analysis.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
