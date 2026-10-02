"""Deep diagnostic of the low-evidence K2/K3/K4 failure cluster.

Uses the frozen pre-HPR fold-3 audit. The subclustering is unsupervised and
does not use true/predicted K. Labels are used only for post-hoc interpretation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 27341

FEATURES = (
    "retained_candidates","full_candidates","candidate_count_log",
    "local_count_mean_norm","local_count_weighted_norm","local_birth_max",
    "proposal_mean","proposal_max","router_mean",
    "foreign_births_31","foreign_births_added_tail","own_births_added_tail",
    "carried_notes_at_start","overlapping_notes_31","offsets_31",
    "foreign_assigned_births_31","unassigned_births_31",
    "eligible_births","contested_births","max_simultaneous_notes",
    "rms_dbfs","post_minus_pre_db","spectral_flatness",
    "native_positive_peak","native_flux_peak",
)
COUNT_LIKE = {
    "retained_candidates","full_candidates","foreign_births_31",
    "foreign_births_added_tail","own_births_added_tail","carried_notes_at_start",
    "overlapping_notes_31","offsets_31","foreign_assigned_births_31",
    "unassigned_births_31","eligible_births","contested_births",
    "max_simultaneous_notes",
}

def find_one(root: Path, name: str) -> Path:
    hits=list(root.rglob(name))
    if len(hits)!=1:
        raise RuntimeError(f"expected one {name}, found {len(hits)}")
    return hits[0]

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_assignments(path):
    rows=[]
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k:r[k] for k in r})
    return rows

def choose_cluster_a(assignments):
    by={}
    for r in assignments:
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0,"over":0})
        d["n"]+=1; d["under"]+=p<y; d["over"]+=p>y
    cid=max(by, key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    return cid,by

def matrix(acoustic,context):
    cols=[]; names=[]
    for n in FEATURES:
        src=acoustic if n in acoustic else context
        if n not in src: continue
        x=np.asarray(src[n],np.float64).reshape(-1)
        ok=np.isfinite(x)
        if not ok.all():
            repl=np.median(x[ok]) if ok.any() else 0.
            x=np.where(ok,x,repl)
        if n in COUNT_LIKE: x=np.log1p(np.maximum(x,0))
        if np.var(x)<=1e-12: continue
        cols.append(x);names.append(n)
    return np.column_stack(cols),names

def expected_k(prob):
    return prob @ np.arange(prob.shape[1],dtype=np.float64)

def compare_to_correct(values,true_k,pred,rows):
    out={}
    for name,x in values.items():
        if name not in FEATURES: continue
        x=np.asarray(x,np.float64)
        rec={}
        for k in (2,3,4):
            a=rows[true_k[rows]==k]
            c=np.flatnonzero((true_k==k)&(pred==k))
            if len(a) and len(c):
                rec[str(k)]={
                    "cluster_mean":float(np.mean(x[a])),
                    "correct_mean":float(np.mean(x[c])),
                    "delta":float(np.mean(x[a])-np.mean(x[c])),
                }
        out[name]=rec
    return out

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
    true_k=acoustic["k"].astype(int);pred=acoustic["predicted"].astype(int)
    gid=acoustic["global_index"].astype(np.int64)
    gid_to_row={int(v):i for i,v in enumerate(gid)}
    cluster_a,cluster_stats=choose_cluster_a(assignments)
    rows=np.array([gid_to_row[int(r["global_index"])] for r in assignments if int(r["cluster"])==cluster_a],dtype=int)

    X,names=matrix(acoustic,context)
    scaler=StandardScaler().fit(X[rows])
    Z=scaler.transform(X[rows])
    candidates=[];best=None
    for k in range(2,6):
        model=KMeans(n_clusters=k,n_init=50,random_state=SEED)
        labels=model.fit_predict(Z)
        counts=np.bincount(labels,minlength=k)
        sil=float(silhouette_score(Z,labels))
        candidates.append({"k":k,"silhouette":sil,"min_rows":int(counts.min()),"max_rows":int(counts.max())})
        admissible=counts.min()>=20
        if admissible and (best is None or sil>best[0]): best=(sil,model,labels)
    if best is None:
        k=max(candidates,key=lambda x:x["silhouette"])["k"]
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

    sub=[]
    for cid in range(model.n_clusters):
        loc=np.flatnonzero(labels==cid); rr=rows[loc]
        center=Z[loc].mean(0)
        order=np.argsort(-np.abs(center))
        top=[{"feature":names[j],"mean_z":float(center[j])} for j in order[:8]]
        trans=Counter(f"{true_k[r]}→{pred[r]}" for r in rr)
        sub.append({
            "subcluster":cid,
            "rows":int(len(rr)),
            "share_of_A":float(len(rr)/len(rows)),
            "true_k":{str(k):int(np.sum(true_k[rr]==k)) for k in (2,3,4)},
            "undercount_rate":float(np.mean(pred[rr]<true_k[rr])),
            "transitions":[{"transition":t,"rows":int(n)} for t,n in trans.most_common(8)],
            "flags":{n:{"rows":int(np.sum(v[loc])),"rate":float(np.mean(v[loc]))} for n,v in flags.items()},
            "support_expectedK":{
                "candidate_context_mean":float(np.mean(candidate_support[rr])),
                "spectral_context_mean":float(np.mean(spectral_support[rr])),
                "tail_frames_mean":float(np.mean(tail_support[rr])),
            },
            "top_standardized_features_within_A":top,
        })

    enough=retained[rows]>=true_k[rows]
    summary_flags={n:{"rows":int(v.sum()),"rate":float(v.mean())} for n,v in flags.items()}
    result={
        "status":"completed",
        "scope":"deep audit of pre-HPR low-evidence failure cluster A",
        "cluster_a_id":int(cluster_a),
        "cluster_a_selection":"highest undercount share among first-stage unsupervised clusters",
        "first_stage_cluster_stats":cluster_stats,
        "rows":int(len(rows)),
        "undercounts":int(np.sum(pred[rows]<true_k[rows])),
        "overcounts":int(np.sum(pred[rows]>true_k[rows])),
        "flags":summary_flags,
        "candidate_shortage_union":{
            "rows":int(np.sum(retained[rows]<true_k[rows])),
            "rate":float(np.mean(retained[rows]<true_k[rows])),
        },
        "enough_candidates_but_weak_birth":{
            "rows":int(np.sum(enough & flags["weak_birth_vs_sameK_correct_median"])),
            "rate_among_A":float(np.mean(enough & flags["weak_birth_vs_sameK_correct_median"])),
            "rate_among_enough_candidates":float(np.mean(flags["weak_birth_vs_sameK_correct_median"][enough])) if enough.any() else None,
        },
        "support_expectedK":{
            "candidate_context_mean":float(np.mean(candidate_support[rows])),
            "spectral_context_mean":float(np.mean(spectral_support[rows])),
            "tail_frames_mean":float(np.mean(tail_support[rows])),
            "interpretation":"positive means that component raises expected K relative to zeroing it; zeroing is an OOD sensitivity probe, not a causal correction",
        },
        "subcluster_candidates":candidates,
        "selected_subclusters":int(model.n_clusters),
        "selected_silhouette":float(sil),
        "subclusters":sub,
        "feature_comparison_to_sameK_correct":compare_to_correct({**acoustic,**context},true_k,pred,rows),
        "limitations":[
            "One development fold and seed.",
            "Same-K medians are used only after clustering for interpretation.",
            "Zero-latent and tail-zero probes may be out of distribution.",
            "Candidate shortage does not prove the audio itself lacks the note.",
            "No source-separated per-note energy is available in this frozen audit.",
        ],
    }
    (a.output/"v273-cluster-a-analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Décomposition du cluster A — perte d'évidence",
        "",
        f"Cluster A : **{len(rows)} cas**, dont **{result['undercounts']} sous-comptages**.",
        "",
        "## Où l'information disparaît",
        "",
        "| Test descriptif | Cas | Part de A |",
        "|---|---:|---:|",
    ]
    for n in ("full_candidate_shortage","retention_shortage","enough_retained_candidates",
              "weak_birth_vs_sameK_correct_median","weak_count_vs_sameK_correct_median",
              "carried_note_present","offset_present"):
        x=summary_flags[n]
        lines.append(f"| {n} | {x['rows']} | {100*x['rate']:.1f} % |")
    e=result["enough_candidates_but_weak_birth"]
    lines += [
        "",
        f"Parmi les cas où le système conserve déjà au moins K candidats, **{e['rows']}** ont malgré tout un local_birth_max inférieur à la médiane des cas corrects du même K.",
        "",
        "## Sensibilité des deux contextes",
        "",
        "| Composante retirée | Contribution moyenne à E[K] |",
        "|---|---:|",
        f"| candidate context | {result['support_expectedK']['candidate_context_mean']:+.3f} |",
        f"| spectral context | {result['support_expectedK']['spectral_context_mean']:+.3f} |",
        f"| tail frames | {result['support_expectedK']['tail_frames_mean']:+.3f} |",
        "",
        f"Le sous-clustering de A retient **{model.n_clusters} sous-familles** (silhouette {sil:.4f}).",
        "",
    ]
    for s in sub:
        lines += [
            f"### A{s['subcluster']} — {s['rows']} cas ({100*s['share_of_A']:.1f} %)",
            f"- sous-comptage : {100*s['undercount_rate']:.1f} %",
            "- transitions : "+", ".join(f"{x['transition']} ({x['rows']})" for x in s["transitions"][:5]),
            "- pénurie full_candidates<K : "+f"{100*s['flags']['full_candidate_shortage']['rate']:.1f} %",
            "- perte au filtrage retained<K : "+f"{100*s['flags']['retention_shortage']['rate']:.1f} %",
            "- naissance locale faible : "+f"{100*s['flags']['weak_birth_vs_sameK_correct_median']['rate']:.1f} %",
            "- note antérieure active : "+f"{100*s['flags']['carried_note_present']['rate']:.1f} %",
            "- offset présent : "+f"{100*s['flags']['offset_present']['rate']:.1f} %",
            "- signature : "+", ".join(f"{x['feature']} {x['mean_z']:+.2f}σ" for x in s["top_standardized_features_within_A"][:5]),
            "",
        ]
    lines += [
        "## Interprétation",
        "",
        "Cette analyse distingue une vraie pénurie de propositions d'un cas où suffisamment de candidats existent mais où leur évidence de naissance/comptage reste faible. Elle ne modifie aucune prédiction.",
    ]
    (a.output/"v273-cluster-a-analysis.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
