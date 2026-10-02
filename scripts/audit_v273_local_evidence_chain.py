"""Audit the V8.8 -> V9/V27 local-evidence chain on pre-HPR Cluster A."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

LOCAL_HALF_WINDOW_NORM = 0.5

def find_one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name}, found {len(h)}")
    return h[0]

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_csv(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))

def select_A(assignments):
    stats={}
    for r in assignments:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=stats.setdefault(c,{"n":0,"under":0})
        d["n"]+=1;d["under"]+=p<y
    cid=max(stats,key=lambda c:(stats[c]["under"]/stats[c]["n"],stats[c]["n"]))
    ids={int(r["global_index"]) for r in assignments if int(r["cluster"])==cid}
    return cid,ids,stats

def desc(x):
    x=np.asarray(x,float)
    return {
        "n":int(len(x)),
        "mean":float(np.mean(x)) if len(x) else None,
        "median":float(np.median(x)) if len(x) else None,
        "p10":float(np.percentile(x,10)) if len(x) else None,
        "p90":float(np.percentile(x,90)) if len(x) else None,
    }

def rate(mask):
    mask=np.asarray(mask,bool)
    return {"rows":int(mask.sum()),"rate":float(mask.mean()) if len(mask) else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--low-k",type=Path,required=True)
    ap.add_argument("--acoustic",type=Path,required=True)
    ap.add_argument("--clustering",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)

    c=load_npz(find_one(a.low_k,"row-context.npz"))
    r=load_npz(find_one(a.acoustic,"rows.npz"))
    ass=load_csv(a.clustering/"v273-failure-clustering-rows.csv")
    np.testing.assert_array_equal(c["global_index"],r["global_index"])
    np.testing.assert_array_equal(c["k"],r["k"])

    y=r["k"].astype(int);p=r["predicted"].astype(int);gid=r["global_index"].astype(np.int64)
    cid,gids,first=select_A(ass)
    A=np.array([i for i,g in enumerate(gid) if int(g) in gids],int)
    correct=np.flatnonzero(np.isin(y,(2,3,4))&(p==y))

    fused_mean=np.asarray(c["proposal_mean"],float)
    fused_max=np.asarray(c["proposal_max"],float)
    router=np.asarray(c["router_mean"],float)
    local_mean=3.0*np.asarray(c["local_count_mean_norm"],float)
    local_weighted=3.0*np.asarray(c["local_count_weighted_norm"],float)
    local_nonzero=np.asarray(c["local_birth_max"],float)
    width=np.asarray(c["group_width"],float)
    retained=np.asarray(c["retained_candidates"],float)
    full=np.asarray(c["full_candidates"],float)

    weighted_delta=local_weighted-local_mean
    deficit_weighted=y-local_weighted
    deficit_mean=y-local_mean

    metrics={
        "fused_birth_mean":fused_mean,
        "fused_birth_max":fused_max,
        "router_mean":router,
        "local_expected_count_mean":local_mean,
        "local_expected_count_fused_weighted":local_weighted,
        "local_nonzero_max":local_nonzero,
        "weighted_minus_unweighted_local_count":weighted_delta,
        "group_width_norm_40ms":width,
        "retained_candidates":retained,
        "full_candidates":full,
        "trueK_minus_local_weighted":deficit_weighted,
        "trueK_minus_local_mean":deficit_mean,
    }

    by_k={}
    for k in (2,3,4):
        aa=A[y[A]==k];cc=np.flatnonzero((y==k)&(p==k))
        section={"A_rows":int(len(aa)),"correct_rows":int(len(cc)),"metrics":{}}
        for name,x in metrics.items():
            section["metrics"][name]={"A":desc(x[aa]),"correct":desc(x[cc])}
        section["rates"]={
            "group_span_gt_20ms_proxy_A":rate(width[aa]>LOCAL_HALF_WINDOW_NORM),
            "group_span_gt_20ms_proxy_correct":rate(width[cc]>LOCAL_HALF_WINDOW_NORM),
            "enough_candidates_A":rate(retained[aa]>=k),
            "local_weighted_below_K_minus_half_A":rate(local_weighted[aa]<(k-.5)),
            "local_mean_below_K_minus_half_A":rate(local_mean[aa]<(k-.5)),
            "fused_weighting_lowers_local_count_A":rate(weighted_delta[aa]<0),
        }
        by_k[str(k)]=section

    composition={k:int(np.sum(y[A]==k)) for k in (2,3,4)}
    def matched_correct_mean(x):
        vals=[];weights=[]
        for k,n in composition.items():
            cc=np.flatnonzero((y==k)&(p==k))
            if len(cc):
                vals.append(float(np.mean(x[cc])));weights.append(n)
        return float(np.average(vals,weights=weights))
    global_compare={}
    for name,x in metrics.items():
        global_compare[name]={
            "A_mean":float(np.mean(x[A])),
            "sameK_correct_composition_matched_mean":matched_correct_mean(x),
            "delta":float(np.mean(x[A])-matched_correct_mean(x)),
        }

    corrs={}
    for label,rows in (("A",A),("correct",correct)):
        corrs[label]={
            "width_vs_local_mean":float(np.corrcoef(width[rows],local_mean[rows])[0,1]),
            "width_vs_local_weighted":float(np.corrcoef(width[rows],local_weighted[rows])[0,1]),
            "width_vs_router":float(np.corrcoef(width[rows],router[rows])[0,1]),
            "width_vs_local_nonzero":float(np.corrcoef(width[rows],local_nonzero[rows])[0,1]),
        }

    enough=retained[A]>=y[A]
    weak_local=local_weighted[A]<(y[A]-.5)
    wide=width[A]>LOCAL_HALF_WINDOW_NORM
    low_fused=fused_max[A] < np.array([
        np.median(fused_max[(y==int(k))&(p==y)]) for k in y[A]
    ])
    intersections={
        "enough_candidates_and_weak_local_count":rate(enough&weak_local),
        "enough_candidates_wide_and_weak_local_count":rate(enough&wide&weak_local),
        "enough_candidates_weak_local_but_not_low_fused_birth":rate(enough&weak_local&~low_fused),
        "wide_group":rate(wide),
        "wide_group_and_weak_local":rate(wide&weak_local),
    }

    result={
        "status":"completed",
        "scope":"pre-HPR fold-3 Cluster A; no prediction changes",
        "cluster_A":int(cid),
        "rows":int(len(A)),
        "first_stage_cluster_stats":first,
        "code_semantics":{
            "v88_local_cardinality_target":"number of reference onsets within +/-20 ms of each candidate, clipped to 3+",
            "v88_router_target":"1 iff that candidate-local cardinality is >=2",
            "v90_group_exactK_target":"number of assigned onsets for the whole candidate group",
            "v90_group_window_ms":40,
            "stats_fused_birth_mean":"historical row-context key proposal_mean = mean V8.8 fused_birth",
            "stats_fused_birth_max":"historical row-context key proposal_max = max V8.8 fused_birth",
            "stats_local_nonzero_max":"historical key local_birth_max = max(1-P(local_cardinality=0)); not the V8.8 fused_birth head",
        },
        "global_compare":global_compare,
        "by_true_k":by_k,
        "correlations":corrs,
        "intersections":intersections,
        "limitations":[
            "group_width is candidate-span, not directly the annotated onset-span",
            "the 20 ms span test is a structural proxy; candidate-local labels were not re-derived row by row in this frozen artifact",
            "one development fold and seed",
            "this is a diagnostic, not a correction rule",
        ],
    }
    (a.output/"v273-local-evidence-chain.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Audit de la chaine local evidence vers Exact-K",
        "",
        f"Cluster A : **{len(A)} cas**.",
        "",
        "## Ce que les variables signifient reellement dans le code",
        "",
        "- local_cardinality V8.8 : nombre d'attaques de reference dans +/-20 ms autour de chaque candidat, classes 0/1/2/3+.",
        "- router V8.8 : cible 1 seulement si cette cardinalite locale vaut au moins 2.",
        "- Exact-K du groupe : nombre d'attaques attribuees au groupe entier.",
        "- proposal_mean/max dans l'ancien audit sont en realite moyenne/max de fused_birth V8.8.",
        "- local_birth_max est en realite max(1-P(local_cardinality=0)), pas la sortie fused_birth.",
        "",
        "## Cluster A vs cas corrects du meme K",
        "",
        "| Signal | A | Correct | Delta |",
        "|---|---:|---:|---:|",
    ]
    for n in ("fused_birth_mean","fused_birth_max","router_mean","local_expected_count_mean",
              "local_expected_count_fused_weighted","local_nonzero_max",
              "weighted_minus_unweighted_local_count","group_width_norm_40ms"):
        q=global_compare[n]
        lines.append(f"| {n} | {q['A_mean']:.3f} | {q['sameK_correct_composition_matched_mean']:.3f} | {q['delta']:+.3f} |")
    lines += [
        "",
        "## Tests structurels dans A",
        "",
        "| Test | Cas | Taux |",
        "|---|---:|---:|",
    ]
    for n,q in intersections.items():
        lines.append(f"| {n} | {q['rows']} | {100*q['rate']:.1f}% |")
    lines += ["","## Par vrai K",""]
    for k,s in by_k.items():
        lines += [
            f"### K{k}",
            f"- A: {s['A_rows']} cas ; corrects: {s['correct_rows']}",
            f"- groupe >20 ms (proxy): A {100*s['rates']['group_span_gt_20ms_proxy_A']['rate']:.1f}% vs correct {100*s['rates']['group_span_gt_20ms_proxy_correct']['rate']:.1f}%",
            f"- assez de candidats: {100*s['rates']['enough_candidates_A']['rate']:.1f}%",
            f"- local weighted < K-0.5: {100*s['rates']['local_weighted_below_K_minus_half_A']['rate']:.1f}%",
            f"- ponderation fused_birth abaisse le local count: {100*s['rates']['fused_weighting_lowers_local_count_A']['rate']:.1f}%",
            "",
        ]
    lines += [
        "## Lecture",
        "",
        "Si le local_expected_count est deja bas alors que fused_birth reste comparable aux controles et que la ponderation fused ne l'abaisse pas fortement, la perte apparait dans la tete local_cardinality ou sa semantique temporelle avant la fusion Exact-K. Si les groupes >20 ms sont surrepresentes, cela soutient un mismatch entre la supervision candidat-centrique +/-20 ms et la cible du groupe entier.",
    ]
    (a.output/"v273-local-evidence-chain.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
