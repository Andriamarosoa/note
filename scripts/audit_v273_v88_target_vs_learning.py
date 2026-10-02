"""Audit whether Cluster A is caused by V8.8 prediction error or target mismatch."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.evaluate_v8_boundaries import _reference_positions
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM, LOCAL_CLUSTER_SAMPLES
from scripts.train_v90_structured_cluster_cardinality import FROZEN_CANDIDATE_DIM

def find_one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name}, found {len(h)}")
    return h[0]

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_csv(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))

def choose_A(assignments):
    stats={}
    for r in assignments:
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=stats.setdefault(c,{"n":0,"under":0})
        d["n"]+=1; d["under"]+=p<y
    cid=max(stats,key=lambda c:(stats[c]["under"]/stats[c]["n"],stats[c]["n"]))
    ids={int(r["global_index"]) for r in assignments if int(r["cluster"])==cid}
    return cid,ids,stats

def load_outer(root):
    parts=[]
    for path in sorted(Path(root).rglob("v100-spectral-shard-*.npz")):
        with np.load(path,allow_pickle=False) as z:
            parts.append({k:np.asarray(z[k]) for k in z.files})
    if not parts: raise RuntimeError("no spectral shards")
    keys=("sequence","mask","exact","members","cluster_start_samples","candidate_samples")
    out={k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}
    out["members"]=out["members"].astype(str)
    return out

def references(dataset,members):
    wanted=set(map(str,members))
    indexed={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    if set(indexed)!=wanted:
        raise RuntimeError("dataset members incomplete")
    result={}
    for member,track in indexed.items():
        audio=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        refs,_=_reference_positions(track,audio.frame_count)
        result[member]=np.asarray(refs,np.int64)
    return result

def rate(x):
    x=np.asarray(x,bool)
    return {"rows":int(x.sum()),"rate":float(x.mean()) if len(x) else None}

def summarize_candidates(records):
    if not records:
        return {}
    n=len(records)
    true=np.array([r["true_local"] for r in records],int)
    arg=np.array([r["pred_local"] for r in records],int)
    exp=np.array([r["expected_local"] for r in records],float)
    rt=np.array([r["router_target"] for r in records],int)
    rp=np.array([r["router_prob"] for r in records],float)
    return {
        "candidate_rows":n,
        "local_argmax_accuracy":float(np.mean(arg==true)),
        "local_argmax_under_rate":float(np.mean(arg<true)),
        "local_expected_mae":float(np.mean(np.abs(exp-true))),
        "local_expected_bias":float(np.mean(exp-true)),
        "router_accuracy_050":float(np.mean((rp>=.5)==rt)),
        "router_false_negative_rate":float(np.mean((rp<.5)&(rt==1))) if np.any(rt==1) else None,
        "true_local_histogram":np.bincount(true,minlength=4).tolist(),
        "pred_local_histogram":np.bincount(arg,minlength=4).tolist(),
    }

def row_records(rows,outer,refs):
    result=[]
    for row in rows:
        member=str(outer["members"][row])
        ref=refs[member]
        keep=outer["mask"][row].astype(bool)
        seq=np.asarray(outer["sequence"][row][keep],float)
        samples=np.asarray(outer["candidate_samples"][row][keep],np.int64)
        if len(seq)!=len(samples): raise RuntimeError("candidate alignment")
        router=seq[:,V88_FEATURE_DIM]
        card=seq[:,V88_FEATURE_DIM+1:V88_FEATURE_DIM+5]
        fused=seq[:,V88_FEATURE_DIM+7]
        if card.shape[1]!=4 or FROZEN_CANDIDATE_DIM!=V88_FEATURE_DIM+8:
            raise RuntimeError("V8.8 feature layout changed")
        true_local=np.array([min(3,int(np.sum(np.abs(ref-s)<=LOCAL_CLUSTER_SAMPLES))) for s in samples],int)
        expected=card@np.arange(4,dtype=float)
        pred=card.argmax(1)
        records=[]
        for j in range(len(samples)):
            records.append({
                "row":int(row),"sample":int(samples[j]),
                "true_local":int(true_local[j]),"pred_local":int(pred[j]),
                "expected_local":float(expected[j]),
                "router_target":int(true_local[j]>=2),"router_prob":float(router[j]),
                "fused_birth":float(fused[j]),
            })
        result.append((int(row),records,true_local,expected,fused))
    return result

def row_summary(items,y):
    rows=[]
    for row,recs,true_local,expected,fused in items:
        w=np.maximum(fused,1e-6)
        oracle_mean=float(np.mean(true_local))
        oracle_weighted=float(np.sum(true_local*w)/np.sum(w))
        pred_mean=float(np.mean(expected))
        pred_weighted=float(np.sum(expected*w)/np.sum(w))
        rows.append({
            "row":row,"k":int(y[row]),
            "candidate_count":len(recs),
            "max_true_local":int(np.max(true_local)),
            "oracle_local_mean":oracle_mean,
            "oracle_local_weighted":oracle_weighted,
            "pred_local_mean":pred_mean,
            "pred_local_weighted":pred_weighted,
        })
    return rows

def aggregate_rows(rows):
    if not rows:return {}
    k=np.array([r["k"] for r in rows],int)
    maxloc=np.array([r["max_true_local"] for r in rows],float)
    om=np.array([r["oracle_local_mean"] for r in rows],float)
    ow=np.array([r["oracle_local_weighted"] for r in rows],float)
    pm=np.array([r["pred_local_mean"] for r in rows],float)
    pw=np.array([r["pred_local_weighted"] for r in rows],float)
    return {
        "rows":len(rows),
        "no_candidate_local_target_reaches_group_K":rate(maxloc<k),
        "perfect_local_weighted_below_K_minus_half":rate(ow<(k-.5)),
        "predicted_local_weighted_below_K_minus_half":rate(pw<(k-.5)),
        "oracle_weighted_bias_vs_groupK":float(np.mean(ow-k)),
        "predicted_weighted_bias_vs_groupK":float(np.mean(pw-k)),
        "v88_prediction_bias_relative_to_local_target":float(np.mean(pw-ow)),
        "oracle_local_mean":float(np.mean(om)),
        "oracle_local_weighted":float(np.mean(ow)),
        "pred_local_mean":float(np.mean(pm)),
        "pred_local_weighted":float(np.mean(pw)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--low-k",type=Path,required=True)
    ap.add_argument("--acoustic",type=Path,required=True)
    ap.add_argument("--clustering",type=Path,required=True)
    ap.add_argument("--outer",type=Path,required=True)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)

    context=load_npz(find_one(a.low_k,"row-context.npz"))
    acoustic=load_npz(find_one(a.acoustic,"rows.npz"))
    ass=load_csv(a.clustering/"v273-failure-clustering-rows.csv")
    outer=load_outer(a.outer)

    np.testing.assert_array_equal(context["k"],outer["exact"])
    np.testing.assert_array_equal(context["member"].astype(str),outer["members"])
    np.testing.assert_array_equal(context["cluster_start_samples"],outer["cluster_start_samples"])

    y=acoustic["k"].astype(int); p=acoustic["predicted"].astype(int); gid=acoustic["global_index"].astype(np.int64)
    cid,gids,first=choose_A(ass)
    A=np.array([i for i,g in enumerate(gid) if int(g) in gids],int)
    controls=np.flatnonzero(np.isin(y,(2,3,4))&(p==y))
    refs=references(a.dataset,outer["members"])

    A_items=row_records(A,outer,refs)
    C_items=row_records(controls,outer,refs)
    A_rows=row_summary(A_items,y)
    C_rows=row_summary(C_items,y)
    A_candidates=[r for _,records,_,_,_ in A_items for r in records]
    C_candidates=[r for _,records,_,_,_ in C_items for r in records]

    by_k={}
    for k in (2,3,4):
        ar=[r for r in A_rows if r["k"]==k]
        cr=[r for r in C_rows if r["k"]==k]
        ac=[r for r in A_candidates if y[r["row"]]==k]
        cc=[r for r in C_candidates if y[r["row"]]==k]
        by_k[str(k)]={
            "A_rows":aggregate_rows(ar),
            "correct_rows":aggregate_rows(cr),
            "A_candidate_prediction":summarize_candidates(ac),
            "correct_candidate_prediction":summarize_candidates(cc),
        }

    overall={
        "A_rows":aggregate_rows(A_rows),
        "correct_rows":aggregate_rows(C_rows),
        "A_candidate_prediction":summarize_candidates(A_candidates),
        "correct_candidate_prediction":summarize_candidates(C_candidates),
    }

    result={
        "status":"completed",
        "scope":"pre-HPR fold-3 Cluster A; candidate-level V8.8 prediction vs reconstructed local target",
        "cluster_A":int(cid),
        "rows_A":len(A),
        "rows_correct_controls":len(controls),
        "v88_target_definition":{
            "local_radius_samples":int(LOCAL_CLUSTER_SAMPLES),
            "local_radius_ms":20.0,
            "classes":[0,1,2,"3+"],
            "router_target":"local cardinality >=2",
        },
        "structural_fact":"V8.8 local cardinality is clipped at 3+, so its scalar expected count cannot exceed 3 even when group Exact-K is 4 or more.",
        "overall":overall,
        "by_true_k":by_k,
        "interpretation_rule":{
            "learning_failure":"candidate true local target is high but V8.8 predicted local count is materially lower",
            "target_mismatch":"even a perfect V8.8 local target aggregate remains materially below the group Exact-K target",
        },
        "limitations":[
            "The oracle aggregation reuses current mean and fused-birth weighting; it is diagnostic, not an alternative decoder.",
            "Local target is clipped to 3 exactly as V8.8 training.",
            "One development fold and one seed.",
            "No model is retrained and no prediction is changed.",
        ],
    }
    (a.output/"v273-v88-target-vs-learning.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# V8.8 : erreur d'apprentissage ou mauvaise cible ?",
        "",
        f"Cluster A : **{len(A)} lignes** ; controles Exact-K corrects K2/K3/K4 : **{len(controls)}**.",
        "",
        "## Resultat global",
        "",
        "| Mesure | A | Corrects |",
        "|---|---:|---:|",
    ]
    for key,label in (
        ("local_argmax_accuracy","accuracy local_cardinality candidat"),
        ("local_argmax_under_rate","sous-prediction locale candidat"),
        ("local_expected_mae","MAE local_cardinality"),
        ("local_expected_bias","biais prediction locale vs cible locale"),
    ):
        av=overall["A_candidate_prediction"][key]; cv=overall["correct_candidate_prediction"][key]
        lines.append(f"| {label} | {av:.3f} | {cv:.3f} |")
    lines += [
        "",
        "| Test de mismatch cible | A | Corrects |",
        "|---|---:|---:|",
    ]
    for key,label in (
        ("no_candidate_local_target_reaches_group_K","aucun candidat local n'atteint le K du groupe"),
        ("perfect_local_weighted_below_K_minus_half","meme cible locale parfaite reste < K-0.5"),
        ("predicted_local_weighted_below_K_minus_half","prediction V8.8 reste < K-0.5"),
    ):
        av=overall["A_rows"][key]["rate"];cv=overall["correct_rows"][key]["rate"]
        lines.append(f"| {label} | {100*av:.1f}% | {100*cv:.1f}% |")
    lines += [
        "",
        f"Biais de la cible locale parfaite vs K dans A : **{overall['A_rows']['oracle_weighted_bias_vs_groupK']:+.3f}**.",
        f"Biais additionnel introduit par V8.8 par rapport a sa propre cible locale : **{overall['A_rows']['v88_prediction_bias_relative_to_local_target']:+.3f}**.",
        "",
        "## Par vrai K",
        "",
    ]
    for k in ("2","3","4"):
        x=by_k[k]
        lines += [
            f"### K{k}",
            f"- A : {x['A_rows']['rows']} lignes.",
            f"- Aucun candidat local n'atteint K : {100*x['A_rows']['no_candidate_local_target_reaches_group_K']['rate']:.1f}%.",
            f"- Meme avec cibles locales parfaites, agregat < K-0.5 : {100*x['A_rows']['perfect_local_weighted_below_K_minus_half']['rate']:.1f}%.",
            f"- V8.8 argmax local accuracy : {100*x['A_candidate_prediction']['local_argmax_accuracy']:.1f}%.",
            f"- Biais V8.8 vs vraie cible locale : {x['A_candidate_prediction']['local_expected_bias']:+.3f}.",
            "",
        ]
    lines += [
        "## Verdict de lecture",
        "",
        "Si la cible locale parfaite reste deja nettement sous K dans A, le defaut est structurel : V8.8 peut predire correctement sa propre cible tout en fournissant a Exact-K une statistique incompatible avec le compte du groupe. Si la cible parfaite est suffisante mais V8.8 la sous-predit, le defaut est principalement d'apprentissage.",
    ]
    (a.output/"v273-v88-target-vs-learning.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
