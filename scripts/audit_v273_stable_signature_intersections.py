"""Common-pattern intersection audit over the five stable inference-safe signals.

The five directions are frozen from completed run 37530033363:
- lower onset contrast in regressions
- higher pre-onset normalized energy in regressions
- lower triplet amplitude range in regressions
- higher amp3/amp1 in regressions
- lower triplet amplitude CV in regressions

For each held-out fold, thresholds are medians computed ONLY on the other three
internal folds. No threshold search is performed. We then measure individual
conditions, score>=k intersections, and all fixed conjunction subsets.

This is descriptive diagnosis only: no Exact-K output is changed and fold 3 is excluded.
"""
from __future__ import annotations
import argparse,json,itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_inference_safe_signature_proxy import safe_features
from scripts.v273_residual_audit import FOLDS,require

GROUPS=("K2_corrected","K3_regressed")
SIGNALS=(
 ("onset_low","nov_onset_contrast_norm_median","low"),
 ("pre_high","nov_pre_norm_median","high"),
 ("amp_range_low","amp_range","low"),
 ("amp3_ratio_high","amp3_over_amp1","high"),
 ("amp_cv_low","amp_cv","low"),
)

def condition(v,thr,direction):
    return bool(v<=thr) if direction=="low" else bool(v>=thr)

def metrics(items):
    n=len(items); k3=sum(r["group"]=="K3_regressed" for r in items); k2=n-k3
    return {
      "n":n,"k3_regressions":k3,"k2_corrections":k2,
      "precision_k3":None if n==0 else k3/n,
      "recall_k3":k3/125.0,
      "veto_net":k3-k2,
    }

def fold_metrics(items):
    out=[]
    for f in FOLDS:
        rr=[r for r in items if r["fold"]==f]
        m=metrics(rr);m["fold"]=int(f);out.append(m)
    return out

def subset_key(names): return "+".join(names)

def main():
    ap=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()]
    cases=[r for r in cases if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    require(all(r["fold"] in FOLDS and r["fold"]!=3 for r in cases),"fold leak")

    wanted={r["recording_id"] for r in cases}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in cases:by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in sorted(by[member],key=lambda z:z["row_id"]):
            sf=safe_features(r,samples)
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],
                         "features":{key:float(sf[key]) for _,key,_ in SIGNALS}})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)

    # OOF median thresholds; direction is fixed before this run.
    thresholds={}
    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f]
        thresholds[str(f)]={}
        for alias,key,direction in SIGNALS:
            thr=float(np.median([r["features"][key] for r in fit]))
            thresholds[str(f)][alias]={"feature":key,"direction":direction,"threshold":thr}
    for r in rows:
        t=thresholds[str(r["fold"])]
        bits={}
        for alias,key,direction in SIGNALS:
            bits[alias]=condition(r["features"][key],t[alias]["threshold"],direction)
        r["signature_bits"]=bits
        r["signature_score"]=sum(bits.values())

    individual={}
    for alias,_,_ in SIGNALS:
        hit=[r for r in rows if r["signature_bits"][alias]]
        individual[alias]={"overall":metrics(hit),"per_fold":fold_metrics(hit),
                           "k2_rate":sum(r["group"]=="K2_corrected" and r["signature_bits"][alias] for r in rows)/108.0,
                           "k3_rate":sum(r["group"]=="K3_regressed" and r["signature_bits"][alias] for r in rows)/125.0}

    score_rules={}
    for k in range(1,6):
        hit=[r for r in rows if r["signature_score"]>=k]
        score_rules[str(k)]={"overall":metrics(hit),"per_fold":fold_metrics(hit)}

    combos=[]
    aliases=[a for a,_,_ in SIGNALS]
    for size in range(2,6):
        for subset in itertools.combinations(aliases,size):
            hit=[r for r in rows if all(r["signature_bits"][s] for s in subset)]
            overall=metrics(hit);pf=fold_metrics(hit)
            fold_prec=[q["precision_k3"] for q in pf if q["n"]>0]
            min_precision=min(fold_prec) if fold_prec and all(q["n"]>0 for q in pf) else None
            combos.append({
              "pattern":subset_key(subset),"size":size,"overall":overall,"per_fold":pf,
              "min_fold_precision":min_precision,
              "all_folds_nonempty":all(q["n"]>0 for q in pf),
              "min_fold_support":min(q["n"] for q in pf),
            })
    # Descriptive ranking only; not a selected deployable rule.
    combos.sort(key=lambda q:(
        q["all_folds_nonempty"],
        -1 if q["min_fold_precision"] is None else q["min_fold_precision"],
        q["overall"]["precision_k3"] if q["overall"]["precision_k3"] is not None else -1,
        q["overall"]["veto_net"],
        q["overall"]["n"]),reverse=True)

    distributions={}
    for g in GROUPS:
        rr=[r for r in rows if r["group"]==g]
        distributions[g]={str(s):sum(r["signature_score"]==s for r in rr) for s in range(6)}

    robust_candidates=[q for q in combos if q["all_folds_nonempty"] and q["min_fold_support"]>=3
                       and q["min_fold_precision"] is not None and q["min_fold_precision"]>=.60
                       and q["overall"]["veto_net"]>0]

    rep={"status":"completed","experiment":"v273_stable_signature_intersections",
         "cases":233,"population":{"K2_corrected":108,"K3_regressed":125},
         "outer_fold_3_used":False,"prediction_changes":False,"threshold_search":False,
         "threshold_rule":"OOF fit-fold median only","signals":[{"alias":a,"feature":k,"direction":d} for a,k,d in SIGNALS],
         "thresholds":thresholds,"individual":individual,"score_rules":score_rules,
         "signature_score_distribution":distributions,
         "robust_descriptive_patterns":robust_candidates,"all_conjunctions":combos}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))

    lines=["# Stable five-signal intersection audit","",
           "OOF fit-fold medians only; no threshold search. Fold 3 excluded. No Exact-K output changed.","",
           "## Individual regression-like conditions","",
           "| condition | K3 hit rate | K2 hit rate | K3/K2 among hits | veto net |",
           "|---|---:|---:|---:|---:|"]
    for alias,_,_ in SIGNALS:
        q=individual[alias];o=q["overall"]
        lines.append(f"| {alias} | {q['k3_rate']:.3f} | {q['k2_rate']:.3f} | {o['precision_k3']:.3f} | {o['veto_net']:+d} |")
    lines+=["","## Number of regression-like conditions satisfied","",
            "| require score >= | cases | K3 regressions | K2 corrections | K3 precision | veto net |",
            "|---:|---:|---:|---:|---:|---:|"]
    for k in range(1,6):
        o=score_rules[str(k)]["overall"]
        lines.append(f"| {k} | {o['n']} | {o['k3_regressions']} | {o['k2_corrections']} | {o['precision_k3']:.3f} | {o['veto_net']:+d} |")
    lines+=["","## Robust descriptive conjunctions","",
            "| pattern | cases | K3 | K2 | precision | min-fold precision | min-fold support | veto net |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for q in robust_candidates[:15]:
        o=q["overall"]
        lines.append(f"| {q['pattern']} | {o['n']} | {o['k3_regressions']} | {o['k2_corrections']} | {o['precision_k3']:.3f} | {q['min_fold_precision']:.3f} | {q['min_fold_support']} | {o['veto_net']:+d} |")
    lines+=["","Patterns are descriptive only; no pattern is promoted automatically."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
