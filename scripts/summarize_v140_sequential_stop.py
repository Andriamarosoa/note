"""Aggregate the five outer-clean V14 folds. No locked12 access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Optional, Sequence
import numpy as np

MODELS = ("v101", "v102", "v104", "v130", "v140")
STEPS = 6


def _metric_sum(rows):
    rows = [r for r in rows if r]
    tp = sum(int(r.get("true_positive", 0)) for r in rows)
    fp = sum(int(r.get("false_positive", 0)) for r in rows)
    fn = sum(int(r.get("false_negative", 0)) for r in rows)
    pred = sum(int(r.get("prediction_count", 0)) for r in rows)
    ref = sum(int(r.get("reference_count", 0)) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"f1": f1, "precision": p, "recall": rec, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "prediction_count": pred,
            "reference_count": ref, "prediction_reference_ratio": pred / ref if ref else None}


def _card(k, p):
    k = np.asarray(k, dtype=np.int32); p = np.asarray(p, dtype=np.int32)
    exact = p == k; birth = k == 1; poly = k >= 2
    return {"accuracy": float(np.mean(exact)),
            "birth_accuracy": float(np.mean(exact[birth])) if np.any(birth) else None,
            "poly_accuracy": float(np.mean(exact[poly])) if np.any(poly) else None,
            "mae": float(np.mean(np.abs(p-k))),
            "mean_predicted_count": float(np.mean(p)), "mean_true_count": float(np.mean(k)),
            "predicted_histogram": {str(v): int(np.sum(p==v)) for v in range(7)},
            "target_histogram": {str(v): int(np.sum(k==v)) for v in range(7)}}


def _player(m): return str(m).split("_",1)[0]
def _mode(m):
    s=str(m)
    return "comp" if s.endswith("_comp.jams") else "solo" if s.endswith("_solo.jams") else "other"
def _genre(m):
    s=str(m).split("_",1)[1] if "_" in str(m) else str(m)
    x=re.match(r"^([A-Za-z]+)",s)
    return x.group(1) if x else "unknown"


def summarize(args):
    reports=[]; parts=[]
    for fold in range(5):
        rp=sorted(args.input_dir.glob(f"**/report-fold-{fold}.json")); npz=sorted(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"))
        if len(rp)!=1 or len(npz)!=1: raise RuntimeError(f"fold {fold}: expected one report+prediction shard")
        r=json.loads(rp[0].read_text()); p=r["protocol"]
        assert p["historical_validation_or_locked12_indexed_or_evaluated"] is False
        assert p["outer_fold_used_for_training"] is False and p["outer_fold_used_for_epoch_selection"] is False
        assert p["categorical_cardinality_head_exists"] is False and p["stop_threshold_tuned"] is False
        reports.append(r)
        with np.load(npz[0],allow_pickle=False) as z: parts.append({k:np.asarray(z[k]) for k in z.files})
    keys=set(parts[0])
    if any(set(p)!=keys for p in parts): raise RuntimeError("prediction shard schema mismatch")
    merged={k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}
    order=np.argsort(merged["global_index"],kind="stable")
    merged={k:v[order] for k,v in merged.items()}
    gi=np.asarray(merged["global_index"],dtype=np.int64)
    if len(np.unique(gi))!=len(gi): raise RuntimeError("outer prediction overlap")
    if set(np.asarray(merged["outer_fold"],dtype=np.int32).tolist())!=set(range(5)): raise RuntimeError("missing fold")

    members=np.asarray(merged["member"]).astype(str); players=np.asarray([_player(x) for x in members]); modes=np.asarray([_mode(x) for x in members]); genres=np.asarray([_genre(x) for x in members])
    k=np.asarray(merged["k"],dtype=np.int32)
    pred={m:np.asarray(merged[f"pred{m[1:] if m.startswith('v') else m}"],dtype=np.int32) for m in MODELS}
    masks={"aggregate":np.ones(len(k),bool),"comp":modes=="comp","solo":modes=="solo","player00":players=="00","player00_comp":(players=="00")&(modes=="comp"),"player00_rock_comp":(players=="00")&(modes=="comp")&(genres=="Rock")}
    for pl in ("00","01","02","03","04"): masks[f"player{pl}"]=players==pl

    strata={}
    for s,mask in masks.items():
        row={"clusters":int(np.sum(mask))}
        for m in MODELS:
            pieces=[]
            for r in reports:
                sr=r["strata"].get(s)
                if sr and m in sr: pieces.append(sr[m]["metrics"]["global"])
            row[m]={"metrics":{"global":_metric_sum(pieces)},"cardinality":_card(k[mask],pred[m][mask])}
        strata[s]=row

    per_k={}
    for value in range(7):
        mask=k==value; row={"clusters":int(np.sum(mask))}
        for m in MODELS:
            pp=pred[m][mask]
            row[m]={"exact":float(np.mean(pp==value)) if len(pp) else None,
                    "under_rate":float(np.mean(pp<value)) if len(pp) else None,
                    "over_rate":float(np.mean(pp>value)) if len(pp) else None,
                    "mae":float(np.mean(np.abs(pp-value))) if len(pp) else None}
        per_k[str(value)]=row

    cont=np.asarray(merged["continue_probability"],dtype=np.float64)
    tfm=np.asarray(merged["tf_mass"],dtype=np.float64); cm=np.asarray(merged["candidate_mass"],dtype=np.float64)
    conditional={}
    for q in range(6):
        reachable=k>=q; positive=k>q; neg=reachable & ~positive; pos=reachable & positive
        conditional[str(q)]={
            "reachable":int(np.sum(reachable)),"positive":int(np.sum(pos)),"negative":int(np.sum(neg)),
            "mean_probability_positive":float(np.mean(cont[pos,q])) if np.any(pos) else None,
            "mean_probability_negative":float(np.mean(cont[neg,q])) if np.any(neg) else None,
            "tf_mass_positive":float(np.mean(tfm[pos,q])) if np.any(pos) else None,
            "tf_mass_negative":float(np.mean(tfm[neg,q])) if np.any(neg) else None,
            "candidate_mass_positive":float(np.mean(cm[pos,q])) if np.any(pos) else None,
            "candidate_mass_negative":float(np.mean(cm[neg,q])) if np.any(neg) else None,
        }

    f104=strata["aggregate"]["v104"]["metrics"]["global"]["f1"]; f130=strata["aggregate"]["v130"]["metrics"]["global"]["f1"]; f140=strata["aggregate"]["v140"]["metrics"]["global"]["f1"]
    result={
        "schema_version":1,
        "protocol":{"five_outer_composition_folds":True,"every_row_evaluated_once_outer_clean":True,"historical_validation_or_locked12_indexed_or_evaluated":False,"categorical_cardinality_head_exists":False,"conditional_stop_decoder":True,"explaining_away":True,"stop_threshold_tuned":False,"stop_threshold":0.5},
        "data":{"clusters":int(len(k)),"unique_global_indices":int(len(np.unique(gi))),"outer_folds":5},
        "strata":strata,"per_true_k":per_k,"conditional_steps":conditional,
        "folds":{str(r["outer_fold"]):{"selected_epochs":r["data"]["selected_epochs"],"v104_f1":r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],"v130_f1":r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],"v140_f1":r["strata"]["aggregate"]["v140"]["metrics"]["global"]["f1"]} for r in reports},
        "comparison":{"v140_minus_v104_global_f1":f140-f104,"v140_minus_v130_global_f1":f140-f130,"v140_minus_v104_player00_rock_comp_f1":strata["player00_rock_comp"]["v140"]["metrics"]["global"]["f1"]-strata["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"],"v140_minus_v130_player00_rock_comp_f1":strata["player00_rock_comp"]["v140"]["metrics"]["global"]["f1"]-strata["player00_rock_comp"]["v130"]["metrics"]["global"]["f1"],"v140_minus_v104_poly_exact":strata["aggregate"]["v140"]["cardinality"]["poly_accuracy"]-strata["aggregate"]["v104"]["cardinality"]["poly_accuracy"],"v140_minus_v130_poly_exact":strata["aggregate"]["v140"]["cardinality"]["poly_accuracy"]-strata["aggregate"]["v130"]["cardinality"]["poly_accuracy"],"v140_minus_v104_k6_exact":per_k["6"]["v140"]["exact"]-per_k["6"]["v104"]["exact"],"folds_won_vs_v104":int(sum(r["strata"]["aggregate"]["v140"]["metrics"]["global"]["f1"]>r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"] for r in reports)),"folds_won_vs_v130":int(sum(r["strata"]["aggregate"]["v140"]["metrics"]["global"]["f1"]>r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"] for r in reports)),"promotion_candidate":bool(f140>f104 and strata["player00_rock_comp"]["v140"]["metrics"]["global"]["f1"]>=strata["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"])}
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(args.output_dir/"predictions.npz",**merged)
    print(json.dumps({"global":{m:strata["aggregate"][m]["metrics"]["global"]["f1"] for m in MODELS},"precision":{m:strata["aggregate"][m]["metrics"]["global"]["precision"] for m in ("v104","v130","v140")},"recall":{m:strata["aggregate"][m]["metrics"]["global"]["recall"] for m in ("v104","v130","v140")},"player00_rock_comp":{m:strata["player00_rock_comp"][m]["metrics"]["global"]["f1"] for m in ("v104","v130","v140")},"poly":{m:strata["aggregate"][m]["cardinality"]["poly_accuracy"] for m in ("v104","v130","v140")},"k6":{m:per_k["6"][m]["exact"] for m in ("v104","v130","v140")},"counts":{m:strata["aggregate"][m]["cardinality"]["mean_predicted_count"] for m in ("v104","v130","v140")},"comparison":result["comparison"]},indent=2,sort_keys=True))
    return result


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); return p

def main(argv:Optional[Sequence[str]]=None): summarize(parser().parse_args(argv)); return 0
if __name__=="__main__": raise SystemExit(main())
