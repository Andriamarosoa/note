"""S122 infer constraints of S4/S5 original 527D column families from tree cuts.

Unlabelled structural audit: where a hypothesis maps a feature to a ratio
mathematically in [0,1], original split thresholds significantly outside
[0,1] are evidence against THAT mapping. A lack of violations DOES NOT
prove feature identity. No fitting/calibration or accuracy.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np,joblib
from scripts.s105_s5_frozen_forward import load as load_s5

STYLES=("amplitude_lag_first","amplitude_activation_first","log_activation_first")
def zones(style):
    """Return assumed feature semantics for 4x104 unknown columns."""
    ratio=[]; counts=[];classes={}
    for zone in range(4):
        base=111+104*zone
        if style=="amplitude_lag_first":
            starts=[(base+21*l,21,"amp") for l in range(4)]+[(base+84,20,"amp")]
        else:
            starts=[(base,20,"log" if style=="log_activation_first" else "amp")]
            starts += [(base+20+21*l,21,
                        "log" if style=="log_activation_first" else "amp") for l in range(4)]
        for ix,width,kind in starts:
            for j in range(width):
                idx=ix+j
                if j==9:
                    ratio.append(idx);classes[idx]="pitch_concentration"
                elif (j==18 and kind=="amp") or j==19:
                    ratio.append(idx);classes[idx]="temporal_ratio"
                elif 11<=j<=14:
                    counts.append(idx);classes[idx]="pitch_peak_count"
                elif j==10:
                    classes[idx]="effective_pitch_count"
                elif j==20:
                    classes[idx]="negative_mass"
                else:
                    classes[idx]="continuous"
    assert len(classes)==416 and len(set(ratio))==len(ratio) and len(set(counts))==len(counts)
    return ratio,counts,classes
def extracts(model):
    cuts=defaultdict(list)
    assert model.n_features_in_==527
    for trees in model._predictors:
        for tree in trees:
            nodes=tree.nodes
            mask=nodes["is_leaf"]==0
            for i,v in zip(nodes["feature_idx"][mask],nodes["num_threshold"][mask]):
                idx=int(i);threshold=float(v)
                if 111<=idx<527 and np.isfinite(threshold):cuts[idx].append(threshold)
    return cuts
def classify(all_models,style):
    ratio,counts,classes=zones(style)
    per_family={}
    for family,models in all_models.items():
        allcuts=defaultdict(list)
        for m in models:
            for k,v in extracts(m).items():allcuts[k].extend(v)
        ratio_cuts=np.array([c for k in ratio for c in allcuts[k]],float)
        count_cuts=np.array([c for k in counts for c in allcuts[k]],float)
        assert len(ratio_cuts)>0 and len(count_cuts)>0
        outside=(ratio_cuts<(-1e-6))|(ratio_cuts>1+1e-6)
        count_bounds=(count_cuts<0)|(count_cuts>49)
        near_half=np.abs((count_cuts-0.5)-np.rint(count_cuts-0.5))
        offenders=[]
        for col in ratio:
            values=np.asarray(allcuts[col],float)
            if len(values):
                violations=int(np.sum((values<(-1e-6))|(values>1+1e-6)))
                if violations:
                    offenders.append({"feature":int(col),
                       "expected":classes[col],"split_count":len(values),
                       "outside_unit":violations,"minimum":float(values.min()),
                       "maximum":float(values.max())})
        offenders.sort(key=lambda v:v["outside_unit"],reverse=True)
        per_family[family]={
            "ratio_columns_assumed":len(ratio),
            "ratio_tree_split_count":len(ratio_cuts),
            "ratio_splits_outside_0_1":int(outside.sum()),
            "ratio_outside_fraction":float(outside.mean()),
            "count_columns_assumed":len(counts),
            "count_tree_split_count":len(count_cuts),
            "count_outside_0_49_fraction":float(count_bounds.mean()),
            "count_cuts_close_to_half_integer_fraction":float((near_half<.05).mean()),
            "top_ratio_bound_conflicts":offenders[:20]}
    return per_family
def run(s4,s5,out):
    if out.exists():raise ValueError("immutable output path required")
    models={"S4":[],"S5":load_s5(s5)}
    for fold in (0,1,2):
        for poly in (1,2):
            state=joblib.load(s4/f"fold{fold}__owned__poly{poly}.joblib")
            if state["family"]!=f"owned__poly{poly}" or state["held_fold"]!=fold:
                raise ValueError("S4 original model provenance mismatch")
            models["S4"].append(state["model"])
    result={"status":"S122_ORIGINAL_S4_S5_RATIO_AND_COUNT_SEMANTIC_AUDIT",
        "original_model_count":10,
        "assumption":"named 416D type layouts are candidate mappings only",
        "hypotheses":{name:classify(models,name) for name in STYLES},
        "original_416D_order_proven":False,"accuracy_measured":False,
        "S80_promoted":False}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"ratios_by_style":{
        name:{fam:{"ratio_outside_fraction":record["ratio_outside_fraction"],
                   "count_half_integer_fraction":record["count_cuts_close_to_half_integer_fraction"],
                   "top_conflicts":record["top_ratio_bound_conflicts"][:4]}
              for fam,record in styles.items()}
        for name,styles in result["hypotheses"].items()},
        "accuracy_measured":False},indent=2))
def selftest():
    for name in STYLES:
        ratio,counts,types=zones(name)
        assert len(types)==416 and len(ratio) in (40,60) and len(counts)==80
        assert len(set(ratio)&set(counts))==0
    print("S122 PASS: 4x104 candidate column signatures and bounds")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s4-models",type=Path)
    p.add_argument("--s5-models",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        if not all((a.s4_models,a.s5_models,a.out)):p.error("all model and output args required")
        run(a.s4_models,a.s5_models,a.out)
