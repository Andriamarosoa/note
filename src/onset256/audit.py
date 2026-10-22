"""Independently audit signed FIT80 Onset trained256 reports without retraining.

No policy selection is made using FIT80 results. A scientist may inspect SHADOW
results, but the historically selected native policy remains fixed.
"""
from collections import defaultdict
import json

COMPOSITIONS=("BN1","Funk2","Jazz1","Rock3","SS1")
REFERENCE={"tp":8528,"fp":3489,"fn":2739}

def statistics(counts):
    tp,fp,fn=(int(counts[k]) for k in ("tp","fp","fn"))
    if min(tp,fp,fn)<0:raise ValueError("negative confusion count")
    return dict(tp=tp,fp=fp,fn=fn,
        precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,
        onset_f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)

def audit(report):
    rows=report.get("tracks",[])
    if not report.get("complete") or len(rows)!=80:
        raise ValueError("FIT80 result incomplete")
    if (report.get("decision_hop_samples")!=256
        or report.get("feature_hop_samples")!=256
        or report.get("s613_training_hop_samples")!=256
        or report.get("s701_training_hop_samples")!=256):
        raise ValueError("not genuine trained and inferred PCM256")
    if not report.get("no_s708_temporal_checkpoint"):
        raise ValueError("S708 legacy network is not allowed")
    if report.get("s701_preselected_policy")!="native" or report.get("shadow_promoted"):
        raise ValueError("post-selection or shadow promotion in research evidence")
    counts=defaultdict(int)
    by_composition=defaultdict(lambda:defaultdict(int))
    changed=[]
    for row in rows:
        comp=row["composition"]
        if comp not in COMPOSITIONS:raise ValueError("unknown GuitarSet fold")
        counts[comp]+=1
        by_composition[comp]["frames"]+=int(row["frames"])
        original=row["counts"]
        shadow=row["shadow_counts"]
        for key in ("tp","fp","fn"):
            for stem,source in (("native_",original),("s701_shadow_",shadow)):
                by_composition[comp][stem+key]+=int(source[key])
        if any(int(original[k])!=int(shadow[k]) for k in ("tp","fp","fn")):
            changed.append(dict(name=row["name"],composition=comp,
                delta_tp=int(shadow["tp"])-int(original["tp"]),
                delta_fp=int(shadow["fp"])-int(original["fp"]),
                delta_fn=int(shadow["fn"])-int(original["fn"])))
    if any(counts[x]!=16 for x in COMPOSITIONS):
        raise ValueError("not exactly 16 tracks per GuitarSet composition")
    if sum(x["frames"] for x in rows)!=306536:
        raise ValueError("not a signed 306536-frame FIT80 PCM256 replay")
    total={}
    for policy,prefix in (("native","native_"),("s701_shadow","s701_shadow_")):
        observed={k:sum(by_composition[c][prefix+k] for c in COMPOSITIONS)
                  for k in ("tp","fp","fn")}
        key="totals" if policy=="native" else "s701_shadow_totals"
        expected=report[key]
        if any(int(expected[k])!=observed[k] for k in observed):
            raise ValueError(f"{policy} score aggregation mismatch")
        total[policy]=statistics(observed)
    reference=statistics(REFERENCE)
    fold={}
    for comp in COMPOSITIONS:
        source=by_composition[comp]
        fold[comp]=dict(tracks=counts[comp],frames=source["frames"],
            native=statistics({k:source["native_"+k] for k in ("tp","fp","fn")}),
            s701_shadow=statistics({k:source["s701_shadow_"+k] for k in ("tp","fp","fn")}))
    selected=total["native"]
    return dict(complete=True,axis="onset",source_run=39716211840,
        decision_hop=256,trained_hop=256,tracks=len(rows),
        frames=306536,preselected_policy="native",
        scientific_verdict=("PASS" if selected["onset_f1"]>=reference["onset_f1"] else "REJECT_NONREGRESSION"),
        accepted_for_promotion=False,external_generalization_proven=False,
        reference=reference,totals=total,
        delta_s701_minus_native=dict(
            tp=total["s701_shadow"]["tp"]-selected["tp"],
            fp=total["s701_shadow"]["fp"]-selected["fp"],
            fn=total["s701_shadow"]["fn"]-selected["fn"],
            onset_f1=total["s701_shadow"]["onset_f1"]-selected["onset_f1"]),
        delta_native_minus_reference_f1=selected["onset_f1"]-reference["onset_f1"],
        changed_tracks=len(changed),changed_track_details=changed,
        compositions=fold)

def main():
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=audit(json.loads(a.report.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2)+"\n")
    print("ONSET_256_REJECTED_CAUSAL_AUDIT",json.dumps(
        {k:data[k] for k in ("scientific_verdict","changed_tracks","totals","delta_s701_minus_native")},sort_keys=True))
if __name__=="__main__":
    main()
