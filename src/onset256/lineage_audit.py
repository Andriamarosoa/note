"""Onset PCM256: immutable train/held/FIT80 overlap forensic audit.

Distinct from inference leakage. No promotion when trained/held waveforms
reoccur in a FIT80 score report, regardless of that report's numerical F1.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path

def metric(rows,field):
    tp=sum(int(r[field]["tp"]) for r in rows)
    fp=sum(int(r[field]["fp"]) for r in rows)
    fn=sum(int(r[field]["fn"]) for r in rows)
    return {"tp":tp,"fp":fp,"fn":fn,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}

def analyze(report,ledger):
    if not report.get("complete") or len(report.get("tracks",[]))!=80:
        raise ValueError("expected complete 80-track FIT80 report")
    tracks=report["tracks"]
    fit=[r["name"] for r in tracks]
    if len(set(fit))!=80 or set(fit)!=set(ledger["fit80_names"]):
        raise ValueError("FIT80 names do not match independently recovered S701 logs")
    if report.get("decision_hop_samples")!=256:
        raise ValueError("not a PCM256 result")
    train613=set(ledger["s613_train_names"])
    train701=set(ledger["s701_train_names"])
    held701=set(ledger["s701_held_names"])
    if train701 & held701:raise ValueError("S701 TRAIN and HELD overlap")
    if (len(train613)!=36 or len(train701)!=80 or len(held701)!=64):
        raise ValueError("unexpected provenance source counts")
    def identity(names):
        return sorted(set(fit)&set(names))
    groups={}
    for label,names in (("s701_train",train701),("s701_held",held701),("s613_train",train613)):
        subset=[r for r in tracks if r["name"] in names]
        groups[label]=dict(count=len(subset),names=identity(names),
            native=metric(subset,"counts"),s701=metric(subset,"shadow_counts"))
    textures={}
    for label in ("comp","solo"):
        subset=[r for r in tracks if r["name"].endswith("_"+label)]
        textures[label]=dict(tracks=len(subset),native=metric(subset,"counts"),
                             s701=metric(subset,"shadow_counts"))
    if (groups["s701_train"]["count"]!=48 or groups["s701_held"]["count"]!=32 or
        groups["s613_train"]["count"]!=20):
        raise AssertionError("historic Onset training overlap not reproduced")
    all_seen=(train701|held701)&set(fit)
    training_overlap=bool(groups["s701_train"]["count"] or groups["s613_train"]["count"])
    prior_held_overlap=bool(groups["s701_held"]["count"])
    return dict(schema=ledger["schema"],source_run=39716211840,fit80_tracks=len(fit),
        native_256_training_and_inference=True,
        runtime_annotation_inference_leakage=False,
        training_evaluation_waveform_overlap=training_overlap,
        previous_model_selection_held_overlap=prior_held_overlap,
        independent_outer_test=False,
        s701_fit80_all_previously_seen=(len(all_seen)==80),
        validation_passed=False,
        promotion_allowed=False,
        counts=groups,textures=textures,
        overall=dict(native=metric(tracks,"counts"),s701=metric(tracks,"shadow_counts")))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=analyze(json.loads(args.report.read_text()),
                   json.loads(args.ledger.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("ONSET_PCM256_TRAIN_FIT80_LINEAGE_AUDIT",json.dumps(dict(
        training_evaluation_waveform_overlap=result["training_evaluation_waveform_overlap"],
        prior_model_selection_held_overlap=result["previous_model_selection_held_overlap"],
        independent_outer_test=result["independent_outer_test"],
        counts={k:v["count"] for k,v in result["counts"].items()},
        textures={k:v["native"] for k,v in result["textures"].items()})))
if __name__=="__main__":main()
