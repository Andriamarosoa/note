"""S650: independent frozen S648 event-birth per-source improvement/FP audit.

Reads only the immutable report from S648; NEVER trains or reselects a
policy, NEVER opens original TEST. Counts actual corrected/reference note
events only as net TP differences, not unverified event-level 1:1 matches.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json


def audit(report):
    if report["original_TEST_not_accessed"] is not True:
        raise AssertionError("Do not audit a report that opened original TEST")
    if report["fit_sources"]!=24 or report["calibration_sources"]!=11 or report["DEV_sources"]!=17:
        raise AssertionError("No hidden train/dev manifest changes")
    name=report["lock_from_CALIBRATION_before_DEV"]
    if name not in [p["name"] for p in report["policies"]]:
        raise AssertionError("Selected policy not among predeclared variants")
    out={}
    for stage in ("CALIBRATION","DEV"):
        rows=report["metrics"][stage]["by_source"]
        stage_result={}
        for policy in (p["name"] for p in report["policies"]):
            differences=[]
            for row in rows:
                baseline=row["policies"]["S614_original"]
                proposal=row["policies"][policy]
                diff={k:proposal["metrics"][k]-baseline["metrics"][k]
                      for k in ("TP","FP","FN","F1")}
                if diff["TP"]!=-diff["FN"]:
                    raise AssertionError("Reference truth changed")
                differences.append(dict(source=row["name"],corpus=row["corpus"],
                    **diff,actual_emitted_assists=proposal["emitted"],
                    noteon_count=proposal["noteons"]))
            aggregates={}
            for corpus in ("GuitarSet","URMP"):
                selected=[d for d in differences if d["corpus"]==corpus]
                sums={k:sum(row[k] for row in selected) for k in ("TP","FP","FN")}
                sums["source_wins"]=sum(d["F1"]>0 for d in selected)
                sums["source_losses"]=sum(d["F1"]<0 for d in selected)
                sums["source_equal"]=sum(d["F1"]==0 for d in selected)
                sums["assisted_noteons"]=sum(d["actual_emitted_assists"] for d in selected)
                aggregates[corpus]=sums
                source_score=report["metrics"][stage]["score"][policy][corpus]
                original=report["metrics"][stage]["score"]["S614_original"][corpus]
                for k in ("TP","FP","FN"):
                    if sums[k]!=source_score[k]-original[k]:
                        raise AssertionError("Totals disagree with original S648 training artifact")
            stage_result[policy]=dict(net_by_corpus=aggregates,per_source=differences)
        out[stage]=stage_result
    return dict(status="S650_FROZEN_S648_NO_ORACLE_EFFECT_AUDIT",
        locked_calibration_choice=name,locked_choice_DEV_delta=out["DEV"][name]["net_by_corpus"],
        scores=out,trained_checkpoint_sha256=report["checkpoint_sha256"],
        original_reference_unchanged=True,not_independent_new_test=True,
        no_promotion=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("Evidence immutable")
    src=json.loads(args.input.read_text())
    result=audit(src)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("S650 EVENT DELTAS",json.dumps(dict(
        locked=result["locked_calibration_choice"],
        locked_DEV=result["locked_choice_DEV_delta"],
        per_variant_DEV={n:x["net_by_corpus"]
             for n,x in result["scores"]["DEV"].items()}),indent=2))


if __name__=="__main__":main()
