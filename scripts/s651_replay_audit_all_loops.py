"""S651 retrospective loop-by-loop replay with mandatory pre-DEV guard.

Uses four real immutable completed GitHub Actions reports. A candidate
is written to the ledger, including exact source regressions, BEFORE
the next candidate is considered. Rejected experiments are never lost.

The score is aggregate TP/FP/FN; event-level attribution / musical
Exact-K has not been computed by these historical artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json

from causal_note.s651_loop_guard import (
    GateLimits,audit_policy_records,frozen_selection,
)


SOURCE_SPECS={
 "S642":dict(key="dev_per_source",n=17,baseline="S614_original"),
 "S644":dict(key="sections",n=52,baseline="S614"),
 "S645":dict(key="stages",n=52,baseline="S614_original"),
 "S648":dict(key="metrics",n=28,baseline="S614_original"),
}


def read_stage(report,loop,stage):
    if loop=="S642":
        if stage!="DEV":return None
        raw=report["dev_per_source"]
        return [dict(name=r["source"],corpus=r["corpus"],
           policies={name:{k:float(v[k]) for k in ("TP","FP","FN","F1")}
                     for name,v in r["policies"].items()}) for r in raw]
    if loop=="S644":
        raw=report["sections"][stage]["per_source"]
        return [dict(name=r["name"],corpus=r["corpus"],
           policies={name:v["metric"] for name,v in r["variants"].items()})
                for r in raw]
    if loop=="S645":
        raw=report["stages"][stage]["by_source"]
        return [dict(name=r["name"],corpus=r["corpus"],
           policies={name:v["score"] for name,v in r["policies"].items()})
                for r in raw]
    if loop=="S648":
        raw=report["metrics"][stage]["by_source"]
        return [dict(name=r["name"],corpus=r["corpus"],
           policies={name:v["metrics"] for name,v in r["policies"].items()})
                for r in raw]
    raise ValueError("Unknown experiment family")


def policy_order(records,baseline):
    names=tuple(records[0]["policies"])
    if baseline not in names:raise AssertionError("Missing frozen S614 variant")
    for item in records:
        if tuple(item["policies"])!=names:
            raise AssertionError("Candidate set changed between sources")
    return [key for key in names if key!=baseline]


def audit_historical(report,loop):
    if loop not in SOURCE_SPECS:raise ValueError("Unrecognized frozen benchmark")
    if loop=="S648" and report.get("original_TEST_not_accessed") is not True:
        raise AssertionError("Original TEST was opened")
    if loop=="S645" and report.get("TEST_never_opened") is not True:
        raise AssertionError("Original TEST was opened")
    if loop=="S644" and report.get("original_test_not_read") is not True:
        raise AssertionError("Original TEST was opened")
    baseline=SOURCE_SPECS[loop]["baseline"]
    train_stage={"S642":None,"S644":"TRAIN","S645":"TRAIN","S648":"CALIBRATION"}[loop]
    stages=[train_stage,"DEV"] if train_stage else ["DEV"]
    stages=[s for s in stages if s]
    audit_log=[]
    locked=baseline
    for stage in stages:
        records=read_stage(report,loop,stage)
        if not records:raise AssertionError("Source ledger missing")
        expected={"TRAIN":35,"CALIBRATION":11,"DEV":17}[stage]
        if len(records)!=expected:
            raise AssertionError(f"{loop} {stage} expected {expected} sources; received {len(records)}")
        audits=[]
        for challenger in policy_order(records,baseline):
            # Every iteration emits independent audit AND retains failed
            # source identities before the next proposal.
            evidence=audit_policy_records(records,baseline,challenger,GateLimits())
            audits.append(evidence)
            audit_log.append(dict(stage=stage,loop_iteration=len(audits),
                proposal=challenger,accepted_for_next_stage=evidence["eligible_for_next_stage"],
                rejected_reasons=evidence["rejected_reasons"],
                corpus_summaries={c:evidence["by_corpus_and_role"][c]["ALL"]
                                  for c in ("GuitarSet","URMP")},
                adverse_sources=[dict(source=r["source"],corpus=r["corpus"],
                                      role=r["role"],delta=r["delta"])
                     for r in evidence["per_source"] if r["state"]=="regression"],
                evidence=evidence))
        if stage==train_stage:
            locked=frozen_selection(audits,baseline)
    if train_stage:
        claimed={"S644":report.get("train_locked_policy"),
                 "S645":report.get("selected_on_TRAIN_before_DEV"),
                 "S648":report.get("lock_from_CALIBRATION_before_DEV")}[loop]
        if locked!=claimed:
            raise AssertionError("Guard's frozen decision disagrees with historical train lock")
    else:
        locked=baseline
    dev=[a for a in audit_log if a["stage"]=="DEV"]
    risk={}
    for corpus in ("GuitarSet","URMP"):
        risk[corpus]=[]
        for item in dev:
            v=item["corpus_summaries"][corpus]
            risk[corpus].append(dict(proposal=item["proposal"],
                net_TP=v["delta"]["TP"],net_FP=v["delta"]["FP"],
                net_F1_points=100*v["delta"]["F1"],
                source_regressions=len(v["losses"]),
                fp_per_net_TP=v["fp_per_net_tp"],
                rejected=item["rejected_reasons"]))
    return dict(status="S651_AUDIT_EACH_ATTEMPT_BEFORE_NEXT",
        source_loop=loop,baseline=baseline,train_locked_reference=locked,
        stage_order=stages,number_of_audited_attempts=len(audit_log),
        attempts=audit_log,dev_risk_summary=risk,
        not_independent_new_test=True,no_models_changed=True,
        no_s614_s80_promotion=True,
        exact_k_not_available_from_summary=True,
        cannot_claim_new_corrected_note_ids_from_net_tp=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s642",type=Path,required=True)
    ap.add_argument("--s644",type=Path,required=True)
    ap.add_argument("--s645",type=Path,required=True)
    ap.add_argument("--s648",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("S651 evidence may never be overwritten")
    reports={name:json.loads(getattr(args,name.lower()).read_text())
             for name in ("S642","S644","S645","S648")}
    out={name:audit_historical(reports[name],name) for name in reports}
    for name,r in out.items():
        if r["train_locked_reference"]!=r["baseline"]:
            raise AssertionError("Unexpected change to any frozen original reference")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(status="S651_HISTORICAL_IN_LOOP_AUDIT",
       original_models_untouched=True,evidence=out),indent=2)+"\n")
    print("S651 LOOP GUARD VERDICT",json.dumps({
      name:dict(replayed_attempts=report["number_of_audited_attempts"],
         preselected=report["train_locked_reference"],
         dev_risk=report["dev_risk_summary"])
      for name,report in out.items()},indent=2),flush=True)


if __name__=="__main__":main()
