"""S651 — invariant audit run INSIDE candidate-selection loops.

This module is deliberately model-free: every evaluated policy gets a
persisted *accepted or rejected* ledger, preserving adverse cases.

All acceptance decisions are made on TRAIN/CALIBRATION and frozen
before DEV. A positive aggregate score CANNOT overrule per-source
losses, false-positive budget, invalid NoteOn/Off, or missing matches.

The summary metrics cannot identify individual corrected/refused note
IDs or historical Exact-K; these require original event/label exports.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,asdict
import math


METRICS=("TP","FP","FN")


@dataclass(frozen=True)
class GateLimits:
    max_additional_fp_per_corpus: int=8
    max_regressed_sources_per_corpus: int=1
    require_both_corpora_f1_gain: bool=True
    require_no_extra_fp_per_net_tp: float | None=None


def _check_score(metrics):
    data={name:int(metrics[name]) for name in METRICS}
    if any(value<0 for value in data.values()):
        raise AssertionError("Negative confusion-matrix count")
    total=2*data["TP"]+data["FP"]+data["FN"]
    f1=2*data["TP"]/max(1,total)
    if "F1" in metrics and not math.isclose(
            float(metrics["F1"]),f1,rel_tol=0.,abs_tol=1e-7):
        raise AssertionError("F1 is not derived from original TP/FP/FN")
    return dict(**data,F1=f1)


def _role(record):
    name=record["name"]
    if record.get("corpus")=="URMP":return "URMP"
    if name.endswith("_solo"):return "solo"
    if name.endswith("_comp"):return "accompaniment"
    return "unspecified"


def _delta(before,after):
    if before["TP"]+before["FN"]!=after["TP"]+after["FN"]:
        raise AssertionError("Ground-truth note count changed by a policy")
    return dict(TP=after["TP"]-before["TP"],
        FP=after["FP"]-before["FP"],
        FN=after["FN"]-before["FN"],
        F1=after["F1"]-before["F1"])


def audit_policy_records(records,baseline,challenger,limits=GateLimits()):
    """Audit every source, preserve regressions and a gate verdict.

    Caller supplies ORIGINAL immutable per-source confusion metrics.
    This audit cannot reselect an option using heldout DEV labels.
    """
    if baseline==challenger:raise ValueError("Need a distinct candidate")
    if not records:raise ValueError("No source evidence")
    seen=set()
    cases=[]
    for source in records:
        identity=(source["corpus"],source["name"])
        if identity in seen:raise AssertionError("Source duplicated in loop")
        seen.add(identity)
        if "policies" not in source or baseline not in source["policies"] or challenger not in source["policies"]:
            raise AssertionError("Candidate or baseline metrics missing")
        original=_check_score(source["policies"][baseline])
        changed=_check_score(source["policies"][challenger])
        change=_delta(original,changed)
        cases.append(dict(source=source["name"],corpus=source["corpus"],
                          role=_role(source),baseline=original,challenger=changed,
                          delta=change,
                          state=("correction" if change["F1"]>1e-9 else
                                 "regression" if change["F1"]< -1e-9 else "unchanged")))
    out={}
    by_group=defaultdict(list)
    for row in cases:
        by_group[(row["corpus"],row["role"])].append(row)
        by_group[(row["corpus"],"ALL")].append(row)
    for (corpus,role),rows in by_group.items():
        b={key:sum(r["baseline"][key] for r in rows) for key in METRICS}
        n={key:sum(r["challenger"][key] for r in rows) for key in METRICS}
        b=_check_score(b); n=_check_score(n)
        changes=_delta(b,n)
        wins=[r["source"] for r in rows if r["state"]=="correction"]
        losses=[r["source"] for r in rows if r["state"]=="regression"]
        out.setdefault(corpus,{})[role]=dict(
            baseline=b,challenger=n,delta=changes,
            wins=wins,losses=losses,n_sources=len(rows),
            net_tp_gained=changes["TP"],net_fp_added=changes["FP"],
            fp_per_net_tp=(changes["FP"]/changes["TP"] if changes["TP"]>0 else None))
    reasons=[]
    required=("GuitarSet","URMP")
    for corpus in required:
        if corpus not in out or "ALL" not in out[corpus]:
            reasons.append(corpus+":missing_sources")
            continue
        total=out[corpus]["ALL"]
        if limits.require_both_corpora_f1_gain and total["delta"]["F1"]<=0:
            reasons.append(corpus+":not_positive_F1")
        if total["delta"]["FP"]>limits.max_additional_fp_per_corpus:
            reasons.append(corpus+":FP_budget_exceeded")
        if len(total["losses"])>limits.max_regressed_sources_per_corpus:
            reasons.append(corpus+":source_regressions_exceeded")
        if (limits.require_no_extra_fp_per_net_tp is not None
            and total["delta"]["TP"]>0
            and total["delta"]["FP"]/total["delta"]["TP"]>limits.require_no_extra_fp_per_net_tp):
            reasons.append(corpus+":FP_cost_per_recovery_exceeded")
    return dict(challenger=challenger,baseline=baseline,
       eligible_for_next_stage=not reasons,
       rejected_reasons=reasons,by_corpus_and_role=out,
       per_source=cases,
       aggregate_only=True,
       exact_k_breakdown_unavailable_without_event_labels=True,
       event_level_corrections_not_proven_from_net_TP=True)


def frozen_selection(audits,original="S614_original"):
    """Never read DEV here; only a caller's designated TRAIN/CALIB audits."""
    options=[x for x in audits if x["eligible_for_next_stage"]]
    if not options:return original
    def score(audit):
        return (min(audit["by_corpus_and_role"][corpus]["ALL"]["delta"]["F1"]
                    for corpus in ("GuitarSet","URMP")),
                audit["challenger"])
    return max(options,key=score)["challenger"]
