"""Second-pass observational S627 interaction checks from IMMUTABLE event audit.

Do not retrain/reselect anything or claim cause. This uses only output
from previously verified original 17 DEV S613 hybrid forensic replay.
Stratifies by multiple simultaneously observed characteristics and
reports denominators, composition support and risk differences.
"""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path


INTERACTIONS=(
  ("corpus","active_K"),
  ("corpus","coactive_interval"),
  ("active_K","coactive_interval"),
  ("register","active_K"),
  ("register","coactive_interval"),
  ("nearest_other_onset","coactive_interval"),
  ("first_frame_onset_bin","active_K"),
  ("corpus","register"),
)


def rates(rows,dimensions,min_count=15,min_compositions=3):
    groups=defaultdict(list)
    for row in rows:
        key=tuple(str(row[k]) for k in dimensions)
        groups[key].append(row)
    output=[]
    for key,vals in groups.items():
        total=len(vals);missed=sum(not v["matched"] for v in vals)
        sources={v["source"] for v in vals}
        compositions={(v["corpus"],v["composition"]) for v in vals}
        output.append(dict(dimension=dict(zip(dimensions,key)),n=total,
             missed=missed,detected=total-missed,
             missed_rate=round(missed/total,5),
             recordings=len(sources),composition_groups=len(compositions),
             limited_evidence=total<min_count or len(compositions)<min_compositions))
    return sorted(output,key=lambda r:(-r["missed"],-r["n"]))


def controlled_interval_contrasts(rows):
    """Harmonic-interval risk contrast within corpus+true K+pitch register.

    Excludes 'none' etc from nothing; control includes every OTHER
    interval category. Report each stratum and signs; avoids pooling
    known major confounders but doesn't rule out residual confounders.
    """
    grouped=defaultdict(lambda:dict(harm=[],other=[]))
    for row in rows:
        key=(row["corpus"],row["active_K"],row["register"])
        group="harm" if row["coactive_interval"]=="harmonic_interval_12_19_24" else "other"
        grouped[key][group].append(row)
    eligible=[];excluded=[]
    for key,v in grouped.items():
        a=v["harm"];b=v["other"]
        if len(a)<5 or len(b)<5:
            excluded.append(key)
            continue
        missa=sum(not x["matched"] for x in a)
        missb=sum(not x["matched"] for x in b)
        ratea=missa/len(a);rateb=missb/len(b)
        eligible.append(dict(
            corpus=key[0],active_K=key[1],register=key[2],
            harmonic_n=len(a),harmonic_missed=missa,
            other_n=len(b),other_missed=missb,
            rate_harmonic=round(ratea,5),rate_other=round(rateb,5),
            risk_difference=round(ratea-rateb,5),
            compositions_harmonic=len(set((x["corpus"],x["composition"]) for x in a)),
            compositions_other=len(set((x["corpus"],x["composition"]) for x in b)),
        ))
    weights=[r["harmonic_n"] for r in eligible]
    return dict(
        strata_eligible=len(eligible),
        strata_excluded=len(excluded),
        weighted_risk_difference=round(
          sum(r["risk_difference"]*w for r,w in zip(eligible,weights))
          /max(1,sum(weights)),5),
        strata_harmonic_higher=sum(r["risk_difference"]>0 for r in eligible),
        strata_harmonic_lower=sum(r["risk_difference"]<0 for r in eligible),
        strata_equal=sum(r["risk_difference"]==0 for r in eligible),
        strata=sorted(eligible,key=lambda r:-r["harmonic_n"]),
        warning=("Descriptive standardized contrast based on only DEV "
                 "recordings; confounding by composition, pitch and timing "
                 "may remain, and bins are not independent.")
    )


def analyze(input_path,out):
    if out.exists():raise FileExistsError(out)
    full=json.loads(input_path.read_text())
    assert full["frozen_hybrid_baseline"]["TP"]==1241
    assert full["frozen_hybrid_baseline"]["FN"]==838
    assert full["frozen_hybrid_baseline"]["FP"]==734
    truth=full["per_note_truth_rows"]
    assert len(truth)==2079
    summary={"+".join(dims):rates(truth,dims) for dims in INTERACTIONS}
    diagnostic=controlled_interval_contrasts(truth)
    output=dict(status="S627_MULTIFACTOR_INTERACTIONS_OBSERVATIONAL_ONLY",
        source_run="38889998300",test_sources_read=0,
        baseline=full["frozen_hybrid_baseline"],truth_note_events=len(truth),
        cross_tabs=summary,controlled_interval_contrasts=diagnostic,
        caution=("All estimates are conditioned on DEV annotated/observed "
                 "notes; no held-out generalization, and no causality "
                 "is established. Small K/pitch/corpus cells often unreliable."))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+"\n")
    for dims,rows in summary.items():
        valid=[r for r in rows if not r["limited_evidence"]]
        print("S627_INTERACTION "+json.dumps(dict(
            dimensions=dims,fully_supported_bins=len(valid),
            high_failure_counts=sorted(valid,key=lambda x:-x["missed"])[:8],
            high_failure_rates=sorted(valid,key=lambda x:-x["missed_rate"])[:8],
        )),flush=True)
    print("S627_CONDITIONAL_HARMONIC_ASSOCIATION "+json.dumps(diagnostic),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-audit",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    opts=ap.parse_args()
    analyze(opts.source_audit,opts.out)


if __name__=="__main__":
    main()
