"""Aggregate exactly six independent NOTE audit artifacts; fail closed.

This is a reproducible reviewer gate, not promotion authority. Research
experiment is never promoted to official MIDI solely due to passing audits.
"""
import argparse
import json
from pathlib import Path
import sys

ROLES=("data","acoustics","architecture","training","metrics","live")

def consensus(report_dir):
    results={}
    issues=[]
    for role in ROLES:
        path=report_dir/f"role_{role}.json"
        if not path.is_file():
            issues.append(f"missing role artifact: {role}")
            continue
        try:
            report=json.loads(path.read_text())
            if report["role"]!=role or report["schema"]!="NOTE_SIX_ROLE_AUDIT_V1":
                raise ValueError("role/schema identity mismatch")
            if report["experimental_promotion_allowed"] is not False:
                raise ValueError("auditor illegally approved experimental promotion")
            results[role]=report
            if not report["passed"] or report["errors"]:
                issues.append(f"{role}: {'; '.join(report['errors'])}")
        except Exception as exc:
            issues.append(f"{role}: {type(exc).__name__}: {exc}")
    if len(results)!=6:
        issues.append(f"only {len(results)}/6 auditor reports loaded")
    passed=not issues and len(results)==6
    metrics=results.get("metrics",{}).get("findings",{})
    return dict(schema="NOTE_SIX_ROLE_CONSENSUS_V1",
                roles_required=list(ROLES),audits_received=len(results),
                passed=passed,official_promotion_eligible=False,
                shadow_trial_exact_k=metrics.get("CAL_trial_shadow_exactk"),
                baseline_S708_official=metrics.get("signed_S708_official_six_metrics"),
                issues=issues,
                auditors={k:dict(passed=v["passed"],findings=v["findings"],errors=v["errors"])
                          for k,v in results.items()})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reports-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=consensus(a.reports_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print("S787_SIX_AGENT_AUDIT_CONSENSUS",json.dumps({
        "passed":data["passed"],"count":data["audits_received"],
        "issues":data["issues"],"shadow_trial_exact_k":data["shadow_trial_exact_k"],
        "official_promotion_eligible":data["official_promotion_eligible"]
    }),flush=True)
    if not data["passed"]:
        sys.exit(1)

if __name__=="__main__":
    main()
