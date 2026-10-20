"""Six deterministic S798 security/protocol roles: cannot substitute Kimi advice."""
import argparse
import json
import sys
from pathlib import Path
from scripts.s787_audit_consensus import ROLES

def audit(role, adviser, workflow, tests):
    code = Path(adviser).read_text(encoding="utf-8")
    yml = Path(workflow).read_text(encoding="utf-8")
    unit = Path(tests).read_text(encoding="utf-8")
    def require(b, why):
        if not b: raise AssertionError(why)
    require("moonshotai/kimi-k3" in code, "model changed")
    require("NVIDIA_API_KEY" in code, "env secret missing")
    require("NVIDIA_API_KEY: \${{ secrets.NVIDIA_API_KEY }}" in yml,
            "no GitHub secret injection")
    require("promotion_authorized\": False" in code, "external authority creep")
    if role == "data":
        require("ALLOWLIST" in code and 'series/S797/resultat.md' in code
                and 'src/causal_note' not in code and 'test.py' not in code,
                "unsafe project data egress")
        require("SECRET_RE" in code and "safe_context" in code,
                "no secret screening")
        return {"data_sent":"research summaries allowlist only",
                "source_audio_held_out_excluded":True}
    if role == "acoustics":
        require("harmoniques" in code or "harmoniques" in code.lower()
                or "harmoni" in code,"acoustic audit prompt absent")
        require("Frame Pitch F1" in code and "Active Exact-K" in code,
                "target metric protocol incomplete")
        return {"question":"independent scientific hypotheses",
                "not_measurements":True}
    if role == "architecture":
        require("NoRedirect" in code and "redirect_request" in code,
                "Authorization redirect forbidden")
        require("opinion" in code.lower() and "review_text" in code,
                "not a shadow reviewer")
        return {"independent_adviser":True,"heads_dynamic_not_fixed_six":True}
    if role == "training":
        require("model.fit(" not in code and "optimizer.step(" not in code,
                "provider used as training loop")
        require("urllib.request" in code and "request_review(" in unit,
                "network is not mocked in unit tests")
        return {"retraining":False,"quality_selection_from_provider":False}
    if role == "metrics":
        require("recommendations_unvalidated" in code
                and "real_guitarset_scores_measured" in code,
                "external feedback masquerades as metrics")
        require("six_deterministic_audits_unchanged" in code,
                "risk of replacing signed metrics")
        return {"signed_S708_preserved":True,
                "external_result_not_f1":True}
    if role == "live":
        require("NoRedirect" in code and "workflow_dispatch" in yml,
                "automatic provider egress risk")
        require("github.event_name == 'workflow_dispatch'" in yml
                and "contents: read" in yml,
                "external call must be manual and read-only")
        return {"live_runtime_unchanged":True,
                "manual_external_invocation":True,
                "physical_latency":"not measured"}
    raise AssertionError("unknown audit role")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    for n in ("adviser","workflow","tests","out"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args()
    r={"schema":"NOTE_SIX_ROLE_AUDIT_V1","role":a.role,
       "passed":False,"experimental_promotion_allowed":False,
       "findings":{},"errors":[]}
    try:
        r["findings"]=audit(a.role,a.adviser,a.workflow,a.tests)
        r["passed"]=True
    except Exception as e:
        r["errors"]=[type(e).__name__+": "+str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("S798_STATIC_"+a.role, "PASS" if r["passed"] else "FAIL",flush=True)
    if not r["passed"]:sys.exit(1)

if __name__=="__main__":main()
