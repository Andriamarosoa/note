"""S788: six independent audits of S786 cross-run reproducibility."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
from scripts.s787_six_role_audit import ROLES, confusion_audit, need

def evaluate(role, a):
    reports = [json.loads(p.read_text()) for p in (a.first, a.second)]
    signed = json.loads(a.signed.read_text())
    for r in reports:
        need(r["schema"] == "S786_CAUSAL_PITCH_TOPOLOGY_FRAGMENT_EXPERTS" and r["complete"], "invalid S786 report")
        need(not r["promoted"] and r["no_CAL_training_or_selection"], "selection or promotion policy violation")
    x, y = reports
    evidence = {}
    if role == "data":
        need(hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == signed["checkpoint_sha256"], "S708 hash mismatch")
        for r in reports:
            need(r["fit_recordings_done"] == 80 and r["cal_recordings_done"] == 32, "split size")
            need(not r["TEST_opened"] and not r["HELD_opened"], "protected split opened")
            need(set(r["FIT"]["baseline_by_composition"]) == {"BN1","Funk2","Jazz1","Rock3","SS1"}, "FIT compositions")
            need(set(r["CAL"]["by_composition"]) == {"BN2","Jazz2"}, "CAL compositions")
        evidence = {"FIT": 80, "CAL": 32, "sha256": signed["checkpoint_sha256"]}
    elif role in ("acoustics", "architecture"):
        trees = [ast.parse(p.read_text()) for p in (a.source_a,a.source_b)]
        removed = [n for n in trees[0].body if isinstance(n,ast.FunctionDef) and n.name == "candidate"]
        need(len(removed) == 1, "expected unused function missing")
        trees[0].body = [n for n in trees[0].body if n not in removed]
        need(ast.dump(trees[0],include_attributes=False) == ast.dump(trees[1],include_attributes=False), "executed source changed")
        need(x["topological_dim"] == y["topological_dim"] == 200, "feature ABI drift")
        need(x["selected_policy"] is None and y["selected_policy"] is None, "policy selection changed")
        evidence = {"executed_AST_equivalent": True, "feature_dim": 200, "selected": None}
    elif role == "training":
        need(not x["FIT"]["eligible"] and not y["FIT"]["eligible"], "FIT selection changed")
        keys = sorted(x["FIT"]["candidates"])
        need(len(keys) == 6 and keys == sorted(y["FIT"]["candidates"]), "candidate list drift")
        diffs = {k:x["FIT"]["candidates"][k]["exact_accuracy"]-y["FIT"]["candidates"][k]["exact_accuracy"] for k in keys}
        evidence = {"baseline_a":x["FIT"]["baseline"]["exact_accuracy"],"baseline_b":y["FIT"]["baseline"]["exact_accuracy"],"policy_deltas":diffs,"exactly_reproducible":not any(diffs.values())}
    elif role == "metrics":
        for i,r in enumerate(reports):
            for key in ("baseline",):
                confusion_audit(r["FIT"][key],f"FIT run {i}")
            for key in ("native","frozen"):
                confusion_audit(r["CAL"][key],f"CAL {key} run {i}")
        need(x["FIT"]["baseline"]["frames"] == y["FIT"]["baseline"]["frames"],"FIT frame drift")
        need(x["CAL"]["native"]["confusion"] == y["CAL"]["native"]["confusion"],"CAL confusion drift")
        need(x["CAL"]["frozen"]["confusion"] == y["CAL"]["frozen"]["confusion"],"CAL trial confusion drift")
        delta = x["FIT"]["baseline"]["exact_accuracy"]-y["FIT"]["baseline"]["exact_accuracy"]
        evidence = {"FIT_delta":delta,"FIT_correct_delta":sum(x["FIT"]["baseline"]["confusion"][k][k]-y["FIT"]["baseline"]["confusion"][k][k] for k in range(7)),"CAL_exact_k":x["CAL"]["native"]["exact_accuracy"],"CAL_K6":x["CAL"]["native"]["per_k"][6]["accuracy"],"promotion_eligible":False}
    elif role == "live":
        for r in reports:
            need(not r["LIVE_changed"] and not r["final_MIDI_changed"] and not r["ONSET_OFFSET_changed"], "LIVE or MIDI changed")
            need(r["decision_clock_samples"] == 1024,"shadow clock changed")
        need(signed["official_LIVE_changed"] is False,"signed LIVE changed")
        evidence = {"LIVE_unchanged":True,"official_MIDI_unchanged":True,"shadow_clock":1024}
    return evidence

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    for key in ("first","second","signed","checkpoint","source-a","source-b","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    result={"schema":"NOTE_SIX_ROLE_AUDIT_V1","role":a.role,"passed":False,"experimental_promotion_allowed":False,"findings":{},"errors":[]}
    try:
        result["findings"]=evaluate(a.role,a)
        result["passed"]=True
    except Exception as e:
        result["errors"]=[str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S788",json.dumps(result),flush=True)
    if not result["passed"]:sys.exit(1)
if __name__=="__main__":main()
