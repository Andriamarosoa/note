"""S796 archived S795 failure: six independent read-only audits, no retraining."""
import argparse, json, sys
from pathlib import Path
from scripts.s787_audit_consensus import ROLES

SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
HEADS=("general","bass","polyphonic","transient","harmonic")

def require(condition,message):
    if not condition: raise AssertionError(message)

def inspect(role,report,signed,bank):
    r=json.loads(Path(report).read_text())
    s=json.loads(Path(signed).read_text())
    fit=r["FIT"]; candidates=fit["candidate_results"]; base=fit["baseline"]
    require(r["schema"]=="S795_CAUSAL_MULTIHEAD_ACTIVEK_REAL_MIDI256","wrong artifact")
    require((r["fit_done"],r["cal_done"])==(80,32),"wrong 80/32 counts")
    require(not r["complete"] and r["CAL"]=={},"false validated CAL")
    require(len(candidates)==25 and "source_identity" in candidates,"policy sweep absent")
    if role=="data":
        require(len(fit["compositions"])==5 and set(fit["training"])==set(fit["compositions"]),
                "not five composition OOF folds")
        require(not any(r[k] for k in ("HELD_opened","TEST_opened",
            "cal_used_for_training","cal_used_for_selection")),"split leakage")
        return dict(fit80_complete=True,cal32_processed=True,
                    cal32_signed_valid=False,compositions=fit["compositions"])
    if role=="acoustics":
        require(tuple(r["heads"])==HEADS and r["fitted_source_head_count"]==5,
                "not five trained acoustic heads")
        return dict(experts=HEADS,measured_gain="FIT-only, regressions elsewhere")
    if role=="architecture":
        require(r["selected_candidate"]==fit["selected"]=="source_identity"
            and r["retained_head_count"]==0 and fit["kept_experts"]==[],
            "unsafe selection")
        code=Path(bank).read_text()
        require("closed_this_frame=set()" in code and "pitch in closed_this_frame" in code,
                "S796 same-decision retrigger guard missing")
        return dict(retained_heads=0,rearm_same_timestamp="blocked by code and unit test")
    if role=="training":
        compbase=candidates["source_identity"]["by_composition"]
        accepted=[]
        for name,v in candidates.items():
            t=v["metrics"]
            valid=(v["mask"]==0 or
                (t["active_exact_k"]>=base["active_exact_k"]+.001
                 and all(t[k]>=base[k]-1e-12 for k in SCORES if k!="active_exact_k")
                 and all(v["by_composition"][c]["active_exact_k"]>=
                         compbase[c]["active_exact_k"]-.005 for c in compbase)))
            require(v["eligible"]==valid,"FIT eligibility inconsistent "+name)
            if valid: accepted.append(name)
        require(accepted==["source_identity"],"unexpected eligible candidate")
        return dict(candidates=25,accepted=accepted,retrained=False)
    if role=="metrics":
        name,v=max(candidates.items(),key=lambda kv:kv[1]["metrics"]["active_exact_k"])
        delta={k:round((v["metrics"][k]-base[k])*100,5) for k in SCORES}
        require(delta["active_exact_k"]>0 and delta["full_note_f1"]<0
                and delta["offset_accuracy"]<0,"failure signature lost")
        require(s["frozen_candidate"]["CAL"]["k_correct"]==78366,"signed counter changed")
        return dict(best_candidate=name,deltas_pp=delta,signed_k_correct=78366,
                    observed_k_correct= "78367 from S795 run logs; not contained in archived report",
                    official_CAL_S795="not validated")
    if role=="live":
        require(not any(r[k] for k in ("production_LIVE_modified","promoted",
            "physical_microphone_latency_measured")),"unsupported LIVE promotion")
        require(r["zero_added_scheduling"] and r["source_midi_events_untouched"],
                "S793 assertion missing")
        return dict(additional_scheduling="zero asserted",
                    physical_cpu_latency="not measured",promotion_allowed=False)
    raise AssertionError("unrecognized role")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",required=True,choices=ROLES)
    for name in ("report","signed","bank","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,
                passed=False,experimental_promotion_allowed=False,
                findings={},errors=[])
    try:
        result["findings"]=inspect(a.role,a.report,a.signed,a.bank)
        result["passed"]=True
    except Exception as error:
        result["errors"]=[f"{type(error).__name__}: {error}"]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S796_ROLE_AUDIT",json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:sys.exit(1)

if __name__=="__main__": main()
