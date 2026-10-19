"""S787: six independently executed audits for every NOTE experiment loop.

This is a reproducible six-role AUDIT ORCHESTRATOR, not six native Work
agents. Each GitHub Actions matrix job loads source, signed S708, the tested
experiment report and changed paths independently and writes its own
machine-readable finding. The consensus job rejects any missing/failed audit.
It NEVER promotes experimental models or confuses shadow 1024 with official
256-sample MIDI metrics. Only FIT selects; CAL is diagnostic.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

ROLES=("data","acoustics","architecture","training","metrics","live")
OFFICIAL=("onset_f1","exact_k","active_exact_k","offset_accuracy",
          "full_note_f1","frame_pitch_f1")
SOURCE_SCHEMA="S786_CAUSAL_PITCH_TOPOLOGY_FRAGMENT_EXPERTS"
SIGNED_SCHEMA="S708_K_GUIDED_S706_CAL_LOOPS"

def need(ok,message):
    if not ok:
        raise AssertionError(message)

def eq(a,b,description,atol=1e-8):
    need(math.isclose(float(a),float(b),rel_tol=0.,abs_tol=atol),
         f"{description}: {a} != {b}")

def loaded(args):
    report=json.loads(args.experiment.read_text())
    signed=json.loads(args.signed.read_text())
    source=args.source.read_text(encoding="utf-8")
    tree=ast.parse(source)
    need(report["schema"]==SOURCE_SCHEMA,"unexpected experiment schema")
    need(signed["schema"]==SIGNED_SCHEMA,"unexpected signed S708 schema")
    return report,signed,source,tree

def function(tree,name):
    found=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
           and n.name==name]
    need(len(found)==1,f"missing/duplicate function {name}")
    return found[0]

def confusion_audit(metrics,who):
    counts=metrics["confusion"]
    need(len(counts)==7 and all(len(row)==7 for row in counts),
         f"{who}: expected 7x7 confusion")
    need(all(isinstance(x,int) and x>=0 for row in counts for x in row),
         f"{who}: malformed confusion entries")
    frames=sum(sum(r) for r in counts)
    need(frames==metrics["frames"],f"{who}: confusion/frame mismatch")
    correct=sum(counts[k][k] for k in range(7))
    eq(metrics["exact_accuracy"],correct/max(frames,1),
       f"{who}: exact-k")
    for k,rec in enumerate(metrics["per_k"]):
        n=sum(counts[k])
        need(rec["k"]==k and rec["n"]==n,f"{who}: support for K{k}")
        need(rec["correct"]==counts[k][k],f"{who}: correct K{k}")
        need(rec["under"]==sum(counts[k][:k]),f"{who}: undercount K{k}")
        need(rec["over"]==sum(counts[k][k+1:]),f"{who}: overcount K{k}")
        if n:
            eq(rec["accuracy"],counts[k][k]/n,f"{who}: recall K{k}")
        else:
            need(rec["accuracy"] is None,f"{who}: empty K{k} must be None")
    return dict(frames=frames,correct=correct,
                k5=metrics["per_k"][5]["accuracy"],
                k6=metrics["per_k"][6]["accuracy"],
                accuracy=metrics["exact_accuracy"])

def robust_score(metrics_by_comp,overall):
    def recall(m,k):
        return float(m["per_k"][k]["accuracy"] or 0.)
    k6=[recall(m,6) for m in metrics_by_comp.values()
        if m["per_k"][6]["n"]>0]
    k5=[recall(m,5) for m in metrics_by_comp.values()
        if m["per_k"][5]["n"]>0]
    return (min(k6) if k6 else 0.,min(k5) if k5 else 0.,
            float(overall["exact_accuracy"]))

def audit_data(r,s,src,tree,args):
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    need(digest==s["checkpoint_sha256"],"signed S708 checkpoint hash differs")
    need(r["fit_recordings_done"]==80 and r["cal_recordings_done"]==32,
         "incomplete FIT80 or CAL32")
    need(r["complete"] is True,"experiment report incomplete")
    need(r["no_CAL_training_or_selection"] is True,"CAL used for tuning")
    need(r["TEST_opened"] is False and r["HELD_opened"] is False,
         "independent TEST/HELD must stay protected")
    need(r["promoted"] is False,"experiment performed unexpected promotion")
    fit_keys=set(r["FIT"]["baseline_by_composition"])
    cal_keys=set(r["CAL"]["by_composition"])
    need(len(fit_keys)==5,"FIT validation not by five compositions")
    need(cal_keys=={"BN2","Jazz2"},"wrong CAL compositions")
    need(not fit_keys.intersection(cal_keys),"composition overlap")
    need("sha=digest((args.s708_dir/\"s708_selected.pt\").read_bytes())" in src,
         "source missing signed checkpoint comparison")
    official=s["frozen_candidate"]["CAL"]
    need(not official["all_targets_above_80"],
         "signed S708 unexpectedly asserts six >80")
    return {"signed_sha256":digest,"fit_recordings":80,"cal_recordings":32,
            "fit_compositions":sorted(fit_keys),"cal_compositions":sorted(cal_keys)}

def audit_acoustics(r,s,src,tree,args):
    fn=function(tree,"harmonic_features")
    names={node.id for node in ast.walk(fn) if isinstance(node,ast.Name)}
    need(not ({"labels","truth","gt","real_notes"}&names),
         "ground truth leaked into acoustic feature computation")
    need({"sustain","onset","energy","parent","conflict","supported",
          "harmonic_density"}.issubset(names),
         "acoustic physical components absent")
    need(r["topological_dim"]==200,"200D topology ABI absent")
    vals={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("DIM","INTERVALS"):
                    vals[target.id]=ast.literal_eval(node.value)
    need(vals.get("DIM")==200,"compiled DIM != report DIM")
    intervals=tuple(vals.get("INTERVALS",()))
    need(intervals==(12,19,24,28,31,36),
         "frozen harmonic interval map changed")
    # Expected integer pitch transport on 49 distinct frequencies:
    # each d constrains ordered fundamental -> overtone pitch slots.
    pairs={(root,root+d) for d in intervals for root in range(49-d)}
    need((0,12) in pairs and (0,19) in pairs and (0,31) in pairs,
         "E2 harmonic topology missing")
    need("sustain" in src and "np.concatenate((base,sustain,onset,energy" in src,
         "pitch-order was discarded by sorting")
    return {"feature_dim":200,"pitch_bins":49,"intervals":list(intervals),
            "harmonic_pairs":len(pairs),
            "note":"AST plus invariant audit; S786 CI unit tests separately execute synthetic features"}

def audit_architecture(r,s,src,tree,args):
    fn=function(tree,"gated_candidate")
    names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
    need(not ({"gt","truth","labels","real_notes","future"} & names),
         "oracle used by per-fragment selector")
    text=ast.get_source_segment(src,fn) or ""
    need("harmonic>=gate_threshold" in text,
         "not selecting expert at each fragment")
    need("new_out=np.where((native<3)&(new_out>=5),native,new_out)" in text,
         "low-K tail promotion guard missing")
    need("base_predicted(baseline_model" in text,
         "S781 alternative expert absent")
    need(r["selected_policy"]==r["FIT"]["selected"],
         "report selected policy mismatch")
    need(r["decision_clock_samples"]==1024,
         "shadow clock changed without new protocol")
    return {"selector":"per-fragment harmonic gate","head_count":2,
            "tail_guard":"native K0-K2 cannot become K5/K6",
            "selected":r["FIT"]["selected"],"clock_samples":1024}

def audit_training(r,s,src,tree,args):
    fit=r["FIT"]
    baseline=fit["baseline"]
    candidates=fit["candidates"]
    by_comp=fit["by_composition"]
    bcomp=fit["baseline_by_composition"]
    need(len(candidates)==6 and set(candidates)==set(by_comp),
         "expected six FIT OOF policies")
    group_keys=set(bcomp)
    for key in candidates:
        need(set(by_comp[key])==group_keys,
             f"{key}: absent OOF composition")
        confusion_audit(candidates[key],f"FIT {key}")
        for comp,m in by_comp[key].items():
            confusion_audit(m,f"FIT {key} {comp}")
    need(sum(m["frames"] for m in bcomp.values())==baseline["frames"],
         "FIT five-fold frame count mismatch")
    limit_micro=float(fit["selection_micro_regression_limit"])
    limit_comp=float(fit["selection_comp_regression_limit"])
    eq(limit_micro,.002,"FIT micro guard")
    eq(limit_comp,.01,"FIT composition guard")
    expected=[key for key in candidates
        if candidates[key]["exact_accuracy"]>=baseline["exact_accuracy"]-limit_micro
        and all(by_comp[key][c]["exact_accuracy"]>=
                bcomp[c]["exact_accuracy"]-limit_comp for c in group_keys)]
    need(set(expected)==set(fit["eligible"]),
         "FIT feasible policy set does not follow guards")
    selected=(max(expected,key=lambda k:robust_score(by_comp[k],candidates[k]))
              if expected else None)
    need(selected==fit["selected"],
         "FIT policy was not chosen by declared objective")
    fn=function(tree,"evaluate")
    code=ast.get_source_segment(src,fn) or ""
    position_choice=code.find("best=max(eligible")
    position_cal=code.find("for i,spec in enumerate(cal_specs):")
    need(0<=position_choice<position_cal,
         "FIT/CAL ordering not preserved")
    return {"tested_policies":len(candidates),
            "eligible_policies":len(expected),
            "selected":selected,
            "selection_recomputed_independently":True,
            "fit_base_exactk":baseline["exact_accuracy"],
            "fit_selected_exactk":candidates[selected]["exact_accuracy"]
                                   if selected else baseline["exact_accuracy"]}

def audit_metrics(r,s,src,tree,args):
    cal=r["CAL"]
    native=confusion_audit(cal["native"],"CAL frozen native")
    frozen=confusion_audit(cal["frozen"],"CAL frozen candidate")
    for comp,entry in cal["by_composition"].items():
        confusion_audit(entry["overall"],f"CAL comp {comp}")
    hist=cal["confusion_k5_k6"]
    for k in ("5","6"):
        need(len(hist[k])==7 and sum(hist[k])==cal["frozen"]["per_k"][int(k)]["n"],
             f"CAL harmonic tail histogram K{k} mismatch")
    baseline=s["frozen_candidate"]["CAL"]
    official={k:baseline[k] for k in OFFICIAL}
    need(all(isinstance(v,(int,float)) and 0<=v<=1
             for v in official.values()),"signed 6 official metrics absent")
    need("official_LIVE_changed" in s and s["official_LIVE_changed"] is False,
         "signed S708 had unintended LIVE change")
    require_80=all(v>.8 for v in official.values())
    need(not r["promoted"] and r["decision_clock_samples"]==1024,
         "shadow K metrics misrepresented as official MIDI")
    # Distinguish raw 1024-head score from unchanged 256-stream signed metrics.
    return {"CAL_native_shadow_exactk":native["accuracy"],
            "CAL_trial_shadow_exactk":frozen["accuracy"],
            "CAL_delta_shadow_exactk":frozen["accuracy"]-native["accuracy"],
            "CAL_trial_K5":frozen["k5"],"CAL_trial_K6":frozen["k6"],
            "signed_S708_official_six_metrics":official,
            "all_official_over_80":require_80,
            "promotion_eligible":False,
            "note":"S786 is shadow 1024; official 256-hop MIDI never changed"}

def audit_live(r,s,src,tree,args):
    need(all(r[k] is False for k in (
        "LIVE_changed","final_MIDI_changed","ONSET_OFFSET_changed")),
         "LIVE/NoteOn/NoteOff or final MIDI were modified")
    need(r["decision_clock_samples"]==1024,
         "cannot label shadow head as streaming final decoder")
    changed=[line.strip() for line in args.changed_files.read_text().splitlines()
             if line.strip()]
    permitted={
        "scripts/s786_harmonic_topology_fragment_experts.py",
        "test/test_s786_harmonic_topology_fragment_experts.py",
        ".github/workflows/note-s786-harmonic-topology-fragment.yml",
        "scripts/s787_six_role_audit.py",
        "scripts/s787_audit_consensus.py",
        "test/test_s787_six_role_audit.py",
        ".github/workflows/note-s787-six-role-audits.yml",
        "docs/NOTE_SIX_ROLE_LOOP_PROTOCOL.md"}
    unexpected=sorted(set(changed)-permitted)
    need(not unexpected,
         f"unauthorized changed paths, cannot guarantee LIVE preserved: {unexpected}")
    fn=function(tree,"harmonic_features")
    text=ast.get_source_segment(src,fn) or ""
    need("np.maximum(0,np.arange(n)-1)" not in text,
         "S786 feature must not assume any future context")
    need("np.maximum" in text,"S786 missing harmonic normalization")
    return {"modified_paths":sorted(changed),
            "protected_live_code_unchanged":True,
            "tested_model_is_not_live_promoted":True,
            "shadow_clock_samples":1024}

HANDLERS={
    "data":audit_data,
    "acoustics":audit_acoustics,
    "architecture":audit_architecture,
    "training":audit_training,
    "metrics":audit_metrics,
    "live":audit_live}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--signed",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--changed-files",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,
                passed=False,experimental_promotion_allowed=False,
                findings={},errors=[])
    try:
        report,signed,source,tree=loaded(a)
        result["findings"]=HANDLERS[a.role](report,signed,source,tree,a)
        result["passed"]=True
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(f"S787_AUDITOR_{a.role.upper()}",json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:
        sys.exit(1)

if __name__=="__main__":
    main()
