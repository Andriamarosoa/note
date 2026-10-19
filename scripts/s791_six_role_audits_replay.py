"""S791 audit replay: fixed six-role validation of COUNT+F1 expert summaries.

Replays immutable 39280680273 model evidence, without refitting or opening
CAL/HELD/TEST. Only difference from initial auditor: expert-only summaries
store TP/FP/FN/F1 (not precision/recall) so recompute/verify those four; all
regular aggregate summaries still REQUIRE precision and recall.
NumPy is explicitly installed in the replay's audit jobs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from causal_note.s791_framepitch_expert_bank import EXPERT_NAMES,ALL_MASKS,retained_from_fit
from scripts.s787_audit_consensus import ROLES

SCHEMA="S791_DATA_DRIVEN_FRAMEPITCH_COMPLEMENTARY_EXPERTS"
def require(ok,why):
    if not ok:raise AssertionError(why)

def validate_score(s,who):
    a,b,c=(s[k] for k in ("TP","FP","FN"))
    require(all(isinstance(v,int) and v>=0 for v in (a,b,c)),who+" counts")
    for key,expected in (("f1",2*a/max(2*a+b+c,1)),
                         ("precision",a/max(a+b,1)),
                         ("recall",a/max(a+c,1))):
        # Expert contribution_table is deliberately a *counts + F1*
        # summary and has no stored precision/recall. Check that F1 was
        # correctly derived; do not fabricate precision/recall fields.
        if key not in s:
            require(key!="f1" and who in
                    {"source","general","bass","harmonic","transition"},
                    f"{who}: missing required {key}")
            continue
        require(math.isclose(s[key],expected,abs_tol=1e-9),
                f"{who}: invalid {key}")
    return s

def source(args):
    r=json.loads(args.report.read_text())
    s=json.loads(args.signed.read_text())
    require(r["schema"]==SCHEMA and r["complete"],"invalid/incomplete S791")
    require(r["fit_recordings_done"]==80 and r["cal_recordings_done"]==32,
            "missing FIT80 or CAL32")
    require(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            ==s["checkpoint_sha256"],"signed S708 checkpoint changed")
    require(s["schema"]=="S708_K_GUIDED_S706_CAL_LOOPS","signed score schema wrong")
    return r,s

def files_changed(args):
    return {v.strip() for v in args.changed.read_text().splitlines() if v.strip()}

def audit_data(r,s,args):
    comps=r["FIT"]["fit_compositions"]
    cal=set(r["CAL"]["frozen_by_composition"])
    require(len(comps)==5 and len(set(comps))==5,"FIT five compositions missing")
    require(cal=={"BN2","Jazz2"} and set(comps).isdisjoint(cal),"CAL/FIT leak")
    require(not r["HELD_opened"] and not r["TEST_opened"],"closed holdouts touched")
    require(r["no_CAL_training_or_selection"],"CAL used as search")
    return dict(training_recordings=80,diagnostic_recordings=32,
                fit_compositions=comps,cal_compositions=sorted(cal),
                signed_s708_sha=r["signed_sha256"],holdouts_closed=True)

def audit_acoustics(r,s,args):
    source=args.bank.read_text()
    tree=ast.parse(source)
    find={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name in ("eligibility","candidate_offers","frame_descriptors"):
        require(name in find,f"missing expert acoustic function {name}")
        terms={n.id for n in ast.walk(find[name]) if isinstance(n,ast.Name)}
        require(not({"truth","labels","oracle_mask_labels","gt"}&terms),
                f"oracle leaked into {name}")
    require("parents=np.stack" in source and "sustain" in source
            and "transition" in source and "bass" in source,
            "physical specialization missing")
    require(tuple(EXPERT_NAMES)==("general","bass","harmonic","transition"),
            "unreviewed expert domains added")
    return dict(specializations=list(EXPERT_NAMES),
        bass_midi="40..47",harmonic_intervals="12,19,24,28,31 semitones",
        lookahead="forbidden, prefix-invariance unit test required",
        caveat="S791 feature audit does not directly inspect raw PCM waveform")

def audit_architecture(r,s,args):
    src=args.train.read_text()
    tree=ast.parse(src)
    routines={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    require({"train_bank","router_model","route","run"}<=routines,
            "missing head training or dynamic router")
    require(r["candidate_masks"]==2**len(EXPERT_NAMES)==16,
            "mask count must come from actual number of candidate heads")
    selected,allowed=retained_from_fit(r["FIT"]["expert_oof"])
    require(selected==r["retained_specialists"],"expert count not FIT-selected")
    require(list(allowed)==r["FIT"]["admissible_masks"],
            "selector allows unvalidated experts")
    require(r["FIT"]["actual_retained_expert_count"]==len(selected),
            "reported count of useful heads inconsistent")
    require("for comp in compositions:" in src and
            "router=router_model(oof,other,allowed)" in src,
            "router must be held-composition OOF by fragment")
    return dict(candidate_experts=len(EXPERT_NAMES),source_heads=1,
                retained_expert_count=len(selected),retained=selected,
                allowed_fragment_masks=len(allowed),maximum_masks=16,
                rule="number determined by FIT cross-composition contribution")

def audit_training(r,s,args):
    fit=r["FIT"]
    table=fit["expert_oof"]
    require(set(table)==set(EXPERT_NAMES)|{"source"},
            "all complementary expert reports missing")
    for name,v in table.items():
        validate_score(v,name)
        for comp,f1 in v["by_composition"].items():
            require(0<=f1<=1, f"{name}: invalid {comp} f1")
    base=validate_score(fit["baseline"],"FIT baseline")
    trial=validate_score(fit["router"],"FIT router")
    bc=fit["baseline_by_composition"]
    tc=fit["router_by_composition"]
    require(set(bc)==set(tc)==set(fit["fit_compositions"]),
            "FIT router missing composition")
    for comp in bc:
        validate_score(bc[comp],comp+" baseline")
        validate_score(tc[comp],comp+" router")
    expected=bool(r["retained_specialists"]) and (
        trial["f1"]>=base["f1"]+fit["min_global_gain"] and
        all(tc[c]["f1"]>=bc[c]["f1"]-fit["max_composition_regression"]
            for c in bc))
    require(bool(fit["router_selected"])==expected,
            "router eligibility not FIT-only")
    return dict(fit_oof_base=base["f1"],fit_oof_router=trial["f1"],
                retained=r["retained_specialists"],
                selected=expected,compositions=list(bc))

def audit_metrics(r,s,args):
    cal=r["CAL"]
    base=validate_score(cal["baseline"],"CAL base shadow")
    trial=validate_score(cal["frozen"],"CAL frozen shadow")
    for prefix in ("baseline","frozen"):
        for field in ("by_composition","by_texture"):
            groups=cal[f"{prefix}_{field}"]
            for k,row in groups.items():validate_score(row,prefix+" "+k)
            for k in ("TP","FP","FN"):
                require(sum(v[k] for v in groups.values())==cal[prefix][k],
                        f"{prefix} {field} {k}: tallies disagree")
    for reg,v in cal["by_register"].items():
        validate_score(v["baseline"],"region "+reg+" baseline")
        validate_score(v["frozen"],"region "+reg+" trial")
    for prefix in ("baseline","frozen"):
        for k in ("TP","FP","FN"):
            require(sum(v[prefix][k] for v in cal["by_register"].values())
                    ==cal[prefix][k],"by-register frame counts differ")
    official=s["frozen_candidate"]["CAL"]
    require(not r["promoted"] and r["decision_clock_samples"]==1024,
            "SHADOW score incorrectly promoted as official 256-hop")
    return dict(cal_shadow_baseline_f1=base["f1"],
                cal_shadow_trial_f1=trial["f1"],
                delta_points=(trial["f1"]-base["f1"])*100,
                frame_pitch_official_S708=official["frame_pitch_f1"],
                other_official_metric_types="unchanged source S708",
                promotion_allowed=False,
                warning="shadow occupation mask is NOT final causal MIDI output")

def audit_live(r,s,args):
    for key in ("promoted","LIVE_changed","final_MIDI_changed",
                "ONSET_OFFSET_changed","HELD_opened","TEST_opened"):
        require(r[key] is False,"S791 protected source was changed: "+key)
    allowed={
       "series/S791/resume.md","series/S791/resultat.md",
       "src/causal_note/s791_framepitch_expert_bank.py",
       "scripts/s791_train_framepitch_complementary.py",
       "scripts/s791_six_role_audits.py",
       "test/test_s791_framepitch_expert_bank.py",
       "test/test_s791_train_framepitch_complementary.py",
       ".github/workflows/note-s791-framepitch.yml"
    }
    changed=files_changed(args)
    require(changed<=allowed,"protected LIVE/midi source touched "+repr(changed-allowed))
    return dict(source_untouched=True,changed_paths=sorted(changed),
                recorded_sample_clock=1024,
                live_latency_not_measured=True,actual_live_midi_not_modified=True)

HANDLERS=dict(data=audit_data,acoustics=audit_acoustics,
  architecture=audit_architecture,training=audit_training,
  metrics=audit_metrics,live=audit_live)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--signed",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--changed",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
      passed=False,experimental_promotion_allowed=False,findings={},errors=[])
    try:
        r,s=source(args)
        result["findings"]=HANDLERS[args.role](r,s,args)
        result["passed"]=True
    except Exception as e:
        result["errors"]=[f"{type(e).__name__}: {e}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S791_AUDIT_"+args.role.upper(),json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:sys.exit(1)

if __name__=="__main__":main()
