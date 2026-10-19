"""S790 six independent audit roles: each can veto Frame Pitch shadow trial."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

from scripts.s787_audit_consensus import ROLES

SOURCE_SCHEMA="S790_INDEPENDENT_CAUSAL_FRAMEPITCH_FIT80_CAL32"
SCALAR_KEYS=("TP","FP","FN")

def check(ok,message):
    if not ok:raise AssertionError(message)

def score(m,where):
    tp,fp,fn=[m[k] for k in SCALAR_KEYS]
    check(all(isinstance(x,int) and x>=0 for x in (tp,fp,fn)),
          f"{where}: invalid counts")
    expected=2*tp/max(2*tp+fp+fn,1)
    check(math.isclose(m["f1"],expected,abs_tol=1e-9),
          f"{where}: F1 formula mismatch")
    check(math.isclose(m["precision"],tp/max(tp+fp,1),abs_tol=1e-9),
          f"{where}: precision")
    check(math.isclose(m["recall"],tp/max(tp+fn,1),abs_tol=1e-9),
          f"{where}: recall")
    return m

def load(args):
    r=json.loads(args.report.read_text())
    s=json.loads(args.signed.read_text())
    src=args.source.read_text(encoding="utf-8")
    tree=ast.parse(src)
    check(r["schema"]==SOURCE_SCHEMA,"wrong S790 experiment schema")
    check(s["schema"]=="S708_K_GUIDED_S706_CAL_LOOPS","wrong S708 schema")
    check(r["complete"] and r["fit_recordings_done"]==80
          and r["cal_recordings_done"]==32,"FIT/CAL incomplete")
    check(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
          ==s["checkpoint_sha256"],"signed S708 changed")
    return r,s,src,tree

def fn(tree,name):
    functions=[x for x in tree.body
        if isinstance(x,ast.FunctionDef) and x.name==name]
    check(len(functions)==1,f"missing function {name}")
    return functions[0]

def data_audit(r,s,src,tree,args):
    check(r["no_CAL_training_or_selection"] and not r["TEST_opened"]
          and not r["HELD_opened"],"training/TEST/HELD contamination")
    comp=r["FIT"]["fit_compositions"]
    check(len(comp)==5 and len(set(comp))==5,"FIT composition OOF missing")
    check(set(r["CAL"]["by_composition_frozen"])=={"BN2","Jazz2"},
          "CAL split contamination")
    check(set(comp).isdisjoint(r["CAL"]["by_composition_frozen"]),
          "FIT/CAL share composition")
    check(r["decision_clock_samples"]==1024,
          "pitch lattice grid modified")
    return dict(FIT_records=80,CAL_records=32,FIT_compositions=comp,
                CAL_compositions=["BN2","Jazz2"],
                signed_s708_sha=s["checkpoint_sha256"],
                exclusion="HELD & TEST not opened")

def acoustic_audit(r,s,src,tree,args):
    f=fn(tree,"causal_features")
    terms={node.id for node in ast.walk(f) if isinstance(node,ast.Name)}
    check("truth" not in terms and "labels" not in terms
          and "targets" not in terms and "y" not in terms,
          "annotations leaked into acoustic extractor")
    check({"spectrum","parents","harmonic","sustain","old","old2",
           "source","active"}.issubset(terms), "not all causal acoustic components present")
    check("INTERVALS=(12,19,24,28,31)" in src,
          "E2 harmonic neighbors changed")
    check(r["feature_dimension"]==40,"feature ABI mismatch")
    return dict(dimensions=40,pitches=49,harmonics=[12,19,24,28,31],
                observations="current and 2 past frames only",
                caution="model's causal features do not guarantee live integration")

def architecture_audit(r,s,src,tree,args):
    f=fn(tree,"evaluate_policy")
    text=ast.get_source_segment(src,f) or ""
    check("harm" in text and "sustain" in text and "base" in text,
          "per-fragment pitch gate not linked to source and acoustics")
    check("np.where(enough_evidence,proposal,base)" in text,
          "not per-pitch per-fragment arbitration")
    check(not {"truth","targets","labels"} & {n.id for n in ast.walk(f)
          if isinstance(n,ast.Name)},"oracle in selector")
    check(r["FIT"]["number_of_policies"]==9,"policy search changed")
    check(r["selected_policy"]==r["FIT"]["selected"],
          "selected policy inconsistent")
    return dict(heads=("signed emitted MIDI", "independent 40D 49-pitch"),
                selection="causal per-frame per-pitch",
                policies=9,selected=r["selected_policy"])

def training_audit(r,s,src,tree,args):
    fit=r["FIT"]
    base=score(fit["baseline"],"FIT base")
    for comp,b in fit["baseline_by_composition"].items():
        score(b,"FIT base "+comp)
    check(len(fit["candidates"])==9 and
          set(fit["candidates"])==set(fit["by_composition"]),
          "missing policy candidates")
    viable=[]
    for key,values in fit["candidates"].items():
        score(values,"FIT "+key)
        per=fit["by_composition"][key]
        check(set(per)==set(fit["baseline_by_composition"]),
              "composition OOF incomplete")
        for comp,m in per.items():score(m,"FIT "+key+" "+comp)
        if values["f1"]>=base["f1"]+fit["min_gain"] and all(
            per[c]["f1"]>=fit["baseline_by_composition"][c]["f1"]-
                 fit["max_comp_regression"]
            for c in per):
            viable.append(key)
    check(set(viable)==set(fit["eligible"]),
          "policy selected using undeclared filters")
    best=max(viable,key=lambda n:(fit["candidates"][n]["f1"],
        min(fit["by_composition"][n][c]["f1"]
            for c in fit["baseline_by_composition"])),default=None)
    check(best==fit["selected"],"chosen policy is not FIT-only maximizer")
    f=fn(tree,"evaluate")
    code=ast.get_source_segment(src,f) or ""
    check(code.find("best=max(eligible")!=-1
          and code.find("best=max(eligible")<
                code.find("for i,spec in enumerate(cal_specs):"),
          "CAL may contaminate policy selection")
    return dict(oof_folds=5,policies=9,eligible=len(viable),
                selected=best,source_only_FIT=True,
                oof_base_f1=base["f1"],
                oof_selected_f1=fit["candidates"][best]["f1"]
                                   if best else base["f1"])

def metrics_audit(r,s,src,tree,args):
    cal=r["CAL"]
    a=score(cal["baseline"],"CAL base")
    b=score(cal["frozen"],"CAL candidate")
    for stage in ("baseline","frozen"):
        by=cal["by_composition_"+stage]
        bytexture=cal["by_texture_"+stage]
        for name,m in by.items():score(m,"composition "+name)
        for name,m in bytexture.items():score(m,"texture "+name)
        for key in SCALAR_KEYS:
            check(sum(v[key] for v in by.values())==cal[stage][key],
                  "CAL composition tally failed "+key)
            check(sum(v[key] for v in bytexture.values())==cal[stage][key],
                  "CAL texture tally failed "+key)
    for reg,stages in cal["by_register"].items():
        score(stages["baseline"],"base "+reg)
        score(stages["trial"],"trial "+reg)
    for key in SCALAR_KEYS:
        check(sum(v["baseline"][key] for v in cal["by_register"].values())
              ==a[key],f"CAL base by-pitch register missing {key}")
        check(sum(v["trial"][key] for v in cal["by_register"].values())
              ==b[key],f"CAL trial by-pitch register missing {key}")
    official=s["frozen_candidate"]["CAL"]
    check(not official["all_targets_above_80"],"new model promotion unverified")
    check(not r["promoted"] and r["decision_clock_samples"]==1024,
          "shadow precision/recall wrongly promoted")
    return dict(shadow_baseline_f1=a["f1"],shadow_trial_f1=b["f1"],
                delta_points=100*(b["f1"]-a["f1"]),
                by_register={k:v["trial"]["f1"] for k,v in cal["by_register"].items()},
                signed_s708_framepitch_final=official["frame_pitch_f1"],
                signed_s708_official_six={k:official[k] for k in (
                    "onset_f1","exact_k","active_exact_k","offset_accuracy",
                    "full_note_f1","frame_pitch_f1")},
                no_official_midi_promotion=True)

def live_audit(r,s,src,tree,args):
    for key in ("LIVE_changed","final_MIDI_changed","ONSET_OFFSET_changed",
                "promoted","HELD_opened","TEST_opened"):
        check(not r[key],f"forbidden runtime mutation: {key}")
    allowed={
        "series/S790/resume.md",
        "series/S790/resultat.md",
        "scripts/s790_independent_framepitch_expert.py",
        "scripts/s790_six_role_audits.py",
        "test/test_s790_independent_framepitch_expert.py",
        "test/test_s790_six_role_audits.py",
        ".github/workflows/note-s790-framepitch.yml"
    }
    changed={x.strip() for x in args.changed.read_text().splitlines() if x.strip()}
    check(changed<=allowed,f"protected file(s) changed: {sorted(changed-allowed)}")
    f=fn(tree,"causal_features")
    code=ast.get_source_segment(src,f) or ""
    check("np.concatenate((sustain[:1],sustain[:-1])" in code,
          "temporal input lacks causal past-only shift")
    check("occupancy_from_emitted" in code,
          "candidate does not use native emitted MIDI state")
    return dict(changed_paths=sorted(changed),
                final_midi_unchanged=True,onset_offset_unchanged=True,
                shadow_clock=1024,latency_measured=False,
                live_integration_verified=False)

HANDLERS=dict(data=data_audit,acoustics=acoustic_audit,
              architecture=architecture_audit,training=training_audit,
              metrics=metrics_audit,live=live_audit)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--signed",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--changed",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    response=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
                  passed=False,experimental_promotion_allowed=False,
                  findings={},errors=[])
    try:
        result,signed,source,tree=load(args)
        response["findings"]=HANDLERS[args.role](
            result,signed,source,tree,args)
        response["passed"]=True
    except Exception as error:
        response["errors"]=[f"{type(error).__name__}: {error}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(response,indent=2,sort_keys=True)+"\n")
    print("S790_AUDIT_"+args.role.upper(),json.dumps(response,sort_keys=True),flush=True)
    if not response["passed"]:sys.exit(1)

if __name__=="__main__":
    main()
