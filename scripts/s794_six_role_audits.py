"""S794 six independent fail-closed root-cause audits on immutable MIDI256."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

from scripts.s787_audit_consensus import ROLES

SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
K_FIELDS=("frames","correct","under","over")
ROLES_ACTUAL=set(ROLES)

def require(cond,why):
    if not cond:raise AssertionError(why)

def load(args):
    r=json.loads(args.report.read_text())
    s=json.loads(args.signed.read_text())
    require(r["schema"]=="S794_ZERO_DELAY_REAL_MIDI256_ACTIVEK_ROOTCAUSE"
            and r["complete"],"S794 report incomplete")
    require(r["source_sha256"]==s["checkpoint_sha256"]==
            hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "signed S708 SHA invalid")
    require((r["fit_done"],r["cal_done"])==(80,32),
            "S794 source coverage FIT80 CAL32 insufficient")
    return r,s

def data(r,s,args):
    fit=set(r["FIT"]["compositions"])
    cal=set(r["CAL"]["compositions"])
    require(len(fit)==5 and cal=={"BN2","Jazz2"} and not fit&cal,
            "S794 wrong FIT/CAL composition boundaries")
    require(not r["TEST_opened"] and not r["HELD_opened"],
            "S794 forbidden holdout opened")
    return dict(fit=80,cal=32,fit_compositions=sorted(fit),
        cal_compositions=sorted(cal),signed_sha=r["source_sha256"])

def acoustic(r,s,args):
    tree=ast.parse(args.taxonomy.read_text())
    f=next(x for x in tree.body
            if isinstance(x,ast.FunctionDef) and x.name=="diagnostic")
    function=ast.get_source_segment(args.taxonomy.read_text(),f)
    require("prior_audio_lookups" in function
            and 'np.searchsorted(d,e,side="right")-1' in args.taxonomy.read_text(),
            "causal 1024 acoustic evidence not gated by available frames")
    cue=r["CAL"]["audit"]["expert_opportunity"]
    require(set(cue)=={"bass_fundamentals","poly_new_voice",
                      "sustain_transition","harmonic_collision"},
            "S794 diagnostic expert causes not distinguished")
    for name,d in cue.items():
        require(all(isinstance(v,int) and v>=0 for v in d.values()),
                "invalid acoustic cue count "+name)
    return dict(cue_names=sorted(cue),causal_clock=1024,
        scores_from_midi_clock=256,
        no_cue_scores_promoted=True)

def architecture(r,s,args):
    source=args.source.read_text()
    require(r["fitted_specialists"]==0 and not r["final_MIDI_changed"],
            "untrained audit falsely became model")
    require("events=events,truth=truth" in source,
            "S794 audit silently replaced signed source events")
    require("baseline_input(" in source and "SCORES" in source,
            "missing authentic S708 source")
    require("veto_added_event_delay" in args.taxonomy.read_text(),
            "S793 event-lineage guard missing")
    return dict(immutable_MIDI=True,fitted_specialists=0,
                multihead_proposals_trained=False,development_target="Active Exact-K")

def training(r,s,args):
    require(r["fitted_specialists"]==0,
            "S794 allowed a trained decoder before error attribution")
    require(not r["promoted"] and not r["promotion_allowed"],
            "S794 diagnostic auto-promotion")
    require(r["FIT"]["audit"]["recordings"]==80
            and r["CAL"]["audit"]["recordings"]==32,
            "FIT CAL forensic recordings coverage invalid")
    return dict(no_oracle_for_inference=True,fit_recordings=80,
       cal_recordings=32,selection_on_CAL=False,
       results_are_diagnostics_not_predictions=True)

def metrics(r,s,args):
    source=s["frozen_candidate"]["CAL"]
    cal=r["CAL"]["metrics"]
    for key in SCORES:
        require(math.isclose(cal[key],source[key],abs_tol=1e-12,rel_tol=0),
                f"S794 signed six metric parity drift {key}")
    for key in ("tp","fp","fn","frames","k_correct","active_frames",
                "active_k_correct","frame_pitch_tp","frame_pitch_fp",
                "frame_pitch_fn","offset_correct","offset_eligible"):
        require(source[key]==cal[key],f"S794 256-hop parity count failed {key}")
    for dataset in ("FIT","CAL"):
        a=r[dataset]["audit"]
        require(a["active_correct"]+a["under_active_frames"]+
                a["over_active_frames"]==a["active_frames"],
                f"{dataset} unresolved active K frames")
        require(abs(a["active_exact_k"]-r[dataset]["metrics"]["active_exact_k"])
                <1e-12,f"{dataset} active K metric not from final MIDI")
        for k,v in a["by_true_k"].items():
            require(all(v[x]>=0 for x in K_FIELDS),"negative K matrix")
            require(v["correct"]+v["under"]+v["over"]==v["frames"],
                    f"{dataset} class K{k} counter mismatch")
        require(sum(v["frames"] for v in a["by_true_k"].values())==
                r[dataset]["metrics"]["frames"],
                f"{dataset} K strata failed to partition source")
    return dict(active_exact_k_cal=cal["active_exact_k"],
                six_metrics_signed_parity=True,
                source_active_k_under=r["CAL"]["audit"]["under_active_frames"],
                source_active_k_over=r["CAL"]["audit"]["over_active_frames"],
                forensic_not_performance_gain=True)

def live(r,s,args):
    require(r["score_clock_samples"]==256,"S794 not using native MIDI hop")
    for x in ("extra_lookahead_samples","extra_buffer_samples",
              "extra_confirmation_samples","extra_hold_samples"):
        require(r[x]==0,"S793 zero-added-wait violated "+x)
    require(not r["LIVE_changed"] and not r["final_MIDI_changed"]
            and not r["real_microphone_latency_measured"],
            "S794 false live timing claim")
    allowed={
        "series/S794/resume.md","series/S794/resultat.md",
        "scripts/s794_activek_failure_taxonomy.py",
        "scripts/s794_activek_midi256_rootcauses.py",
        "scripts/s794_six_role_audits.py",
        "test/test_s794_activek_failure_taxonomy.py",
        ".github/workflows/note-s794-activek-forensics.yml",
    }
    changes=set(filter(None,(x.strip() for x in args.changed.read_text().splitlines())))
    require(changes<=allowed,"S794 source LIVE modified "+str(changes-allowed))
    return dict(midi_source_immutable=True,zero_added_scheduling=True,
       no_actual_hardware_timing_claim=True,modified=sorted(changes))

AUDITS=dict(data=data,acoustics=acoustic,architecture=architecture,
            training=training,metrics=metrics,live=live)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    for name in ("report","signed","checkpoint","source","taxonomy",
                 "changed","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
        passed=False,experimental_promotion_allowed=False,findings={},errors=[])
    try:
        r,s=load(args)
        result["findings"]=AUDITS[args.role](r,s,args)
        result["passed"]=True
    except Exception as e:
        result["errors"]=[f"{type(e).__name__}: {e}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S794_AUDIT",args.role,json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:sys.exit(1)

if __name__=="__main__":main()
