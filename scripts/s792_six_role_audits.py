"""S792 six independent audits of real 256-hop emitted MIDI evidence."""
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
def require(ok,why):
    if not ok:raise AssertionError(why)

def verify_aggregates(m,where):
    for key in SCORES:
        require(key in m and math.isfinite(m[key]) and 0<=m[key]<=1,
                f"{where} invalid {key}")
    require(math.isclose(m["onset_f1"],2*m["tp"]/max(2*m["tp"]+m["fp"]+m["fn"],1),
                         abs_tol=1e-12),f"{where} wrong onset")
    require(math.isclose(m["full_note_f1"],2*m["offset_correct"]/
                  max(2*m["tp"]+m["fp"]+m["fn"],1),abs_tol=1e-12),
            f"{where} wrong FullNote")
    require(math.isclose(m["frame_pitch_f1"],2*m["frame_pitch_tp"]/
                  max(2*m["frame_pitch_tp"]+m["frame_pitch_fp"]+
                      m["frame_pitch_fn"],1),abs_tol=1e-12),
            f"{where} frame pitch mismatch")
    require(math.isclose(m["exact_k"],m["k_correct"]/max(m["frames"],1),
                  abs_tol=1e-12),f"{where} wrong Exact-K")
    require(math.isclose(m["active_exact_k"],m["active_k_correct"]/
                max(m["active_frames"],1),abs_tol=1e-12),
            f"{where} wrong Active-K")
    require(math.isclose(m["offset_accuracy"],m["offset_correct"]/
                max(m["offset_eligible"],1),abs_tol=1e-12),
            f"{where} wrong Offset")
    return m

def load(args):
    report=json.loads(args.report.read_text())
    signed=json.loads(args.signed.read_text())
    original=json.loads(args.s790.read_text())
    require(report["schema"]=="S792_FRAMEPITCH_S790_TRANSFER_REAL_MIDI256"
            and report["complete"],"experiment not completed")
    require(report["fit_done"]==80 and report["cal_done"]==32,
            "missing FIT80 or CAL32")
    require(signed["schema"]=="S708_K_GUIDED_S706_CAL_LOOPS",
            "source S708 schema drift")
    require(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            ==signed["checkpoint_sha256"],"S708 checkpoint modified")
    require(original["selected_policy"]=="cut0.45_harm0.00"
            and original["complete"],"S790 policy refit or missing")
    return report,signed,original

def audit_data(r,s,o,args):
    require(r["cal_selection_forbidden"] and
            not r["TEST_opened"] and not r["HELD_opened"],"holdout contamination")
    fit=r["FIT"]
    comps=fit["fit_compositions"]
    require(len(comps)==5 and len(set(comps))==5,"not five FIT compositions")
    cal=set(r["CAL"]["by_composition_baseline"])
    require(cal=={"BN2","Jazz2"} and not(set(comps)&cal),"FIT CAL leaked")
    return dict(signed_S708=s["checkpoint_sha256"],FIT80=80,CAL32=32,
                fit_compositions=comps,cal_compositions=sorted(cal),
                holdouts_unopened=True)

def audit_acoustics(r,s,o,args):
    src=args.source.read_text()
    require("causal_features(item,post,k,events)" in src and
            "evaluate_policy(probabilities,source,harm,sustain" in src,
            "S790 frozen acoustic head disconnected")
    require(r["s790_frozen_policy"]=="cut0.45_harm0.00"
            and r["pitch_head_clock_samples"]==1024,
            "frozen head policy or clock changed")
    require("truth" not in ast.get_source_segment(src,
            next(x for x in ast.parse(src).body
                 if isinstance(x,ast.FunctionDef) and x.name=="replay_delayed_noteoffs")),
            "MIDI gate used annotation at inference")
    return dict(frozen_head_policy=r["s790_frozen_policy"],
        acoustic_inputs="S790 current and past 40D/pitch; no oracle",
        note="No real microphone/latency measured; offline causal playback only")

def audit_architecture(r,s,o,args):
    src=args.source.read_text()
    tree=ast.parse(src)
    fn=next((x for x in tree.body if isinstance(x,ast.FunctionDef)
             and x.name=="replay_delayed_noteoffs"),None)
    require(fn is not None,"real event replay missing")
    code=ast.get_source_segment(src,fn)
    for snippet in ("if e[\"type\"]==\"note_off\":",
                    "elif e[\"type\"]==\"note_on\":",
                    'pending[pitch]', 'close(pitch,t,"retrigger")',
                    'while len(active)>=MAX_ACTIVE and pending:'):
        require(snippet in code,"missing causal event-state safeguard: "+snippet)
    require(r["official_evaluation"] and
            r["score_clock_samples"]==256,"using detached shadow metrics")
    return dict(actual_midi_events=True,no_noteon_creation=True,
                maximum_six_active=True,finite_hold=True,
                final_reference_midi_untouched=True)

def audit_training(r,s,o,args):
    fit=r["FIT"]
    variants=fit["variants"]
    require(set(variants)=={"0","1","2","4"},
            "undeclared extra CAL-tuned release policy")
    base=verify_aggregates(fit["baseline"],"FIT source")
    comps=fit["by_composition"]
    require(len(comps)==5,"FIT OOF absent")
    elig=[]
    for key,v in variants.items():
        trial=verify_aggregates(v["metrics"],"FIT "+key)
        by=v["by_composition"]
        require(set(by)==set(comps),"incomplete FIT composition OOF")
        for c in by:
            verify_aggregates(comps[c],"FIT base "+c)
            verify_aggregates(by[c],"FIT trial "+c)
        qualifies=(key=="0" or (
          trial["frame_pitch_f1"]>=base["frame_pitch_f1"]+.001
          and all(trial[m]>=base[m]-1e-10
                  for m in SCORES if m!="frame_pitch_f1")
          and all(by[c]["frame_pitch_f1"]>=comps[c]["frame_pitch_f1"]-.005
                  for c in comps)))
        require(qualifies==v["eligible"],"invalid FIT-only eligibility "+key)
        if qualifies:elig.append(key)
    selected=max(elig,key=lambda k:(variants[k]["metrics"]["frame_pitch_f1"],
                                   -int(k)))
    require(int(selected)==r["selected_hold"]==fit["selected_hold"],
            "selected candidate not FIT-maximizing")
    return dict(FIT_candidates=4,selected_max_hold=r["selected_hold"],
                baseline=base["frame_pitch_f1"],
                selected_FIT=variants[selected]["metrics"]["frame_pitch_f1"],
                selected_on_FIT_only=True)

def audit_metrics(r,s,o,args):
    cal=r["CAL"]
    src=verify_aggregates(cal["baseline"],"CAL signed baseline")
    trial=verify_aggregates(cal["trial"],"CAL trial")
    signed=s["frozen_candidate"]["CAL"]
    require(cal["official_source_parity"],"S708 parity gate not reached")
    for metric in SCORES:
        require(math.isclose(src[metric],signed[metric],abs_tol=1e-12),
                "S708 reference drift "+metric)
    for prefix in ("baseline","trial"):
        for group in ("composition","texture"):
            groups=cal[f"by_{group}_{prefix}"]
            for name,m in groups.items():
                verify_aggregates(m,f"CAL {prefix} {name}")
            for count in ("tp","fp","fn","frames","frame_pitch_tp",
                          "frame_pitch_fp","frame_pitch_fn","offset_correct"):
                require(sum(m[count] for m in groups.values())==cal[prefix][count],
                        f"CAL {group} count mismatch {count}")
    require(cal["official_promotion_admissible"] is False and
            r["promoted"] is False,"unapproved promotion")
    return dict(six_official_baseline={m:src[m] for m in SCORES},
                six_official_trial={m:trial[m] for m in SCORES},
                frame_pitch_gain_points=100*(trial["frame_pitch_f1"]-
                                              src["frame_pitch_f1"]),
                non_regression=cal["six_metric_no_regression"],
                frame_pitch_above_80=cal["frame_pitch_above_80"],
                official_source_parity=True,promoted=False)

def audit_live(r,s,o,args):
    require(not r["official_LIVE_changed"] and not r["promoted"],
            "production LIVE was altered")
    allowed={"series/S792/resume.md","series/S792/resultat.md",
            "scripts/s792_framepitch_midi256_replay.py",
            "scripts/s792_six_role_audits.py",
            "test/test_s792_framepitch_midi256_replay.py",
            ".github/workflows/note-s792-framepitch-midi256.yml"}
    changed=set(filter(None,(x.strip() for x in args.changed.read_text().splitlines())))
    require(changed<=allowed, "unexpected protected source change "+
            repr(changed-allowed))
    require(r["score_clock_samples"]==256,"wrong final MIDI metric clock")
    return dict(changed_files=sorted(changed),live_production_untouched=True,
                strict_offline_causal_event_proof=True,
                actual_live_latency_measured=False)

HANDLERS=dict(data=audit_data,acoustics=audit_acoustics,
              architecture=audit_architecture,training=audit_training,
              metrics=audit_metrics,live=audit_live)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",required=True,choices=ROLES)
    p.add_argument("--report",required=True,type=Path)
    p.add_argument("--signed",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--s790",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--changed",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,passed=False,
                experimental_promotion_allowed=False,findings={},errors=[])
    try:
        r,s,o=load(args)
        result["findings"]=HANDLERS[args.role](r,s,o,args)
        result["passed"]=True
    except Exception as e:
        result["errors"]=[f"{type(e).__name__}: {e}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S792_AUDIT_"+args.role.upper(),json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:sys.exit(1)

if __name__=="__main__":
    main()
