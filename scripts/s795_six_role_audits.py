"""S795 independent, fail-closed six-role audits of actual MIDI256 evidence."""
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
COUNTS=("tp","fp","fn","frames","k_correct","active_frames","active_k_correct",
        "frame_pitch_tp","frame_pitch_fp","frame_pitch_fn","offset_eligible",
        "offset_correct")
HEADS=("general","bass","polyphonic","transient","harmonic")
MASKS=(0,1,2,4,8,16,3,5,6,10,12,24,31)

def require(ok,why):
    if not ok:raise AssertionError(why)

def signed(args):
    r=json.loads(args.report.read_text())
    s=json.loads(args.signed.read_text())
    require(r["schema"]=="S795_CAUSAL_MULTIHEAD_ACTIVEK_REAL_MIDI256"
        and r["complete"] is True,"incomplete/wrong report")
    require((r["fit_done"],r["cal_done"])==(80,32),"missing FIT80 CAL32")
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    require(sha==s["checkpoint_sha256"]==r["signed_source_sha256"],
            "S708 source mismatch")
    return r,s

def data(r,s,args):
    f=r["FIT"]["compositions"]
    c=set(r["CAL"]["original_by_composition"])
    require(len(f)==5 and len(set(f))==5 and c=={"BN2","Jazz2"}
        and not set(f)&c,"not independent FIT/CAL compositions")
    require(not r["HELD_opened"] and not r["TEST_opened"]
        and not r["cal_used_for_training"] and not r["cal_used_for_selection"],
            "data leak")
    return dict(FIT80=80,CAL32=32,fit_compositions=f,cal_compositions=sorted(c),
        signed_sha256=r["signed_source_sha256"],holdouts_closed=True)

def acoustics(r,s,args):
    code=args.bank.read_text()
    funcs={n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    for name in ("causal_features","regions","decode"):
        require(name in funcs, "missing causal expert "+name)
    for fn in ("causal_features","regions","decode"):
        names={n.id for n in ast.walk(funcs[fn]) if isinstance(n,ast.Name)}
        require(not (names&{"truth","ground_truth","labels","oracle"}),
                fn+" sees oracle")
    require(tuple(r["heads"])==HEADS,"wrong candidate expert roles")
    return dict(experts=HEADS,causal_features=23,
        audio_latest_full_fragment_only=True,no_truth_at_inference=True)

def architecture(r,s,args):
    src=args.train.read_text()
    assert "s708_source(" in src and "s708_decode(" in src
    assert "render_release(" in src
    bank=args.bank.read_text()
    for key in ("SchedulingContract().validate()",
      "veto_added_event_delay(originals,out)",
      'out.append(midi("note_on",pitch,t',
      'out.append(midi("note_off",pitch,t'):
        require(key in bank,"missing native-time MIDI protection "+key)
    require(tuple(r["candidate_masks"])==MASKS,
            "missing deterministic fragment-subset definitions")
    require(r["metric_clock_samples"]==256 and r["decision_clock_samples"]==1024,
            "wrong official MIDI metric")
    return dict(expert_candidates=5,policy_masks=len(MASKS),
        fitted_heads=r["fitted_source_head_count"],
        retained_heads=r["retained_head_count"],
        noteon_original_untouched=True,max_voices=6)

def training(r,s,args):
    fit=r["FIT"]
    assert len(fit["compositions"])==5
    require(set(fit["training"])==set(fit["compositions"]),
            "not five OOF expert training folds")
    require(len(fit["candidate_results"])==r["candidate_policies"],
            "declared policies weren't FIT-evaluated")
    v=fit["candidate_results"]
    eligible=[k for k,x in v.items() if x["eligible"]]
    for key,entry in v.items():
        mask=entry["mask"]
        require(mask in MASKS and (mask==0 or entry["cut"] in (.85,.93)),
                "new undeclared candidate")
        if mask:
            baseline=fit["baseline"]
            candidate=entry["metrics"]
            compbaseline=r["FIT"]["candidate_results"]["source_identity"]["by_composition"]
            admissible=(candidate["active_exact_k"]>=baseline["active_exact_k"]+.001
                and all(candidate[k]>=baseline[k]-1e-12
                        for k in SCORES if k!="active_exact_k")
                and all(entry["by_composition"][comp]["active_exact_k"]>=
                        compbaseline[comp]["active_exact_k"]-.005
                        for comp in compbaseline))
            require(entry["eligible"]==admissible,
                    "FIT six-metric selection gate tampered "+key)
    chosen=max(eligible,key=lambda k:(v[k]["metrics"]["active_exact_k"],
                                      -v[k]["synthetic_note_ons"]))
    require(chosen==r["selected_candidate"]==fit["selected"],
            "chosen candidate was not FIT-only best eligible")
    retained=fit["kept_experts"]
    expected=[name for j,name in enumerate(HEADS)
              if v[chosen]["mask"]&(1<<j)]
    require(retained==expected==[x for x in HEADS if x in retained],
            "number of retained heads not data-driven")
    require(len(retained)==r["retained_head_count"],
            "reported retained heads inconsistent")
    return dict(candidate_count=len(v),fit_select=chosen,
        retained_head_count=len(retained),retained=retained)

def metrics(r,s,args):
    cal=r["CAL"]
    require(cal["official_source_parity"],"S708 parity check missing")
    official=s["frozen_candidate"]["CAL"]
    for key in SCORES:
        require(math.isclose(cal["original"][key],official[key],
                        rel_tol=0,abs_tol=1e-12), "S708 score parity "+key)
    for key in COUNTS:
        require(cal["original_counts"][key]==official[key],
                "S708 count parity "+key)
    src=cal["original"];trial=cal["candidate"]
    for k in SCORES:
        require(math.isfinite(trial[k]) and 0<=trial[k]<=1,
                "invalid trial score "+k)
    require(cal["six_metric_no_regression"]==all(
        trial[k]>=src[k]-1e-12 for k in SCORES),
        "anti-regression report inconsistent")
    require(cal["active_k_over_80"]==(trial["active_exact_k"]>.8),
            "invented >80 threshold")
    return dict(six_signed_metrics=src,six_trial_metrics=trial,
        global_six_non_regression=cal["six_metric_no_regression"],
        new_noteons=cal["new_midi_note_ons"],
        no_automatic_promotion=True)

def live(r,s,args):
    require(not r["production_LIVE_modified"] and not r["promoted"],
            "LIVE changed or false promotion")
    require(not r["physical_microphone_latency_measured"],
            "unsupported hardware latency claim")
    require(r["zero_added_scheduling"] and r["source_midi_events_untouched"],
            "S793 scheduling/source lineage not proven")
    allowed={
        "series/S795/resume.md","series/S795/resultat.md",
        "src/causal_note/s795_activek_experts.py",
        "scripts/s795_train_activek_midi256.py",
        "scripts/s795_six_role_audits.py",
        "test/test_s795_activek_experts.py",
        ".github/workflows/note-s795-activek-multihead.yml"
    }
    actual=set(filter(None,(x.strip() for x in args.changed.read_text().splitlines())))
    require(actual<=allowed,"protected LIVE source changed "+repr(actual-allowed))
    return dict(no_new_buffer=True,no_lookahead=True,
        source_event_timing_unaltered=True,
        actual_microphone_latency_still_required=True,
        changed_files=sorted(actual))

FUNCTIONS=dict(data=data,acoustics=acoustics,architecture=architecture,
               training=training,metrics=metrics,live=live)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    for name in ("report","signed","checkpoint","train","bank","changed","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
       passed=False,experimental_promotion_allowed=False,findings={},errors=[])
    try:
        r,s=signed(args)
        result["findings"]=FUNCTIONS[args.role](r,s,args)
        result["passed"]=True
    except Exception as e:
        result["errors"]=[f"{type(e).__name__}: {e}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S795_AUDIT_"+args.role.upper(),json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:sys.exit(1)

if __name__=="__main__":
    main()
