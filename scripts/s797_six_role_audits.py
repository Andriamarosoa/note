"""S797 six independent read-only audit roles: synthetic research only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from scripts.s787_audit_consensus import ROLES

def require(ok,message):
    if not ok:raise AssertionError(message)

def inspect(role,smoke,failed,router):
    r=json.loads(Path(smoke).read_text())
    p=json.loads(Path(failed).read_text())
    code=Path(router).read_text()
    x=r["experiments"];old=p["FIT"]["candidate_results"]
    require(r["schema"]=="S797_FRAGMENT_DYNAMIC_SYNTHETIC_SMOKE_V1"
        and r["test_domain"]=="SYNTHETIC-FIXTURE-ONLY"
        and r["not_real_audio"],"fixture masquerading as GuitarSet")
    require(len(x)==5 and set(x)=={"identity","consensus","harmonic_veto","poly_gap","adaptive"},
        "non-predeclared / missing loops")
    require(not any(r[k] for k in (
        "fit_score_claimed","cal_score_claimed","model_retrained",
        "selected_on_cal","real_dataset_validated","physical_latency_measured",
        "promoted","official_six_metrics_changed")),"false quality claim")
    if role=="data":
        require(p["fit_done"]==80 and p["cal_done"]==32 and not p["complete"],
            "S795 failure archive mismatch")
        require(not p["HELD_opened"] and not p["TEST_opened"]
            and not p["cal_used_for_training"] and not p["cal_used_for_selection"],
            "held-out leakage")
        return dict(fit80_archive=True,cal32_invalid=True,held_closed=True,
            fixture_only=True)
    if role=="acoustics":
        require(len(p["heads"])==5 and "harmonic" in p["heads"]
            and "bass" in p["heads"],"non-heterogeneous head provenance")
        require(x["consensus"]["false_fixture_attacks"]==1
            and x["harmonic_veto"]["false_fixture_attacks"]==0,
            "harmonic counterexample not tested")
        return dict(source_heads=p["heads"],synthetic_alias_blocked=True,
            no_real_acoustic_gain_claimed=True)
    if role=="architecture":
        require(x["adaptive"]["mask_variety"]>=3,
            "only global/unchanging head mask")
        require("high[name]" in code and "source_count" in code
            and "consensus" in code and "k_gap" in code,
            "dynamic fragment input missing")
        return dict(dynamic_masks=x["adaptive"]["router"]["fragment_masks"],
            per_fragment=True,source_decoder_unchanged=True)
    if role=="training":
        require(p["retained_head_count"]==0 and p["selected_candidate"]=="source_identity",
            "S795 failed selector misrepresented")
        require(len(old)==25 and sum(int(v["eligible"]) for v in old.values())==1,
            "past mask eligibility changed")
        return dict(new_models_trained=0,ablation_modes=list(x),
            deployed_models=0,no_duplicate_fit80_training=True)
    if role=="metrics":
        base=p["FIT"]["baseline"]
        best=max(old.items(),key=lambda kv:kv[1]["metrics"]["active_exact_k"])
        require(best[1]["metrics"]["active_exact_k"]>base["active_exact_k"]
            and best[1]["metrics"]["full_note_f1"]<base["full_note_f1"]
            and best[1]["metrics"]["offset_accuracy"]<base["offset_accuracy"],
            "known cross-metric regressions hidden")
        require(x["adaptive"]["true_fixture_hits"]==3
            and x["adaptive"]["false_fixture_attacks"]==0
            and x["identity"]["true_fixture_hits"]==0,
            "synthetic fixture replay inconsistent")
        return dict(s795_best_fit_active_k=best[1]["metrics"]["active_exact_k"],
            s795_best_fit_fullnote=best[1]["metrics"]["full_note_f1"],
            s797_real_six_metrics="NOT MEASURED",synthetic_only=True)
    if role=="live":
        require(all(v["native_events_unchanged"] and
            v["decoder"]["max_added_native_delay_samples"]==0 and
            v["router"]["extra_lookahead_samples"]==0 and
            v["router"]["extra_hold_samples"]==0 for v in x.values()),
            "latency or source mutation in tests")
        return dict(native_midi_clock_unchanged_in_fixture=True,
            physical_cpu_microphone_latency="NOT MEASURED",
            no_production_live_promotion=True)
    raise AssertionError("unknown role")

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--role",required=True,choices=ROLES)
    for n in ("smoke","failed","router","out"):
        a.add_argument("--"+n,required=True,type=Path)
    opt=a.parse_args()
    r=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=opt.role,
        passed=False,experimental_promotion_allowed=False,findings={},errors=[])
    try:
        r["findings"]=inspect(opt.role,opt.smoke,opt.failed,opt.router)
        r["passed"]=True
    except Exception as exc:
        r["errors"]=[f"{type(exc).__name__}: {exc}"]
    opt.out.parent.mkdir(parents=True,exist_ok=True)
    opt.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print("S797_AUDIT",json.dumps(r,sort_keys=True),flush=True)
    if not r["passed"]:sys.exit(1)
