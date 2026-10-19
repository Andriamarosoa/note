"""Run FIVE predeclared fragment-routing loops on one synthetic MIDI fixture.

NOT GuitarSet, not fitted optimization, not CAL, not a physical latency test.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.s797_fragment_loop_fixture import trial
from causal_note.s797_fragment_router import MODES
from causal_note.s793_zero_added_latency import SOURCE_ID

TRUTH={42,55,60}

def run():
    rows={}
    for mode in MODES:
        attacks,masks,meta,events,changes,original=trial(mode)
        native=[e for e in events if e.get(SOURCE_ID) is not None]
        source_unchanged=(
            [(e["type"],e["pitch"],e["available_sample"]) for e in native] ==
            [(e["type"],e["pitch"],e["available_sample"]) for e in original])
        if not source_unchanged or changes["max_added_native_delay_samples"]!=0:
            raise AssertionError("S797 native MIDI timing mutated")
        rows[mode]=dict(synthetic_only=True,emitted_attacks=sorted(attacks),
            true_fixture_hits=len(attacks&TRUTH),
            false_fixture_attacks=len(attacks-TRUTH),
            missed_fixture_attacks=len(TRUTH-attacks),
            mask_variety=len([v for v in meta["fragment_masks"] if v!="0"]),
            native_events_unchanged=source_unchanged,decoder=changes,router=meta)
    if set(rows)!=set(MODES) or rows["identity"]["emitted_attacks"] or rows["adaptive"]["false_fixture_attacks"]!=0:
        raise AssertionError("S797 five smoke loops or zero-decoy guard failed")
    return dict(schema="S797_FRAGMENT_DYNAMIC_SYNTHETIC_SMOKE_V1",
        test_domain="SYNTHETIC-FIXTURE-ONLY",not_real_audio=True,
        fit_score_claimed=False,cal_score_claimed=False,
        model_retrained=False,selected_on_cal=False,real_dataset_validated=False,
        physical_latency_measured=False,promoted=False,
        experiments=rows,reference="S708",official_six_metrics_changed=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    report=run();a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S797_SYNTHETIC_FIVE_LOOPS",json.dumps({
        name:dict(TP=x["true_fixture_hits"],FP=x["false_fixture_attacks"],
                  FN=x["missed_fixture_attacks"])
        for name,x in report["experiments"].items()}),flush=True)
