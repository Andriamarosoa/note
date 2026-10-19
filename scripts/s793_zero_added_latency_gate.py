"""S793 bridge: zero-extra-latency promotion rule rejects S792 holds.

Historical CI on immutable S792 artifact, without model retraining.
This is NOT a substitute for actual paired hardware microphone timings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from causal_note.s793_zero_added_latency import (
    LatencyViolation,SchedulingContract,SCORES,veto_metric_regression)

def assess_s792(report):
    if report.get('schema')!='S792_FRAMEPITCH_S790_TRANSFER_REAL_MIDI256' \
            or report.get('complete') is not True or report.get('fit_done')!=80 \
            or report.get('cal_done')!=32 or not report.get('CAL',{}).get('official_source_parity'):
        raise LatencyViolation('S792 incomplete, non-parity or unrelated source report')
    variants=report['FIT']['variants']
    if set(variants)!={'0','1','2','4'}:
        raise LatencyViolation('S792 policies not same historical experiment')
    for hold in (1,2,4):
        if variants[str(hold)]['eligible']:
            raise LatencyViolation('non-zero hold erroneously accepted in old research')
        changed=variants[str(hold)]['event_changes']
        if changed['held']<=0:
            raise LatencyViolation('S792 latency experiment missing held NoteOff proof')
    if report['selected_hold']!=0 or report['FIT']['selected_hold']!=0:
        raise LatencyViolation('zero-added-delay selection not preserved')
    base=report['CAL']['baseline']
    chosen=report['CAL']['trial']
    veto_metric_regression(base,chosen)
    for metric in SCORES:
        if base[metric]!=chosen[metric]:
            raise LatencyViolation('S792 selected non-identical output unexpectedly')
    SchedulingContract().validate()
    return dict(schema='S793_ZERO_EXTRA_LATENCY_HISTORICAL_AUDIT',
       historical_run=39328341071,source='S792 real emitted MIDI256',
       accepted_hold_blocks=0,rejected_hold_blocks=[1,2,4],
       reference_metric_identity=True,zero_added_latency_policy=True,
       physical_microphone_latency_measured=False,production_promotion_allowed=False,
       future_candidate_must_supply_repeated_paired_runtime_metrics=True,
       explanation='Historical S792 is a negative delay example, not proof of hardware zero-added-latency')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--s792-report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:
        result=assess_s792(json.loads(args.s792_report.read_text()))
    except Exception as e:
        print('S793_STRICT_LATENCY_REJECT',f'{type(e).__name__}: {e}',flush=True)
        raise SystemExit(1) from e
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('S793_STRICT_LATENCY_POLICY_PASS',json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
