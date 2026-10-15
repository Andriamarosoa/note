"""S641: frozen-output causal delay counterfactual, NOT a new recogniser.

For each S638 NoteOn, simulate a nonnegative post-decision MIDI output hold
without altering note pitch, original neural probabilities, candidate selection
or the true audio. Evaluate exactly the same past-only 1:1 matching rule and
count both new TP and LOST TP for every fixed delay.

This is a latency tradeoff DIAGNOSTIC on already-exposed DEV. The test
does not delay physical audio, does not run an adaptive policy, and cannot
be used to claim a new trained model or changed official F1. Candidate
output after audio EOF is retained for scoring instead of disappearing.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from causal_note.s635_shared_midi_metric import causal_score,extract_reference,extract_events

HOP_MS=256*1000./44100.
DELAYS_HOPS=(0,1,2,3,4,5,6,8,10,12,14)
PITCH_MIN,PITCH_MAX=40,76


def score_delay(reference,original_noteons,additional_frames):
    offset=float(additional_frames)*HOP_MS/1000.
    shifted=[(t+offset,pitch,index) for t,pitch,index in original_noteons]
    return causal_score(reference,shifted,80.)


def evaluate(manifest,s638,s640):
    if s638["overall"]["note"]["80"]["TP"]!=901:
        raise AssertionError("S638 frozen score changed")
    if s640["count_missed_by_both"]!=613 or not s640["matched_original_NOTE_S614_events"]:
        raise AssertionError("S640 missed-note proof changed")
    per_shift={}
    all_rows={h:[] for h in DELAYS_HOPS}
    for clip in manifest["clips"]:
        truth=[r for r in extract_reference(clip["reference"],clip["duration_s"])
               if PITCH_MIN<=r[1]<=PITCH_MAX]
        predicted=[e for e in extract_events(clip["note"]["events"],clip["duration_s"])
                   if PITCH_MIN<=e[1]<=PITCH_MAX]
        baseline=score_delay(truth,predicted,0)
        previous=set(int(pair[0]) for pair in baseline["matches"])
        if baseline["TP"]!=s638["by_clip"][clip["clip_id"]]["note"]["80"]["TP"]:
            raise AssertionError("Original 80ms matched truth must be identical")
        for hops in DELAYS_HOPS:
            current=score_delay(truth,predicted,hops)
            now=set(int(pair[0]) for pair in current["matches"])
            all_rows[hops].append(dict(clip=clip["clip_id"],
              role="solo" if clip["clip_id"].endswith("_solo") else "accompaniment",
              TP=current["TP"],FP=current["FP"],FN=current["FN"],F1=current["F1"],
              gained_truth=len(now-previous),lost_truth=len(previous-now),
              score_changed_sources=int(current["F1"]!=baseline["F1"]),
              late_emissions_after_EOF=sum(
                t+hops*HOP_MS/1000.>clip["duration_s"] for t,pitch,idx in predicted)))
    for hops in DELAYS_HOPS:
        subset=all_rows[hops]
        metrics={key:sum(row[key] for row in subset) for key in
                 ("TP","FP","FN","gained_truth","lost_truth",
                  "late_emissions_after_EOF")}
        tp,fp,fn=metrics["TP"],metrics["FP"],metrics["FN"]
        metrics["F1"]=2*tp/max(1,2*tp+fp+fn)
        metrics["delay_ms"]=hops*HOP_MS
        metrics["improved_sources"]=sum(row["F1"]>all_rows[0][i]["F1"]
                                      for i,row in enumerate(subset))
        metrics["regressed_sources"]=sum(row["F1"]<all_rows[0][i]["F1"]
                                       for i,row in enumerate(subset))
        metrics["by_role"]={}
        for role in ("solo","accompaniment"):
            group=[row for row in subset if row["role"]==role]
            t=sum(x["TP"] for x in group)
            f=sum(x["FP"] for x in group)
            n=sum(x["FN"] for x in group)
            metrics["by_role"][role]=dict(TP=t,FP=f,FN=n,F1=2*t/max(1,2*t+f+n))
        per_shift[str(hops)]=metrics
    base=per_shift["0"]
    for key in ("TP","FP","FN","F1"):
        if abs(base[key]-s638["overall"]["note"]["80"][key])>1e-9:
            raise AssertionError("Original reference score drift")
    ranking=sorted(((v["F1"],int(k)) for k,v in per_shift.items()),reverse=True)
    return dict(status="S641_FROZEN_FIXED_HOLD_DELAY_COUNTERFACTUAL_DEV_ONLY",
        original_model_and_scores_unchanged=True,
        original_S80_unchanged=True,
        no_audio_lookahead_used=True,
        no_independent_heldout_or_promotion=True,
        hypothesis="causal output hold of each NOTE NoteOn; output after EOF retained",
        originally_missed_both=613,originally_missed_with_early_same_pitch=93,
        original_onsets=base["TP"]+base["FP"],
        delays_hops=list(DELAYS_HOPS),
        results=per_shift,
        highest_scoring_exploratory_hops=ranking[0][1],
        highest_scoring_exploratory_F1=ranking[0][0],
        tuning_would_invalidate_independence=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s638",type=Path,required=True)
    ap.add_argument("--s638-metric",type=Path,required=True)
    ap.add_argument("--s640",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Immutable S641 diagnostic evidence")
    s638=json.loads(args.s638.read_text())
    original_metrics=json.loads(args.s638_metric.read_text())
    s640=json.loads(args.s640.read_text())
    result=evaluate(s638,original_metrics,s640)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("S641 FIXED HOLD AUDIT",json.dumps({
        "original_note":result["results"]["0"],
        "delays":[dict(hops=int(k),delay_ms=v["delay_ms"],TP=v["TP"],FP=v["FP"],FN=v["FN"],
                       F1=v["F1"],gained=v["gained_truth"],lost=v["lost_truth"],
                       improved=v["improved_sources"],regressed=v["regressed_sources"])
                  for k,v in result["results"].items()]},indent=2))


if __name__=="__main__":main()
