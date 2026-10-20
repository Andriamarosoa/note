"""S809 independent architectural + genuine 80-track proof, six CI audit roles.

PASS means the observed separation contract was checked, NOT improvements in
musical F1, trained per-axis neural models or measured microphone latency.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames",
 "active_k_correct","frame_pitch_tp","frame_pitch_fp","frame_pitch_fn",
 "offset_eligible","offset_correct")
AXES=("exact_k","active_exact_k","onset","offset","frame_pitch","full_note")
ROLES=("data","acoustics","architecture","training","metrics","live")
def check(role,r):
 def need(ok,msg):
  if not ok:raise AssertionError(msg)
 need(r.get("schema")=="S809_GUITARSET_FIT80_SIX_INDEPENDENT_AXES_REPLAY_V1" and
      r.get("complete"),"signed S809 data absent")
 need(r["tracks"]==80 and len(r["compositions"])==5 and
      r["source_run"]==39352880035,"80 FIT OOF recordings not confirmed")
 need(not any(r[k] for k in ("CAL_opened","HELD_opened","TEST_opened")) and
      r["reference"]=="S708" and not r["promotion"],"official reference or reserved split altered")
 need(r["source_MIDI_unchanged"],"protected full-note stream modified")
 if role=="data":
  need(r["source_counts"]["frames"]==306536 and
       r["source_counts"]["k_correct"]==189124 and
       set(r["source_counts"])==set(COUNT_KEYS),
       "source 12 counters do not match signed S800")
  return {"FIT_track_count":80,"source_parity":True}
 if role=="acoustics":
  need(r["K_posterior_from_S800"]==
       "ARGMAX_ONEHOT_PROXY_NOT_ORIGINAL_PROBABILITIES" and
       r["no_lookahead_samples"]==0,
       "fabricated seven-class original K source or future leakage")
  return {"source_uses_only_available_frames":True,
          "K_probabilities_limit_disclosed":True}
 if role=="architecture":
  need(r["axes_count"]==6 and r["heads_per_axis"]==2 and
       r["selectors_per_axis"]==1 and r["decoders_per_axis"]==1 and
       set(r["axis_names"])==set(AXES) and
       set(r["separate_head_names"])==set(AXES) and
       set(r["separate_selector_names"])==set(AXES),
       "missing actual per-axis head/selector/decoder")
  need(r["all_six_axis_private_state"] and
       r["mutation_scenarios_for_all_six_axes"]==6 and
       r["all_five_other_axes_bitwise_identical_under_each_of_six_mutations"] and
       r["no_single_joint_decoder"],
       "cross-axis contamination")
  need(len(set(r["separate_selector_names"].values()))==6 and
       len({h for v in r["separate_head_names"].values() for h in v})==12,
       "reused/mislabeled heads or selectors")
  return {"six_noninterfering_chains":True,
          "all_thirty_cross_axis_mutations_preserved":True}
 if role=="training":
  need(r["scientific_limitation"].startswith("New independent acoustic signal alternatives are deterministic untrained shadow heads."),
       "unsupported learned independent weights promotion")
  return {"models_are_not_falsely_claimed_trained":True,
          "no_training_occured":True}
 if role=="metrics":
  need(set(r["source_scores"])==set(SCORES) and
       set(r["per_axis_shadow_diagnostics"])==set(AXES) and
       r["per_axis_shadow_metrics_are_NOT_final_midi_six_scores"],
       "mixed final MIDI S704 scores with separate axis diagnostics")
  return {"original_six_MIDI_metrics":True,"separate_shadow_scores":True}
 if role=="live":
  need(not r["production_LIVE_modified"] and not r["CPU_iRig_latency_measured"]
       and r["extra_hold_samples"]==0 and r["no_lookahead_samples"]==0,
       "unmeasured physical latency or future confirmation")
  return {"production_still_S708":True,"physical_latency_not_measured":True}
 raise ValueError("unknown role")
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--role",choices=ROLES,required=True)
 p.add_argument("--report",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 row=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,passed=False,
          findings={},errors=[],experimental_promotion_allowed=False)
 try:
  row["findings"]=check(a.role,json.loads(a.report.read_text()))
  row["passed"]=True
 except Exception as e:
  row["errors"]=[type(e).__name__+": "+str(e)]
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(row,indent=2)+"\n")
 print("S809_REAL_ROLE",a.role,"PASS" if row["passed"] else "FAIL",flush=True)
 if not row["passed"]:sys.exit(1)
if __name__=="__main__":main()
