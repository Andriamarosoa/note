"""S809: verify six independent axis pipelines on 80 real signed GuitarSet FIT OOF tracks.

Immutable native S708 MIDI256 metrics always come from ONE unchanged stream;
separate per-axis SHADOW assessments are not substituted for official scores.
The six counterfactual attacks mutate one axis selector at a time on the same
real audio prefix and assert bit-for-bit identity for each of the OTHER FIVE
axis decisions and selector records, with no joined decoder or shared states.

S800 stores frozen K argmax/entropy but not the full [7] original posterior;
the K provided to isolated shadow experiments is a ONE-HOT *proxy* of saved
argmax. Never claim a calibrated independent Exact-K benchmark from it.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from causal_note.s763_independent_decoders import AXES,Fragment,DECISION_SAMPLES
from causal_note.s809_isolated_axis_pipeline import (
 SCHEMA as ARCH_SCHEMA,SixIndependentAxes,default_pipelines,AxisSelector,HEAD_NAMES,
 SELECTOR_NAMES
)
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA,verify_snapshot
from scripts.s704_joint_metrics import metric,aggregate,SCORES
from scripts.s795_train_activek_midi256 import COUNT_KEYS
from scripts.s613_train_strict_live import event_score

SCHEMA="S809_GUITARSET_FIT80_SIX_INDEPENDENT_AXES_REPLAY_V1"
SOURCE_RUN=39352880035
AUDIT_PREFIX_FRAMES=64

def event(e):
 return dict(type=e.kind,pitch=e.pitch,decision_sample=e.sample,
  available_sample=e.sample,time=float(e.sample/44100),
  velocity=96 if e.kind=="note_on" else 0,reason=e.reason)

def ground_truth(truth,ends):
 times=np.asarray(ends,dtype=np.int64)/44100
 target=np.zeros((len(ends),49),bool)
 counts=np.zeros(len(ends),np.int16)
 for note in truth:
  p=int(note["pitch"])-40
  if not 0<=p<49:raise AssertionError("invalid FIT truth pitch")
  mask=(times>=note["time"])&(times<note["time"]+note["duration"])
  counts+=mask.astype(np.int16)
  target[mask,p]=True
 return target,np.minimum(counts,6)

def shadow_diagnostics(decisions,ending,truth,ends):
 """Axis-local standalone scores, never six S704 final-MIDI ratios."""
 pitches,kreal=ground_truth(truth,ends)
 n=len(ends)
 exact=np.fromiter((d["exact_k"].count for d in decisions),np.int16,count=n)
 active=np.fromiter((d["active_exact_k"].count for d in decisions),np.int16,count=n)
 frame=np.zeros((n,49),bool)
 for t,dec in enumerate(decisions):
  for p in dec["frame_pitch"].pitches:frame[t,p-40]=True
 k_mask=kreal>0
 ons=[event(a) for d in decisions for a in d["onset"].intents]
 offs=[event(a) for d in decisions for a in d["offset"].intents]
 joint=[event(a) for d in decisions for a in d["full_note"].intents]
 joint.extend(event(a) for a in ending)
 onscore,_=event_score(ons,truth,tolerance=.080)
 offtruth=[dict(time=n["time"]+n["duration"],pitch=n["pitch"])
           for n in truth if n["time"]+n["duration"]<ends[-1]/44100]
 offset_score,_=event_score(
   [dict(type="note_on",pitch=e["pitch"],time=e["time"]) for e in offs],
   offtruth,tolerance=.050)
 # This full-note score is for ONLY the axis's standalone paired MIDI stream.
 # It is not copied into reference S708's six score row.
 fullscore=metric(joint,truth,ends)
 return {
  "exact_k":dict(correct=int((exact==kreal).sum()),frames=n),
  "active_exact_k":dict(correct=int(((active==kreal)&k_mask).sum()),
                         active_frames=int(k_mask.sum())),
  "onset":dict(tp=int(onscore["TP"]),fp=int(onscore["FP"]),fn=int(onscore["FN"])),
  "offset":dict(tp=int(offset_score["TP"]),fp=int(offset_score["FP"]),
                fn=int(offset_score["FN"])),
  "frame_pitch":dict(tp=int((frame&pitches).sum()),fp=int((frame&~pitches).sum()),
                     fn=int((~frame&pitches).sum())),
  "full_note":dict(tp=int(fullscore["tp"]),fp=int(fullscore["fp"]),
                   fn=int(fullscore["fn"]),
                   offset_eligible=int(fullscore["offset_eligible"]),
                   offset_correct=int(fullscore["offset_correct"]))}

def frame_input(X,t,end):
 post=X[t,:,:3]
 # Original calibrated K distribution NOT archived in S800, only argmax.
 pred=np.rint(float(X[t,0,13])*6)
 if not np.isfinite(pred):raise AssertionError("invalid signed K proxy")
 k=np.zeros(7,np.float32);k[int(np.clip(pred,0,6))]=1.
 return Fragment(int(end),post,k)

def replay(manifest,traces):
 m=json.loads(Path(manifest).read_text())
 if m.get("schema")!=TRACE_SCHEMA or not m.get("complete") or (
     not m.get("FIT_source_parity_S795")):
  raise AssertionError("real signed S800 source required")
 if len(m["fit_files"])!=80 or len({x["index"] for x in m["fit_files"]})!=80:
  raise AssertionError("need all 80 distinct real FIT tracks")
 if any(m.get(x,False) for x in ("CAL_opened","HELD_opened","TEST_opened")):
  raise AssertionError("no reserved labels can be read")
 per_axis=defaultdict(Counter)
 changes=Counter()
 used=Counter()
 frozen=[]
 comps=defaultdict(list)
 tracks=0
 for meta in sorted(m["fit_files"],key=lambda x:x["index"]):
  r=verify_snapshot(meta,traces)
  with np.load(Path(traces)/f"fit_{meta['index']:03d}.npz",
               allow_pickle=False) as z:
   X=z["features"].astype(np.float32)
   decision_ends=z["decision_ends"]
   ends=z["metric_ends"]
  if len(X)!=len(decision_ends) or len(X)==0 or (
    decision_ends[0]!=DECISION_SAMPLES) or (
    not np.all(np.diff(decision_ends)==DECISION_SAMPLES)):
   raise AssertionError("S809 requires exact zero-wait 1024-sample source")
  src=r["source_events"]
  src_snapshot=json.dumps(src,sort_keys=True,separators=(",",":"))
  score=metric(src,r["fit_truth"],ends)
  if any(int(score[k])!=int(r["source_counts"][k]) for k in COUNT_KEYS):
   raise AssertionError("S809 reference source counters drift")
  frozen.append(score);comps[r["composition"]].append(score)
  base=SixIndependentAxes()
  variants={}
  n=min(AUDIT_PREFIX_FRAMES,len(X))
  for axis in AXES:
   modified=default_pipelines()
   modified[axis].selector=AxisSelector(axis,"causal_contrast",threshold=.12)
   variants[axis]=SixIndependentAxes(modified)
  records=[]
  for t,end in enumerate(decision_ends):
   f=frame_input(X,t,end)
   dec,select=base.step(f)
   records.append(dec)
   if t<n:
    for axis,alternate in variants.items():
     out,selector_state=alternate.step(f)
     used[axis]+=int(selector_state[axis].selected_alternative_pitches)
     changes[axis]+=int(out[axis]!=dec[axis])
     for other in AXES:
      if other!=axis and (out[other]!=dec[other] or
                         selector_state[other]!=select[other]):
       raise AssertionError(f"S809 cross-axis contamination {axis}->{other} on {r['name']} at {end}")
  # Decoder never gets permission to emit a joint MIDI candidate.
  if json.dumps(src,sort_keys=True,separators=(",",":"))!=src_snapshot:
   raise AssertionError("native input MIDI events modified")
  copied=SixIndependentAxes.signed_midi_identity(src)
  if copied!=src or (copied and copied[0] is src[0]):
   raise AssertionError("S708 signed stream identity broken")
  final=base.pipelines["full_note"].decoder.finish(int(r["samples"]))
  local=shadow_diagnostics(records,final,r["fit_truth"],decision_ends)
  for axis,counters in local.items():
   per_axis[axis].update(counters)
  tracks+=1
  print("S809_REAL_FIT_TRACK",meta["index"],r["composition"],
        "private_axes=6",flush=True)
 original=aggregate(frozen)
 if any(int(original[k])!=int(m["S795_archived_fit_counts"][k])
        for k in COUNT_KEYS):
  raise AssertionError("80-track source counts mismatch")
 if len(comps)!=5 or tracks!=80:raise AssertionError("incomplete FIT fold coverage")
 return {"schema":SCHEMA,"complete":True,"tracks":tracks,"compositions":sorted(comps),
  "source_run":SOURCE_RUN,"source_counts":{k:int(original[k]) for k in COUNT_KEYS},
  "source_scores":{k:float(original[k]) for k in SCORES},
  "axis_names":list(AXES),"separate_head_names":{k:list(v) for k,v in HEAD_NAMES.items()},
  "separate_selector_names":dict(SELECTOR_NAMES),
  "axes_count":6,"heads_per_axis":2,"selectors_per_axis":1,"decoders_per_axis":1,
  "all_six_axis_private_state":True,
  "per_axis_shadow_diagnostics":{k:dict(v) for k,v in per_axis.items()},
  "per_axis_shadow_metrics_are_NOT_final_midi_six_scores":True,
  "K_posterior_from_S800":"ARGMAX_ONEHOT_PROXY_NOT_ORIGINAL_PROBABILITIES",
  "counterfactual_prefix_per_track":AUDIT_PREFIX_FRAMES,
  "mutation_scenarios_for_all_six_axes":len(AXES),
  "independent_selector_cells_activated":dict(used),
  "independent_target_decisions_changed":dict(changes),
  "all_five_other_axes_bitwise_identical_under_each_of_six_mutations":True,
  "no_single_joint_decoder":True,"production_LIVE_modified":False,
  "source_MIDI_unchanged":True,"no_lookahead_samples":0,"extra_hold_samples":0,
  "CPU_iRig_latency_measured":False,
  "CAL_opened":False,"HELD_opened":False,"TEST_opened":False,
  "reference":"S708","promotion":False,
  "scientific_limitation":"New independent acoustic signal alternatives are deterministic untrained shadow heads. Source K is an argmax-only proxy. No performance improvement or independent calibrated K score is claimed."}

def main():
 p=argparse.ArgumentParser()
 for key in ("manifest","traces","out"):
  p.add_argument("--"+key,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError("immutable scientific result")
 result=replay(a.manifest,a.traces)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
 print("S809_SIX_WAY_REAL_FIT80_INDEPENDENCE_PASS",result["tracks"],
       result["independent_target_decisions_changed"],flush=True)

if __name__=="__main__":main()
