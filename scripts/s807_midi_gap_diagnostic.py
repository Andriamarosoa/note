"""S807 post-model falsification: exact gaps on signed S800 FIT80 MIDI events.

A retrospective observer, not a new note decoder. One missing pitch frame is
classified into exactly one of four temporal occupancy segments. No truth
inference, no CAL/HELD/TEST, no advice from external LLMs, no LIVE changes.
"""
from __future__ import annotations
import argparse,json,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

SCHEMA="S807_FIT80_TRUE_NOTE_STATE_GAP_DIAGNOSTIC_V1"
SR,HOP=44100,256
TYPES=("entirely_missed_run","late_coverage_prefix",
       "early_coverage_suffix","middle_gap")

def audit_track(item):
 nframes=int(item["samples"])//HOP
 ends=np.arange(1,nframes+1,dtype=np.int64)*HOP
 times=ends/SR
 truth=np.zeros((nframes,49),bool)
 count=np.zeros(nframes,np.int16)
 for note in item["fit_truth"]:
  pitch=int(note["pitch"])-40
  if not 0<=pitch<49:raise ValueError("out-of-range ground truth pitch")
  mask=(times>=note["time"])&(times<note["time"]+note["duration"])
  truth[mask,pitch]=True
  count[mask]+=1
 pred=np.zeros_like(truth)
 active={}
 for event in item["source_events"]:
  p=int(event["pitch"])-40
  at=int(np.searchsorted(ends,int(event["available_sample"])))
  if event["available_sample"]<event["decision_sample"]:
   raise AssertionError("future/backdated MIDI event")
  if not 0<=p<49:raise AssertionError("invalid source MIDI pitch")
  if event["type"]=="note_on":
   if p in active:raise AssertionError("source duplicate pitch")
   active[p]=at
  elif event["type"]=="note_off":
   if p not in active:raise AssertionError("source unpaired NoteOff")
   pred[active.pop(p):at,p]=True
  else:raise ValueError("unknown event type")
 for p,start in active.items():pred[start:,p]=True
 missed=truth&~pred
 labels={name:np.zeros_like(missed) for name in TYPES}
 for p in range(49):
  t=truth[:,p]
  starts=np.flatnonzero(t&~np.concatenate(([False],t[:-1])))
  ends0=np.flatnonzero(t&~np.concatenate((t[1:],[False])))+1
  for lo,hi in zip(starts,ends0):
   covered=np.flatnonzero(pred[lo:hi,p])
   if not len(covered):
    labels["entirely_missed_run"][lo:hi,p]=True
   else:
    first=lo+int(covered[0]);last=lo+int(covered[-1])+1
    labels["late_coverage_prefix"][lo:first,p]=True
    labels["early_coverage_suffix"][last:hi,p]=True
    labels["middle_gap"][first:last,p]=truth[first:last,p]&~pred[first:last,p]
 summed=sum(v.astype(np.int8) for v in labels.values())
 if not np.array_equal(summed,missed.astype(np.int8)):
  raise AssertionError("missing pitch frames do not partition")
 pc=pred.sum(axis=1)
 delta=np.minimum(pc,6)-np.minimum(count,6)
 under=(count>0)&(delta<0)
 over=(count>0)&(delta>0)
 out=Counter(active_frames=int(np.count_nonzero(count)),
  correct=int(np.count_nonzero((count>0)&(delta==0))),
  under=int(under.sum()),over=int(over.sum()),
  frame_pitch_fn=int(missed.sum()),
  frame_pitch_fp=int(np.count_nonzero(pred&~truth)),
  unison_frames=int(np.count_nonzero((count>truth.sum(1))&(count>0))),
  under_unison_frames=int(np.count_nonzero(under&(count>truth.sum(1)))))
 for name,mask in labels.items():
  out[name+"_frames"]=int(mask.sum())
  out["under_with_"+name]=int(np.count_nonzero(under&np.any(mask,axis=1)))
 for name,field in (("active_frames","active_frames"),
                    ("correct","active_k_correct"),
                    ("frame_pitch_fn","frame_pitch_fn"),
                    ("frame_pitch_fp","frame_pitch_fp")):
  if out[name]!=int(item["source_counts"][field]):
   raise AssertionError("signed MIDI source count drift "+field)
 return out

def calculate(archive):
 total=Counter();comps=defaultdict(Counter)
 with zipfile.ZipFile(archive) as z:
  files=sorted(x for x in z.namelist() if x.startswith("fit_") and x.endswith(".json"))
  if len(files)!=80:raise AssertionError("not 80 authentic GuitarSet FIT OOF traces")
  for name in files:
   r=json.loads(z.read(name))
   if r["schema"]!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or r.get("cal_annotations_opened"):
    raise AssertionError("FIT-only signed source requirement broken")
   counts=audit_track(r)
   total.update(counts);comps[r["composition"]].update(counts)
 expected={"active_frames":249411,"correct":142213,
           "under":80654,"over":26544,"frame_pitch_fn":146984,
           "frame_pitch_fp":76787}
 for key,value in expected.items():
  if total[key]!=value:raise AssertionError("S794/S800 FIT parity "+key)
 if sum(total[n+"_frames"] for n in TYPES)!=total["frame_pitch_fn"]:
  raise AssertionError("missed-pitch cause accounting broken")
 return {"schema":SCHEMA,"complete":True,"tracks":80,"split":"FIT80_ONLY",
  "source_run":39352880035,"reference":"S708",
  "input_ground_truth_used_only_for_retrospective_metric":True,
  "native_events_unchanged":True,
  "source_counters_parity_pass":True,
  "CAL_opened":False,"HELD_opened":False,"TEST_opened":False,
  "definitions":{
    "entirely_missed_run":"true pitch occupancy run with zero overlapping predicted pitch-occupied frames",
    "late_coverage_prefix":"missing true pitch frames before first predicted overlap",
    "early_coverage_suffix":"missing true pitch frames after last predicted overlap",
    "middle_gap":"missing true pitch frames inside the first and last predicted overlaps",
    "limitations":"Temporal occupancy gaps, not proven acoustic cause or trained solution. Types are per pitch frame. Under-count frame coverage by type may overlap."},
  "counts":dict(total),"by_composition":{k:dict(v) for k,v in sorted(comps.items())}}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--traces-zip",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError("immutable forensic audit")
 result=calculate(a.traces_zip)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print("S807_REAL_FIT80_MIDI_GAP_ACCOUNTING_PASS",result["counts"])
if __name__=="__main__":main()
