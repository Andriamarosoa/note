"""S807 blind evidence: exact same signed FIT-only numbers and numbered code for 3 independent NIM models.

Never transmit CAL32, HELD, TEST, individual recordings, audio, truth labels,
checkpoint weights, secrets, past model opinions, or assistant conclusions.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

SCHEMA="S807_BLIND_ACTIVEK_FIT80_EVIDENCE_V1"
SOURCE="S794_ZERO_DELAY_REAL_MIDI256_ACTIVEK_ROOTCAUSE"
# Each reviewer gets this identical set in an identical order.
SOURCE_WINDOWS={
 "scripts/s704_joint_metrics.py":((1,97),),
 "scripts/s794_activek_failure_taxonomy.py":((28,122),),
 "src/causal_note/s795_activek_experts.py":((20,101),(118,210)),
 "src/causal_note/s797_fragment_router.py":((1,73),),
 "src/causal_note/s763_independent_decoders.py":((98,128),(254,287)),
 "src/causal_note/s740_multitarget_dynamic.py":((24,62),(163,205),(277,310)),
 "src/causal_note/s741_exactk_sources.py":((1,65),),
}
SECRET=re.compile(r"(?:nvapi-[A-Za-z0-9_-]{15,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
def digest(payload):
 return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def build(root,report_path):
 root=Path(root)
 original=json.loads(Path(report_path).read_text())
 if original.get("schema")!=SOURCE or not original.get("complete") or not original.get("source_identity_confirmed") or original.get("fit_done")!=80:
  raise AssertionError("missing signed S794 FIT80 source and MIDI parity")
 if original.get("HELD_opened") or original.get("TEST_opened") or original.get("final_MIDI_changed"):
  raise AssertionError("historical source identity or holdout lock violated")
 f=original["FIT"]
 a=f["audit"];m=f["metrics"]
 if not (a["active_frames"]==249411 and a["active_correct"]==142213 and a["under_active_frames"]==80654 and a["over_active_frames"]==26544):
  raise AssertionError("S794 signed Active-K FIT counters drift")
 if a["active_correct"]+a["under_active_frames"]+a["over_active_frames"]!=a["active_frames"]:
  raise AssertionError("FIT accounting broken")
 if m["k_correct"]!=189124 or m["frames"]!=306536 or m["active_k_correct"]!=142213:
  raise AssertionError("FIT original all-frame counter mismatch")
 if a["acoustic_evidence_available_undercount_frames"]+a["no_acoustic_evidence_undercount_frames"]!=a["under_active_frames"]:
  raise AssertionError("FIT undercount cue partition mismatch")
 # FIT split only and tightly allowlisted aggregation, no CAL record ever copied.
 numeric={
  "origin":"GuitarSet 80 composition-grouped FIT recordings, original S708 emitted MIDI256",
  "source_run":39336923230,
  "source_schema":SOURCE,
  "split":"FIT80_ONLY",
  "fit_count":80,
  "active_frames":a["active_frames"],
  "active_correct":a["active_correct"],
  "under_active_frames":a["under_active_frames"],
  "over_active_frames":a["over_active_frames"],
  "acoustic_evidence_available_undercount_frames":a["acoustic_evidence_available_undercount_frames"],
  "no_acoustic_evidence_undercount_frames":a["no_acoustic_evidence_undercount_frames"],
  "missed_pitch_frames":a["missed_pitch_frames"],
  "spurious_pitch_frames":a["spurious_pitch_frames"],
  "by_true_k":a["by_true_k"],
  "by_register":a["by_register"],
  "expert_opportunity":a["expert_opportunity"],
  "six_metrics":{k:m[k] for k in ("onset_f1","exact_k","active_exact_k","offset_accuracy","full_note_f1","frame_pitch_f1")},
  "source_counts":{k:m[k] for k in ("frames","k_correct","active_frames","active_k_correct","tp","fp","fn","offset_eligible","offset_correct","frame_pitch_tp","frame_pitch_fp","frame_pitch_fn")},
  "composition_fit_diagnostic":{comp:{
      "active_frames":row["audit"]["active_frames"],
      "active_correct":row["audit"]["active_correct"],
      "under_active_frames":row["audit"]["under_active_frames"],
      "over_active_frames":row["audit"]["over_active_frames"],
      "active_exact_k":row["metrics"]["active_exact_k"]
    } for comp,row in sorted(f["by_composition"].items())}
 }
 excerpts=[]
 for path,windows in SOURCE_WINDOWS.items():
  file=root/path
  if not file.is_file():raise AssertionError("missing reviewed file "+path)
  data=file.read_text(encoding="utf-8")
  if SECRET.search(data):raise AssertionError("possible credential in reviewed file")
  lines=data.splitlines()
  pieces=[]
  for lo,hi in windows:
   if lo>len(lines):raise AssertionError("missing source window")
   pieces.extend(f"{i}: {lines[i-1][:290]}" for i in range(lo,min(hi,len(lines))+1))
   pieces.append("... [unshown source lines omitted]")
  excerpts.append({
   "path":path,"sha256":digest(data),
   "total_lines":len(lines),"windows":[list(w) for w in windows],
   "excerpt":"\n".join(pieces)})
 out={"schema":SCHEMA,"complete":True,"numeric":numeric,
      "source_code_excerpts":excerpts,
      "reviewer_priming":"No hypotheses, prior assistant recommendations, model opinions or suggestions provided.",
      "limitations":"Only bounded numbered hot-path excerpts; not full repository/audio. FIT observations are retrospective; no claims about unseen split or LIVE physical latency.",
      "official_reference_untouched":"S708",
      "CAL_contents_sent":False,"HELD_contents_sent":False,"TEST_contents_sent":False}
 out["evidence_sha256"]=digest(json.dumps(out,sort_keys=True,ensure_ascii=False))
 return out
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--root",type=Path,default=Path("."))
 p.add_argument("--report",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError("immutable S807 evidence")
 result=build(a.root,a.report)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 print("S807_SAME_BLIND_CONTEXT",result["evidence_sha256"],
       "tracks=80 under="+str(result["numeric"]["under_active_frames"]),
       "over="+str(result["numeric"]["over_active_frames"]))
if __name__=="__main__":main()
