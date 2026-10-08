"""Combined energy + signed flux + birth/death continuity audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_residual_audit import FOLDS, require
from scripts.extract_v273_energy_flow import energy_flow_features, SAMPLE_RATE
from scripts.extract_v273_energy_continuity import continuity_features

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--fold",type=int,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(a.fold in FOLDS and a.fold != 3,"bad fold"); require(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    require(len(cases)==488,f"cohort drift {len(cases)}")
    rows=[r for r in cases if int(r["fold"])==a.fold]
    require(rows and all(int(r["true_K"]) in (2,3) for r in rows),"bad cohort")
    wanted={r["recording_id"] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage incomplete")
    audio={}
    for member,track in tracks.items():
        wav=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(int(wav.sample_rate)==SAMPLE_RATE,"sample-rate drift")
        audio[member]=np.asarray(wav.samples,np.float64)/32768.0
    out=[]
    for row in rows:
        ef=energy_flow_features(audio[row["recording_id"]],int(row["start_sample"]))
        cf=continuity_features(audio[row["recording_id"]],int(row["start_sample"]))
        features={**ef,**cf}
        require(len(features)==len(ef)+len(cf),"feature collision")
        require(np.isfinite(np.asarray(list(features.values()),np.float64)).all(),"nonfinite")
        out.append({"row_id":int(row["row_id"]),"fold":a.fold,"true_k":int(row["true_K"]),"features":features})
    names=sorted(out[0]["features"]); require(all(sorted(r["features"])==names for r in out),"schema drift")
    a.output.mkdir(parents=True)
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in out))
    rep={"status":"completed","experiment":"v273_energy_flux_combined_extract","fold":a.fold,
         "rows":len(out),"K2":sum(r["true_k"]==2 for r in out),"K3":sum(r["true_k"]==3 for r in out),
         "feature_count":len(names),"labels_used_during_feature_extraction":False,
         "fold3_used":False,"automatic_promotion":False}
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
