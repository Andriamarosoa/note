"""Label-blind harmonic trajectories for EVERY native baseline prediction K2/K3/K4.

This is a restricted-application correction study, NOT a three-class test.
Frozen original native cohort: all 59,309 events K0..K6, folds 0,1,2,4.
Eligibility depends solely on the untouched baseline prediction, never true K.
Only 7,493 eligible positions get waveform features and can be corrected.
The final evaluation keeps the other 51,816 baseline predictions unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_flow import SAMPLE_RATE
from scripts.extract_v273_harmonic_trajectory import trajectory_features
from scripts.yourmt3_exactk_common import FOLDS,digest,require

EXPERIMENT="v273_harmonic_global_candidate_extract"
CANDIDATES=(2,3,4)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--fold",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require(args.fold in FOLDS and not args.output.exists(),"fold/output")
    manifest=json.loads((args.cohort/"manifest.json").read_text())
    require(manifest.get("status")=="verified" and
            manifest.get("fold_3_evaluated") is False and
            manifest.get("player_05_evaluated") is False,
            "native cohort verification not acceptable")
    cohort_file=args.cohort / f"fold-{args.fold}.npz"
    require(digest(cohort_file)==manifest["by_fold"][str(args.fold)]["cohort_sha256"],"frozen cohort checksum mismatch")
    with np.load(cohort_file,allow_pickle=False) as z:
        idx=z["global_index"].astype(np.int64)
        members=z["member"].astype(str)
        start=z["starts"].astype(np.int64)
        true_k=z["k"].astype(int)
        base=z["baseline"].astype(int)
    require(len(idx)==len(start)==len(true_k)==len(base)==len(members),"cohort array length drift")
    require(all(m[:2] in ("00","01","02","03","04") for m in members),"player05 used")
    require(np.array_equal(np.unique(idx).size, len(idx)),"duplicate indices")
    require(np.isin(base,range(7)).all() and np.isin(true_k,range(7)).all(),"out of range")
    eligible=np.flatnonzero(np.isin(base,CANDIDATES))
    require(len(eligible)>0,"empty correction coverage")
    wanted={members[i] for i in eligible}
    tracks={t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"incomplete audio membership")
    audio={}
    for name,track in tracks.items():
        wav=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(int(wav.sample_rate)==SAMPLE_RATE,"sample rate mismatch")
        audio[name]=np.asarray(wav.samples,np.float64)/32768.
    output=[]
    for i in eligible:
        ident=int(idx[i]);name=members[i];ts=int(start[i])
        signal=audio[name]
        output.append(dict(
            global_index=ident,fold=int(args.fold),recording_id=name,start_sample=ts,
            baseline_k=int(base[i]),true_k=int(true_k[i]),
            features=trajectory_features(signal,ts),
            detuned=trajectory_features(signal,ts,"detuned"),
            fundamental=trajectory_features(signal,ts,"fundamental"),
            scrambled=trajectory_features(signal,ts,"scramble-time",seed=ident+27084),
        ))
    names=sorted(output[0]["features"])
    require(all(all(sorted(r[f])==names for f in ("features","detuned","fundamental","scrambled")) for r in output),"feature schema drift")
    args.output.mkdir(parents=True)
    (args.output/"rows.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in output))
    result=dict(experiment=EXPERIMENT,status="completed",fold=args.fold,
                full_rows=len(idx),eligible_rows=len(output),
                selection="untouched baseline prediction belongs to K2/K3/K4 ONLY",
                candidate_original_prediction_counts={str(k):int(np.sum(base[eligible]==k)) for k in CANDIDATES},
                true_label_counts={str(k):int(np.sum(true_k[eligible]==k)) for k in range(7)},
                recordings=len(wanted),features=len(names),
                source_cohort_sha256=digest(cohort_file),
                fold3_used=False,player05_used=False,labels_used_in_features=False,
                full_global_evaluation_required=True,source_model_modified=False,
                automatic_promotion=False)
    (args.output/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":main()
