"""Balanced K2/K3/K4 energy-transport feature extraction on verified native events.

Selection is a predeclared, fixed-seed label-stratified sample of 150 K2,
150 K3, 150 K4 events per fold from the pre-existing YourMT3+ native
cohort. Only folds 0,1,2,4 and players 00..04; no player05 or fold3.
K labels are used solely for stratified sampling/assessment, never in any
energy/transport/persistence computation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_transport import energy_observable
from scripts.extract_v273_energy_flow import SAMPLE_RATE
from scripts.yourmt3_exactk_common import FOLDS,digest,require

PER_K=150
CLASSES=(2,3,4)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--fold",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(a.fold in FOLDS and not a.output.exists(),"fold/output")
    manifest=json.loads((a.cohort/"manifest.json").read_text())
    require(manifest["status"]=="verified" and manifest.get("player_05_evaluated")==False
            and manifest.get("fold_3_evaluated")==False,"cohort provenance drift")
    src=a.cohort/f"fold-{a.fold}.npz"
    require(digest(src)==manifest["by_fold"][str(a.fold)]["cohort_sha256"],"cohort sha drift")
    with np.load(src,allow_pickle=False) as z:
        required=("global_index","member","starts","k","baseline")
        require(all(k in z.files for k in required),"schema")
        ids=z["global_index"].astype(np.int64)
        member=z["member"].astype(str)
        starts=z["starts"].astype(np.int64)
        labels=z["k"].astype(int)
        base=z["baseline"].astype(int)
    require(len(ids)==len(labels)==len(member)==len(starts)==len(base),"length mismatch")
    require(all(m[:2] in ("00","01","02","03","04") for m in member),"player leak")
    chosen=[]
    inventory={}
    for k in CLASSES:
        pool=np.flatnonzero(labels==k)
        require(len(pool)>=PER_K,f"not enough K{k} in fold")
        # One seed and one sample size for every fold and class; fixed in source.
        chosen.extend(np.random.default_rng(20261008+int(a.fold)*7+k).permutation(pool)[:PER_K].tolist())
        inventory[str(k)]=len(pool)
    chosen=sorted(chosen,key=lambda i:int(ids[i]))
    require(len(chosen)==3*PER_K,"sampling")
    wanted={member[i] for i in chosen}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"track coverage")
    audio={}
    for name,t in tracks.items():
        wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
        require(int(wav.sample_rate)==SAMPLE_RATE,"sample rate drift")
        audio[name]=np.asarray(wav.samples,np.float64)/32768.
    output=[]
    for i in chosen:
        row_id=int(ids[i]);x=audio[member[i]];start=int(starts[i]);seed=17064+row_id
        output.append(dict(
          global_index=row_id,member=member[i],start_sample=start,
          fold=int(a.fold),true_k=int(labels[i]),base_pred=int(base[i]),
          features=energy_observable(x,start),
          scrambled=energy_observable(x,start,"scramble-time",seed=seed),
          permuted=energy_observable(x,start,"permute-bands",seed=seed),
        ))
    feature_names=sorted(output[0]["features"])
    require(all(sorted(r[key])==feature_names for r in output for key in ("features","scrambled","permuted")),"schema drift")
    a.output.mkdir(parents=True)
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in output))
    info=dict(status="completed",experiment="v273_energy_transport_multik_extract",
              fold=a.fold,rows=len(output),classes=list(CLASSES),per_class=PER_K,
              initial_class_counts=inventory,recordings=len(tracks),features=len(feature_names),
              source_cohort_sha256=digest(src),player05_used=False,fold3_used=False,
              features_calculated_without_labels=True,
              selection="fixed-seed label-stratified sample; evaluation conditional on balancing",
              model_unchanged=True,automatic_promotion=False)
    (a.output/"report.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    print(json.dumps(info,sort_keys=True),flush=True)

if __name__=="__main__":main()
