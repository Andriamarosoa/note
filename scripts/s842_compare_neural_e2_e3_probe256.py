"""Evaluate trained S842 neural latent octave views vs same plain encoder.

IMPORTANT: strictly compares preselected CAL thresholds on OUTER compositions,
on real audio *selected near labeled note intervals*. This is a NOTE-ACTIVITY
PROXY, not a replay of NoteOn timestamps or official GuitarSet 80-WAV F1.
No reference MIDI/Onset/other head is touched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from onset256.learned_latent_transposer256 import NeuralLatentTransposer256


def safe_f1(y,positive):
    tp=int(np.count_nonzero(y&positive))
    fp=int(np.count_nonzero(y&~positive))
    fn=int(np.count_nonzero(~y&positive))
    return dict(tp=tp,fp=fp,fn=fn,
        precision=tp/max(1,tp+fp),
        recall=tp/max(1,tp+fn),
        f1=2*tp/max(1,2*tp+fp+fn))


def pick_threshold(prob,truth):
    candidates=np.arange(.05,.951,.05)
    best=None
    for cut in candidates:
        score=safe_f1(prob>=cut,truth)
        rank=(score["f1"],score["tp"],-score["fp"])
        if best is None or rank>best[0]:
            best=(rank,float(round(cut,2)))
    return best[1]


def get_probs(model,records,batch=256):
    model.eval()
    plain=[];fused=[];weight=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch):
            x=torch.from_numpy(records[start:start+batch])
            a,b,w=model(x,shifts=(-12,12))
            plain.append(torch.sigmoid(b).numpy())
            fused.append(torch.sigmoid(a).numpy())
            weight.append(w.numpy())
    return dict(plain=np.concatenate(plain),transposed=np.concatenate(fused),
                mean_view_weights=np.concatenate(weight).mean(0).tolist())


def evaluate(probe_path,checkpoint_path,out):
    if Path(out).exists():
        raise FileExistsError("immutable evaluation")
    with np.load(probe_path,allow_pickle=False) as ds:
        fields=("pcm","note_mask","kind","split","track","composition","end_sample")
        if any(k not in ds for k in fields):raise ValueError("not a real GT probe")
        d={k:ds[k].copy() for k in fields}
    if d["pcm"].ndim!=2 or d["pcm"].shape[1]!=256:
        raise ValueError("must use exact PCM256, never longer context")
    if d["note_mask"].shape!=(len(d["pcm"]),49):
        raise ValueError("missing GT 49D")
    record=torch.load(checkpoint_path,weights_only=True,map_location="cpu")
    if record.get("abi")!="ONSET_LEARNED_LATENT_TRANSPOSE_RAW256_V1":
        raise ValueError("unknown model provenance")
    model=NeuralLatentTransposer256()
    model.load_state_dict(record["model_state"],strict=True)
    torch.set_num_threads(1)
    result=dict(schema="S842_E2_E3_GT_ACTIVITY_PROXY_V1",complete=True,
                calibration="CAL-only thresholds, OUTER never chooses",
                strict_raw_samples=256,future_audio=0,added_wait=0,
                classification_proxy_not_Onset_event_F1=True,
                reference_Onset_F1_80_WAV=.6925843310420176,
                from_independently_recorded_octave_pairs=True,
                no_DSP_pitch_shift=True,no_midi_changes=True,no_promotion=True,
                heads={})
    cal=np.flatnonzero(d["split"]=="cal")
    outer=np.flatnonzero(d["split"]=="outer")
    if len(cal)==0 or len(outer)==0:raise ValueError("missing CAL/OUTER samples")
    for key in ("plain","transposed"):
        result["heads"][key]=dict()
    cal_prob=get_probs(model,d["pcm"][cal])
    outer_prob=get_probs(model,d["pcm"][outer])
    result["cal_samples"]=len(cal)
    result["outer_samples"]=len(outer)
    result["outer_kind_distribution"]={
        str(kind):int(np.count_nonzero(d["kind"][outer]==kind))
        for kind in np.unique(d["kind"][outer])}
    for name in ("plain","transposed"):
        for label,index in (("E2",0),("E3",12)):
            train_y=d["note_mask"][cal,index]>.5
            outer_y=d["note_mask"][outer,index]>.5
            cut=pick_threshold(cal_prob[name][:,index],train_y)
            pred=outer_prob[name][:,index]>=cut
            result["heads"][name][label]=dict(
                cut_cal=cut,groundtruth_outer_positives=int(outer_y.sum()),
                **safe_f1(pred,outer_y))
        # Conjunction: must detect both simultaneous GT active pitches.
        truth_cal=(d["note_mask"][cal,0]>.5)&(d["note_mask"][cal,12]>.5)
        truth_outer=(d["note_mask"][outer,0]>.5)&(d["note_mask"][outer,12]>.5)
        qcal=np.minimum(cal_prob[name][:,0],cal_prob[name][:,12])
        qouter=np.minimum(outer_prob[name][:,0],outer_prob[name][:,12])
        cut=pick_threshold(qcal,truth_cal)
        result["heads"][name]["E2_E3_JOINT"]=dict(
            cut_cal=cut,groundtruth_outer_positives=int(truth_outer.sum()),
            **safe_f1(qouter>=cut,truth_outer))
    result["mean_outer_view_weights"]=outer_prob["mean_view_weights"]
    result["outer_delta_F1_points"]={
        k:100*(result["heads"]["transposed"][k]["f1"]-
               result["heads"]["plain"][k]["f1"])
        for k in ("E2","E3","E2_E3_JOINT")}
    out=Path(out);out.mkdir(parents=True)
    (out/"probe_report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S842_REAL_PROBE_OUTER",json.dumps({
        "n_cal":len(cal),"n_outer":len(outer),
        "plain":result["heads"]["plain"],
        "transposed":result["heads"]["transposed"],
        "delta_F1_points":result["outer_delta_F1_points"],
        "not_onset_f1":True},sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    evaluate(a.probe,a.checkpoint,a.out)
