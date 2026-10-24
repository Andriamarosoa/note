"""S843 genuine mic/hex evaluation, honest raw-audio vs note hypotheses.

CAL guitarist 05 freezes thresholds. OUTER guitarists 00..03 never choose
thresholds or model checkpoint. This scores ACTIVITY PER PCM256 HOP only,
not NoteOn event F1, and the hex-vs-mic domain is not perfectly matched.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from onset256.real_audio_transpose_head256 import (
    ABI,HOP,NeuralAcousticTransposeHead256,CausalRealAudioTransposeStream256)


def f1_counts(pred,truth):
    pred=np.asarray(pred,bool)
    truth=np.asarray(truth,bool)
    if pred.shape!=truth.shape:raise ValueError("GT/pred shape mismatch")
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth))
    fn=int(np.sum(~pred&truth))
    return dict(tp=tp,fp=fp,fn=fn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def best_cut_cal_only(prob,truth):
    if not bool(np.any(truth)):
        return None
    good=None
    for cut in np.arange(.05,.951,.05):
        c=f1_counts(prob>=cut,truth)
        rank=(c["f1"],c["tp"],-c["fp"])
        if good is None or rank>good[0]:good=(rank,float(round(cut,2)))
    return good[1]


def load_data(path):
    with np.load(path,allow_pickle=False) as f:
        d={k:f[k].copy() for k in ("mic","hex","note_gt","string_gt")}
    if (d["mic"].ndim!=3 or d["mic"].shape[-1]!=HOP or
        d["hex"].shape!=(*d["mic"].shape[:2],6,HOP) or
        d["note_gt"].shape!=(*d["mic"].shape[:2],49) or
        d["string_gt"].shape!=(*d["mic"].shape[:2],6)):
        raise ValueError("real audio data not aligned exactly 256")
    return d


def predictions(net,d,batch_size=24):
    net.eval()
    buffers={k:[] for k in ("plain_note_logits","note_logits","stem_presence_logits",
                            "stems","mic_reconstructed")}
    with torch.inference_mode():
        for idx in range(0,len(d["mic"]),batch_size):
            x=torch.from_numpy(d["mic"][idx:idx+batch_size])
            result,_=net(x,views=(-12,12))
            for name in buffers:
                buffers[name].append(result[name].detach().cpu().numpy())
    return {k:np.concatenate(v,axis=0) for k,v in buffers.items()}


def measure_one(name,cal_p,outer_p,cal_y,outer_y):
    threshold=best_cut_cal_only(cal_p,cal_y)
    if threshold is None:
        return dict(status="UNASSESSED_NO_POSITIVE_CAL",threshold=None,
                    outer_positive=int(outer_y.sum()))
    score=f1_counts(outer_p>=threshold,outer_y)
    return dict(status="MEASURED_GT_ACTIVITY_NOT_ONSET_F1",
                threshold_cal=threshold,outer_positive=int(outer_y.sum()),
                **score)


def evaluation(model_file,data_dir,out):
    out=Path(out);data_dir=Path(data_dir)
    if out.exists():raise FileExistsError("immutable real-acoustics evaluation")
    torch.set_num_threads(1)
    record=torch.load(model_file,map_location="cpu",weights_only=True)
    if (record["abi"]!=ABI or record["trained_hop_samples"]!=HOP or
        record["promoted"]):
        raise ValueError("S843 only trained 256 raw waveform shadow")
    net=NeuralAcousticTransposeHead256()
    net.load_state_dict(record["state_dict"],strict=True)
    cal=load_data(data_dir/"cal.npz")
    outer=load_data(data_dir/"outer.npz")
    result=dict(schema="S843_REAL_HEX_RECONSTRUCTION_OUTER_ACTIVITY_V1",
                complete=True,cal_source_player="05",outer_source_players=["00","01","02","03"],
                input_samples=256,sample_rate=44100,
                audio_reconstruction_is_neural_estimate=True,
                acoustic_targets_real_hex_same_performance=True,
                mic_hex_domain_and_alignment_limitations=True,
                no_synthetic_audio_training=True,
                extra_wait_samples=0,lookahead_samples=0,
                no_onset_events_evaluated=True,
                official_onset_F1_unchanged=.6925843310420176,
                no_midi_changes=True,promoted=False,metrics={})
    cp=predictions(net,cal)
    op=predictions(net,outer)
    labels={"E2":(0,),"E3":(12,),"E2_E3_SIMULTANEOUS":(0,12)}
    result["outer_pcm_blocks"]=int(np.prod(outer["mic"].shape[:2]))
    result["cal_pcm_blocks"]=int(np.prod(cal["mic"].shape[:2]))
    for head,key in (("no_transposition","plain_note_logits"),
                     ("learned_transposition","note_logits")):
        result["metrics"][head]={}
        qcal=1./(1.+np.exp(-np.clip(cp[key],-30,30)))
        qouter=1./(1.+np.exp(-np.clip(op[key],-30,30)))
        for label,notes in labels.items():
            # A real E2+E3 means BOTH pitches physically GT-active now.
            if len(notes)==1:
                cal_p=qcal[...,notes[0]]
                out_p=qouter[...,notes[0]]
                cal_gt=cal["note_gt"][...,notes[0]]>.5
                out_gt=outer["note_gt"][...,notes[0]]>.5
            else:
                cal_p=np.minimum(qcal[...,0],qcal[...,12])
                out_p=np.minimum(qouter[...,0],qouter[...,12])
                cal_gt=(cal["note_gt"][...,0]>.5)&(cal["note_gt"][...,12]>.5)
                out_gt=(outer["note_gt"][...,0]>.5)&(outer["note_gt"][...,12]>.5)
            result["metrics"][head][label]=measure_one(
                label,cal_p, out_p,cal_gt,out_gt)
    # Domain-aware error: targets are pickup signals; fidelity is not an
    # independent confirmation of true microphone-phase reconstruction.
    target=outer["hex"]
    reconstructed=op["stems"]
    result["metrics"]["audio_hex"]=dict(
        mean_absolute_error=float(np.mean(np.abs(reconstructed-target))),
        target_absolute_mean=float(np.mean(np.abs(target))),
        estimated_absolute_mean=float(np.mean(np.abs(reconstructed))),
        mean_absolute_error_within_paired_pickup_domain=True,
        exact_mic_hex_string_equivalence_unproven=True)
    result["metrics"]["mic"]=dict(
        mean_absolute_error=float(np.mean(np.abs(op["mic_reconstructed"]-outer["mic"]))))
    real_present=outer["string_gt"]>.5
    qcal=1/(1+np.exp(-np.clip(cp["stem_presence_logits"],-30,30)))
    qout=1/(1+np.exp(-np.clip(op["stem_presence_logits"],-30,30)))
    cut=best_cut_cal_only(qcal,cal["string_gt"]>.5)
    result["metrics"]["strings"]=dict(threshold_cal=cut,**f1_counts(qout>=cut,real_present))
    # CPU timing is ONE experimental Linux process and not actual Windows/iRig
    # audio end-to-end with 6 independent axes. Never assert 5.805 ms p99.
    stream=CausalRealAudioTransposeStream256(net)
    sample=outer["mic"][0].reshape(-1,HOP)
    times=[]
    for i,block in enumerate(sample):
        start=time.perf_counter_ns()
        stream.push(np.asarray(block,np.float32),start_sample=i*HOP)
        times.append((time.perf_counter_ns()-start)/1e6)
    result["linux_cpu_single_stream_ms"]=dict(
        p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
        p99=float(np.percentile(times,99)),n=len(times),
        realtime_whole_note_pipeline_NOT_proven=True)
    out.mkdir(parents=True)
    (out/"evaluation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S843_REAL_AUDIO_EVAL",json.dumps({
        "blocks":result["outer_pcm_blocks"],
        "without_transposition":result["metrics"]["no_transposition"],
        "with_transposition":result["metrics"]["learned_transposition"],
        "hex_mae":result["metrics"]["audio_hex"]["mean_absolute_error"],
        "Linux_CPU":result["linux_cpu_single_stream_ms"],
        "NOT_ONSET_F1":True},sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    evaluation(a.model,a.data,a.out)
