"""Experimental training of an actual *learned* latent transposer, not DSP.

Requires a separately supplied NPZ of independently RECORDED audio pairs whose
ground-truth notes differ by the declared semitone displacement. No synthesized
pitch-shift audio, S708 teacher, or historic NOTE model may substitute.
TRAIN/CAL/OUTER are disjoint recording IDs. Printed F1 is a FRAME PROXY,
never official Onset event F1; all outputs are research shadow only.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from onset256.learned_latent_transposer256 import (
    NeuralLatentTransposer256,MODEL_ABI,SR,HOP,
    recorded_pair_training_loss,verify_recorded_pairs)
SPLITS=("train","cal","outer")
KEYS=("source_pcm","target_pcm","source_note_mask","target_note_mask",
      "semitones","split","source_track","target_track")


def validate_partition(data):
    visited={k:set() for k in SPLITS}
    if any(str(k) not in SPLITS for k in data["split"]):
        raise ValueError("only train, cal and outer are allowed")
    for i in range(len(data["split"])):
        part=str(data["split"][i])
        a,b=str(data["source_track"][i]),str(data["target_track"][i])
        if not a or not b or a==b:
            raise ValueError("pair needs distinct real original recordings")
        visited[part].update((a,b))
    if any(not np.any(data["split"]==p) for p in SPLITS):
        raise ValueError("TRAIN, CAL, OUTER all need recordings")
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if visited[a]&visited[b]:
                raise ValueError("GT source or target recording leaks across "+a+"/"+b)


def load_verified_pairs(dataset,manifest):
    """Checks declared origin and SHA; authenticity still needs human WAV audit."""
    info=json.loads(manifest.read_text(encoding="utf8"))
    expected=dict(
        schema="ONSET_REAL_RECORDED_NOTE_TRANSPOSE_PAIRS_256_V1",
        sample_rate_hz=SR,pcm_block_samples=HOP,
        audio_pair_source="independently_recorded_real_audio",
        dsp_pitch_shift_used=False,lookahead_samples=0,
        source_s708_teacher_used=False)
    if any(type(info.get(k)) is not type(v) or info.get(k)!=v
           for k,v in expected.items()):
        raise ValueError("original real-audio PCM256 source provenance missing")
    sha=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if info.get("paired_npz_sha256")!=sha:
        raise ValueError("signed pair dataset hash changed")
    with np.load(dataset,allow_pickle=False) as archive:
        if any(k not in archive for k in KEYS):
            raise ValueError("NPZ missing original audio, labels or split IDs")
        data={k:archive[k].copy() for k in KEYS}
    n=len(data["source_pcm"])
    if n<6 or any(len(x)!=n for x in data.values()):
        raise ValueError("need at least 6 matching recording pairs")
    for k in ("source_pcm","target_pcm"):
        x=data[k]
        if x.shape!=(n,256) or x.dtype!=np.float32 or not np.isfinite(x).all() or np.max(np.abs(x))>1:
            raise ValueError("only genuine mono float32 PCM [N,256] accepted")
    for k in ("source_note_mask","target_note_mask"):
        if data[k].shape!=(n,49) or data[k].dtype!=np.float32:
            raise ValueError("GT must be float32 note masks [N,49]")
    if data["semitones"].shape!=(n,) or data["semitones"].dtype.kind not in "iu":
        raise ValueError("need integer semitone labels per real pair")
    validate_partition(data)
    verify_recorded_pairs(
        torch.from_numpy(data["source_pcm"]),torch.from_numpy(data["target_pcm"]),
        torch.from_numpy(data["source_note_mask"]),torch.from_numpy(data["target_note_mask"]),
        torch.from_numpy(data["semitones"].astype("int64")))
    return data,sha


def frame_micro_f1(logits,gt):
    """Frame-note only. NOT an Onset event F1, no 80ms matching."""
    prediction=torch.sigmoid(logits)>.5
    expected=gt>.5
    tp=int((prediction&expected).sum())
    fp=int((prediction&~expected).sum())
    fn=int((~prediction&expected).sum())
    return 2.*tp/max(1,2*tp+fp+fn)


def measure(model,data,ids,batch_size):
    model.eval()
    full=[];plain=[];gt=[]
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            ix=ids[start:start+batch_size]
            x=torch.from_numpy(data["source_pcm"][ix])
            logits,control,_=model(x)
            full.append(logits);plain.append(control)
            gt.append(torch.from_numpy(data["source_note_mask"][ix]))
    truth=torch.cat(gt)
    return dict(transposed_frame_note_F1=frame_micro_f1(torch.cat(full),truth),
                plain_frame_note_F1=frame_micro_f1(torch.cat(plain),truth))


def train(dataset,manifest,out,*,epochs=5,batch_size=128,seed=839):
    if out.exists():raise FileExistsError("never overwrite trained experiment")
    if epochs<1 or batch_size<1 or seed<0:raise ValueError("invalid training arguments")
    data,sha=load_verified_pairs(dataset,manifest)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    rng=np.random.default_rng(seed)
    ids={k:np.flatnonzero(data["split"]==k) for k in SPLITS}
    model=NeuralLatentTransposer256()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=-1.;best_epoch=None;weights=None;history=[]
    for epoch in range(epochs):
        model.train();rng.shuffle(ids["train"])
        losses=[]
        for start in range(0,len(ids["train"]),batch_size):
            ix=ids["train"][start:start+batch_size]
            src=torch.from_numpy(data["source_pcm"][ix])
            dst=torch.from_numpy(data["target_pcm"][ix])
            src_y=torch.from_numpy(data["source_note_mask"][ix])
            dst_y=torch.from_numpy(data["target_note_mask"][ix])
            shift=torch.from_numpy(data["semitones"][ix].astype("int64"))
            loss,_=recorded_pair_training_loss(model,src,dst,src_y,dst_y,shift)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            losses.append(float(loss.detach()))
        cal=measure(model,data,ids["cal"],batch_size)
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),cal=cal))
        if cal["transposed_frame_note_F1"]>best:
            best=cal["transposed_frame_note_F1"]
            best_epoch=epoch+1
            weights=copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    outer=measure(model,data,ids["outer"],batch_size)
    out.mkdir(parents=True)
    torch.save(dict(model_state=weights,abi=MODEL_ABI,dataset_sha256=sha,
                    best_epoch_cal_only=best_epoch,promoted=False),
               out/"neural_latent_transposer256.pt")
    report=dict(schema=MODEL_ABI,complete=True,training_pcm_samples=HOP,
        inference_pcm_samples=HOP,added_wait_samples=0,lookahead_samples=0,
        source_dataset_sha256=sha,source_manifest=str(manifest),
        outer_labels_not_used_for_selection=True,
        real_recordings_must_be_externally_authenticated=True,
        train_pairs=int(len(ids["train"])),cal_pairs=int(len(ids["cal"])),
        outer_pairs=int(len(ids["outer"])),best_epoch=best_epoch,
        cal_history=history,outer_frame_proxy=outer,
        onset_event_f1_80_wav_NOT_MEASURED=True,
        reference_pcm256_onset_event_f1=.6925843310420176,
        no_other_axes_changed=True,public_midi_changed=False,promoted=False)
    (out/"resultat.json").write_text(json.dumps(report,indent=2)+"\n")
    print("TRANSPOSE256_FRAME_PROXY_ONLY",json.dumps(outer,sort_keys=True))
    print("ONSET80_EVENT_F1_NOT_MEASURED")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("pairs","manifest","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=839)
    a=parser.parse_args()
    train(a.pairs,a.manifest,a.out,epochs=a.epochs,batch_size=a.batch_size,seed=a.seed)
