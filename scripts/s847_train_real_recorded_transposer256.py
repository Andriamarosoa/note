"""S847 real-recording neural shift selector: genuine guitar notes, no audio DSP.

TRAIN/CAL/OUTER sourced from signed GuitarSet microphone/hex/JAMS artifacts
prepared by S844. TRAIN guitarist=04, CAL=05, OUTER=00..03. OUTER is unseen
during optimization and selection; test metrics are FRAME note hypotheses,
never original-note Onset F1. Output uses actual, real hex-channel recordings
from separate physical performances (NOT direct recording reconstruction).
"""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path
from collections import Counter,defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from onset256.real_recorded_transposer256 import (
    ABI,HOP,SR,PITCHES,PITCH_LOW,NeuralShiftSelector256,supervised_loss,
    shift_truth_for_supervision_only,make_recorded_bank,
    RecordedAudioNeuralTransposeStream256)
from scripts.s843_train_real_hex_transpose256 import (
    authenticate,unique_zip_name,wav_in_zip,read_jams_note_midi)
from scripts.s843_evaluate_real_hex_reconstruction256 import (
    f1_counts,best_cut_cal_only)

SEED=847
SHIFTS=(-12,0,12)


def true_note_bank(mic_zip,hex_zip,jams_zip,output,*,train_player="04"):
    """Extract verbatim guitar audio from GT-audited HEX string channel.

    String index is from original JAMS note_midi annotation order and MUST
    be separately acoustically checked if exact correspondence is claimed.
    """
    output=Path(output)
    if output.exists():raise FileExistsError("original captured note bank is immutable")
    checks=authenticate(mic_zip,hex_zip,jams_zip)
    collected={}
    source={}
    with zipfile.ZipFile(mic_zip) as microphones, zipfile.ZipFile(hex_zip) as hex_wavs, zipfile.ZipFile(jams_zip) as jamz:
        tracks=sorted(Path(x).name[:-8] for x in microphones.namelist()
                      if x.endswith("_mic.wav") and Path(x).name.startswith(train_player+"_"))
        for name in tracks:
            real_hex=wav_in_zip(hex_wavs,name+"_hex_cln.wav",6)
            notes=read_jams_note_midi(jamz,name)
            # Prefer a full physically played sustained note when available;
            # never fabricate waveform by pitch DSP or clip-stitching.
            for begin,end,pitch,ch in notes:
                if pitch in collected or not 0<=ch<6:continue
                start=int((begin+.020)*SR/HOP)+1
                end_hop=min(len(real_hex)//HOP,int(end*SR/HOP))
                frames=end_hop-start
                if frames<16:continue
                window=real_hex[start*HOP:(start+min(frames,128))*HOP,ch]
                window=np.ascontiguousarray(window,np.float32)
                if len(window)%HOP!=0 or len(window)<HOP*16:continue
                rms=float(np.sqrt(np.mean(window.astype(np.float64)**2)))
                if rms < .0002:continue
                collected[pitch]=window
                source[pitch]=dict(track=name,guitarist=train_player,
                    original_note=pitch,hex_channel_from_JAMS_order=ch,
                    original_note_GT_onset_s=begin,
                    audio_excerpt_start_s=start*HOP/SR,
                    captured_samples=len(window),original_rms=rms)
            if len(collected)==49:break
    if not collected:raise ValueError("no genuine independently recorded string note")
    output.mkdir(parents=True)
    physical={f"note_{p}":clip for p,clip in collected.items()}
    np.savez_compressed(output/"physically_recorded_guitar_notes256.npz",**physical)
    details=dict(schema="S847_GENUINE_GUITAR_RECORDED_NOTE_EXEMPLAR_BANK_V1",
        source_checksums=checks,source_guitarist=train_player,
        true_live_mic_audio_for_model=True,
        recorded_guitar_note_audio_not_synthesized=True,
        real_note_audio_not_pitch_shifted_by_DSP=True,
        hex_and_JAMS_string_index_alignment_not_acoustically_proven=True,
        training_only_source_tracks_no_outer_leakage=True,
        bank_real_recorded_note_count=len(collected),
        available_recorded_midi=sorted(int(p) for p in collected),
        original_physical_recordings=source,
        output_guitar_performance_is_not_a_true_replayed_transposition=True)
    (output/"recorded_bank_provenance.json").write_text(json.dumps(details,indent=2)+"\n")
    print("S847_GUITAR_REAL_WAVE_BANK",json.dumps({
        "n":len(collected),"available":sorted(collected),
        "E2_present":40 in collected,"E3_present":52 in collected},sort_keys=True),flush=True)
    return details


def read_recorded_bank(directory):
    directory=Path(directory)
    info=json.loads((directory/"recorded_bank_provenance.json").read_text())
    if info["schema"]!="S847_GENUINE_GUITAR_RECORDED_NOTE_EXEMPLAR_BANK_V1":
        raise ValueError("unverified real guitar audio bank")
    with np.load(directory/"physically_recorded_guitar_notes256.npz",allow_pickle=False) as data:
        notes=[int(key.removeprefix("note_")) for key in data.files]
        clips=[data[f"note_{p}"].copy() for p in notes]
    out=make_recorded_bank(clips,notes,[info["original_physical_recordings"][str(p)]["track"] for p in notes])
    if set(out)!=set(info["available_recorded_midi"]):
        raise ValueError("real recorded bank changed from signed index")
    return out,info


def read_split(directory,split):
    with np.load(Path(directory)/(split+".npz"),allow_pickle=False) as ds:
        pcm=ds["mic"].copy()
        truth=ds["note_gt"].copy()
        ids=ds["track_id"].copy()
    if (pcm.ndim!=3 or pcm.shape[-1]!=HOP or
        truth.shape!=(*pcm.shape[:2],PITCHES) or ids.shape!=(len(pcm),) or
        pcm.dtype!=np.float32 or truth.dtype!=np.float32):
        raise ValueError("real split violates signed raw PCM256 and GT shape")
    return dict(pcm=pcm,truth=truth,track_id=ids)


def train_only_weights(data):
    rows=data["truth"].reshape(-1,PITCHES)
    positives=rows.sum(axis=0)
    ratio=(len(rows)-positives)/np.maximum(positives,1)
    weights=np.where(positives>0,np.minimum(np.sqrt(np.maximum(ratio,1)),25.),1.).astype(np.float32)
    return torch.from_numpy(weights),positives


def eval_probs(net,data,shift,batch_size=128):
    rows=[];plain=[]
    with torch.inference_mode():
        for i in range(0,len(data["pcm"]),batch_size):
            source=torch.from_numpy(data["pcm"][i:i+batch_size])
            sh=torch.full((len(source),),shift,dtype=torch.int64)
            logits,_=net(source,sh)
            rows.append(torch.sigmoid(logits["target_logits"]).numpy())
            plain.append(torch.sigmoid(logits["source_logits"]).numpy())
    return np.concatenate(rows,axis=0),np.concatenate(plain,axis=0)


def binary_metrics_from_real_gt(cal,outer,shift,cal_score,outer_score,pitch):
    cal_gt=torch.from_numpy(cal["truth"])
    out_gt=torch.from_numpy(outer["truth"])
    c=shift_truth_for_supervision_only(cal_gt,
                     torch.full((len(cal_gt),),shift,dtype=torch.int64)).numpy()[...,pitch-PITCH_LOW]>.5
    o=shift_truth_for_supervision_only(out_gt,
                     torch.full((len(out_gt),),shift,dtype=torch.int64)).numpy()[...,pitch-PITCH_LOW]>.5
    if not np.any(c):
        return dict(status="UNASSESSABLE_ZERO_POSITIVE_CAL",positive_cal=0,
                    positive_outer=int(o.sum()),f1=None)
    if not np.any(o):
        return dict(status="UNASSESSABLE_ZERO_POSITIVE_OUTER",
                    positive_cal=int(c.sum()),positive_outer=0,f1=None)
    threshold=best_cut_cal_only(cal_score[...,pitch-PITCH_LOW],c)
    report=f1_counts(outer_score[...,pitch-PITCH_LOW]>=threshold,o)
    return dict(status="HELD_OUT_FRAME_NOTE_NOT_ONSET_F1",
                positive_cal=int(c.sum()),positive_outer=int(o.sum()),
                cut_from_cal=threshold,**report)


def train_and_test(real_dataset,model_dir,*,epochs=7,seed=SEED):
    real_dataset=Path(real_dataset);model_dir=Path(model_dir)
    if model_dir.exists():raise FileExistsError("S847 report immutable")
    if epochs<1:raise ValueError("train at least one epoch")
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=read_split(real_dataset,"train")
    cal=read_split(real_dataset,"cal")
    outer=read_split(real_dataset,"outer")
    pos_weight,positives=train_only_weights(train)
    print("S847_REAL_TRAIN_GT_POSITIVES",dict(E2=int(positives[0]),
          E3=int(positives[12]),frames=int(train["truth"].shape[0]*train["truth"].shape[1])),flush=True)
    net=NeuralShiftSelector256()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    selected=None;lowest=float("inf");best_epoch=0;history=[]
    for epoch in range(epochs):
        net.train()
        losses=[]
        for ids in np.array_split(rng.permutation(len(train["pcm"])),
                                  max(1,int(np.ceil(len(train["pcm"])/64)))):
            x=torch.from_numpy(train["pcm"][ids])
            gt=torch.from_numpy(train["truth"][ids])
            shifts=torch.from_numpy(rng.choice(SHIFTS,len(ids)).astype(np.int64))
            loss,_=supervised_loss(net,x,gt,shifts,pos_weight)
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.inference_mode():
            validation=[]
            for ids in np.array_split(np.arange(len(cal["pcm"])),
                                      max(1,int(np.ceil(len(cal["pcm"])/64)))):
                x=torch.from_numpy(cal["pcm"][ids]);y=torch.from_numpy(cal["truth"][ids])
                shifts=torch.full((len(ids),),12,dtype=torch.int64)
                loss,_=supervised_loss(net,x,y,shifts,pos_weight)
                validation.append(float(loss))
        loss_cal=float(np.mean(validation))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),
                            cal_shift12_loss=loss_cal))
        print("S847_REAL_SHIFT_TRAIN_EPOCH",epoch+1,losses[-1],loss_cal,flush=True)
        if loss_cal<lowest:
            lowest=loss_cal;best_epoch=epoch+1
            selected={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(selected)
    net.eval()
    frame_metrics={}
    for shift in (-12,0,12):
        cal_scores,_=eval_probs(net,cal,shift)
        outer_scores,plain=eval_probs(net,outer,shift)
        pitch_report={}
        for pitch in ((40,52) if shift==-12 else (52,64) if shift==12 else (40,52)):
            pitch_report[str(pitch)]=binary_metrics_from_real_gt(
                cal,outer,shift,cal_scores,outer_scores,pitch)
        frame_metrics[str(shift)]=pitch_report
    model_dir.mkdir(parents=True)
    torch.save(dict(abi=ABI,model_state=selected,epoch_selected_by_CAL_only=best_epoch,
         trained_with_original_PCM_HOP=HOP,source_dataset_from_real_recordings=True,
         fully_synthetic_source=False,model_produced_waveform=False,
         already_deployed_in_NOTE=False,promoted=False),
         model_dir/"s847_neural_target_selector.pt")
    report=dict(schema="S847_REAL_NOTE_TO_RECORDED_AUDIO_NEURAL_TRANSPOSER",
        complete=True,epoch_selected_by_CAL_only=best_epoch,history=history,
        TRAIN_guitarist="04",CAL_guitarist="05",OUTER_guitarists=["00","01","02","03"],
        shift_logits_learned_without_frequency_formula=True,
        GT_pitch_shift_only_computed_during_offline_training=True,
        actual_recorded_audio_samples_not_neural_synthetic_audio=True,
        exact_256_new_samples_per_live_decision=True,
        audio_is_not_same_guitar_performance_transposed=True,
        instrument_audio_fidelity_does_not_equal_pitch_accuracy=True,
        all_six_official_NOTE_heads_and_MIDI_unchanged=True,
        Onset_F1_80WAV_NOT_MEASURED=True,
        historical_Onset_PCM256_F1=.6925843310420176,
        report_frame_note_not_NoteOn_F1=frame_metrics,
        score_not_promoted=True)
    (model_dir/"resultat.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S847_REAL_AUDIO_SHIFT_FINAL",json.dumps(frame_metrics),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("bank","train"))
    parser.add_argument("--mic-zip",type=Path)
    parser.add_argument("--hex-zip",type=Path)
    parser.add_argument("--annotation-zip",type=Path)
    parser.add_argument("--data",type=Path,default=Path("/tmp/s847_real_dataset"))
    parser.add_argument("--bank-out",type=Path,default=Path("/tmp/s847_recorded_bank"))
    parser.add_argument("--out",type=Path,default=Path("/tmp/s847_selector"))
    parser.add_argument("--epochs",type=int,default=7)
    args=parser.parse_args()
    if args.phase=="bank":
        if not all((args.mic_zip,args.hex_zip,args.annotation_zip)):
            parser.error("bank requires 3 authentic original archives")
        true_note_bank(args.mic_zip,args.hex_zip,args.annotation_zip,args.bank_out)
    else:
        train_and_test(args.data,args.out,epochs=args.epochs)
