"""S848 fair A/B: trained S847 neural real-recording selector vs STANDARD
offline pitch shifter on EXACTLY the SAME physically recorded guitar input.

The baseline is librosa.effects.pitch_shift (standard signal-processing pitch
shifter), used ONLY in the independent offline benchmark, NEVER in the
learned S847 network. It may use noncausal windows and must NOT be claimed
real-time at 256 hop. S847 sees ONLY one 256 sample hop per decision, plus
its own past GRU state, and plays actual physically recorded GuitarSet hex.
Results are comparative AUDIO metrics, not Onset NoteOn F1 or proof that a
true performance has been faithfully transposed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import welch

from onset256.real_recorded_transposer256 import (
    HOP,SR,ABI,NeuralShiftSelector256,RecordedAudioNeuralTransposeStream256)
from scripts.s847_train_real_recorded_transposer256 import read_recorded_bank

SHIFT=12
DECISION_THRESHOLD=.23  # set before OUTER: S847 CAL pitch gates around .21-.25


def read_original(path):
    sr,raw=wavfile.read(path)
    if sr!=SR or raw.ndim!=1:
        raise ValueError("same original mono recorded guitar must be 44100 Hz")
    if raw.dtype==np.int16:
        x=raw.astype(np.float32)/32768
    elif raw.dtype==np.float32:
        x=raw.copy()
    else:
        raise ValueError("only real original mono PCM16 / float32, no synth")
    if len(x)<SR*2 or not np.isfinite(x).all():
        raise ValueError("need at least 2s valid physically recorded guitar")
    return x


def spec_profile(wave):
    x=np.asarray(wave,dtype=np.float64)
    if x.ndim!=1 or len(x)<8192:
        raise ValueError("not enough audio for rigorous Fourier-profile audit")
    fs,power=welch(x,fs=SR,nperseg=8192)
    total=float(power.sum())
    if total<1e-13:
        return dict(silent=True,rms=0.,median_hz=None,centroid_hz=None,
                    over_10khz=None,under_250hz=None)
    median=float(fs[np.searchsorted(np.cumsum(power),total/2.)])
    rms=float(np.sqrt(np.mean(x*x)))
    return dict(silent=False,rms=rms,median_hz=median,
                centroid_hz=float(np.dot(fs,power)/total),
                over_10khz=float(power[fs>=10000].sum()/total),
                under_250hz=float(power[(fs>=80)&(fs<250)].sum()/total))


def frame_envelopes(x):
    x=np.asarray(x,np.float64)
    frames=len(x)//HOP
    return np.sqrt(np.mean(x[:frames*HOP].reshape(frames,HOP)**2,axis=1))


def envelope_similarity(a,b):
    x=frame_envelopes(a);y=frame_envelopes(b)
    if len(x)!=len(y):raise ValueError("A/B duration must match")
    if np.std(x)<1e-11 or np.std(y)<1e-11:return None
    return float(np.corrcoef(x,y)[0,1])


def log_mel_distance(a,b):
    # OFFLINE evaluation metric; never an inference feature.
    ma=librosa.feature.melspectrogram(y=a,sr=SR,n_fft=2048,hop_length=HOP,
                                      n_mels=80,power=2.0)
    mb=librosa.feature.melspectrogram(y=b,sr=SR,n_fft=2048,hop_length=HOP,
                                      n_mels=80,power=2.0)
    ea=librosa.power_to_db(ma,ref=np.max)
    eb=librosa.power_to_db(mb,ref=np.max)
    return float(np.mean(np.abs(ea-eb)))


def estimate_note_histogram(probs,bank):
    # Raw per-block neural scores and chosen label; not proof of correct pitch
    filtered=[p for p in probs if p is not None]
    return dict(played_hops=len(filtered),silence_hops=len(probs)-len(filtered),
                played_note_histogram={str(k):v for k,v in
                      sorted(Counter(filtered).items(),key=lambda z:-z[1])[:15]},
                playable_note_count=len(bank))


def make_comparison(original_path,model_path,bank_dir,destination,*,seconds=3.5):
    out=Path(destination)
    if out.exists():raise FileExistsError("S848 immutable experimental audio")
    source_full=read_original(original_path)
    n=min(round(seconds*SR),len(source_full))
    start=0
    if n<len(source_full):
        candidates=range(0,len(source_full)-n+1,SR//4)
        start=max(candidates,key=lambda i:float(np.mean(source_full[i:i+n]**2)))
    source=np.ascontiguousarray(source_full[start:start+n],dtype=np.float32)

    checkpoint=torch.load(model_path,map_location="cpu",weights_only=True)
    if (checkpoint.get("abi")!=ABI or
        checkpoint.get("trained_with_original_PCM_HOP")!=HOP or
        checkpoint.get("promoted") is not False):
        raise ValueError("untrained or wrong S847 real note shift selector")
    neural=NeuralShiftSelector256()
    neural.load_state_dict(checkpoint["model_state"],strict=True)
    torch.set_num_threads(1)
    bank,bank_info=read_recorded_bank(bank_dir)
    stream=RecordedAudioNeuralTransposeStream256(
         neural,bank,shift=SHIFT,threshold=DECISION_THRESHOLD)
    out_ia=[]
    note_choices=[]
    for t in range(0,len(source),HOP):
        fresh=np.zeros(HOP,np.float32)
        length=min(HOP,len(source)-t)
        fresh[:length]=source[t:t+length]
        step=stream.push(fresh,start_sample=t)
        out_ia.append(step.physical_recorded_wave256[:length].copy())
        note_choices.append(step.note)
    ai=np.concatenate(out_ia).astype(np.float32)
    # Conventional REFERENCE, computed after AI evaluation. Never used in
    # training, latent features, PCM256 decisions or sample-bank selection.
    conventional=np.asarray(
       librosa.effects.pitch_shift(source,sr=SR,n_steps=SHIFT),
       dtype=np.float32)
    if len(conventional)!=len(source) or len(ai)!=len(source):
        raise AssertionError("standard/AI must have identical original timing")

    in_prof=spec_profile(source)
    std_prof=spec_profile(conventional)
    ai_prof=spec_profile(ai)
    a=estimate_note_histogram(note_choices,bank)
    out.mkdir(parents=True)
    def save(name,wave):
        wavfile.write(out/name,SR,
                      (np.clip(wave,-1.,1.-1/32768)*32768).astype(np.int16))
    save("01_entree_guitare_reelle.wav",source)
    save("02_transposeur_standard_plus12.wav",conventional)
    save("03_transposeur_IA_S847_plus12.wav",ai)
    pause=np.zeros(int(.3*SR),np.float32)
    save("04_comparaison_entree_standard_IA.wav",
         np.concatenate((source,pause,conventional,pause,ai)))
    result=dict(schema="S848_REAL_GUITAR_AI_VS_CONVENTIONAL_PLUS12_PCM256",
      complete=True,original_wav=str(original_path),
      original_excerpt_start_s=start/SR,segment_seconds=len(source)/SR,
      neural_checkpoint=str(model_path),real_bank_guitarist=bank_info["source_guitarist"],
      source_audio_is_physically_recorded=True,
      standard_method="librosa.effects.pitch_shift offline reference (noncausal)",
      standard_used_for_AI_training=False,
      AI_method="learned note selection plus physical captured guitar sample replay",
      AI_not_faithful_reconstruction_of_performance=True,
      standard_not_evaluated_for_strict_live_latency=True,
      original_new_samples_per_AI_inference=256,
      AI_lookahead_samples=0,AI_added_hold_samples=0,
      shift_semitones=SHIFT,AI_threshold_predeclared_from_CAL=DECISION_THRESHOLD,
      analysis_NOT_Onset_F1=True,
      note_decisions=a,
      acoustic=dict(original=in_prof,standard=std_prof,AI=ai_prof),
      comparison=dict(
         AI_vs_standard_logmel_abs_db=log_mel_distance(ai,conventional),
         AI_vs_standard_envelope_correlation=envelope_similarity(ai,conventional),
         standard_vs_original_envelope_correlation=envelope_similarity(conventional,source),
         AI_vs_original_envelope_correlation=envelope_similarity(ai,source),
         AI_output_silent=ai_prof["silent"],
         original_and_standard_non_silent=not in_prof["silent"] and not std_prof["silent"]),
      notes="Audio comparison shows benchmark differences, not guaranteed musical correctness.")
    (out/"comparaison_resultat.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S848_REAL_GUITAR_AI_VS_STANDARD_AB",json.dumps(dict(
        choices=result["note_decisions"],
        spectrum=result["acoustic"],
        comparison=result["comparison"]),sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--wav",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    make_comparison(a.wav,a.model,a.bank,a.out)
