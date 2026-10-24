"""Create ONE audible A/B demonstration from a physically recorded guitar WAV.

Part 1: real acoustic guitar microphone audio, unchanged pitch.
Part 2: S845 neural estimated audio from learned +12 latent view.

The second is NOT an independently recorded true +12 guitar performance,
nor a successful transposition guarantee. Its amplitude is normalized ONLY
for audibility. No DSP pitch shift, oscillators, sine tones, or MIDI notes
are generated; all model decisions receive precisely 256 new PCM samples.
This is an honest failed/experimental model listening example, not a quality
or Onset-F1 demonstration.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import json

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly, welch
from onset256.real_audio_transpose_head256 import (
    ABI,SR,HOP,NeuralAcousticTransposeHead256,CausalRealAudioTransposeStream256)


def read_real_guitar(path):
    rate,pcm=wavfile.read(path)
    if pcm.ndim == 2:
        pcm=pcm.astype(np.float32).mean(axis=1)
    elif pcm.ndim != 1:
        raise ValueError("expected genuine single or stereo audio file")
    if np.issubdtype(pcm.dtype,np.integer):
        pcm=pcm.astype(np.float32)/float(max(abs(np.iinfo(pcm.dtype).min),
                                               np.iinfo(pcm.dtype).max))
    elif np.issubdtype(pcm.dtype,np.floating):
        pcm=pcm.astype(np.float32)
    else:
        raise ValueError("unsupported real audio format")
    if rate!=SR:
        # Offline sample-rate adaptation ONLY to the fixed model rate, NOT
        # pitch-shifting or resynthesizing guitar content.
        g=math.gcd(rate,SR)
        pcm=resample_poly(pcm,SR//g,rate//g).astype(np.float32)
    if pcm.size < SR*2 or not np.isfinite(pcm).all():
        raise ValueError("need >=2 seconds of recorded guitar audio")
    return pcm


def _spectrum_statistics(x, sr=SR):
    """Independent audit of audio ENERGY LOCATION, not proof of correct pitch."""
    x=np.asarray(x,np.float64)
    if x.ndim!=1 or len(x)<8192 or not np.isfinite(x).all():
        raise ValueError("not enough finite audio for independent quality audit")
    freqs,psd=welch(x,fs=sr,nperseg=8192)
    total=float(psd.sum())
    if not np.isfinite(total) or total<1e-15:
        raise ValueError("audible reconstructed waveform energy is effectively zero")
    centroid=float((freqs*psd).sum()/total)
    median=float(freqs[np.searchsorted(np.cumsum(psd),total*.5)])
    high=float(psd[freqs>=10000].sum()/total)
    bass=float(psd[(freqs>=80)&(freqs<250)].sum()/total)
    return dict(spectral_centroid_hz=centroid,spectral_median_hz=median,
                energy_above_10khz_fraction=high,energy_80_to_250hz_fraction=bass)


def audit_transposed_audio(genuine,estimated):
    """FAIL CLOSED for clearly invalid neural transposition.

    256-hop causality is NOT sufficient evidence of meaningful audio.
    This catches high-frequency stationary noise from S845; passing it alone
    never proves that an octave was reconstructed correctly.
    """
    if len(genuine)!=len(estimated):
        raise ValueError("original and learned audio lengths differ")
    source=_spectrum_statistics(genuine)
    predicted=_spectrum_statistics(estimated)
    # Guitar in E2-E5 generally has a spectral median well below 1.5 kHz.
    # A learned +12 approximation may change timbre, so these are broad
    # sanity bounds, *not* a DSP pitch-shift or inverse note calculation.
    reasons=[]
    ratio=predicted["spectral_median_hz"]/max(source["spectral_median_hz"],1.)
    if source["spectral_median_hz"]<1500. and ratio>7.:
        reasons.append("predicted spectral median is >7x source, not an octave-like guitar")
    if (source["energy_above_10khz_fraction"]<.15 and
        predicted["energy_above_10khz_fraction"]>.45):
        reasons.append("prediction dominated by high-frequency noise absent in the recorded guitar")
    # A constant synthetic-esque output cannot reproduce genuine playing
    # dynamics, even if raw samples and amplitude normalization are legal.
    size=max(1,int(round(.25*SR)))
    N=len(genuine)//size
    src_env=np.array([np.sqrt(np.mean(genuine[k*size:(k+1)*size]**2))
                      for k in range(N)])
    est_env=np.array([np.sqrt(np.mean(estimated[k*size:(k+1)*size]**2))
                      for k in range(N)])
    source_cv=float(src_env.std()/max(src_env.mean(),1e-12))
    estimated_cv=float(est_env.std()/max(est_env.mean(),1e-12))
    if source_cv>.30 and estimated_cv<.04:
        reasons.append("estimated dynamics nearly constant across changing recorded performance")
    return dict(passed=not bool(reasons),fail_reasons=reasons,
                original=source,estimated=predicted,
                spectral_median_ratio=ratio,
                source_envelope_cv=source_cv,
                estimated_envelope_cv=estimated_cv)


def produce(genuine_recording, checkpoint, target, *,seconds=3.5):
    target=Path(target)
    if target.exists():raise FileExistsError("never overwrite an existing demo")
    source=read_real_guitar(genuine_recording)
    # Choose a clear contiguous guitar segment OFFLINE. Does not affect
    # individual causal 256-hop decisions.
    n=min(len(source),round(SR*seconds))
    if len(source)>n:
        stride=SR//4
        possible=[(float(np.sqrt(np.mean(source[i:i+n]**2))),i)
                  for i in range(0,len(source)-n+1,stride)]
        start=max(possible)[1]
    else:start=0
    original=np.ascontiguousarray(source[start:start+n],np.float32)
    payload=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if payload.get("abi")!=ABI or payload.get("trained_hop_samples")!=HOP:
        raise ValueError("only genuine trained S845 recurrent-256 checkpoint allowed")
    model=NeuralAcousticTransposeHead256()
    model.load_state_dict(payload["state_dict"],strict=True)
    torch.set_num_threads(1)
    stream=CausalRealAudioTransposeStream256(model,views=(-12,12))
    processed=[];estimated=[];confidence=[]
    for i,offset in enumerate(range(0,len(original),HOP)):
        window=original[offset:offset+HOP]
        valid=len(window)
        if valid<HOP:
            buf=np.zeros(HOP,np.float32)
            buf[:valid]=window
            window=buf
        view=stream.push(np.ascontiguousarray(window,np.float32),
                         start_sample=offset)
        # Learned +12 waveform estimated from transposed LATENT, no fixed
        # frequency-multiplying algorithm and no true hex at inference.
        audio=np.mean(view.transposed_acoustic_estimates[1],axis=0)
        processed.append(audio[:valid])
        estimated.append(np.mean(view.strings,axis=0)[:valid])
        confidence.append(float(view.note_probabilities.max()))
    plus12=np.concatenate(processed)
    direct=np.concatenate(estimated)
    acoustic_audit=audit_transposed_audio(original,plus12)
    print("S846_TRANSPOSE_AUDIO_QUALITY_GATE",json.dumps(acoustic_audit),flush=True)
    if not acoustic_audit["passed"]:
        raise ValueError("reject false S845 guitar transposition demo: "+
                         "; ".join(acoustic_audit["fail_reasons"]))
    def rms(x):return float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
    signal_rms=rms(original)
    predicted_rms=rms(plus12)
    direct_rms=rms(direct)
    if signal_rms<1.e-8:raise ValueError("original recording is silent")
    if predicted_rms<1.e-9:
        raise ValueError("S845 produced no audible learned-transposition signal")
    # Adjust VOLUME only, not musical pitches. A weak estimate may sound
    # very noisy after gain; report exact values in accompanying metadata.
    gain=min(signal_rms/predicted_rms,2000)
    amp=plus12*gain
    maxpeak=float(np.max(np.abs(amp)))
    clip_gain=min(1.,.80/max(maxpeak,1.e-9))
    amp=amp*clip_gain
    # One WAV: physically recorded original followed by an experimental
    # estimated +12 output, separated by genuine silence.
    separation=np.zeros(round(.35*SR),np.float32)
    final=np.concatenate((original,separation,amp))
    fade=min(1024, len(original)//10)
    window=np.sin(np.linspace(0,np.pi/2,fade,dtype=np.float32))**2
    final[:fade]*=window
    final[len(original)-fade:len(original)]*=window[::-1]
    k=len(original)+len(separation)
    final[k:k+fade]*=window
    final[-fade:]*=window[::-1]
    if float(np.max(np.abs(final)))>1.:
        final=final/max(1.,np.max(np.abs(final)))*.95
    target.parent.mkdir(parents=True,exist_ok=True)
    wavfile.write(target,SR,np.int16(np.clip(final,-1,1)*32767))
    summary=dict(schema="S845_REAL_GUITAR_AUDIO_AB_DEMO_NO_DSP",
        source_name=Path(genuine_recording).name,checkpoint_name=Path(checkpoint).name,
        original_sample_rate=SR,source_excerpt_start_s=start/SR,
        segment_duration_s=len(original)/SR,separator_s=len(separation)/SR,
        original_rms=signal_rms,estimated_plus12_raw_rms=predicted_rms,
        estimated_direct_raw_rms=direct_rms,
        estimated_plus12_volume_gain=gain,additional_safety_gain=clip_gain,
        plus12_output_is_learned_neural_estimate_not_original_guitar=True,
        valid_native_PCM256_decisions=int(math.ceil(len(original)/HOP)),
        no_synthetic_guitar_source=True,no_DSP_pitch_shifter_used=True,
        no_guarantee_of_correct_musical_transposition=True,
        sonic_sanity_checks_not_pitch_accuracy_proof=True,
        acoustic_quality_gate=acoustic_audit,
        official_note_MIDI_unchanged=True,
        original_duration_s=len(original)/SR,
        final_duration_s=len(final)/SR,
        first_part="ORIGINAL REAL RECORDED GUITAR",
        second_part="S845 NEURAL ESTIMATED LEARNED PLUS 12, VOLUME NORMALIZED",
        evaluation_E2_E3_not_inferred_from_demo=True)
    target.with_suffix(".json").write_text(json.dumps(summary,indent=2)+"\n")
    print("S845_REAL_GUITAR_DEMO_CREATED",json.dumps(summary),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--wav",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    produce(args.wav,args.model,args.out)
