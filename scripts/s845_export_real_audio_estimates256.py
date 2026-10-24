"""Use a *trained* S845 real-guitar neural acoustic reconstruction model on
an original microphone WAV. This is a forensic sound-preview tool, NOT MIDI.

Outputs estimated six-string RAW acoustic signals and *estimated* learned
octave-transport waveforms. The estimates are NOT the original mic/hex stems,
not guaranteed true transposed performances, and cannot independently prove
the existence of E3 while E2 is sounding.

At each hop only the CURRENT 256 recorded PCM samples are fed in; previous
state is a recurrent hidden vector, no lookahead, no buffering/hold.
No oscillator, pitch-shifting DSP, vocoder, FFT or resampling exists here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from onset256.real_audio_transpose_head256 import (
    ABI,HOP,SR,NeuralAcousticTransposeHead256,
    CausalRealAudioTransposeStream256)


def _audio_44100_mono(path):
    rate,pcm=wavfile.read(path)
    if rate != SR or pcm.ndim!=1:
        raise ValueError("only ORIGINAL mono microphone 44100 Hz WAV input")
    if pcm.dtype == np.int16:
        samples=pcm.astype(np.float32)/32768.
    elif pcm.dtype==np.float32:
        samples=pcm
    else:
        raise ValueError("WAV must be uncompressed real mono PCM16 or float32")
    if samples.size==0 or not np.isfinite(samples).all() or np.max(np.abs(samples))>1.:
        raise ValueError("input WAV contains invalid real PCM values")
    return samples


def _save_wav(path,pcm):
    """PCM16 encoding is for output file compatibility, not note synthesis."""
    pcm=np.asarray(pcm,np.float32)
    clipped=np.clip(pcm,-1,1-1/32768)
    wavfile.write(path,SR,np.round(clipped*32768).astype(np.int16))


def run(real_mic_wav,checkpoint,output,*,include_shift_audio=True):
    destination=Path(output)
    if destination.exists():
        raise FileExistsError("never replace previous real-audio forensic preview")
    audio=_audio_44100_mono(real_mic_wav)
    bundle=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (bundle.get("abi")!=ABI or bundle.get("trained_hop_samples")!=256
            or bundle.get("promoted") is not False):
        raise ValueError("checkpoint lacks known trained S845-compatible ABI")
    net=NeuralAcousticTransposeHead256()
    net.load_state_dict(bundle["state_dict"],strict=True)
    torch.set_num_threads(1)
    stream=CausalRealAudioTransposeStream256(net,views=(-12,12))
    stems=[[] for _ in range(6)]
    mic=[]
    plus=[[] for _ in range(6)]
    minus=[[] for _ in range(6)]
    metadata=[]
    for i,start in enumerate(range(0,len(audio),HOP)):
        window=audio[start:start+HOP]
        if len(window)!=HOP:
            # Only final EOF may be zero-padded. Truncate such output to the
            # original length; no future data or next block is consulted.
            padded=np.zeros(HOP,np.float32)
            padded[:len(window)]=window
            window=padded
        result=stream.push(np.ascontiguousarray(window,np.float32),
                           start_sample=start)
        length=min(HOP,len(audio)-start)
        for channel in range(6):
            stems[channel].append(result.strings[channel,:length])
            if include_shift_audio:
                minus[channel].append(result.transposed_acoustic_estimates[0][channel,:length])
                plus[channel].append(result.transposed_acoustic_estimates[1][channel,:length])
        mic.append(result.mic_reconstruction[:length])
        strongest=np.argsort(result.note_probabilities)[-5:][::-1]
        metadata.append(dict(sample_end=int(min(result.sample,len(audio))),
                             top5_midi_notes=[int(k)+40 for k in strongest],
                             top5_probabilities=[float(result.note_probabilities[k]) for k in strongest]))
    destination.mkdir(parents=True)
    for channel in range(6):
        _save_wav(destination/f"estimated_string_{channel+1}.wav",np.concatenate(stems[channel]))
        if include_shift_audio:
            _save_wav(destination/f"estimated_learned_minus12_string_{channel+1}.wav",
                      np.concatenate(minus[channel]))
            _save_wav(destination/f"estimated_learned_plus12_string_{channel+1}.wav",
                      np.concatenate(plus[channel]))
    _save_wav(destination/"estimated_microphone.wav",np.concatenate(mic))
    report=dict(schema="S845_RECORDED_AUDIO_ESTIMATE_ONLY_V1",
                source_wav=str(real_mic_wav),source_sample_count=len(audio),
                source_rate=SR,hop=HOP,future_audio_samples=0,added_hold_samples=0,
                model_checkpoint=str(checkpoint),
                estimated_six_strings_are_not_ground_truth=True,
                estimated_shift_audio_not_proven_real_transposition=True,
                no_synthetic_training_audio_assumed_from_verified_checkpoint=True,
                exact_midi_note_f1_not_measured=True,
                official_note_pipeline_unchanged=True,
                note_probabilities_per_block=metadata)
    (destination/"audit_estimated_only.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print("S845_REAL_WAV_RECONSTRUCTION_ESTIMATES",dict(
        source_samples=len(audio),real_input=True,estimated_stems=6,
        transposed_audio_estimates=bool(include_shift_audio),
        no_midi=True,not_validated_as_true_strings=True),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wav",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--no-shift-audio",action="store_true")
    args=parser.parse_args()
    run(args.wav,args.model,args.out,include_shift_audio=not args.no_shift_audio)
