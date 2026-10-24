"""One AUDIO REFERENCE built entirely from physically recorded GuitarSet notes.

This is deliberately NOT called a demo of faithful AI guitar transposition.
It plays an authentic captured E2 and authentic captured E3, each stored
verbatim in the verified original GuitarSet six-channel hex source. The AI
will separately learn which target note to select from microphone audio.
No oscillator/synth, vocoder, FFT pitch shifting, tempo change, or synthetic
musical target note is generated.

Unlike S845's false whine, this reference refuses near-silence and output
dominated by very high-frequency noise. A clean reference DOES NOT certify
AI note-selection accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from scripts.s847_train_real_recorded_transposer256 import read_recorded_bank

SR=44100
HOP=256


def true_capture_profile(clip):
    if len(clip)<HOP*16:
        raise ValueError("need >=16 captured real waveform hops")
    frequencies,power=welch(np.asarray(clip,np.float64),fs=SR,
                            nperseg=min(len(clip),8192))
    total=float(np.sum(power))
    if total < 1e-14:raise ValueError("recorded target essentially silent")
    high_fraction=float(np.sum(power[frequencies>10000])/total)
    rms=float(np.sqrt(np.mean(np.asarray(clip,np.float64)**2)))
    median=float(frequencies[np.searchsorted(np.cumsum(power),total*.5)])
    return dict(rms=rms,energy_above_10khz=high_fraction,
                spectral_median_hz=median,samples=len(clip))


def create_recorded_reference(real_bank_dir,out):
    out=Path(out)
    if out.exists():raise FileExistsError("original evidence immutable")
    bank,meta=read_recorded_bank(real_bank_dir)
    clips=[];diagnostic={}
    for note in (40,52):
        if note not in bank:
            raise ValueError("authentic GuitarSet TRAIN E2 or E3 clip missing")
        clip=bank[note]
        profile=true_capture_profile(clip)
        if profile["rms"]<.0002 or profile["energy_above_10khz"]>.45:
            raise ValueError("original bank selected inaudible or high-frequency corrupted sample")
        diagnostic[str(note)]=profile
        clips.append(clip.copy())
    # Original clips retain their physical recording amplitude and pitch.
    # The inserted silence is an A/B separator, not a generated musical note.
    pause=np.zeros(11025,np.float32)
    wave=np.concatenate((clips[0],pause,clips[1]))
    if np.max(np.abs(wave))>1.:raise ValueError("original physical capture clips out of range")
    out.parent.mkdir(parents=True,exist_ok=True)
    wavfile.write(out,SR,(np.clip(wave,-1.,1.-1/32768)*32768).astype(np.int16))
    report=dict(
        schema="S847_AUTHENTIC_RECORDED_GUITAR_E2_VS_E3_REFERENCE_NOT_AI_TRANSPOSE",
        real_capture=True,
        no_synthesized_note_waveform=True,
        no_DSP_pitch_shift=True,
        no_artificial_frequency_change=True,
        no_guarantee_of_transposed_input_performance=True,
        sound_part_1="real GuitarSet note E2 from TRAIN guitarist 04",
        sound_part_2="real GuitarSet note E3 from TRAIN guitarist 04",
        source_provenance=meta["source_checksums"],
        measured=diagnostic,
        measured_model_GuitarSet_F1_NOT_IMPLIED=True)
    out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print("S847_AUDITABLE_PHYSICALLY_RECORDED_GUITAR_AUDIO",json.dumps({
        "recorded_original_notes":(40,52),"profiles":diagnostic,
        "no_AI_transposition_claim":True},sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    create_recorded_reference(args.bank,args.out)
