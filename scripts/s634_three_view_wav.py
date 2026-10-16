"""S634 off-line three-window diagnostic on real WAV; DOES NOT output MIDI.

Report distinct candidate note attacks from frozen NOTE at 1/2, 1, 2x
frequency. Rescued pitches are remapped into original MIDI coordinates.
No annotation means these are proposals, NOT verified correct notes.
The original K prediction remains the only cardinality source.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP,SR,load_heads
from causal_note.three_view_pitch import ThreeViewPitchProbe,EXTENDED_HISTORY
from scripts.live_strict_notes import DEFAULT_MODEL
from scripts.s613_prepare_strict_live import digest


def load_audio(path,seconds):
    rate,pcm=wavfile.read(path)
    if pcm.dtype==np.int16:
        audio=pcm.astype(np.float32)/32768
    elif pcm.dtype==np.int32:
        audio=pcm.astype(np.float32)/2147483648
    elif np.issubdtype(pcm.dtype,np.floating):
        audio=pcm.astype(np.float32)
    else:
        raise ValueError("Only 16/32-bit signed PCM or floating-point WAV supported")
    if audio.ndim==2:audio=audio.mean(1)
    elif audio.ndim!=1:raise ValueError("Mono or stereo WAV only")
    if rate!=SR:
        g=np.gcd(rate,SR)
        audio=resample_poly(audio,SR//g,rate//g).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite original audio")
    return audio[:int(seconds*SR)]


def diagnostic(audio,heads,*,confidence=.6):
    if not 0<confidence<1:
        raise ValueError("Confidence must be within (0,1)")
    probe=ThreeViewPitchProbe(heads)
    previous={name:np.zeros(49,bool) for name in ("original","bass_x2","high_half")}
    counts=Counter();rows=[]
    start=time.perf_counter()
    for at in range(0,len(audio),512):
        for view in probe.push(audio[at:at+512],start_sample=at):
            # Only original pitch bins that can be mapped without extrapolation.
            views={
                "original":(view["native"],np.ones(49,bool)),
                "bass_x2":(view["bass_rescue"],np.arange(40,89)<=63),
                "high_half":(view["high_rescue"],np.arange(40,89)>=76),
            }
            for name,(p,valid_pitch) in views.items():
                active=(p[:,0]>=confidence)&(p[:,1]>=.5)&valid_pitch
                if name!="original" and not view["valid_rescue"]:
                    active[:]=False
                newcomers=np.flatnonzero(active & ~previous[name])
                for ix in newcomers:
                    rows.append(dict(
                        view=name,
                        detected_pitch=int(ix+40),
                        time_s=round(view["decision_sample"]/SR,5),
                        attack_confidence=round(float(p[ix,0]),4),
                        sustain_confidence=round(float(p[ix,1]),4),
                        native_active_K=int(np.argmax(view["native_k"])),
                    ))
                counts[name+"_candidates"]+=len(newcomers)
                previous[name]=active.copy()
    elapsed=time.perf_counter()-start
    report=dict(
        kind="S634_three_view_shadow_candidates_only",
        sound_realtime_notes_unchanged=True, pitch_remapped=True,
        frequency_factors=dict(bass_x2=2,original=1,high_half=.5),
        pitch_transposition_semitones=dict(bass_x2=12,original=0,high_half=-12),
        all_three_native_K_source="original", native_checkpoint_frozen=True,
        physical_future_lookahead_ms=0,
        maximum_history_ms=round(1000*EXTENDED_HISTORY/SR,2),
        warn="Candidate crossings are not accuracy, recall or F1 measurements",
        original_samples=len(audio),frames_analyzed=probe.records,
        runtime_seconds=round(elapsed,3),
        model_audio_seconds=round(len(audio)/SR,3),
        worker_over_audio_ratio=round(elapsed/max(len(audio)/SR,.001),3),
        candidate_counts=dict(counts),candidates=rows)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wav",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--seconds",type=float,default=8.)
    ap.add_argument("--confidence",type=float,default=.6)
    args=ap.parse_args()
    if not 1.<=args.seconds<=120.:
        ap.error("--seconds must be in 1..120")
    if args.out.exists():ap.error("Refusing to overwrite previous evidence")
    model_sha=json.loads(DEFAULT_MODEL.with_name("provenance.json").read_text())["model_sha256"]
    if digest(DEFAULT_MODEL.read_bytes())!=model_sha:
        raise ValueError("Frozen reference NOTE checkpoint changed")
    with threadpool_limits(limits=1):
        heads=load_heads(DEFAULT_MODEL,expected_sha256=model_sha)
        report=diagnostic(load_audio(args.wav,args.seconds),heads,confidence=args.confidence)
    report["reference_sha256"]=model_sha
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in
         ("kind","frames_analyzed","candidate_counts","worker_over_audio_ratio","warn")},indent=2))


if __name__=="__main__":
    main()
