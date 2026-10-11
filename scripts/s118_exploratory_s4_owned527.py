"""S118 exploratory S4 owned 527D candidate from real S97 GOAT trajectories.

This is NOT the recovered historical feature builder. 111D prefix follows
S116, 416D suffix follows a reproducible, explicitly assumed 4-zone,
4-lag statistics contract derived from the S4 written protocol.
Never substitute this candidate for proven S4/S5 provenance or promote S80.
No MIDI/truth keys are read or accepted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SHIFTS = (1, 2, 4, 8)
BANNED = {'true_K', 'truth', 'target', 'label', 'labels', 'y', 'midi', 'annotation'}
INPUT_CONTROL = {'row_id', 'control111', 'freeze_K', 'geometry46'}
REQUIRED_AUDIO = {'row_id', 'sequence42x49'}
ZONE_NAMES = ('native', 'dilated2', 'before', 'after')

def check(condition, message):
    if not condition:
        raise ValueError(message)

def descriptors(magnitude: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """20D fixed, scale-preserving candidate summary; no pitch identity."""
    values = np.asarray(magnitude[selected], np.float64)
    if len(values) == 0:
        return np.zeros(20, np.float32)
    check(values.ndim == 2 and values.shape[1] == 49, 'pitch/frame contract')
    per_pitch = np.max(values, axis=0)
    peak = np.sort(per_pitch)[::-1][:8]
    total = float(per_pitch.sum())
    max_peak = float(peak[0])
    concentration = max_peak / (total + 1e-12)
    effective = (total ** 2) / (float(np.square(per_pitch).sum()) + 1e-12)
    counts = np.array([(per_pitch > max_peak * t).sum() if max_peak > 0 else 0
                       for t in (0.02, 0.05, 0.10, 0.20)], np.float64)
    time_flux = np.sum(values, axis=1)
    temporal = np.array([time_flux.max(), time_flux.mean(), time_flux.std(),
                         np.quantile(time_flux, .95),
                         time_flux.max() / (time_flux.sum() + 1e-12)], np.float64)
    out = np.r_[peak, total, concentration, effective, counts, temporal]
    check(out.shape == (20,) and np.isfinite(out).all(), 'bad 20D descriptors')
    return out.astype(np.float32)

def zone_masks(native: np.ndarray) -> list[np.ndarray]:
    check(native.shape == (42,), 'expected 42 native ownership frames')
    own = np.asarray(native, bool)
    extended = own.copy()
    for delta in (1, 2):
        extended[delta:] |= own[:-delta]
        extended[:-delta] |= own[delta:]
    ix = np.flatnonzero(own)
    before = np.zeros(42, bool)
    after = np.zeros(42, bool)
    if len(ix):
        before[:ix[0]] = True
        after[ix[-1]+1:] = True
    return [own, extended, before, after]

def extra416(sequence: np.ndarray, native: np.ndarray) -> np.ndarray:
    """4 zones * [20 activation + 4 shifts * (20 positive + 1 negative)] = 416.

    The 20D statistics and their ordering are an S118 HYPOTHESIS, not proof
    of original historical feature ordering or normalization.
    """
    check(sequence.shape == (42, 49), 'expected 42x49 harmonic signal')
    check(np.isfinite(sequence).all(), 'nonfinite waveform feature')
    check((sequence >= 0).all(), 'expected nonnegative log1p activation')
    result = []
    for zone in zone_masks(native):
        result.extend(descriptors(sequence, zone))
        for lag in SHIFTS:
            delta = np.zeros_like(sequence, dtype=np.float64)
            delta[lag:] = sequence[lag:] - sequence[:-lag]
            positive = np.maximum(delta, 0)
            negative = np.maximum(-delta, 0)
            result.extend(descriptors(positive, zone))
            result.append(float(negative[zone].sum()))
    out = np.asarray(result, np.float32)
    check(out.shape == (416,) and np.isfinite(out).all(), 'candidate 416D invalid')
    return out

def build(audio_path: Path, control_path: Path, out_dir: Path) -> dict:
    check(not out_dir.exists(), 'never overwrite external audit')
    with np.load(audio_path, allow_pickle=False) as a:
        check(REQUIRED_AUDIO.issubset(a.files), 'missing authentic acoustic evidence')
        check(not BANNED.intersection(a.files), 'labels forbidden in S118 audio')
        audio_keys = a['row_id'].copy()
        signals = a['sequence42x49'].copy()
    with np.load(control_path, allow_pickle=False) as c:
        check(set(c.files) == INPUT_CONTROL, 'S116 contract mismatch')
        keys = c['row_id'].copy()
        control = c['control111'].copy()
        geometry = c['geometry46'].copy()
    check(np.array_equal(audio_keys, keys), 'S97/S116 GOAT event mismatch')
    check(len(keys) == 57 and len(set(map(str,keys))) == 57,
          'GOAT event count/uniqueness drift')
    check(signals.shape == (57, 42, 49) and geometry.shape == (57, 46)
          and control.shape == (57, 111), 'signal/geometry dimensions invalid')
    check(np.all((geometry[:,:42] == 0) | (geometry[:,:42] == 1)),
          'ownership masks must be binary')
    suffix = np.stack([extra416(s, g[:42]) for s,g in zip(signals,geometry)])
    design = np.column_stack([control,suffix]).astype(np.float32)
    check(design.shape == (57,527) and np.isfinite(design).all(),
          '527D prototype matrix invalid')
    check(np.array_equal(design[:,:111],control),
          'control111 prefix changed')
    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir/'candidate_527.npz', row_id=keys,design527=design)
    report = {
       'status':'S118_REPRODUCIBLE_EXPLORATORY_OWNED527_CANDIDATE',
       'rows':57, 'feature_shape':[57,527], 'feature_breakdown':[111,416],
       'suffix_rule':'4 zones * (20 activation + 4 lags * (20 positive + 1 negative))',
       'shift_frames':list(SHIFTS), 'zones':list(ZONE_NAMES),
       'original_S4_owned_column_equivalence_proven':False,
       'original_S5_feature_equivalence_proven':False,
       'labels_or_midi_used':False,
       'independent_S80_accuracy':None, 'S80_promoted':False,
       'sha256':hashlib.sha256(design.tobytes()).hexdigest()}
    (out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

def selftest():
    zeros=np.zeros((42,49),np.float32)
    own=np.zeros(42,np.float32)
    own[10:15]=1
    v=extra416(zeros,own)
    check(v.shape==(416,) and np.all(v==0), 'zero-signal regression')
    s=np.arange(42*49,dtype=np.float32).reshape(42,49)/1000
    v=extra416(s,own)
    check(np.isfinite(v).all() and (v>=0).all(), 'finite/positive contract')
    reversed_pitch=extra416(s[:,::-1],own)
    check(np.allclose(v,reversed_pitch,atol=1e-5),
          'no absolute-pitch identity invariant broken')
    try:
        extra416(np.full((42,49),np.nan),own)
    except ValueError:
        pass
    else:
        raise ValueError('NaN reject invariant broken')
    check(sum(len(mask) for mask in zone_masks(own))==4*42, 'zone shapes')
    print('S118 selftest PASS: dimension, zero, finite, pitch-invariant, NaN gate')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audio',type=Path)
    p.add_argument('--control',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        if not all((args.audio,args.control,args.out)):
            p.error('requires audio/control/out')
        build(args.audio,args.control,args.out)
