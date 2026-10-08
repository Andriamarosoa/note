"""Harmonic-source trajectory probe for V27.3; no fitted note identities.

Fixed nonnegative harmonic dictionaries track candidate f0 energies over 240 ms
and estimate birth/retention/damping. Identifiable sources are NOT guaranteed:
nearby f0, octave aliases and overlapping partials make this a proxy only.
Neither label-derived frequencies nor true-K enters feature extraction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_flow import SAMPLE_RATE, PRE_MS, POST_MS, _safe_segment
from scripts.yourmt3_exactk_common import FOLDS, digest, require

FFT = 4096
HOP = 256
PITCHES = np.arange(40, 89, dtype=int)
PARTIALS = 6
EPS = 1e-10
EXPERIMENT = "v273_harmonic_trajectory_extract"
GROUPS = ("spectral", "source", "birth", "persistence", "damping", "coherence")

def _template_bank(detuned=False):
    freq = np.fft.rfftfreq(FFT, 1.0 / SAMPLE_RATE)
    T = np.zeros((len(PITCHES), len(freq)), np.float64)
    for i, pitch in enumerate(PITCHES):
        f0 = 440.0 * 2 ** ((int(pitch) - 69) / 12.0)
        for h in range(1, PARTIALS + 1):
            center = f0 * h
            if detuned and h > 1:
                center *= (1.06 if h % 2 == 0 else 0.94)
            if center >= freq[-1] - 100:
                continue
            sigma = max(1.15 * SAMPLE_RATE / FFT, 0.006 * center)
            weight = h ** -1.25
            T[i] += weight * np.exp(-0.5 * ((freq - center) / sigma) ** 2)
        T[i] /= max(float(np.linalg.norm(T[i])), EPS)
    return T

HARMONIC_BANK = _template_bank()
DETUNED_BANK = _template_bank(detuned=True)

def _frames(samples, start):
    pre = int(round(PRE_MS * SAMPLE_RATE / 1000.0))
    post = int(round(POST_MS * SAMPLE_RATE / 1000.0))
    segment = _safe_segment(samples, int(start), pre, post)
    padded = np.pad(segment, (FFT // 2, FFT // 2))
    centers = np.arange(0, len(segment), HOP, dtype=int)
    window = np.hanning(FFT)
    framed = np.stack([padded[s:s + FFT] * window for s in centers])
    spec = np.abs(np.fft.rfft(framed, axis=1))
    t = centers * 1000.0 / SAMPLE_RATE - PRE_MS
    require(np.isfinite(spec).all(), "nonfinite spectrogram")
    return spec, t

def _states(spec, bank):
    """Nonnegative fixed-template multiplicative reconstruction (not transcription)."""
    q = spec @ bank.T
    gram = bank @ bank.T
    A = np.maximum(q, EPS)
    penalty = float(max(0.015 * np.max(q), EPS))
    for _ in range(24):
        A *= np.maximum(q - penalty, 0.0) / (A @ gram + EPS)
        A = np.maximum(A, 0.0)
    require(np.isfinite(A).all() and np.min(A) >= 0.0, "nonfinite activation")
    A = uniform_filter1d(A, size=3, axis=0, mode="nearest")
    return A

def _moment(out, prefix, v):
    a = np.asarray(v, float).reshape(-1)
    require(len(a) > 0 and np.isfinite(a).all(), "bad moment " + prefix)
    out[prefix + "_mean"] = float(np.mean(a))
    out[prefix + "_std"] = float(np.std(a))
    out[prefix + "_max"] = float(np.max(a))

def features_from_spectrogram(spec, times, bank=None):
    bank = HARMONIC_BANK if bank is None else bank
    spec = np.asarray(spec, float)
    times = np.asarray(times, float)
    require(spec.shape == (len(times), FFT // 2 + 1), "spectrum mismatch")
    A = _states(spec, bank)
    # Entire track scaling is label-independent and shared by all families.
    A /= max(float(np.median(np.sum(A, axis=1))), EPS)
    pre = (times >= -65) & (times < -18)
    on = (times >= -12) & (times <= 42)
    late = (times > 70) & (times < 140)
    require(pre.sum() >= 5 and on.sum() >= 5 and late.sum() >= 5, "phase coverage")
    out = {}
    norm = A / (np.sum(A, axis=1, keepdims=True) + EPS)
    for name, sel in (("pre", pre), ("on", on), ("late", late)):
        mass = np.sum(A[sel], axis=1)
        entropy = -np.sum(norm[sel] * np.log(norm[sel] + EPS), axis=1) / np.log(A.shape[1])
        out[f"spectral__mass_{name}"] = float(np.mean(mass))
        out[f"spectral__entropy_{name}"] = float(np.mean(entropy))
        out[f"spectral__active_{name}"] = float(np.mean(np.sum(norm[sel] > 0.05, axis=1)))
        out[f"spectral__dominant_{name}"] = float(np.mean(np.max(norm[sel], axis=1)))
    out["spectral__late_pre_ratio"] = float(np.log((np.mean(A[late].sum(axis=1)) + EPS) /
                                                 (np.mean(A[pre].sum(axis=1)) + EPS)))
    out["spectral__on_pre_ratio"] = float(np.log((np.mean(A[on].sum(axis=1)) + EPS) /
                                               (np.mean(A[pre].sum(axis=1)) + EPS)))

    before = np.mean(A[pre], axis=0)
    onset = np.mean(A[on], axis=0)
    after = np.mean(A[late], axis=0)
    before_rel = before / (np.sum(before) + EPS)
    on_rel = onset / (np.sum(onset) + EPS)
    after_rel = after / (np.sum(after) + EPS)
    persist = np.minimum(before, after) / (before + EPS)
    gain = np.maximum(onset - before, 0.)
    decay = np.maximum(before - after, 0.)
    # Do not silently count individual pitch templates as independent notes.
    out["source__stable_presence_fraction"] = float(np.sum(np.minimum(before_rel, after_rel)))
    out["source__novelty_l1"] = float(np.sum(np.abs(on_rel - before_rel)))
    out["source__continuation_l1"] = float(np.sum(np.abs(after_rel - on_rel)))
    out["source__effective_onset_sources"] = float(1.0 / (np.sum(on_rel ** 2) + EPS))
    out["source__effective_late_sources"] = float(1.0 / (np.sum(after_rel ** 2) + EPS))
    out["source__onset_below_pre_fraction"] = float(np.mean(onset < before))
    out["source__post_below_pre_fraction"] = float(np.mean(after < before))
    out["source__spectral_shift_cents"] = float(
        np.sum((on_rel - before_rel) * PITCHES) * 100.
    )

    delta = np.diff(A, axis=0)
    ft = (times[:-1] + times[1:]) / 2.
    ev = (ft >= -12.) & (ft <= 120.)
    de = delta[ev]
    t_ev = ft[ev]
    birth = np.maximum(de, 0.)
    sink = np.maximum(-de, 0.)
    birth_mass = np.sum(birth, axis=0)
    sink_mass = np.sum(sink, axis=0)
    out["birth__novel_mass"] = float(np.sum(gain))
    out["birth__positive_flow"] = float(birth.sum())
    out["birth__positive_fraction"] = float(birth.sum() / (np.abs(de).sum() + EPS))
    out["birth__novel_track_fraction"] = float(
        np.sum((before < 0.35 * np.maximum(onset, after)) *
               (np.maximum(onset, after) > 0.03 * np.max(np.maximum(onset, after)))) / len(before)
    )
    out["birth__positive_track_concentration"] = float(
        np.sum((birth_mass / (birth_mass.sum() + EPS)) ** 2)
    )
    _moment(out, "birth__frame", birth.sum(axis=1))

    out["damping__lost_mass"] = float(np.sum(decay))
    out["damping__negative_flow"] = float(sink.sum())
    out["damping__sink_fraction"] = float(sink.sum() / (np.abs(de).sum() + EPS))
    out["damping__vanishing_track_fraction"] = float(
        np.mean((after < 0.35 * np.maximum(before, onset)) &
                (np.maximum(before, onset) > 0.03 * np.max(np.maximum(before, onset))))
    )
    out["damping__sink_track_concentration"] = float(
        np.sum((sink_mass / (sink_mass.sum() + EPS)) ** 2)
    )
    _moment(out, "damping__frame", sink.sum(axis=1))

    out["persistence__pre_post_retention"] = float(
        np.sum(np.minimum(before, after)) / (np.sum(before) + EPS)
    )
    out["persistence__new_source_retention"] = float(
        np.sum(gain * np.minimum(1.0, after / (onset + EPS))) / (np.sum(gain) + EPS)
    )
    out["persistence__mean_survival"] = float(np.mean(np.minimum(persist, 2.0)))
    out["persistence__stable_track_fraction"] = float(np.mean(
        (before_rel > 0.03) & (after_rel > 0.03)
    ))
    for lag in (2, 5, 10):
        aa = A[:-lag]
        bb = A[lag:]
        out[f"persistence__retention_{lag}"] = float(
            np.sum(np.minimum(aa, bb)) / (np.sum(aa) + EPS)
        )
        out[f"coherence__temporal_cos_{lag}"] = float(
            np.mean(np.sum(aa * bb, axis=1) /
                    ((np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)) + EPS))
        )
    both = np.sum((birth > 0.) & (sink > 0.), axis=1)
    out["coherence__simultaneous_birth_death"] = float(np.mean(both))
    out["coherence__source_occupancy_var"] = float(np.std(np.sum(norm > 0.05, axis=1)))
    # A trajectory must be stable in absolute note-pitch ordering.
    pitch_prob = A / (np.sum(A, axis=1, keepdims=True) + EPS)
    moment = pitch_prob @ PITCHES
    out["coherence__pitch_centroid_drift"] = float(np.mean(np.abs(np.diff(moment))))
    out["coherence__pitch_centroid_late_pre"] = float(
        np.mean(moment[late]) - np.mean(moment[pre])
    )
    for mname,m in (("early",(t_ev >= -12)&(t_ev < 35)),
                    ("mid",(t_ev >= 35)&(t_ev < 75)),
                    ("late",(t_ev >= 75)&(t_ev <= 120))):
        require(m.any(), "no event phase")
        out[f"birth__{mname}_mass_fraction"] = float(birth[m].sum() / (birth.sum() + EPS))
        out[f"damping__{mname}_mass_fraction"] = float(sink[m].sum() / (sink.sum() + EPS))
    require(all(np.isfinite(x) for x in out.values()), "nonfinite harmonic feature")
    return out

def trajectory_features(samples, start, perturb=None, seed=0):
    spec, t = _frames(samples, start)
    if perturb == "scramble-time":
        idx = np.flatnonzero((t >= -12) & (t <= 120))
        spec = spec.copy()
        spec[idx] = spec[np.random.default_rng(seed).permutation(idx)]
    elif perturb not in (None,"detuned"):
        raise ValueError("unknown perturbation")
    return features_from_spectrogram(spec,t,DETUNED_BANK if perturb=="detuned" else None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--fold",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(a.fold in FOLDS and not a.output.exists(),"fold or existing output")
    manifest=json.loads((a.cohort/"manifest.json").read_text())
    require(manifest.get("status")=="verified" and manifest.get("fold_3_evaluated")==False
            and manifest.get("player_05_evaluated")==False,"bad cohort")
    path=a.cohort/f"fold-{a.fold}.npz"
    require(digest(path)==manifest["by_fold"][str(a.fold)]["cohort_sha256"],"cohort SHA")
    with np.load(path,allow_pickle=False) as z:
        ids=z["global_index"].astype(np.int64)
        member=z["member"].astype(str)
        starts=z["starts"].astype(np.int64)
        truth=z["k"].astype(int)
        base=z["baseline"].astype(int)
    from scripts.extract_v273_energy_transport_multik import PER_K, CLASSES
    selected=[]
    for k in CLASSES:
        pool=np.flatnonzero(truth==k)
        require(len(pool)>=PER_K,"insufficient class")
        selected += np.random.default_rng(20261008+int(a.fold)*7+k).permutation(pool)[:PER_K].tolist()
    selected.sort(key=lambda i:int(ids[i]))
    require(len(selected)==450,"selected count")
    wanted={member[i] for i in selected}
    require(all(x[:2] in {"00","01","02","03","04"} for x in wanted),"excluded player")
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"missing audio")
    decoded={}
    for name,track in tracks.items():
        wave=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(int(wave.sample_rate)==SAMPLE_RATE,"sample rate")
        decoded[name]=np.asarray(wave.samples,np.float64)/32768.
    output=[]
    for i in selected:
        ident=int(ids[i]); x=decoded[member[i]]; s=int(starts[i])
        output.append(dict(global_index=ident,member=member[i],fold=a.fold,
                           true_k=int(truth[i]),base_pred=int(base[i]),
                           features=trajectory_features(x,s),
                           scrambled=trajectory_features(x,s,"scramble-time",seed=ident+27084),
                           detuned=trajectory_features(x,s,"detuned")))
    keys=sorted(output[0]["features"])
    require(all(sorted(r[z])==keys for r in output for z in ("features","scrambled","detuned")),"schema drift")
    a.output.mkdir(parents=True)
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in output))
    report=dict(status="completed",experiment=EXPERIMENT,fold=a.fold,
                rows=450,per_class=150,features=len(keys),
                groups={g:sum(n.startswith(g+"__") for n in keys) for g in GROUPS},
                recordings=len(wanted),fold3_used=False,player05_used=False,
                labels_used_during_feature_extraction=False,
                vocabulary="MIDI pitches 40..88; six harmonics; nonnegative fixed template activation",
                limits="coherent harmonic spectral proxy; cannot identify independent notes from overlapping harmonics",
                model_unchanged=True,automatic_promotion=False)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":main()
