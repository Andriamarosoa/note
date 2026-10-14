"""S439 -- prospective label-blind, K-locked joint pitch-set challenger.

Input: S437's SHA-frozen real guitar MIC note names, original MIC WAV audio.
This does NOT recompute Exact-K, read JAMS, train models, or alter S430.
Two fixed hypotheses (conservative/exploratory) must be frozen BEFORE S440
inspects the original GuitarSet annotations. They are challengers only.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import decode_pcm
from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    PRE, ON, LATE, harmonic_pitch_scores, note_name, sha,
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import need

FIRST = 40
LAST = 88
SR = 44100
PRESETS = {"conservative": 0.35, "exploratory": 0.70}
MAX_CANDIDATES = 14
ANCHOR_REWARD = 0.34
ACOUSTIC_REWARD = 0.48
BIRTH_REWARD = 0.20
DELTA_TO_INTERVENE = 0.04

def note_midi(value):
    for p in range(FIRST, LAST + 1):
        if note_name(p) == value:
            return p
    raise ValueError("S439 pitch outside original MIDI 40..88 range: " + str(value))

def candidate_pool(baseline, scores):
    """Never omit an incumbent note; other options come only from real audio."""
    cand = set(baseline)
    sorted_pitch = np.argsort(-scores, kind="stable") + FIRST
    cand.update(int(p) for p in sorted_pitch[:9])
    for p in baseline:
        for step in (-19, -12, -7, -2, -1, 1, 2, 7, 12, 19):
            q = p + step
            if FIRST <= q <= LAST:
                cand.add(q)
    # Bound combinatorial search, preserving all original K notes.
    rest = sorted((p for p in cand if p not in baseline),
                  key=lambda p: (-float(scores[p-FIRST]), p))
    return sorted(set(baseline) | set(rest[:max(0, MAX_CANDIDATES-len(baseline))]))

def physical_pitch_evidence(seq):
    """Audio-only evidence; no lyrics, pitch labels, chord truth, or teacher."""
    x = np.asarray(seq, np.float32)
    need(x.shape == (42,49) and np.isfinite(x).all()
         and np.min(x) >= 0, "S439 invalid genuine acoustic trajectory")
    a = np.expm1(x.astype(np.float64))
    on = np.mean(a[ON], axis=0)
    pre = np.mean(a[PRE], axis=0)
    # Contains real onset novelty, not just the strength of a spectral peak.
    birth = np.maximum(on-pre, 0.0)/(np.maximum(on, pre)+1e-6)
    raw = harmonic_pitch_scores(x[None])[0].astype(np.float64)
    acoustic = raw/(max(float(np.max(raw)), 1e-8))
    # Cross-note trajectory similarity indicates a possible harmonic alias,
    # but cannot by itself prove that two independent notes did not sound.
    novel = np.maximum(a-a[PRE].mean(axis=0)[None, :], 0.0)
    norm = np.linalg.norm(novel, axis=0)+1e-8
    corr = (novel.T @ novel)/(norm[:, None]*norm[None, :])
    return acoustic, birth, np.clip(corr, 0, 1)

def optimize_joint(baseline, seq, competition):
    """Compare whole K-note sets with a low-fundamental alias penalty."""
    k = len(baseline)
    if k == 0:
        return [], dict(candidates=0, objective_margin=0., intervened=False)
    acoustic, birth, corr = physical_pitch_evidence(seq)
    pool = candidate_pool(baseline, acoustic)
    value = {p: float(ACOUSTIC_REWARD*acoustic[p-FIRST]
                +BIRTH_REWARD*birth[p-FIRST]
                +ANCHOR_REWARD*(p in baseline)) for p in pool}

    def objective(combo):
        score = sum(value[p] for p in combo)
        for p, q in itertools.combinations(combo,2):
            low,high = min(p,q), max(p,q)
            interval = high-low
            if interval not in (12,19):
                continue
            # Only a strong co-moving attack is penalized. This cannot
            # claim physical causality: true simultaneous octaves may occur.
            similar = max(0., float(corr[low-FIRST,high-FIRST])-.68)/.32
            low_birth = float(birth[low-FIRST])
            high_birth = float(birth[high-FIRST])
            subordinate = max(0., min(1., low_birth-high_birth+.45))
            score -= competition*similar*subordinate
        return score

    old = tuple(sorted(baseline))
    original = objective(old)
    best = old
    best_score = original
    # No ground truth, note-name teacher or learned test-set threshold.
    for combo in itertools.combinations(pool, k):
        v = objective(combo)
        if v > best_score+1e-10 or (
            abs(v-best_score)<1e-10 and combo < best):
            best, best_score = combo, v
    margin = best_score-original
    chosen = list(best) if margin > DELTA_TO_INTERVENE else list(old)
    need(len(chosen)==k and len(set(chosen))==k,
         "S439 Exact-K pitch identity/cardinality violated")
    return chosen, dict(candidates=len(pool),
        objective_margin=round(float(margin), 6),
        intervened=chosen!=list(old))

def source_audio_events(mic_archive, rows):
    grouped = collections.defaultdict(list)
    for j,row in enumerate(rows):
        grouped[row["member"]].append((j,row))
    results=[None]*len(rows)
    with zipfile.ZipFile(mic_archive) as z:
        paths=collections.defaultdict(list)
        for path in z.namelist():
            stem=PurePosixPath(path).name
            if stem.endswith("_mic.wav"):
                paths[stem[:-8]+".jams"].append(path)
        for member,events in grouped.items():
            need(len(paths[member])==1,
                 "S439 cannot match unique frozen real MIC recording")
            audio=decode_pcm(z.read(paths[member][0]))
            maxlen=min(len(audio), 15*SR)
            # Exact same quantization as S437, no domain/substitute mismatch.
            pcm=np.clip(np.rint(audio[:maxlen]*32768),-32768,32767).astype("<i2")
            waveform=pcm.astype(np.float32)/32768.
            for j,row in events:
                start=int(row["start_sample"])
                end=int(row["end_sample"])
                need(0 <= start < maxlen and end+int(.25*SR)<maxlen,
                     "S439 onset/context violates original S437 audio clip")
                spec,t=_frames(waveform,start)
                _,activation=features_from_spectrogram(spec,t,return_states=True)
                need(activation.shape==(42,49),"S439 original S437 42x49 mismatch")
                results[j]=np.log1p(activation).astype(np.float32)
    need(all(x is not None for x in results),"S439 missing MIC audio trajectory")
    return results

def run(names,report,mic_archive,out):
    need(not out.exists(),"S439 immutable prospective output already exists")
    src=json.loads(report.read_text())
    need(src["status"]=="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES"
         and src["true_K_labels_and_JAMS_NEVER_loaded_for_sampling_or_inference"]
         and src["number_distinct_original_compositions"]==12
         and sha(names)==src["output_SHA256"],
         "S439 requires untouched S437 frozen note-name predictions")
    rows=[json.loads(line) for line in names.read_text().splitlines() if line.strip()]
    need(len(rows)==174 and len({row["original_row_id"] for row in rows})==len(rows),
         "S439 original S437 174 frozen rows altered")
    need(sha(mic_archive)==src["original_GuitarSet_MIC_audio_zip_SHA256"],
         "S439 real original MIC WAV input SHA mismatch")
    audio=source_audio_events(mic_archive,rows)
    out.mkdir(parents=True)
    outputs={}
    for mode,strength in PRESETS.items():
        results=[]
        changes=collections.Counter()
        for row,seq in zip(rows,audio):
            k=int(row["original_S80_K"])
            baseline=[note_midi(n) for n in row["notes"]]
            need(k==len(baseline) and len(set(baseline))==k
                 and 0 <= k <= 6, "S439 frozen original K invalid")
            proposed,diagnostic=optimize_joint(baseline,seq,strength)
            record=dict(row)
            record["notes"]=[note_name(p) for p in proposed]
            record["candidate_type"]="S439_"+mode
            record["unchanged_original_S80_K"]=k
            record["S437_baseline_notes"]=row["notes"]
            record["label_blind_joint_diagnostic"]=diagnostic
            results.append(record)
            if diagnostic["intervened"]:
                changes[k]+=1
        path=out/("S439_"+mode+"_FROZEN_REAL_MIC_NOTE_NAMES.jsonl")
        with path.open("w") as h:
            for item in results:
                h.write(json.dumps(item,sort_keys=True)+"\n")
        reread=[json.loads(x) for x in path.read_text().splitlines()]
        need(len(reread)==len(rows)
             and all(r["original_row_id"]==x["original_row_id"]
                     and r["original_S80_K"]==x["original_S80_K"]
                     and len(r["notes"])==int(x["original_S80_K"])
                     for r,x in zip(reread,rows)),
             "S439 frozen alternative changes Exact-K or row identity")
        outputs[mode]=dict(path=path.name,sha256=sha(path),
                           proposed_changes_by_predicted_K={str(k):int(v) for k,v in changes.items()},
                           proposed_changes=int(sum(changes.values())))
    result=dict(status="S439_K_LOCKED_LABEL_BLIND_REAL_MIC_JOINT_NOTE_SET_CHALLENGERS",
        n_events=len(rows),distinct_compositions=12,
        original_K_source="S437_original_S80_frozen_predicted_K",
        S80_K_regressions=0,all_Original_S80_K_unchanged=True,
        no_JAMS_annotations_opened=True,no_truth_or_teacher_used=True,
        source_is_original_MIC_WAV_not_synthetic=True,
        same_S437_Mic_SHA256=True,
        original_S430_S428_models_and_baseline_untouched=True,
        fixed_weights_before_S440_labels=True,
        fixed_preset_coefficients=PRESETS,
        not_validated_and_not_promoted=True,
        genuine_audited_scores_not_yet_known=True,
        original_frozen_names_SHA256=sha(names),audio_SHA256=sha(mic_archive),
        candidates=outputs)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S439 FROZEN",json.dumps(outputs),flush=True)
    print("S439 PASS zero K change; challenger note names frozen BEFORE labels",flush=True)

def selftest():
    samples=np.zeros((42,49),np.float32)
    # Synthetic signal is ONLY a decoder unit test; real test uses MIC WAV.
    samples[ON,12]=np.log1p(8.)
    samples[ON,24]=np.log1p(4.)
    for k in range(7):
        old=list(range(FIRST+1,FIRST+k+1))
        newer,info=optimize_joint(old,samples,0.70)
        assert len(newer)==k and len(set(newer))==k
        assert info["intervened"]==(newer!=sorted(old))
    assert note_midi("E3")==52
    assert candidate_pool([52,64],np.arange(49,dtype=float))[:]
    print("S439 SELFTEST PASS label-blind joint K0..K6, note E3, strict K lock")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--names",type=Path)
    parser.add_argument("--s437-report",type=Path)
    parser.add_argument("--mic-zip",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.self_test:
        selftest()
    else:
        for flag in ("names","s437_report","mic_zip","out"):
            need(getattr(args,flag) is not None,"S439 missing "+flag)
        run(args.names,args.s437_report,args.mic_zip,args.out)
