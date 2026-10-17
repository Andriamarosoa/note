"""S643: can a learned head distinguish one guitar note from a true octave duet?

Corpus: REAL GuitarSet TRAIN isolated recordings, paired digitally to form
controlled single-note vs two-note octave stimuli. E2-alone is unavailable
in the original protocol, so test multiple available octave pairs instead.

The split is by WHOLE GUITARSET COMPOSITIONS, including BOTH members of
each stereo/source pair: no original recording from a validation group can
appear in the training group. The split is chosen only from metadata /
inventory counts; labels of the held-out pairs cannot influence split,
threshold, features or fitting. TEST and DEV remain CLOSED.

Important limits:
  - digitally constructed mixtures are NOT natural octave chords,
  - sources have reverb and sometimes other notes outside the onset window,
  - the classifier is a diagnostic, NOT an official NOTE pitch or K head,
  - no claim of strict microphone real-time speed or global NoteOn F1.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR
from scripts.s613_prepare_strict_live import labels,download,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s637_guitarset_three_view_groundtruth import ROOT,raw_guitar_audio
from scripts.s642_annotation_probe import eligible
from scripts.s642_real_octave_pair_comparison import (
    BASE_T,DELAY,DURATION,normalize_loudness,source_clip,
)

SEED=643
BANDS=range(40,65)   # low note; high note is low+12 MIDI
NFFT=(2048,4096,8192)
OFFSETS=(.10,.19)    # physically <= measured decision time
MAX_CLIPS_PER_PITCH_AND_TRACK=2
MAX_PAIRS_PER_PITCH_AND_SPLIT=14
MAX_PAIRS_PER_SPLIT=210
LOW_GAIN=.80
HIGH_GAINS=(.45,.75,1.05)
LIMIT_AUDIO_PEAK=.98


def extract_sources(specs,root,archive):
    sources=[]
    inventory=Counter()
    for spec in specs:
        notes=real_notes(labels(spec,root)["notes"],24.)
        selected={pitch:eligible(notes,pitch)[:MAX_CLIPS_PER_PITCH_AND_TRACK]
                  for pitch in range(40,77)}
        if not any(selected.values()):
            continue
        wav,sha=raw_guitar_audio(spec,archive)
        for pitch,times in selected.items():
            for onset in times:
                try:
                    clip=normalize_loudness(source_clip(wav,onset))
                except ValueError:
                    continue
                # Stop if large incomplete transients prevent alignment.
                if not np.isfinite(clip).all():
                    raise ValueError("Invalid original TRAIN GuitarSet source")
                sources.append(dict(
                    pitch=int(pitch),composition=spec["composition"],
                    name=spec["name"],onset=float(onset),
                    source_sha256=sha,wave=clip))
                inventory[pitch]+=1
    return sources,inventory


def candidate_pairs(sources,compositions,*,max_per_pitch=MAX_PAIRS_PER_PITCH_AND_SPLIT):
    """Deterministic diverse physical pairs from a fixed set of compositions."""
    allowed=set(compositions)
    by_pitch=defaultdict(list)
    for item in sources:
        if item["composition"] in allowed:
            by_pitch[item["pitch"]].append(item)
    pairs=[]
    for pitch in BANDS:
        low=by_pitch[pitch]
        high=by_pitch[pitch+12]
        if not low or not high:
            continue
        # Sorting by recording and onset avoids random sampling / cherry picking.
        possibilities=[(a,b) for a in low for b in high
                       if a["composition"]!=b["composition"]]
        possibilities.sort(key=lambda z:(
            z[0]["composition"],z[1]["composition"],
            z[0]["name"],z[0]["onset"],z[1]["name"],z[1]["onset"]))
        # Spread available anchors rather than overusing one low note.
        unique=[]
        occupied=set()
        for a,b in possibilities:
            mark=(a["name"],a["onset"],b["name"],b["onset"])
            if mark in occupied:continue
            occupied.add(mark)
            unique.append((pitch,a,b))
        if not unique:continue
        order=np.linspace(0,len(unique)-1,min(len(unique),max_per_pitch)).astype(int)
        pairs.extend(unique[int(j)] for j in order)
    pairs.sort(key=lambda z:(z[0],z[1]["composition"],z[2]["composition"],
                              z[1]["name"],z[2]["name"]))
    return pairs[:MAX_PAIRS_PER_SPLIT]


def select_composition_halves(sources):
    """Choose 4/5 composition partition from source metadata only.

    Both TRAIN folds must possess diverse octave pairs and no shared
    recording; pair labels/data are never inspected for partitioning.
    """
    compositions=sorted({s["composition"] for s in sources})
    if len(compositions)!=9:
        raise ValueError(f"Expected 9 original TRAIN compositions, got {compositions}")
    candidates=[]
    allset=set(compositions)
    for left in combinations(compositions,4):
        right=tuple(sorted(allset-set(left)))
        a=candidate_pairs(sources,left)
        b=candidate_pairs(sources,right)
        coverage_a=len({p for p,_,_ in a})
        coverage_b=len({p for p,_,_ in b})
        # First maximize the smallest composition-separated sample count,
        # then coverage, then total. No features or labels used.
        score=(min(len(a),len(b)),min(coverage_a,coverage_b),
               len(a)+len(b),coverage_a+coverage_b)
        candidates.append((score,left,right,a,b))
    best=max(candidates,key=lambda v:v[0])
    _,left,right,a,b=best
    if len(a)<12 or len(b)<12 or len({p for p,_,_ in a})<3 or len({p for p,_,_ in b})<3:
        raise ValueError("Not enough clean composition-separated octave pairs")
    assert set(left).isdisjoint(right)
    assert all(v["composition"] in left for _,low,high in a for v in (low,high))
    assert all(v["composition"] in right for _,low,high in b for v in (low,high))
    return dict(a=left,b=right,ab_pairs=a,ba_pairs=b,
                source_only_criterion=best[0])


def make_conditions(low,high,*,gain=.75,offset=0.):
    """Negative and positive preserve same low note; normalize RMS equally.

    In particular, the negative E2/E3 *harmonics already contain* the
    f_high bin; the task is to detect an independent fundamental above it.
    """
    x=low["wave"]
    y=high["wave"]
    assert x.shape==y.shape==(round(DURATION*SR),)
    companion=np.zeros_like(x)
    gap=round(offset*SR)
    if gap:
        companion[gap:]=y[:-gap]
    else:
        companion[:]=y
    combined=LOW_GAIN*x+float(gain)*companion
    # One final RMS normalization for both classes reduces level shortcuts.
    solo=normalize_loudness(LOW_GAIN*x)
    duo=normalize_loudness(combined)
    return solo,duo


def spectral_signature(audio,pitch,*,cue):
    """Feature for octave p+12 on right-aligned physical PCM, no future samples.

    Absolute f0_high is NOT sufficient: that bin is the second harmonic
    of f0_low. Check energy at harmonic positions, ratios between odd/even
    partials, spectral change after the onset, and neighbouring-bin leakage.
    """
    data=np.asarray(audio,np.float32)
    if len(data)!=round(DURATION*SR) or pitch<40 or pitch>64:
        raise ValueError("Expected fixed GuitarSet source and octave pitch")
    now=int(round(cue*SR))
    low_hz=440.*2.**((pitch-69)/12.)
    values=[float(pitch/127.)]
    for n in NFFT:
        if now<n:raise ValueError("No physical history for FFT")
        taper=np.hanning(n).astype(np.float32)
        # End-aligned FFT: all timestamps <= cue. Evaluate exactly
        # once per (window, cue), not at a future target.
        block=data[now-n:now]
        spectrum=(2./taper.sum())*np.abs(np.fft.rfft(block*taper))
        spectrum=np.asarray(spectrum,np.float32)
        bins=np.clip(np.rint(np.arange(1,9)*low_hz*n/SR).astype(int),1,n//2-2)
        harmonic=np.array([float(spectrum[b-1:b+2].max()) for b in bins])
        h=np.log1p(700.*harmonic)
        values.extend(map(float,h))  # 8
        values.extend((float(h[1]-h[0]),float(h[3]-h[1]),
                       float(h[5]-h[2]),float(h[1]-h[2]),
                       float(h[3]-h[4]),float(h[7]-h[5]),
                       float(h[1]+h[3]-h[0]-h[2]),float(h[1]+h[3]-h[2]-h[4]),
                       float(h[1]-max(h[0],h[2])),float(h[3]-max(h[2],h[4]))))
        # Normalized physically local upper-octave excess: amplitude
        # at upper f0 against ±1 semitone neighbor alternatives.
        bh=int(bins[1])
        if bh>3:
            neighbor=int(round(bh*2.**(1./12.)))
            neighbor=min(len(spectrum)-2,neighbor)
            values.append(float(np.log1p(700*spectrum[bh])-
                                np.log1p(700*max(spectrum[neighbor],spectrum[max(1,bh-3)]))))
        else:
            values.append(0.)
    a=np.asarray(values,np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite spectral feature")
    return a


def extracted_features(audio,pitch,*,onset=BASE_T):
    """Causal pair evidence following candidate onset; no signal beyond .84s."""
    early=spectral_signature(audio,pitch,cue=onset+OFFSETS[0])
    later=spectral_signature(audio,pitch,cue=onset+OFFSETS[1])
    return np.concatenate((early,later, later[1:]-early[1:])).astype(np.float32)


def make_dataset(pairs,*,gain=HIGH_GAINS[1],delay=0.):
    features=[];targets=[];groups=[];identity=[]
    for pitch,low,high in pairs:
        solo,duo=make_conditions(low,high,gain=gain,offset=delay)
        for sound,label in ((solo,0),(duo,1)):
            features.append(extracted_features(sound,pitch,onset=BASE_T+delay))
            targets.append(label)
            groups.append((low["composition"],high["composition"]))
            identity.append(dict(pitch=pitch,low_source=low["name"],
                                 high_source=high["name"],high_gain=gain,
                                 onset_delay=delay,class_label=label))
    return np.asarray(features,np.float32),np.asarray(targets,np.int32),groups,identity


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.2,class_weight="balanced",
                           max_iter=1500,random_state=SEED))


def evaluate(train,test):
    x,y=train[0],train[1]
    v,w=test[0],test[1]
    if not len(x) or set(np.unique(y))!={0,1} or set(np.unique(w))!={0,1}:
        raise ValueError("Octave classes missing")
    model=classifier()
    model.fit(x,y)
    p=model.predict_proba(v)[:,1]
    guess=p>=.5       # fixed, no held-out tuning
    c=confusion_matrix(w,guess,labels=[0,1])
    return dict(
        n=int(len(w)),duet_n=int(w.sum()),
        accuracy=float(accuracy_score(w,guess)),
        precision=float(precision_score(w,guess,zero_division=0)),
        recall=float(recall_score(w,guess,zero_division=0)),
        F1=float(f1_score(w,guess,zero_division=0)),
        TN=int(c[0,0]),FP=int(c[0,1]),FN=int(c[1,0]),TP=int(c[1,1]),
        mean_P_duet_given_alone=float(p[w==0].mean()),
        mean_P_duet_given_duet=float(p[w==1].mean()))


def run(out):
    if out.exists():raise FileExistsError("Immutable S643 evidence only")
    original=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train_specs=[s for s in original["sources"] if s["corpus"]=="GuitarSet"
                 and s["split"]=="train"]
    assert len(train_specs)==36 and len({s["composition"] for s in train_specs})==9
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                              ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=original["guitar_archives"][name]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio_zip:
            clips,inventory=extract_sources(train_specs,root,audio_zip)
        part=select_composition_halves(clips)
        a=part["ab_pairs"];b=part["ba_pairs"]
        print("S643 SOURCE INVENTORY",json.dumps(dict(
            recorded_isolated_clips=len(clips),octave_pitches_used={
                "foldA":dict(Counter(p for p,_,_ in a)),
                "foldB":dict(Counter(p for p,_,_ in b))},
            foldA_compositions=part["a"],foldB_compositions=part["b"],
            foldA_pairs=len(a),foldB_pairs=len(b))),flush=True)
        xa=make_dataset(a)
        xb=make_dataset(b)
        assert set(part["a"]).isdisjoint(part["b"])
        assert len(xa[0])==2*len(a) and len(xb[0])==2*len(b)
        forward=evaluate(xa,xb)
        backward=evaluate(xb,xa)
        print("S643 FOLD A TO B",json.dumps(forward),flush=True)
        print("S643 FOLD B TO A",json.dumps(backward),flush=True)
        # Out-of-distribution comparison: delayed attacks are tested,
        # never used to select the head or its threshold.
        delayed=evaluate(xa,make_dataset(b,delay=DELAY))
        print("S643 DELAYED 120MS",json.dumps(delayed),flush=True)
        combined={k:forward[k]+backward[k] for k in ("TP","FP","TN","FN")}
        precision=combined["TP"]/max(1,combined["TP"]+combined["FP"])
        recall=combined["TP"]/max(1,combined["TP"]+combined["FN"])
        combined.update(precision=precision,recall=recall,
                        F1=2*precision*recall/(precision+recall)
                        if precision+recall else 0)
        print("S643 OOF TWO-SIDED",json.dumps(combined),flush=True)
        report=dict(
            status="S643_REAL_GUITARSET_SINGLE_VS_OCTAVE_DUET_COMPOSITION_SPLIT",
            original_GuitarSet_microphone_wavs=True,
            source_waves_separately_recorded=True,
            manually_constructed_duets=True,
            S614_not_modified=True,
            no_TEST_or_DEV_audio_or_labels=True,
            all_annotations_from_GuitarSet_TRAIN=True,
            physical_input_history_only=True,
            onset_evidence_after_attack_ms=[int(v*1000) for v in OFFSETS],
            note_pair_interval_semitones=12,
            E2_single_clean_training_examples=inventory[40],
            training_and_holdout_compositions_disjoint=True,
            split_selection_only_from_metadata_counts=True,
            foldA_compositions=part["a"],foldB_compositions=part["b"],
            foldA_pairs=len(a),foldB_pairs=len(b),
            foldA_pitch_counts=dict(Counter(p for p,_,_ in a)),
            foldB_pitch_counts=dict(Counter(p for p,_,_ in b)),
            learning_head="StandardScaler+LogisticRegression(C=0.2)",
            decision_threshold=.5,
            octave_f0_not_evidence_alone=True,
            results=dict(A_train_B_test=forward,B_train_A_test=backward,
                         two_sided_disjoint_oof=combined,
                         A_train_B_delayed=delayed),
            examples=dict(foldB=xb[3],foldA=xa[3]),
            no_official_promotion=True,
            note="Diagnostic recognition of digitally mixed guitar octave pairs; NOT natural-chord F1; no independently validated microphone throughput.",
        )
        out.mkdir(parents=True)
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S643 FINAL",json.dumps(report["results"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
