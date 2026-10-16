"""S631 — train-only REAL PCM note overlay, emphasizing masked high-pitch attacks.

A/B: identical frozen S614 front end, S627 architecture/seed/epochs/DEV;
control uses original 35 training recordings, treatment adds mixed train-only
clips. Every mix is made from two distinct TRAIN compositions; annotations
are time-translated and combined. 17 historical DEV sources are never mixed.
No S80/strict-LIVE weights, thresholds, or production code are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from causal_note.strict_live import SR, Selection, CausalSelection, feature_track, load_heads
from causal_note.s627_joint_heads import Fusion, frames, fuse
from scripts.s613_prepare_strict_live import download, digest, labels, urmp_blob
from scripts.s613_train_strict_live import decode, event_score, real_notes, summary
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s627_train_joint_eventK_pitch import (
    ROOT, MODEL, CONTROL, choose, train_network, export_forward, scores_for,
)

CLIP_SECONDS = 3.5
HIGH_MIDI = 76  # E5+; E4..D#5 (64..75) is also reported separately.
SEED = 631
POLICY = Fusion(pitch_weight=.4)
OFFSETS = (.020, .100, .230)
GAINS = (.22, .36, .50, .64)


def load_pcm(spec, root, archive=None):
    """Original TRAIN waveform, never a pitch-shifted label-only approximation."""
    if spec["split"] != "train" or spec["corpus"] not in ("URMP", "GuitarSet"):
        raise ValueError("S631 can only load train audio")
    if spec["corpus"] == "URMP":
        blob = urmp_blob(spec["audio"], root)
    else:
        if archive is None:
            raise ValueError("Original GuitarSet audio ZIP required")
        blob = archive.read(spec["audio"])
    rate, pcm = wavfile.read(io.BytesIO(blob))
    if pcm.dtype not in (np.int16, np.int32):
        raise ValueError("Only original signed PCM16/32 is accepted")
    result = pcm.astype(np.float32) / (32768. if pcm.dtype == np.int16 else 2147483648.)
    if result.ndim == 2:
        result = result.mean(axis=1)
    if rate != SR:
        divisor = np.gcd(rate, SR)
        result = resample_poly(result, SR // divisor, rate // divisor).astype(np.float32)
    return result[:int(spec["seconds"] * SR)].copy()


def candidates(notes, *, target=False, duration=24.):
    """Place the overlay onset after 0.6 s of causal audio."""
    if target:
        return [n for n in notes if HIGH_MIDI <= n["pitch"] <= 88
                and .65 <= n["time"] <= duration - (CLIP_SECONDS - .6)
                and n["duration"] > .05]
    return [n for n in notes if 40 <= n["pitch"] <= 88
            and n["duration"] >= .38 and .65 <= n["time"]
            and n["time"] <= duration - (CLIP_SECONDS - .6)]


def choose_pairs(catalog, count, rng):
    """High note begins 20/100/230 ms AFTER a different sustained pitch.

    Choose only train compositions, and enforce real overlap in annotations.
    No dev/test identity or label is read by pair selection.
    """
    high = [(s, n) for s in catalog for n in candidates(s["notes"], target=True, duration=s["seconds"])]
    bg = [(s, n) for s in catalog for n in candidates(s["notes"], duration=s["seconds"])]
    if not high or not bg:
        raise ValueError("No high-pitch/sustain TRAIN anchors; cannot claim augmentation")
    result = []
    for i in range(count):
        delta = OFFSETS[i % len(OFFSETS)]
        gain = GAINS[(i // len(OFFSETS)) % len(GAINS)]
        for attempt in range(400):
            second, target = high[int(rng.integers(len(high)))]
            pool = [(first, old) for first, old in bg
                    if first["composition"] != second["composition"] or first["corpus"] != second["corpus"]
                    if old["pitch"] != target["pitch"] and old["duration"] > delta + .080]
            if not pool:
                raise ValueError("Not enough distinct composition train backgrounds")
            first, old = pool[int(rng.integers(len(pool)))]
            # Prefer acoustic ambiguities but also include arbitrary intervals.
            interval = abs(old["pitch"] - target["pitch"])
            if i % 3 == 0 and interval not in (1, 2, 7, 12, 19, 24):
                continue
            # Avoid an already held identical pitch at the target instant.
            start_first = float(old["time"]) - .6
            start_second = float(target["time"]) - (.6 + delta)
            if start_first < 0 or start_second < 0:
                continue
            if start_first + CLIP_SECONDS > first["seconds"] + .005:
                continue
            if start_second + CLIP_SECONDS > second["seconds"] + .005:
                continue
            if any(n["pitch"] == target["pitch"] and n["time"] - start_first < .6 + delta
                   and n["time"] - start_first + n["duration"] > .6 + delta
                   for n in first["notes"]):
                continue
            result.append(dict(first=first, second=second, anchor=old, high=target,
                               start_first=start_first, start_second=start_second,
                               gap_s=delta, secondary_gain=gain, interval=interval))
            break
        else:
            raise ValueError("Could not find eligible high/sustained TRAIN pair")
    return result


def cut_audio(audio, start, samples):
    i = round(start * SR)
    if i < 0 or i + samples > len(audio):
        raise ValueError("Mixing requires complete real samples, no synthetic padding")
    return audio[i:i+samples].copy()


def shifted_labels(notes, start, seconds, prefix):
    result = []
    for note in notes:
        t = float(note["time"]) - start
        if .12 <= t < seconds - .04:
            duration = min(float(note["duration"]), seconds-t)
            if duration > .010:
                result.append(dict(time=t, duration=duration,
                                   pitch=int(note["pitch"]), track=prefix))
    return result


def mix_pair(pair, first_audio, second_audio):
    """Actual waveform sum, not feature concatenation or label-only union."""
    n = int(round(CLIP_SECONDS * SR))
    a = cut_audio(first_audio, pair["start_first"], n)
    b = cut_audio(second_audio, pair["start_second"], n)
    audio = .65*a + float(pair["secondary_gain"])*b
    # Smooth artificial clip starts so their discontinuities are not taught as attacks.
    edge = max(1,round(.012*SR))
    envelope = np.linspace(0,1,edge,dtype=np.float32)
    audio[:edge] *= envelope
    audio[-edge:] *= envelope[::-1]
    peak = float(np.max(np.abs(audio)))
    if peak > .98:
        audio *= (.98 / peak)
    audio = np.asarray(audio, np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Invalid audio mix")
    note_rows = shifted_labels(pair["first"]["notes"],pair["start_first"],
                               CLIP_SECONDS,"background")
    note_rows += shifted_labels(pair["second"]["notes"],pair["start_second"],
                                CLIP_SECONDS,"high_overlay")
    truth = real_notes(note_rows,CLIP_SECONDS)
    high_at = .6 + pair["gap_s"]
    if not any(n["pitch"] == pair["high"]["pitch"] and abs(n["time"]-high_at) <= 1/SR
               for n in truth):
        raise AssertionError("High NOTE onset target was lost")
    if not any(n["pitch"] != pair["high"]["pitch"] and n["time"] < high_at
               and n["time"]+n["duration"] > high_at+.02 for n in note_rows
               if n["track"] == "background"):
        raise AssertionError("No actual annotated sustain beneath the high attack")
    return audio,truth


def category(note, all_notes, i):
    pitch = note["pitch"]
    reg = "high_E5_plus" if pitch >= 76 else "mid_E4_to_Dsharp5" if pitch >= 64 else "low"
    start = note["time"]
    has_other = any(j != i and n["pitch"] != pitch
        and n["time"] < start - .02 and n["time"] + n["duration"] > start+.02
        for j,n in enumerate(all_notes))
    return reg, "overlap" if has_other else "nonoverlap"


def targeted_score(row, ckpt, cfg):
    ends,x,p,old_k = row["ends"],row["x"],row["p"],row["k"]
    with np.load(ckpt,allow_pickle=False) as z:
        arr={key:z[key] for key in z.files if key not in ("schema","metadata")}
    q,ks=export_forward(arr,frames(x,p))
    selector=CausalSelection(cfg)
    events=[]
    for t,pitch,k,nq,nk in zip(ends,p,old_k,q,ks):
        events.extend(selector.step(fuse(pitch,nq,nk,POLICY),k,int(t),int(t)))
    events.extend(selector.panic(row["n"]))
    score,matched=event_score(events,row["truth"])
    found={j for _,j in matched}
    by={}
    for i,note in enumerate(row["truth"]):
        keys=(category(note,row["truth"],i)[0],
              "high_overlap" if category(note,row["truth"],i)==("high_E5_plus","overlap") else "all")
        for key in keys:
            bucket=by.setdefault(key,dict(total=0,matched=0))
            bucket["total"]+=1
            bucket["matched"]+=int(i in found)
    return score,by


def fold_counts(rows):
    grouped={}
    for row in rows:
        for key,v in row.items():
            g=grouped.setdefault(key,dict(total=0,matched=0))
            g["total"]+=v["total"];g["matched"]+=v["matched"]
    for g in grouped.values():
        g["recall"]=g["matched"]/g["total"] if g["total"] else None
    return grouped


def run(out, *, epochs=8, mixes=48):
    if out.exists():
        raise FileExistsError("Do not overwrite S631 immutable evidence")
    if epochs < 1 or mixes < 1:
        raise ValueError("Positive epochs/mixes required")
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    sha=digest(MODEL.read_bytes())
    assert sha==frozen["model_sha256"]
    heads=load_heads(MODEL,expected_sha256=sha)
    cfg=Selection(**heads.metadata["selection"])
    all_specs=protocol["sources"]
    train=[s for s in all_specs if s["corpus"]=="URMP" and s["split"]=="train"]
    train+=pick_original_guitar_training(protocol)
    dev=[s for s in all_specs if s["corpus"] in ("URMP","GuitarSet") and s["split"]=="dev"]
    assert len(train)==35 and len(dev)==17
    groups=lambda ss:{(s["corpus"],s["composition"]) for s in ss}
    assert not groups(train)&groups(dev)
    assert all(s["split"]=="train" for s in train)
    start=time.perf_counter()
    out.mkdir(parents=True)
    rng=np.random.default_rng(SEED)
    original=[];proof=[];mixed=[];mix_proof=[];dev_results=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for spec in train:
                row=get_source(spec,root,heads,archive if spec["corpus"]=="GuitarSet" else None)
                sample=choose(row,rng)
                original.append(sample)
                proof.append(dict(source=row["name"],corpus=row["corpus"],**sample[3]))
                print("S631 ORIGINAL",len(original),row["name"],flush=True)
            catalog=[]
            for spec in train:
                item=dict(spec)
                item["notes"]=real_notes(labels(spec,root)["notes"],float(spec["seconds"]))
                catalog.append(item)
            chosen=choose_pairs(catalog,mixes,np.random.default_rng(SEED))
            # Cache original train waveforms; never mix labels or signals from DEV.
            pcm_cache={}
            for i,pair in enumerate(chosen):
                first,second=pair["first"],pair["second"]
                for spec in (first,second):
                    if spec["name"] not in pcm_cache:
                        pcm_cache[spec["name"]]=load_pcm(spec,root,archive)
                audio,truth=mix_pair(pair,pcm_cache[first["name"]],pcm_cache[second["name"]])
                ends,x=feature_track(audio)
                p,k=heads.predict(x)
                row=dict(ends=ends,x=x,p=p,k=k,truth=truth,n=len(audio))
                sample=choose(row,rng)
                mixed.append(sample)
                mix_proof.append(dict(first=first["name"],second=second["name"],
                    first_group=first["composition"],second_group=second["composition"],
                    gap_ms=round(pair["gap_s"]*1000),target_high_midi=pair["high"]["pitch"],
                    interval_semitones=pair["interval"],gain=pair["secondary_gain"],
                    real_pcm_mix_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                    actual_high_onsets=sum(n["pitch"]>=HIGH_MIDI for n in truth),**sample[3]))
                print("S631 PCM OVERLAY",i+1,"/",mixes,first["name"],"+",second["name"],
                      "high",pair["high"]["pitch"],flush=True)
            # Same exact architecture, start RNG and optimizer. Only data differs.
            control_path,control_hist,control_counts=train_network(
                original,out,epochs=epochs)
            # Preserve control before second training writes default checkpoint name.
            moved=out/"S631_original_only_control.npz"
            control_path.rename(moved)
            control_path=moved
            treatment_path,treatment_hist,treatment_counts=train_network(
                original+mixed,out,epochs=epochs)
            treatment_renamed=out/"S631_train_plus_true_overlays.npz"
            treatment_path.rename(treatment_renamed)
            treatment_path=treatment_renamed
            print("S631 TRAIN COMPLETE",flush=True)
            for spec in dev:
                row=get_source(spec,root,heads,archive if spec["corpus"]=="GuitarSet" else None)
                base_events=decode(row["ends"],row["p"],row["k"],cfg,row["n"])
                frozen_score,_=event_score(base_events,row["truth"])
                old=targeted_score(row,control_path,cfg)
                new=targeted_score(row,treatment_path,cfg)
                dev_results.append(dict(source=spec["name"],corpus=spec["corpus"],
                   reference=frozen_score,original_only=old[0],overlaid=new[0],
                   original_registers=old[1],overlaid_registers=new[1]))
                print("S631 DEV",spec["name"],"original",old[0]["F1"],
                      "overlay",new[0]["F1"],flush=True)
    per_corpus={}
    for corpus in ("GuitarSet","URMP"):
        matches=[r for r in dev_results if r["corpus"]==corpus]
        result={key:summary([r[key] for r in matches]) for key in
                ("reference","original_only","overlaid")}
        control=CONTROL[corpus]
        assert all(result["reference"][k]==control[k] for k in ("TP","FP","FN"))
        result["delta_treatment_vs_control_F1_points"]=100*(
            result["overlaid"]["F1"]-result["original_only"]["F1"])
        result["register_control"]=fold_counts([r["original_registers"] for r in matches])
        result["register_overlay"]=fold_counts([r["overlaid_registers"] for r in matches])
        per_corpus[corpus]=result
    report=dict(status="S631_TRAIN_PCM_OVERLAY_HIGH_NOTE_ABLATION_DEV_ONLY",
        frozen_S614_sha256=sha,original_S80_and_production_S614_untouched=True,
        no_future_at_inference=True,synthetic_mix_is_training_only=True,
        mixed_train_only=True,composition_disjoint_from_dev=True,
        dev_preexposed_historical_not_independent=True,not_promoted=True,
        epochs=epochs,train_original_sources=len(train),train_mixes=len(mixed),
        control_checkpoint_sha256=digest(control_path.read_bytes()),
        treatment_checkpoint_sha256=digest(treatment_path.read_bytes()),
        training_original=control_counts,training_augmented=treatment_counts,
        history_control=control_hist,history_augmented=treatment_hist,
        source_proof=proof,mix_proof=mix_proof,
        scoring_policy=dict(pitch_weight=.4,support=0,count_gate=0),
        comparisons=per_corpus,dev_results=dev_results,
        elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S631 FINAL",json.dumps({k:report[k] for k in (
       "train_mixes","comparisons","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--mixes",type=int,default=48)
    a=ap.parse_args()
    run(a.out,epochs=a.epochs,mixes=a.mixes)
