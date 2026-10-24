"""Build ACTUAL GuitarSet raw PCM256 transposition pairs and octave probes.

No synthetic waveforms, phase vocoder, audio pitch math, audio stretching or
new Onset model. Each supervision pair uses two DIFFERENT signed GuitarSet
recordings with verified non-overlapping GT note intervals in the selected
256-sample block. This is a *weak semantic* octave pairing: two independent
performances, not the identical phrase played after retuning. The difference
must be disclosed. Labels are for TRAIN/CAL/OUTER only, never read at Live.

Research-only: pairing can exploit annotation timestamps OFFLINE to select
examples. It MUST NOT be interpreted as a new causal Onset detector.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile

from scripts.onset_event_integrity256 import verify_inputs
from scripts.s841_audit_real_guitarset_e2e3_256 import sha

SR,HOP=44100,256
LO,HI=40,88
GROUPS={"train":("BN1","Funk2","Jazz1"),"cal":("Rock3",),"outer":("SS1",)}
MAX_SOURCE_PER_PITCH=200
MAX_TARGET_PER_PITCH=200
MAX_PROBE_PER_KIND_PER_TRACK=24
SEED=842


def split_of(comp):
    for split,comps in GROUPS.items():
        if comp in comps:return split
    raise ValueError("unknown composition: "+str(comp))


def active_pitches(notes,start,end):
    """Every note active ANYTIME within the [start,end] sampling block."""
    return {p for t,off,p in notes if t < end and off > start}


def is_exactly(notes,start,end,expected):
    """Physical-labelled interval must cover the whole current RAW PCM256 hop."""
    active=active_pitches(notes,start,end)
    if active != set(expected):
        return False
    for pitch in expected:
        if not any(t<=start and off>=end and p==pitch for t,off,p in notes):
            return False
    return True


def candidate_blocks_for_note(t,off):
    """Only *complete* 256 blocks after note onset, with no future at inference."""
    first=(int(math.floor(t*SR/HOP))+2)*HOP
    for step in (0,1,3):
        end_sample=first+step*HOP
        start_sample=end_sample-HOP
        if (start_sample/SR >= t and end_sample/SR <= off and
            start_sample >= 0):
            yield start_sample,end_sample


def sample_window(pcm,start_sample,end_sample):
    raw=pcm[start_sample:end_sample]
    if raw.shape!=(HOP,) or raw.dtype!=np.int16:
        raise ValueError("wrong original 256-sample int16 block")
    return np.ascontiguousarray(raw.astype(np.float32)/32768.)


def choose_windows_for_probe(notes,pcm,max_each=MAX_PROBE_PER_KIND_PER_TRACK):
    """Use real note OFFSETS: pair E2/E3 overlaps and isolated pure notes."""
    candidates=defaultdict(list)
    known=set()
    for t,off,p in notes:
        if p in (40,52):
            for begin,end in candidate_blocks_for_note(t,off):
                pitches=active_pitches(notes,begin/SR,end/SR)
                typ=("E2_ONLY" if pitches=={40} else
                     "E3_ONLY" if pitches=={52} else
                     "E2_E3_SUSTAIN" if {40,52}.issubset(pitches) else
                     "E2_OTHER" if 40 in pitches else
                     "E3_OTHER" if 52 in pitches else "OTHER")
                key=(typ,begin)
                if key not in known:
                    known.add(key)
                    if (typ in ("E2_ONLY","E3_ONLY") and
                        not is_exactly(notes,begin/SR,end/SR,{p})):
                        continue
                    if typ=="E2_E3_SUSTAIN" and not (
                        any(x<=begin/SR and y>=end/SR and q==40 for x,y,q in notes) and
                        any(x<=begin/SR and y>=end/SR and q==52 for x,y,q in notes)):
                        continue
                    candidates[typ].append((begin,end,pitches))
    out=[]
    for typ,blocks in sorted(candidates.items()):
        for begin,end,pitches in blocks[:max_each]:
            y=np.zeros(49,np.float32)
            for p in pitches:
                if LO<=p<=HI:y[p-LO]=1.
            out.append(dict(typ=typ,pcm=sample_window(pcm,begin,end),
                            note_mask=y,end_sample=end))
    return out


def build(root,out):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError("do not overwrite original paired samples")
    manifest=verify_inputs(root)
    bank=defaultdict(list)
    probes=[]
    per_track=[]
    count=Counter()
    with zipfile.ZipFile(root/"GuitarSet_audio.zip") as archive:
        for item in sorted(manifest["fit_files"],key=lambda x:x["index"]):
            comp=item["composition"]
            split=split_of(comp)
            meta_path=root/"traces"/("fit_%03d.json"%item["index"])
            if sha(meta_path)!=item["json_sha256"]:
                raise ValueError("signed GT trace SHA mismatch")
            meta=json.loads(meta_path.read_text())
            if meta["name"]!=item["name"] or meta["composition"]!=comp:
                raise ValueError("signed audio and GT identity mismatch")
            sr,pcm=wavfile.read(io.BytesIO(archive.read(meta["name"]+"_mic.wav")))
            if sr!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                raise ValueError("only authenticated 44100Hz mono 16bit WAV")
            pcm=pcm[:meta["samples"]]
            notes=[]
            for raw in meta["fit_truth"]:
                p=int(raw["pitch"])
                t=float(raw["time"]);dur=float(raw["duration"])
                off=min(len(pcm)/SR,t+dur)
                if t<0 or not math.isfinite(t+dur) or dur<0:
                    raise ValueError("invalid GT note time/duration")
                if LO<=p<=HI and off>t:
                    notes.append((t,off,p))
            if len(notes)==0:continue
            singles=Counter()
            for t,off,p in notes:
                for start,end in candidate_blocks_for_note(t,off):
                    if not is_exactly(notes,start/SR,end/SR,{p}):
                        continue
                    if singles[p]>=20:break
                    singles[p]+=1
                    bank[(split,p)].append(dict(
                        track=meta["name"],composition=comp,texture=(
                            "solo" if meta["name"].endswith("_solo") else "comp"),
                        pcm=sample_window(pcm,start,end),end_sample=end,pitch=p))
            extracted=choose_windows_for_probe(notes,pcm)
            for row in extracted:
                probes.append(dict(row,split=split,track=meta["name"],
                                   comp=comp))
                count[(split,row["typ"])]+=1
            per_track.append(dict(track=meta["name"],split=split,
                                  singles=sum(singles.values()),
                                  probes=len(extracted),
                                  e2_singles=singles[40],
                                  e3_singles=singles[52],
                                  valid_GT_notes=len(notes)))
            print("S842_REAL_NOTE_TRACK",meta["name"],"singles",sum(singles.values()),
                  "E2",singles[40],"E3",singles[52],
                  "probes",len(extracted),flush=True)
    # Assign pair identifiers from *distinct recordings*, never the same mic
    # or the same track. Pairing is semantic, not a simulated transposition.
    rng=np.random.default_rng(SEED)
    pairs=[]
    counts_by_shift=Counter()
    for split in ("train","cal","outer"):
        for p in range(LO,HI-12+1):
            left=list(bank[(split,p)])[:MAX_SOURCE_PER_PITCH]
            right=list(bank[(split,p+12)])[:MAX_TARGET_PER_PITCH]
            if not left or not right:continue
            rng.shuffle(left);rng.shuffle(right)
            for k,src in enumerate(left):
                viable=[x for x in right if x["track"]!=src["track"]]
                if not viable:continue
                target=viable[k%len(viable)]
                pairs.append((src,target,12,split))
                counts_by_shift[(split,12)]+=1
            # Also train inverse octave direction from other distinct WAVs.
            for k,src in enumerate(right):
                viable=[x for x in left if x["track"]!=src["track"]]
                if not viable:continue
                target=viable[k%len(viable)]
                pairs.append((src,target,-12,split))
                counts_by_shift[(split,-12)]+=1
    if not pairs:
        raise AssertionError("no ground-truth octave pairs are available")
    def onehot(p):
        v=np.zeros(49,np.float32)
        v[p-LO]=1.
        return v
    pcm0=np.stack([a["pcm"] for a,b,n,s in pairs]).astype(np.float32)
    pcm1=np.stack([b["pcm"] for a,b,n,s in pairs]).astype(np.float32)
    data=dict(source_pcm=pcm0,target_pcm=pcm1,
              source_note_mask=np.stack([onehot(a["pitch"]) for a,b,n,s in pairs]),
              target_note_mask=np.stack([onehot(b["pitch"]) for a,b,n,s in pairs]),
              semitones=np.asarray([n for a,b,n,s in pairs],np.int64),
              split=np.asarray([s for a,b,n,s in pairs]),
              source_track=np.asarray([a["track"] for a,b,n,s in pairs]),
              target_track=np.asarray([b["track"] for a,b,n,s in pairs]))
    out.mkdir(parents=True)
    path=out/"real_octave_pairs256.npz"
    np.savez_compressed(path,**data)
    info=dict(schema="ONSET_REAL_RECORDED_NOTE_TRANSPOSE_PAIRS_256_V1",
              sample_rate_hz=SR,pcm_block_samples=HOP,
              audio_pair_source="independently_recorded_real_audio",
              dsp_pitch_shift_used=False,lookahead_samples=0,
              source_s708_teacher_used=False,
              paired_npz_sha256=sha(path),
              NOTE="weak semantic pairs of different guitar performances; not identical physical phrase retuned",
              generator="S842 real GuitarSet JAMS GT duration and mono original WAV",
              splits=GROUPS,all_labels_from_signed_GT=True)
    (out/"real_octave_pairs_manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    if probes:
        p=out/"e2_e3_probe256.npz"
        np.savez_compressed(p,
            pcm=np.stack([x["pcm"] for x in probes]),
            note_mask=np.stack([x["note_mask"] for x in probes]),
            kind=np.asarray([x["typ"] for x in probes]),
            split=np.asarray([x["split"] for x in probes]),
            composition=np.asarray([x["comp"] for x in probes]),
            track=np.asarray([x["track"] for x in probes]),
            end_sample=np.asarray([x["end_sample"] for x in probes],dtype=np.int64))
    else:p=None
    pair_counts=Counter(s for a,b,n,s in pairs)
    record=dict(schema="S842_SIGNED_REAL_GUITARSET_OCTAVE_PAIRS_PCM256_V1",
                complete=True,pairs=len(pairs),pairs_by_split=dict(pair_counts),
                pairs_by_shift={str(k):v for k,v in counts_by_shift.items()},
                source_track_and_target_track_always_distinct=all(
                    a["track"]!=b["track"] for a,b,n,s in pairs),
                gt_active_entire_current_256_samples=True,
                sample_rate=SR,hop=HOP,
                probe_frames=len(probes),probe_counts={
                    str(k):v for k,v in sorted(count.items())},
                notes_included=["all singleton MIDI40-88 with real 12-semitone GT pairs",
                                "E2_ONLY","E3_ONLY","E2_E3_SUSTAIN",
                                "E2_OTHER","E3_OTHER"],
                speaker_and_instrument_comparability_not_proven=True,
                strict_train_cal_outer_composition_split=True,
                Onset_F1_NOT_MEASURED=True,
                source_80_WAV_reference_Onset_F1=.6925843310420176,
                data_provenance=info,track_audits=per_track)
    (out/"extraction_report.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("S842_SIGNED_REAL_OCTAVE_PAIRS",json.dumps({
        "pairs":record["pairs"],"by_split":record["pairs_by_split"],
        "probes":record["probe_counts"]},sort_keys=True),flush=True)
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    build(args.root,args.out)
