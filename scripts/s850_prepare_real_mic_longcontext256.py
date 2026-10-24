"""S850: signed real GuitarSet microphone/JAMS TRAIN/CAL/OUTER PCM256.

Only ORIGINAL physically recorded mono guitar waveforms and note annotations.
NOT DSP pitch shifted, not synthesized guitar, not pseudo-labels, and hex
pickup samples are NOT a prerequisite for training the pitch selector.

64 TRAIN guitarist tracks (00..03), 16 CAL (04), 16 OUTER (05).
All samples are independent CONSECUTIVE 256-PCM hops. No future audio used
for an individual model prediction; a 32-hop TRAIN unroll does not insert an
input queue or extra LIVE waiting.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scripts.s843_train_real_hex_transpose256 import read_jams_note_midi

SR=44100
HOP=256
PITCHES=49
SEQUENCE_HOPS=32
SEQUENCES_PER_TRACK=80
SEED=850
SOURCE_SHA256={
  "mic":"237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28",
  "jams":"8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99",
}
PLAYERS={
    "train":("00","01","02","03"),
    "cal":("04",),
    "outer":("05",),
}
PREFERRED=("BN1","Funk2","Jazz1","Rock3","SS1")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def split_of(stem):
    for key,players in PLAYERS.items():
        if stem[:2] in players:return key
    raise ValueError("unknown GuitarSet player "+stem)


def ordered_diverse_tracks(names,player,max_tracks):
    chosen=[]
    groups=[]
    for comp in PREFERRED:
        subset=sorted(n for n in names if n.startswith(player+"_")
                      and n.split("_",1)[1].startswith(comp+"-"))
        groups.append(subset)
    while len(chosen)<max_tracks and any(groups):
        for group in groups:
            if group and len(chosen)<max_tracks:chosen.append(group.pop(0))
    return chosen


def target_mask(notes,begins):
    begins=np.asarray(begins,dtype=np.int64)
    gt=np.zeros((len(begins),PITCHES),np.float32)
    lo=begins/SR
    hi=(begins+HOP)/SR
    for t,off,pitch,ch in notes:
        if not 40<=pitch<=88:continue
        active=(t<hi)&(off>lo)
        gt[active,pitch-40]=1.
    return gt


def sample_sequences(raw,notes,*,seed=SEED,count=SEQUENCES_PER_TRACK,
                     hops=SEQUENCE_HOPS):
    if raw.dtype!=np.int16 or raw.ndim!=1:
        raise ValueError("only original real mono int16 mic")
    n=len(raw)//HOP
    if n<=hops:raise ValueError("recording too short")
    rng=np.random.default_rng(seed)
    maximum=n-hops
    def index(sec):
        return int(np.clip(math.floor(sec*SR/HOP)-2,0,maximum))
    rare=[]
    all_attacks=[]
    for t,off,p,ch in notes:
        if not 40<=p<=88:continue
        all_attacks.append(index(t))
        if p in (40,52):
            rare.extend((index(t),index((t+off)/2)))
    for t,off,p,ch in notes:
        if p!=40:continue
        for a,b,q,channel in notes:
            if q!=52:continue
            lower=max(t,a);upper=min(off,b)
            if upper-lower>=HOP/SR:
                rare.extend((index(lower),index((lower+upper)/2)))
    rng.shuffle(rare)
    rng.shuffle(all_attacks)
    random=list(rng.integers(0,maximum+1,size=count*3))
    starts=list(dict.fromkeys(
        rare[:count//2]+all_attacks[:count//3]+random))[:count]
    if not starts:raise ValueError("no real PCM sequences sampled")
    blocks=[];labels=[]
    for first in starts:
        begin=first*HOP
        end=(first+hops)*HOP
        pcm=raw[begin:end].astype(np.float32).reshape(hops,HOP)/32768.
        starts_for_notes=begin+np.arange(hops)*HOP
        gt=target_mask(notes,starts_for_notes)
        blocks.append(pcm)
        labels.append(gt)
    return np.stack(blocks),np.stack(labels)


def build(mic_path,jams_path,out,*,max_tracks_per_player=16,
          count=SEQUENCES_PER_TRACK,hops=SEQUENCE_HOPS):
    out=Path(out)
    if out.exists():raise FileExistsError("immutable real GuitarSet dataset")
    actual=dict(mic=sha256(mic_path),jams=sha256(jams_path))
    if actual!=SOURCE_SHA256:
        raise ValueError("GuitarSet mic/annotation checksum mismatch: "+str(actual))
    original=defaultdict(lambda:defaultdict(list))
    track_details=[]
    with zipfile.ZipFile(mic_path) as micro,zipfile.ZipFile(jams_path) as jams:
        names=sorted(Path(v).name[:-8] for v in micro.namelist()
                     if v.endswith("_mic.wav"))
        for player in ("00","01","02","03","04","05"):
            chosen=ordered_diverse_tracks(names,player,max_tracks_per_player)
            if len(chosen)<max_tracks_per_player:
                raise ValueError("not enough authentic diverse tracks for "+player)
            for stem in chosen:
                found=[v for v in micro.namelist() if Path(v).name==stem+"_mic.wav"]
                if len(found)!=1:raise ValueError("missing/duplicate original mono WAV")
                sr,raw=wavfile.read(io.BytesIO(micro.read(found[0])))
                if sr!=SR or raw.dtype!=np.int16 or raw.ndim!=1:
                    raise ValueError("not original 44100 int16 mono GuitarSet")
                note_gt=read_jams_note_midi(jams,stem)
                split=split_of(stem)
                seq,gt=sample_sequences(raw,note_gt,seed=SEED+int(player),
                                        count=count,hops=hops)
                original[split]["pcm"].append(seq)
                original[split]["gt"].append(gt)
                track_details.append(dict(name=stem,split=split,
                         sequences=len(seq),actual_gt_notes=len(note_gt),
                         E2_gt_hops=int(gt[...,0].sum()),
                         E3_gt_hops=int(gt[...,12].sum()),
                         both_gt_hops=int(((gt[...,0]>.5)&(gt[...,12]>.5)).sum())))
                print("S850_AUTHENTIC_GUITAR_MIC",stem,split,
                      "E2",int(gt[...,0].sum()),"E3",int(gt[...,12].sum()),
                      flush=True)
    out.mkdir(parents=True)
    record=dict(schema="S850_REAL_MONO_GUITARSET_LONG_CAUSAL_SEQUENCES_PCM256",
       source_sha256=actual,sample_rate=SR,hop=HOP,hops_per_training_sequence=hops,
       training_sequence_future_pcm_available_only_for_training_loss=True,
       no_future_input_in_network=True,no_added_live_delay=True,
       no_synthetic_music=True,no_dsp_pitch_shift_used_for_train=True,
       splits={k:list(v) for k,v in PLAYERS.items()},
       tracks=track_details,split_counts={})
    for split in ("train","cal","outer"):
        d=original[split]
        if not d["pcm"]:raise ValueError("missing "+split)
        x=np.concatenate(d["pcm"],axis=0)
        y=np.concatenate(d["gt"],axis=0)
        path=out/(split+".npz")
        np.savez_compressed(path,pcm=x,truth=y)
        record["split_counts"][split]=dict(
           sequences=len(x),decisions=len(x)*hops,shape=list(x.shape),
           notes_E2=int(y[...,0].sum()),notes_E3=int(y[...,12].sum()),
           notes_both=int(((y[...,0]>.5)&(y[...,12]>.5)).sum()),
           positives_any_pitch=int((y.sum(axis=-1)>0).sum()),
           sha256=sha256(path))
        print("S850_REAL_PCM256_GT_DISTRIBUTION",split,
              json.dumps(record["split_counts"][split]),flush=True)
    (out/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mic",type=Path,required=True)
    p.add_argument("--jams",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--tracks-per-player",type=int,default=16)
    p.add_argument("--sequences-per-track",type=int,default=80)
    p.add_argument("--hops",type=int,default=32)
    args=p.parse_args()
    build(args.mic,args.jams,args.out,max_tracks_per_player=args.tracks_per_player,
          count=args.sequences_per_track,hops=args.hops)
