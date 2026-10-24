"""S843 trainer for REAL matched GuitarSet microphone / 6-string hex audio.

This script NEVER synthesizes musical waveforms. Requires original GuitarSet
mic, debleeded hex and JAMS archives with upstream published checksums.
TRAIN player 04, CAL player 05, OUTER player 00..03, split by guitarist.
No hidden use of OUTER for model/threshold selection and no production MIDI.

Hex pickup is NOT the microphone signal decomposed exactly: string isolation
has bleed and a different microphone/pickup transfer response. Targets are
real same-performance multichannel recordings, not perfect separations.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import time
import zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import torch

from onset256.real_audio_transpose_head256 import (
    ABI,HOP,SR,NeuralAcousticTransposeHead256,
    reconstruction_loss,real_shift_pair_loss)

OFFICIAL={
    "mic_sha256":"237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28",
    "annotation_sha256":"8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99",
    "hex_clean_md5":"c31d97279464c9a67e640cb9061fb0c6",
}
SEED=843
TRAIN_PLAYERS=("04",)
CAL_PLAYERS=("05",)
OUTER_PLAYERS=("00","01","02","03")


def digest_file(path,algorithm="sha256"):
    h=hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def authenticate(mic,hex_audio,annotations):
    values={
        "mic_sha256":digest_file(mic),
        "annotation_sha256":digest_file(annotations),
        "hex_clean_md5":digest_file(hex_audio,"md5")}
    for key,expected in OFFICIAL.items():
        if values[key]!=expected:
            raise ValueError("NOT ORIGINAL GUITARSET: "+key)
    return values


def unique_zip_name(archive,basename):
    names=[n for n in archive.namelist() if Path(n).name==basename]
    if len(names)!=1:
        raise ValueError("missing or duplicate original "+basename)
    return names[0]


def wav_in_zip(archive,basename,channels):
    name=unique_zip_name(archive,basename)
    sr,a=wavfile.read(io.BytesIO(archive.read(name)))
    if sr!=SR or (a.ndim!=1 if channels==1 else
                   a.ndim!=2 or a.shape[1]!=channels):
        raise ValueError("raw GuitarSet audio must be mono mic or six-channel hex 44100Hz")
    if a.dtype.kind=="i":
        scale=float(np.iinfo(a.dtype).max+1)
        a=np.asarray(a,np.float32)/scale
    elif a.dtype.kind=="f":
        a=np.asarray(a,np.float32)
    else:
        raise ValueError("unsupported original audio PCM type")
    if not np.isfinite(a).all():
        raise ValueError("invalid recorded PCM")
    return a


def read_jams_note_midi(archive,stem):
    data=json.loads(archive.read(unique_zip_name(archive,stem+".jams")))
    ann=[a for a in data["annotations"] if a.get("namespace")=="note_midi"]
    if len(ann)!=6:
        raise ValueError("need exactly six original per-string JAMS note_midi annotations")
    # In original GuitarSet, the 6 note_midi entries correspond in channel
    # order to the 6 hex pickup channels. This MUST be validated acoustically
    # before claiming precise reconstruction accuracy.
    notes=[]
    for channel,a in enumerate(ann):
        for row in a["data"]:
            start=float(row["time"]);duration=float(row["duration"])
            pitch=int(round(float(row["value"])))
            if (not math.isfinite(start+duration) or start<0 or duration<0):
                raise ValueError("invalid real JAMS note")
            if 40<=pitch<=88 and duration>0:
                notes.append((start,start+duration,pitch,channel))
    return notes


def target_masks(notes,ends):
    e=np.asarray(ends,np.int64)
    if e.ndim!=1 or len(e)==0 or bool((np.diff(e)!=HOP).any()):
        raise ValueError("training GT must match consecutive exactly PCM256")
    pitches=np.zeros((len(e),49),np.float32)
    strings=np.zeros((len(e),6),np.float32)
    for start,end,pitch,ch in notes:
        # Strict active THROUGH the current hop, not an arbitrary lookahead
        # or annotation feature supplied as a network input.
        active=(e-HOP)/SR < end
        active &= e/SR > start
        pitches[active,pitch-40]=1.
        strings[active,ch]=1.
    return pitches,strings


def player_split(stem):
    player=stem[:2]
    if player in TRAIN_PLAYERS:return "train"
    if player in CAL_PLAYERS:return "cal"
    if player in OUTER_PLAYERS:return "outer"
    raise ValueError("unknown guitarist, impossible player-disjoint split")


def real_blocks(mic,hex_audio,notes,*,seed=0,sequences=28,seq_hops=8):
    n=min(len(mic),len(hex_audio))
    num_frames=n//HOP
    if num_frames < seq_hops+1:
        raise ValueError("too short genuine recording")
    candidate=list(range(0,num_frames-seq_hops+1))
    # Use only valid samples, and choose a mix of random blocks and note
    # neighborhoods. This is offline DATASET SAMPLING, never Live lookahead.
    rng=np.random.default_rng(seed)
    attacks=[max(0,min(num_frames-seq_hops,int(t*SR/HOP)))
             for t,off,p,ch in notes if 40<=p<=88]
    random_starts=rng.choice(candidate,min(sequences,len(candidate)),replace=False).tolist()
    chosen=list(dict.fromkeys((attacks[:sequences//2]+random_starts)))[:sequences]
    inputs=[];targets=[];pitch=[];strings=[]
    for start in chosen:
        begins=start*HOP;end=(start+seq_hops)*HOP
        audio=np.asarray(mic[begins:end],np.float32).reshape(seq_hops,HOP)
        clean=np.asarray(hex_audio[begins:end],np.float32).reshape(seq_hops,HOP,6)
        labels,activity=target_masks(notes,range(begins+HOP,end+1,HOP))
        inputs.append(audio)
        targets.append(clean.transpose(0,2,1))
        pitch.append(labels)
        strings.append(activity)
    return (np.stack(inputs),np.stack(targets),np.stack(pitch),np.stack(strings))


def data_from_real_archives(mic_zip,hex_zip,jams_zip,out,*,sequences=28,
                            max_tracks_per_player=8):
    if out.exists():
        raise FileExistsError("original real datasets and experiment are immutable")
    proven=authenticate(mic_zip,hex_zip,jams_zip)
    rng=np.random.default_rng(SEED)
    split_data=defaultdict(lambda: defaultdict(list))
    sample_meta=[]
    out.mkdir(parents=True)
    with (zipfile.ZipFile(mic_zip) as mic,
          zipfile.ZipFile(hex_zip) as hex_data,
          zipfile.ZipFile(jams_zip) as jams):
        names=sorted(Path(name).name.removesuffix("_mic.wav") for name in mic.namelist()
                     if name.endswith("_mic.wav"))
        for player in sorted({name[:2] for name in names}):
            selected=sorted(n for n in names if n.startswith(player+"_"))[:max_tracks_per_player]
            for stem in selected:
                orig=wav_in_zip(mic,stem+"_mic.wav",1)
                target=wav_in_zip(hex_data,stem+"_hex_cln.wav",6)
                notes=read_jams_note_midi(jams,stem)
                if not notes:
                    continue
                if abs(len(orig)-len(target))>HOP:
                    raise ValueError("mic and hex target track lengths disagree: "+stem)
                split=player_split(stem)
                pcm,stems,gt,stringgt=real_blocks(orig,target,notes,
                    seed=SEED+int(stem[:2]),sequences=sequences)
                for key,value in zip(("mic","hex","note_gt","string_gt"),
                                     (pcm,stems,gt,stringgt)):
                    split_data[split][key].append(value)
                split_data[split]["track_id"].append(np.full(len(pcm),len(split_data[split]["tracks"]),np.int32))
                split_data[split]["tracks"].append(stem)
                sample_meta.append(dict(track=stem,player=stem[:2],
                    split=split,seq_count=len(pcm),samples_per_decision=HOP))
                print("S843_REAL_HEX_TRACK",stem,split,len(pcm),flush=True)
    record=dict(schema="S843_OFFICIAL_HEX_RECORDED_AUDIO_V1",
        source_hashes=proven,train_players=list(TRAIN_PLAYERS),
        cal_players=list(CAL_PLAYERS),outer_players=list(OUTER_PLAYERS),
        raw_samples=HOP,lookahead=0,added_hold=0,
        hex_is_same_take_different_pickup_domain=True,
        hex_channel_mapping_assumed_JAMS_original_order=True,
        physical_mic_hex_alignment_not_independently_verified=True,
        track_info=sample_meta,
        total_tracks=len(sample_meta),split_tracks={},
        no_synthetic_audio=True,onset_F1_not_measured=True)
    for split in ("train","cal","outer"):
        if not split_data[split]["tracks"]:
            raise ValueError("missing required real player split "+split)
        arrays={}
        for key in ("mic","hex","note_gt","string_gt","track_id"):
            arrays[key]=np.concatenate(split_data[split][key],axis=0)
        np.savez_compressed(out/(split+".npz"),**arrays)
        record["split_tracks"][split]=dict(
            tracks=list(split_data[split]["tracks"]),
            examples=len(arrays["mic"]),
            sha256=digest_file(out/(split+".npz")))
    (out/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def pair_pool(real):
    """Semantic octave pairing from single-note real audio only.
    Not identical phase/time or a formulaic DSP-pitch-shift; all in TRAIN.
    """
    d=real
    pcm=d["mic"];notes=d["note_gt"]
    if "track_id" not in d or len(d["track_id"])!=len(pcm):
        raise ValueError("paired recordings require original distinct track provenance")
    bank=defaultdict(list)
    for i in range(len(pcm)):
        for t in range(pcm.shape[1]):
            y=notes[i,t]
            if int(y.sum())==1:
                pitch=int(np.flatnonzero(y)[0])+40
                bank[pitch].append((pcm[i,t].copy(),int(d["track_id"][i])))
    rng=np.random.default_rng(SEED)
    pairs=[]
    for pitch in range(40,77):
        lower=bank[pitch]
        upper=bank[pitch+12]
        if not lower or not upper:continue
        for k,(a,source_id) in enumerate(lower[:32]):
            valid=[(b,target_id) for b,target_id in upper if source_id!=target_id]
            if not valid:continue
            b,target_id=valid[k%len(valid)]
            if not np.array_equal(a,b):
                pairs.append((a,b,pitch,pitch+12,12))
                pairs.append((b,a,pitch+12,pitch,-12))
    rng.shuffle(pairs)
    return pairs


def train(data_dir,out,*,epochs=4,batch_size=32,seed=SEED):
    out=Path(out);root=Path(data_dir)
    if out.exists():raise FileExistsError("no overwrite of model evidence")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    with np.load(root/"train.npz",allow_pickle=False) as f:
        tr={k:f[k].copy() for k in f.files}
    with np.load(root/"cal.npz",allow_pickle=False) as f:
        cal={k:f[k].copy() for k in f.files}
    for ds in (tr,cal):
        if (ds["mic"].ndim!=3 or ds["mic"].shape[-1]!=HOP or
                ds["hex"].shape!=(*ds["mic"].shape[:2],6,HOP) or
                ds["note_gt"].shape!=(*ds["mic"].shape[:2],49) or
                ds["string_gt"].shape!=(*ds["mic"].shape[:2],6)):
            raise ValueError("training must use original aligned RAW PCM256 arrays")
    pool=pair_pool(tr)
    net=NeuralAcousticTransposeHead256()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    history=[]
    best_val=float("inf");best=None;best_epoch=0
    for epoch in range(epochs):
        net.train();indices=rng.permutation(len(tr["mic"]))
        losses=[]
        for step,first in enumerate(range(0,len(indices),batch_size)):
            ids=indices[first:first+batch_size]
            def make(k):return torch.from_numpy(tr[k][ids])
            loss,parts,_=reconstruction_loss(net,make("mic"),make("hex"),
                                               make("note_gt"),make("string_gt"))
            if pool:
                idx=rng.integers(0,len(pool),size=min(16,len(pool)))
                pairs=[pool[int(i)] for i in idx]
                src=torch.tensor(np.stack([x[0] for x in pairs]))[:,None,:]
                tgt=torch.tensor(np.stack([x[1] for x in pairs]))[:,None,:]
                sy=torch.zeros((len(pairs),49));ty=torch.zeros_like(sy)
                for j,(_,_,p,q,offset) in enumerate(pairs):
                    sy[j,p-40]=1.;ty[j,q-40]=1.
                shifts=torch.tensor([p[4] for p in pairs],dtype=torch.long)
                loss=loss+.2*real_shift_pair_loss(net,src,tgt,sy,ty,shifts)
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.inference_mode():
            val=[]
            for start in range(0,len(cal["mic"]),batch_size):
                sl=slice(start,start+batch_size)
                t={k:torch.from_numpy(cal[k][sl]) for k in ("mic","hex","note_gt","string_gt")}
                v,parts,_=reconstruction_loss(net,t["mic"],t["hex"],
                                               t["note_gt"],t["string_gt"])
                val.append(float(v))
            val_loss=float(np.mean(val))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),
                            cal_loss=val_loss))
        if val_loss<best_val:
            best_val=val_loss;best_epoch=epoch+1
            best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        print("S843_REAL_TRAIN_EPOCH",epoch+1,"train_loss",history[-1]["train_loss"],
              "cal_loss",val_loss,flush=True)
    out.mkdir(parents=True)
    torch.save(dict(abi=ABI,state_dict=best,trained_hop_samples=HOP,
                    shift_real_pair_count=len(pool),best_cal_epoch=best_epoch,
                    promoted=False),out/"s843_real_hex_neural.pt")
    results=dict(abi=ABI,real_training_source=root.as_posix(),
        best_cal_epoch=best_epoch,history=history,shift_real_pairs=len(pool),
        trained_on_real_microphone_and_hex_recordings=True,
        pitch_shift_DSP_used=False,has_future_audio=False,added_hold_samples=0,
        guitarset_ONSET_80_F1_NOT_MEASURED=True,
        baseline_Onset_80_F1=.6925843310420176,promoted=False)
    (out/"train_result.json").write_text(json.dumps(results,indent=2)+"\n")
    print("S843_REAL_TRAIN_COMPLETED_NO_OFFICIAL_ONSET_SCORE",
          json.dumps(dict(epoch=best_epoch,real_pair_count=len(pool))),flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("prepare","train"),required=True)
    p.add_argument("--mic-zip",type=Path)
    p.add_argument("--hex-zip",type=Path)
    p.add_argument("--annotation-zip",type=Path)
    p.add_argument("--data",type=Path,default=Path("/tmp/s843_data"))
    p.add_argument("--out",type=Path,default=Path("/tmp/s843_model"))
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--max-tracks-per-player",type=int,default=8)
    a=p.parse_args()
    if a.phase=="prepare":
        if not all((a.mic_zip,a.hex_zip,a.annotation_zip)):
            p.error("all 3 real signed archives are required")
        report=data_from_real_archives(a.mic_zip,a.hex_zip,a.annotation_zip,a.data,
                           max_tracks_per_player=a.max_tracks_per_player)
        print("S843_REAL_SOURCE_TRACKS",report["total_tracks"],flush=True)
    else:
        train(a.data,a.out,epochs=a.epochs)


if __name__=="__main__":
    main()
