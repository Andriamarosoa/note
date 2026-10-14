"""S437 — sample real GuitarSet MIC WAV with balanced PREDICTED S80 K strata.

This never reads GuitarSet JAMS or true K. Song selection based only on
stable recording name hashes and predicted K0..K6. For selected source
events, compute the SAME FFT/harmonic 42x49 representation as S432 and
run genuine pretrained S430 4-checkpoint note identity networks.
No MIDI file and no test labels. DO NOT claim measured accuracy here.

This test intentionally balances predicted K1/K2/K3 vs K0 so that any
later audit cannot inflate the result with many empty K0 events.
Sample is still GuitarSet and S80 historically learned/tuned there.
MIC differs from S258 pickup-mix historical source.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,need)
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import decode_pcm
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.s430_infer_note_names_from_new_S80_sound_events import run as infer
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

SR=44100
N=59309
MAX_TRACKS=12
CLIP_SECONDS=15.
MIN_ONSET=int(.20*SR)
BUCKETS=((0,2),(1,4),(2,4),(3,4),(4,3),(5,2),(6,2))

def sample_indices(ind,K,start,end,clip_samples):
    safe=ind[(start[ind]>MIN_ONSET)&
             (end[ind]+int(.25*SR)<clip_samples)]
    # A *predicted* K only; no true labels, chord correctness or JAMS.
    sel=[]
    for k,limit in BUCKETS:
        candidates=safe[K[safe]==k]
        candidates=candidates[np.argsort(start[candidates],kind="stable")]
        sel.extend(int(x) for x in candidates[:limit])
    return np.asarray(sorted(set(sel),key=lambda i:int(start[i])),np.int32)

def run(audiozip,source,S80,s402,s416,s428,rep419,rep428,out):
    need(not out.exists(),"S437 immutable guitar MIC dataset exists")
    with np.load(source,allow_pickle=False) as z:
        fields={"row_id","member","start_samples","end_samples"}
        need(fields<=set(z.files),"S437 native source timings unavailable")
        r={k:z[k].copy() for k in fields}
    with np.load(S80,allow_pickle=False) as z:
        need({"row_id","s80_K"}<=set(z.files),"S437 original S80 source missing")
        need(not ({"true_K","labels","truth"}&set(z.files)),
             "S437 must not read original source true K")
        K=z["s80_K"].astype(np.int8)
        need(np.array_equal(z["row_id"].astype(str),r["row_id"].astype(str)),
             "S437 S80 rows misaligned")
    row=r["row_id"].astype(str)
    members=r["member"].astype(str)
    start=r["start_samples"].astype(np.int64)
    end=r["end_samples"].astype(np.int64)
    need(len(row)==N and len(set(row))==N and
         int(K.sum())==33475 and np.isin(K,range(7)).all(),
         "S437 original S80 K0-K6 references modified")
    with zipfile.ZipFile(audiozip) as z:
        paths=collections.defaultdict(list)
        for entry in z.namelist():
            stem=PurePosixPath(entry).name
            if stem.endswith("_mic.wav"):
                paths[stem[:-8]+".jams"].append(entry)
        # Choose by hash of NAME only. This pre-registration may not use
        # note labels, a particular prediction error or its score.
        names=sorted(set(members),key=lambda m:
            hashlib.sha256(("S437-original-mic-"+m).encode()).hexdigest())
        chosen=[];seen=set()
        for member in names:
            if len(paths.get(member,[]))!=1:continue
            comp=composition_key(member)
            if comp in seen:continue
            ind=np.flatnonzero(members==member)
            selected=sample_indices(ind,K,start,end,int(CLIP_SECONDS*SR))
            if np.count_nonzero(K[selected]>0)<5 or np.count_nonzero(K[selected]>=2)<2:
                continue
            chosen.append((member,paths[member][0],selected))
            seen.add(comp)
            if len(chosen)>=MAX_TRACKS:break
        need(len(chosen)==MAX_TRACKS,"S437 not enough different real guitar compositions")
        ids=[];groups=[];t0=[];t1=[];counts=[];features=[]
        track_report=[]
        for member,path,indices in chosen:
            waveform=decode_pcm(z.read(path))
            maxlen=min(len(waveform),int(CLIP_SECONDS*SR))
            indices=indices[end[indices]+int(.25*SR)<maxlen]
            need(len(indices)>=5 and np.any(K[indices]>=2),
                 "S437 missing native audio context or poly events")
            real_pcm=np.clip(np.rint(waveform[:maxlen]*32768),-32768,32767).astype('<i2')
            real=real_pcm.astype(np.float32)/32768.
            for i in indices:
                spec,t=_frames(real,int(start[i]))
                physical,activation=features_from_spectrogram(spec,t,return_states=True)
                need(len(physical)==58 and activation.shape==(42,49)
                     and np.isfinite(activation).all()
                     and np.min(activation)>=0,
                     "S437 original harmonic trajectory invalid")
                ids.append(row[i]);groups.append(member)
                t0.append(int(start[i]));t1.append(int(end[i]))
                counts.append(int(K[i]))
                features.append(np.log1p(activation).astype(np.float32))
            track_report.append(dict(member=member,
                original_audio_member=path,
                predicted_K_hist=np.bincount(K[indices],minlength=7).tolist(),
                source_events=len(indices),composition=composition_key(member)))
            print("S437 TRACK",json.dumps(track_report[-1]),flush=True)
    need(len(ids)==len(set(ids)) and len(ids)>=100 and
         len(ids)==len(features),"S437 multi-guitar frozen source size invalid")
    counts=np.asarray(counts,np.int8)
    out.mkdir(parents=True)
    acoustic=out/"original_MIC_acoustic_42x49.npz"
    original=out/"original_S80_K_from_source_59309_aligned.npz"
    np.savez_compressed(acoustic,row_id=np.asarray(ids),
            sequence42x49=np.asarray(features,np.float32))
    np.savez_compressed(original,row_id=np.asarray(ids),s80_K=counts)
    predictions=out/"recognized"
    infer(acoustic,original,s402,s416,s428,rep419,rep428,predictions)
    detected=[json.loads(x) for x in
            (predictions/"notes_from_new_sound.jsonl").read_text().splitlines()]
    outpath=out/"S437_FROZEN_REAL_GUITAR_MIC_EVENT_NOTE_NAMES.jsonl"
    with outpath.open("w",encoding="utf8") as f:
        for j,p in enumerate(detected):
            need(p["source_row_id"]==ids[j] and p["K"]==int(counts[j])
                 and len(p["notes"])==int(counts[j]),
                 "S437 loss of original S80 Exact-K in note-name output")
            f.write(json.dumps(dict(
                original_row_id=ids[j],member=groups[j],
                start_sample=t0[j],end_sample=t1[j],
                original_S80_K=int(counts[j]),notes=p["notes"]))+"\n")
    readback=[json.loads(x) for x in outpath.read_text().splitlines()]
    need(len(readback)==len(ids) and sum(len(x["notes"]) for x in readback)
         ==int(counts.sum()),"S437 independently reparsed notes lost original S80 K")
    report=dict(status="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES",
        number_distinct_original_compositions=len(chosen),
        real_mono_MIC_GuitarSet_recordings=True,
        original_GuitarSet_MIC_audio_zip_SHA256=sha(audiozip),
        original_source_K_original_59309_preds_SHA256=sha(S80),
        original_S258_native_source_SHA256=sha(source),
        source_MIC_different_from_S258_pickup_mix=True,
        original_S80_GuitarSet_historically_exposed=True,
        true_K_labels_and_JAMS_NEVER_loaded_for_sampling_or_inference=True,
        selection_from_hash_song_name_and_predicted_K_only=True,
        sampled_number_audio_events=len(ids),
        sampled_predicted_K_histogram=np.bincount(counts,minlength=7).tolist(),
        sampled_sum_original_K=int(counts.sum()),
        original_K_conserved_in_note_name_JSONL=True,
        no_MIDI_file_export=True,
        no_Live_or_Offset_changes=True,
        no_new_score_claim=True,
        independent_test_accuracy_not_yet_measured=True,
        no_YourMT_teacher_used=True,
        full_raw_WAV_only_S80_inference_not_implemented=True,
        original_trainable_models_unchanged=True,
        official_promotion=False,
        song_samples=track_report,
        output_SHA256=sha(outpath))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S437 FREEZE",json.dumps(dict(
       songs=len(chosen),events=len(ids),K_hist=report["sampled_predicted_K_histogram"],
       new_notes=int(counts.sum()))),flush=True)
    print("S437 PASS multiple authentic guitar MIC WAV→named notes, independent subsequent labels required",flush=True)

def selftest():
    k=np.asarray([2,0,1,1,3,0,4,2,3,1,0,6],np.int8)
    start=np.arange(len(k))*4410+20000
    end=start+256
    got=sample_indices(np.arange(len(k)),k,start,end,200000)
    hist=np.bincount(k[got],minlength=7)
    need(hist[0]==2 and hist[1]==3 and hist[2]==2 and hist[3]==2
         and hist[4]==1 and hist[6]==1,
         "S437 predicted K-balanced sampling not deterministic")
    print("S437 SELFTEST PASS stratify using only predicted S80 K0..K6")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    for name in ("audio-zip","source-events","s80-npz",
                 "s402-models","s416-models","s428-models",
                 "s419-report","s428-report","out"):
        parser.add_argument("--"+name,type=Path)
    x=parser.parse_args()
    if x.self_test:selftest()
    else:
        for a in ("audio_zip","source_events","s80_npz",
                  "s402_models","s416_models","s428_models",
                  "s419_report","s428_report","out"):
            need(getattr(x,a)is not None,"Missing S437 --"+a)
        run(x.audio_zip,x.source_events,x.s80_npz,
            x.s402_models,x.s416_models,x.s428_models,
            x.s419_report,x.s428_report,x.out)
