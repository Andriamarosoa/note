"""S434 — real GuitarSet WAV (not synthetic!) -> actual E3 etc notes.

Run S433 on one ORIGINAL GuitarSet recording using frozen original S80
GuitarSet 59,309 predictions and S258 native event-time archive. Extract
the recording from official MIC zip without using annotations. The MIC
source differs from S258 pickup mix; this is a REAL acoustic compatibility
smoke test, NOT an independently held-out quality benchmark.

Does not infer K, change S80, retrain pitch models, or touch Live/Offsets.
"""
from __future__ import annotations
import argparse,collections,json,wave,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import need
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s411_extract_continuous_GuitarSet_WAV_pitch_survival_features import decode_pcm
from scripts.s433_join_original_S80_native_events_to_WAV_note_names import run as label_sound

SR=44100
MIN_START=round(.20*SR)
CLIP_SEC=10
MAX_EVENTS=30

def prepare(real_zip,original_events,original_s80,out):
    need(not out.exists(),"S434 immutable real-wave source folder exists")
    with np.load(original_events,allow_pickle=False) as f:
        required={"row_id","member","start_samples","end_samples"}
        need(required<=set(f.files),"S434 original S258 source missing real event metadata")
        e={k:f[k].copy() for k in required}
    with np.load(original_s80,allow_pickle=False) as f:
        need({"row_id","s80_K"}<=set(f.files) and
             not {"true_K","label","labels","truth"}&set(f.files),
             "S434 must use ONLY original predicted S80 K, never original truth")
        r={k:f[k].copy() for k in ("row_id","s80_K")}
    keys=e["row_id"].astype(str);ids=r["row_id"].astype(str)
    K=r["s80_K"].astype(np.int8)
    members=e["member"].astype(str)
    start=e["start_samples"].astype(np.int64)
    end=e["end_samples"].astype(np.int64)
    need(len(keys)==len(ids)==59309 and len(set(ids))==59309
         and np.array_equal(keys,ids) and np.isin(K,range(7)).all()
         and int(K.sum())==33475,
         "S434 original 59309 S80 source event IDs or K decisions changed")

    mapping=collections.defaultdict(list)
    with zipfile.ZipFile(real_zip) as z:
        for item in z.namelist():
            base=PurePosixPath(item).name
            if base.endswith("_mic.wav"):
                mapping[base[:-8]+".jams"].append(item)
        chosen=None
        candidates=[]
        for m in sorted(set(members)):
            if len(mapping.get(m,[]))!=1:
                continue
            ind=np.flatnonzero((members==m)&(start>=MIN_START)
                &(end+round(.25*SR)<CLIP_SEC*SR))
            ind=ind[np.argsort(start[ind],kind="stable")]
            if len(ind)<6:
                continue
            ind=ind[:MAX_EVENTS]
            poly=int(np.sum(K[ind]>=2))
            single=int(np.sum(K[ind]==1))
            if single and poly:
                candidates.append((m,ind))
        need(len(candidates)>0,
             "S434 unable to locate official real guitar mic WAV with K1 and K2")
        # Select by original recording identity only; never by its note
        # truth, pitch accuracy or inference results.
        chosen,ind=candidates[0]
        wav_name=mapping[chosen][0]
        pcm=decode_pcm(z.read(wav_name))
    samples=int(min(len(pcm),CLIP_SEC*SR))
    ind=ind[end[ind]+int(.25*SR)<samples]
    need(len(ind)>=4 and np.all(start[ind]>MIN_START),
         "S434 actual guitar WAV is shorter than frozen native S80 events")
    out.mkdir(parents=True)
    wav_out=out/"genuine_GuitarSet_mic_audio_excerpt.wav"
    with wave.open(str(wav_out),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
        buf=np.clip(np.rint(pcm[:samples]*32768),-32768,32767).astype("<i2")
        w.writeframes(buf.tobytes())
    native=out/"original_NATIVE_true_audio_event_IDs_NO_labels.npz"
    np.savez_compressed(native,row_id=keys[ind],member=members[ind],
        start_samples=start[ind],end_samples=end[ind])
    result=dict(status="S434_AUTHENTIC_GUITARSET_MIC_WAV_NO_LABELS_WITH_FROZEN_S80_NATIVE_GROUPS",
        actual_original_GuitarSet_MIC_WAV_archive_sha256=sha(real_zip),
        actual_wav_member=wav_name,
        original_member=chosen,
        original_source_event_count=len(ind),
        original_source_event_K_histogram=np.bincount(K[ind],minlength=7).tolist(),
        original_native_S80_K_unchanged=True,
        genuine_audio_not_synthesized=True,
        MIC_audio_different_from_historical_S258_pickup_mix=True,
        audio_excerpt_seconds=samples/SR,
        no_truth_JAMS_or_labels_opened=True,
        no_YourMT_teacher_inference=True,
        original_S80_offline_training_overlap_with_GuitarSet=True,
        source_genuine_WAV_sha256=sha(wav_out),
        source_event_archive_sha256=sha(native))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    return wav_out,chosen,native,result

def run(wav_zip,source_events,source_s80,s402,s416,s428,rep419,rep428,out):
    need(not out.exists(),"S434 results must be immutable")
    clip,member,native,info=prepare(wav_zip,source_events,source_s80,out/"source")
    label_sound(clip,member,source_s80,native,
                s402,s416,s428,rep419,rep428,out/"recognized")
    combined=json.loads((out/"recognized/report.json").read_text())
    predictions=out/"recognized/note_names/recognized_note_names_from_WAV.jsonl"
    outputs=[json.loads(line) for line in predictions.read_text().splitlines()]
    need(len(outputs)==info["original_source_event_count"]
         and sum(len(item["notes"]) for item in outputs)
             ==combined["number_new_notes"]
         and any(int(item["K"])>=2 for item in outputs),
         "S434 real acoustic note-name output violates original S80")
    result=dict(status="S434_REAL_GUITARSET_WAV_THROUGH_S80_SOURCE_AND_SOUND_NOTE_NAMES",
        real_original_MIC_WAV=True,
        original_S80_predicted_K_genuinely_frozen=True,
        source_original_S80_K_verbatim_join=True,
        real_audio_acoustic_features_computed=True,
        real_notes_output_count=len(outputs),
        real_predicted_new_note_count=sum(len(item["notes"]) for item in outputs),
        real_predicted_poly_event_count=sum(x["K"]>=2 for x in outputs),
        example_musical_note_names=[x["notes"] for x in outputs[:12]],
        independently_measured_pitch_accuracy=False,
        original_MIC_and_pickup_mix_audio_mismatch=True,
        original_S80_GuitarSet_training_overlap=True,
        true_RAW_WAV_alone_to_S80_inference_not_integrated=True,
        original_exactK_models_and_predictions_unchanged=True,
        original_Live_and_Offsets_unchanged=True,
        no_promotion=True,
        output_notes_SHA256=sha(predictions))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S434 REAL GUITAR",json.dumps(dict(
        member=member,source_events=len(outputs),
        poly_events=result["real_predicted_poly_event_count"],
        notes=result["example_musical_note_names"])),flush=True)
    print("S434 PASS genuine recorded guitar -> original note names with original S80 K",flush=True)

if __name__=="__main__":
    q=argparse.ArgumentParser()
    for n in ("audio-zip","source-events","s80-npz","s402-models",
              "s416-models","s428-models","s419-report","s428-report","out"):
        q.add_argument("--"+n,type=Path,required=True)
    a=q.parse_args()
    run(a.audio_zip,a.source_events,a.s80_npz,a.s402_models,
        a.s416_models,a.s428_models,a.s419_report,a.s428_report,a.out)
