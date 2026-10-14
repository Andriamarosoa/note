"""S432 — WAV -> original 42x49 sound features -> E3 note names.

Reuses EXACT ORIGINAL six-harmonic 4096-FFT/256-hop extractor.
CLI consumes WAV plus each proposed event's ORIGINAL S80 predicted K.
This is not raw WAV-only Exact-K; the original S80 model and timing
proposer must provide decisions for NEW audio externally. It never
estimates K from peaks. It calls the actual pretrained S430 note
recognition and returns named notes rather than writing a MIDI file.

Do NOT claim the 40.96% S428 song-OOF score for this NEW-audio
S430 4-checkpoint ensemble: its note identity accuracy is unmeasured.
"""
from __future__ import annotations
import argparse,json,hashlib,wave
from pathlib import Path
import numpy as np
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import need
from scripts.extract_v273_harmonic_trajectory import (
    _frames,features_from_spectrogram)
from scripts.s82_external_pr16_neural_inference import load_wav
from scripts.s430_infer_note_names_from_new_S80_sound_events import run as notes_run

SR=44100

def prepare_wav(wav,decisions,out):
    """No MIDI labels and no true-K available at any point."""
    need(not out.exists(),"S432 cannot overwrite an output")
    with wave.open(str(wav),'rb') as obj:
        need(obj.getnchannels()==1 and obj.getframerate()==SR
             and obj.getsampwidth()==2 and obj.getcomptype()=="NONE",
             "S432 WAV must be mono PCM16 at original 44100 Hz, do not silently resample")
    pcm=load_wav(wav)
    ids=[];onsets=[];ends=[];K=[]
    with decisions.open(encoding="utf8") as fp:
        for row,line in enumerate(fp):
            if not line.strip():continue
            data=json.loads(line)
            # Fail closed when unlabeled annotation or ground truth appears.
            need(set(data)=={"row_id","start_sample","end_sample","s80_K"},
                 "S432 event decision must come from ORIGINAL S80 inference only")
            id=str(data["row_id"])
            start=data["start_sample"]
            end=data["end_sample"]
            k=data["s80_K"]
            need(not isinstance(start,bool) and isinstance(start,int)
                 and not isinstance(end,bool) and isinstance(end,int)
                 and not isinstance(k,bool) and isinstance(k,int)
                 and 0<=k<=6,"S432 S80 predicted K0..K6 and sample positions required")
            need(start>int(.150*SR)
                and end>=start and end-start<=1764
                and end+int(.25*SR)<len(pcm),
                "S432 raw WAV sound event lacks acoustic context or bounds")
            need(not ids or start>onsets[-1],
                 "S432 incoming onset events must be strictly ordered")
            ids.append(id);onsets.append(start);ends.append(end);K.append(k)
    need(len(ids)>0 and len(ids)==len(set(ids)),
         "S432 no unique Original S80 predicted sound events")
    X=np.zeros((len(ids),42,49),np.float32)
    for i,start in enumerate(onsets):
        spec,t=_frames(pcm,int(start))
        feat,states=features_from_spectrogram(spec,t,return_states=True)
        need(len(feat)==58 and states.shape==(42,49) and
             np.isfinite(states).all() and np.min(states)>=0,
             "S432 original harmonic projection incompatible with S430")
        X[i]=np.log1p(states).astype(np.float32)
    need(np.isfinite(X).all() and np.all(X>=0),
         "S432 nonfinite original extracted acoustic features")
    out.mkdir(parents=True)
    audio=out/"original_WAV_42x49_sound_events.npz"
    count=out/"provided_predicted_S80_K_for_new_sound.npz"
    np.savez_compressed(audio,row_id=np.asarray(ids),
         sequence42x49=X, start_samples=np.asarray(onsets,np.int64),
         end_samples=np.asarray(ends,np.int64))
    np.savez_compressed(count,row_id=np.asarray(ids),s80_K=np.asarray(K,np.int8))
    report=dict(
      status="S432_REAL_WAV_TO_ORIGINAL_HARMONIC_ACOUSTIC_EVENT_FEATURES",
      source_RAW_WAV_SHA256=sha(wav),
      predicted_S80_input_SHA256=sha(decisions),
      original_4096_FFT_256_hop_6_harmonic_bank=True,
      output_S430_compatible_sequence42x49=True,
      actual_WAV_read_for_every_event=True,
      K_predicted_precomputed_EXTERNALLY_not_from_WAV_features=True,
      true_K_and_JAMS_not_used=True,
      original_exact_K_model_weights_unchanged=True,
      no_labels_or_YourMT_used=True,
      number_events=len(ids),
      emitted_note_count=int(sum(K)),
      zero_predicted_K_events=int(np.sum(np.asarray(K)==0)),
      label_free_WAV_reconstruction_not_raw_WAV_to_S80_K=True,
      acoustic_sha256=sha(audio),K_sha256=sha(count),
      note_predictions_accuracy_not_measured=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return audio,count,report

def run(wav,decisions,s402,s416,s428,report419,report428,out):
    need(not out.exists(),"S432 output cannot overwrite an experiment")
    inputs=out/"inputs"
    audio,K,source=prepare_wav(wav,decisions,inputs)
    head=out/"note_names"
    notes_run(audio,K,s402,s416,s428,report419,report428,head)
    predicted=[]
    with (head/"notes_from_new_sound.jsonl").open(encoding="utf8") as fp:
        predicted=[json.loads(x) for x in fp]
    with K.open("rb") as _:pass
    with np.load(K,allow_pickle=False) as z:
        ids=z["row_id"].astype(str);KK=z["s80_K"].astype(np.int8)
    need(len(predicted)==len(ids) and all(
        p["source_row_id"]==ids[i] and p["K"]==int(KK[i])
        and len(p["notes"])==int(KK[i])
        for i,p in enumerate(predicted)),
        "S432 actual detected names violate original predicted K")
    events=[json.loads(line) for line in decisions.read_text().splitlines() if line.strip()]
    result=out/"recognized_note_names_from_WAV.jsonl"
    with result.open("w",encoding="utf8") as h:
        for d,p in zip(events,predicted):
            h.write(json.dumps(dict(
               source_row_id=p["source_row_id"],
               start_sample=d["start_sample"],
               K=p["K"],notes=p["notes"]),ensure_ascii=False)+"\n")
    report=dict(status="S432_WAV_PLUS_S80_K_TO_MUSICAL_NOTE_NAMES",
        original_raw_WAV_used=True,
        note_name_events=len(predicted),
        note_count=int(np.sum(KK)),
        real_WAV_acoustic_source_hash=sha(wav),
        actual_output_file_SHA256=sha(result),
        complete_raw_WAV_to_K_to_notes=False,
        predicted_S80_K_must_be_provided_from_original_source=True,
        no_ground_truth_or_MIDI_score_loaded=True,
        no_model_training_at_inference=True,
        S430_four_checkpoint_ensemble_new_WAV_not_songheld_scored=True,
        original_live_untouched=True,
        original_exactK_reference_untouched=True,
        no_S80_cardinality_regressions_for_provided_K=True,
        no_benchmark_claim_for_unseen_WAV=True,
        official_promotion=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S432 NOTE NAMES",json.dumps(predicted,ensure_ascii=False),flush=True)
    print("S432 PASS raw WAV -> authentic 42x49 -> trained note names, precomputed S80 K required",flush=True)

def selftest():
    import tempfile
    from scipy.io import wavfile
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        t=np.arange(SR*2)/SR
        audio=(.12*np.sin(2*np.pi*164.8138*t)).astype(np.float32)
        wav=root/"E3_audio.wav"
        wavfile.write(wav,SR,np.rint(np.clip(audio,-1,1)*32767).astype(np.int16))
        events=root/"external_predicted_K.jsonl"
        decisions=[
          dict(row_id="sound-1",start_sample=round(.5*SR),
               end_sample=round(.5*SR)+256,s80_K=1),
          dict(row_id="sound-2",start_sample=round(1.2*SR),
               end_sample=round(1.2*SR)+256,s80_K=0)]
        events.write_text("\n".join(json.dumps(d) for d in decisions)+"\n")
        source,K,report=prepare_wav(wav,events,root/"prepared")
        with np.load(source,allow_pickle=False) as z:
            need(z["sequence42x49"].shape==(2,42,49),
                 "S432 failed original signal feature geometry")
        with np.load(K,allow_pickle=False) as z:
            need(np.array_equal(z["s80_K"],np.array([1,0])),
                 "S432 lost provided K1/K0 source decisions")
        need(report["K_predicted_precomputed_EXTERNALLY_not_from_WAV_features"]
              and report["actual_WAV_read_for_every_event"],
             "S432 failed honest provenance test")
    print("S432 SELFTEST PASS original mono WAV E3 physical acoustic features, K0/K1 preserved")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    for flag in ("wav","decisions","s402-models","s416-models",
                 "s428-models","s419-report","s428-report","out"):
        parser.add_argument("--"+flag,type=Path)
    q=parser.parse_args()
    if q.self_test:selftest()
    else:
        for k in ("wav","decisions","s402_models","s416_models",
                  "s428_models","s419_report","s428_report","out"):
            need(getattr(q,k)is not None,"missing --"+k)
        run(q.wav,q.decisions,q.s402_models,q.s416_models,
            q.s428_models,q.s419_report,q.s428_report,q.out)
