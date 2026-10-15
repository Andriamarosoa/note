"""S588: unchanged pretrained Basic Pitch on S586 sixth-cohort WAVs.

Verify source audio SHA against S587; no target MIDI or training."""
from __future__ import annotations
import argparse,json,hashlib,io,wave
from pathlib import Path
import numpy as np
from scripts.s587_sixth_original_V88_WAV import fetch_track
from scripts.s277_original_freeze_dataset3_native_prelabel import decode_external_wav,pcm
SR=44100
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def name(midi):
    names=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return names[int(midi)%12]+str(int(midi)//12-1)
def run(manifest,s587,out):
    need(not out.exists(),"S588 source-independent pretrained baseline freeze cannot overwrite")
    m=json.loads(manifest.read_text())
    r=json.loads(s587.read_text())
    need(m["status"]=="S586_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and m["five_next_distinct_composers"]==5
         and r["status"]=="S587_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
         and r["preregistered_S586_SHA"]==sha_file(manifest)
         and r["original_complete_S80_NOT_REPLAYED"]
         and len(r["exact_original_audio_source_fingerprints"])==5
         and r["five_original_piano_recordings"]==5
         and r["prelabel_predictions_frozen"]
         and not r["paired_MIDI_annotation_files_OPENED"],
         "S588 original independent third source WAV metadata/S587 provenance altered")
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict,Model
    est=Model(ICASSP_2022_MODEL_PATH)
    evidence={e["WAV_filename_SHA256"]:e for e in
              r["exact_original_audio_source_fingerprints"]}
    ledger=[];per=[]
    out.mkdir(parents=True)
    for i,item in enumerate(m["chosen_metadata_only"]):
        ident=item["filename_SHA256"]
        need(ident in evidence and
             item["audio_filename"]==evidence[ident]["audio_filename"],
             "S588 original 3 piano WAV selection or hash mismatch")
        raw,url=fetch_track(m["source_audio_mirror"],item["audio_filename"])
        need(sha(raw)==evidence[ident]["original_WAV_SHA256"],
             "S588 WAV bytes differ from already-frozen true V88 source")
        whole,fmt=decode_external_wav(raw)
        cut=int(float(item["audio_excerpt_start_seconds"])*SR)
        size=int(float(item["audio_excerpt_duration_seconds"])*SR)
        need(size==12*SR and len(whole)>=cut+size,
             "S588 original source excerpt time changed")
        piece=out/f"NEVER_UPLOAD_original_piano_{i}.wav"
        piece.write_bytes(pcm(whole[cut:cut+size]))
        try:
            _,_,notes=predict(str(piece),est,
              onset_threshold=0.5,frame_threshold=0.3,
              minimum_note_length=127.7,
              minimum_frequency=None,maximum_frequency=None)
        finally:
            piece.unlink(missing_ok=True)
        complete=[]
        for st,en,pitch,amplitude,pitchbend in notes:
            p=int(pitch)
            if not (40<=p<=88 and 0<=float(st)<12 and float(en)>float(st)):
                continue
            q=dict(WAV_filename_SHA256=ident,
                predicted_onset_s=float(st),
                predicted_offset_s=min(12.,float(en)),
                pitch_MIDI=p,
                note_name=name(p),
                pretrained_note_score=float(amplitude),
                pitch_bend_available=bool(pitchbend))
            complete.append(q);ledger.append(q)
        per.append(dict(WAV_filename_SHA256=ident,composer=item["composer"],
            total_basic_pitch_notes=len(notes),
            MIDI40to88_notes_in_12s=len(complete),
            onset_assessment_NOT_YET_PERFORMED=True))
        print("S588 MULTI-INSTRUMENT NOTES",json.dumps(
            dict(source=i,n=len(notes),supported=len(complete),
                 first_note_names=[v["note_name"] for v in complete[:12]])),flush=True)
    events=out/"S588_FIXED_BASIC_PITCH_PRETRAINED_PIANO_NOTES_PRELABEL.jsonl"
    events.write_text("".join(json.dumps(v)+"\n" for v in ledger))
    report=dict(status="S588_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI",
        source="MAESTRO_v3_train_5_piano_WAV_12s",
        original_source_S586_manifest_SHA256=sha_file(manifest),
        original_S587_WAV_proof_SHA256=sha_file(s587),
        original_WAV_content_SHA_verified=True,
        pretrained_library_version="spotify/basic-pitch==0.4.0",
        default_postprocessing_preserved=True,
        model_fitted_or_labels_tuned=False,
        original_MAESTRO_MIDI_ANNOTATIONS_OPENED=False,
        S586_new_composers_before_WAV_MIDI=True,
        output_note_names_not_MIDI_files=True,
        prediction_windows_piano=5,
        total_notes_in_MIDI40to88=len(ledger),
        source_per_piece=per,
        frozen_notes_SHA256=sha_file(events),
        historical_S80_complete_pipeline_NOT_RUN=True,
        S80_and_live_and_offset_unchanged=True,
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S588 FIXED MODEL PROSPECTIVE",json.dumps(dict(
        true_audio_files=5,notes=len(ledger),MIDI_read=False,
        source_midi_K_predictor_modified=False)),flush=True)
def selftest():
    need(name(52)=="E3" and name(55)=="G3" and name(59)=="B3",
         "S588 pitch-to-name user target must be exactly E3 G3 B3")
    print("S588 SELFTEST PASS real MIDI pitch -> named E3 G3 B3")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("manifest","s587","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("manifest","s587","out"):need(getattr(a,k)is not None,"S588 missing "+k)
        run(a.manifest,a.s587,a.out)
