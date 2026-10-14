"""S487 independent pretrained multi-instrument Basic Pitch on true IDMT guitar audio.

Use exact five original S276 preregistered WAVs and S277 original source
WAV SHA without reading any XML. Execute Spotify Basic Pitch 0.4.0
default unchanged note decoder, output named notes (E3,G3,B3), freeze
before S488 accesses original guitar XML. No train, no promotion.
"""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s464_validate_IDMT_farK0_audio_rms_against_recording import ReliableReadOnlyRangeZIP
from scripts.s277_original_freeze_dataset3_native_prelabel import decode_external_wav,pcm
from scripts.s485_freeze_real_MAESTRO_basic_pitch_pretrained_note_names import name
import urllib.request
SR=44100
def need(c,m):
    if not c:raise ValueError(m)
def sha(blob):return hashlib.sha256(blob).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def run(s276,s277,s277_report,out):
    need(not out.exists(),"S487 genuine pretrained guitar note predictions already frozen")
    sel=json.loads(s276.read_text())
    baseline=json.loads(s277_report.read_text())
    need(sel["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and sel["counts_WAV"]["dataset3"]==5
         and not sel["XML_MIDI_annotation_contents_read"]
         and baseline["status"]=="S277_TRUE_OFFICIAL_FREEZE_ON_DATASET3_FIVE_WAV_UNLABELLED"
         and baseline["sealed_original_predictions_SHA256"]==file_sha(s277)
         and baseline["count_WAV"]==5
         and not baseline["XML_MIDI_annotated_contents_read"],
         "S487 original five IDMT source SHA and MIDI-blind selection required")
    selected=sel["candidate_groups"]["dataset3_first"]
    proofs={p["WAV_filename_SHA256"]:p for p in baseline["original_audio_fingerprints"]}
    need(len(selected)==len(proofs)==5,"S487 original recording keys invalid")
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict,Model
    m=Model(ICASSP_2022_MODEL_PATH)
    req=urllib.request.Request(API,headers={"Accept":"application/json",
                 "User-Agent":"S487-pretrained-BasicPitch-original-IDMT-WAV-NOXML"})
    with urllib.request.urlopen(req,timeout=60) as response:
        src=audit(json.load(response))
    reader=ReliableReadOnlyRangeZIP(src["source_url"],src["size_bytes"])
    notes=[];summaries=[]
    out.mkdir(parents=True)
    with zipfile.ZipFile(reader,"r") as original:
        need(sha("\n".join(sorted(original.namelist())).encode())==
             sel["zip_central_names_sha256"],
             "S487 original preregistered real IDMT ZIP directory SHA has drifted")
        for i,item in enumerate(selected):
            ident=item["wav_filename_sha256"]
            need(ident in proofs
                 and sha(item["wav"].encode())==ident,
                 "S487 source ZIP WAV filename was not immutable")
            raw=original.read(item["wav"])
            need(sha(raw)==proofs[ident]["original_WAV_SHA256"],
                 "S487 original guitar recording bytes differ from S277")
            wave,meta=decode_external_wav(raw)
            need(len(wave)>=2*SR,
                 "S487 source genuine guitar recording shorter than 2s")
            clip=wave[:10*SR]
            temp=out/f"NEVER_UPLOAD_guitar_{i:02d}.wav"
            temp.write_bytes(pcm(clip))
            try:
                _,_,events=predict(str(temp),m,onset_threshold=.5,
                    frame_threshold=.3,minimum_note_length=127.7)
            finally:
                temp.unlink(missing_ok=True)
            accepted=0
            for st,en,pitch,score,bend in events:
                pp=int(pitch)
                if not (40<=pp<=88 and 0<=float(st)<10 and float(en)>float(st)):
                    continue
                notes.append(dict(WAV_filename_SHA256=ident,
                    predicted_onset_s=float(st),predicted_offset_s=float(en),
                    pitch_MIDI=pp,note_name=name(pp),
                    pretrained_note_score=float(score)))
                accepted+=1
            summaries.append(dict(WAV_filename_SHA256=ident,
                original_SHA=proofs[ident]["original_WAV_SHA256"],
                predicted_named_note_events=accepted,
                original_pitch_track_count=len(events),
                preannotation_original_WAV_S277_format=meta))
            print("S487 GUITAR SOURCE",json.dumps(dict(
                source=i,native_basic_pitch_events=len(events),supported=accepted)),
                flush=True)
    path=out/"S487_BASIC_PITCH_FIVE_TRUE_IDMT_GUITAR_PREXML_NOTES.jsonl"
    path.write_text("".join(json.dumps(x)+"\n" for x in notes))
    report=dict(status="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML",
        source="IDMT_SMT_GUITAR_V2_dataset3_5_WAV_10s_original",
        truly_original_WAV_SHA_verified=True,
        original_source_WAV_count=5,
        pretrained_architecture="Spotify Basic Pitch 0.4.0",
        model_no_IDMT_truth_training_or_postprocessing_tuning=True,
        model_default_onset_threshold=.5,
        model_default_frame_threshold=.3,
        model_default_minimum_note_ms=127.7,
        output_note_names_rather_than_MIDI_file=True,
        original_XML_NOT_READ_before_note_freeze=True,
        total_supported_pitch_notes=len(notes),
        per_original_recording=summaries,
        all_output_note_events_sha256=file_sha(path),
        original_S80_and_live_unchanged=True,
        no_promotion=True,
        SHA256=dict(S276= file_sha(s276),
                    S277_original_WAV_ledger=file_sha(s277_report)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S487 GUITAR NOTE FREEZE",json.dumps(dict(
        originalWAVs=5,note_names=len(notes),annotation_read=False)),flush=True)
def selftest():
    need(name(52)=="E3" and name(55)=="G3" and name(59)=="B3",
         "S487 user desired actual note name conversion failed")
    print("S487 SELFTEST PASS E3 G3 B3 original source guitars")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s276","s277","s277-report","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s276","s277","s277_report","out"):
            need(getattr(a,k)is not None,"S487 missing "+k)
        run(a.s276,a.s277,a.s277_report,a.out)
