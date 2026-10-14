"""S306: real S97 acoustic views on S302's original V88 Dataset2 native events.

Every row comes from frozen original native start and recording SHA. Audio is
the EXACT same deterministic mono-44.1k preprocessing as S302. Never opens
XML/MIDI annotations; no model fitting; no claim that S275 is scored.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s302_official_V88_dataset2_rank49plus_16_WAV_source_only import decode_external_wav
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run(s302,s302report,s276,out):
    need(not out.exists(),"S306 must not overwrite prelabel native acoustic views")
    seal=json.loads(s276.read_text())
    ref_report=json.loads(s302report.read_text())
    need(ref_report["status"]=="S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED"
         and not ref_report["XML_MIDI_annotated_contents_read"],
         "must use truly native pretrained frozen unlabelled S302 archive")
    need(sha(s302.read_bytes())==ref_report["sealed_original_predictions_SHA256"],
         "S302 immutable parent scores changed")
    with np.load(s302,allow_pickle=False) as z:
        keys={"row_id","member","start_samples","end_samples",
              "incumbent_K","incumbent_probabilities"}
        need(set(z.files)==keys,"no labels or unrecognized source fields")
        native={k:z[k].copy() for k in z.files}
    rid=native["row_id"]
    member=native["member"].astype(str)
    starts=native["start_samples"]
    n=len(rid)
    need(n==964 and all(r==m+":"+str(int(s)) for r,m,s in zip(rid,member,starts))
         and len(set(rid))==n,
         "S302 genuine native identities must be unchanged")
    selected=seal["candidate_groups"]["dataset2_unexposed_XML_paired_reserve"]
    hashes=[a["wav_filename_sha256"] for a in selected]
    frozen={p["WAV_filename_SHA256"]:p for p in ref_report["original_audio_fingerprints"]}
    need(len(selected)==16 and set(hashes)==set(frozen)
         and set(member).issubset(set(hashes)),
         "S276 WAV identities or source WAV SHA changed")
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         "Accept":"application/json",
         "User-Agent":"S306-IDMT-dataset2-original-S97-no-XML"}),timeout=60) as resp:
        source=audit(json.load(resp))
    reader=HttpRangeZIP(source["source_url"],source["size_bytes"])
    summary=np.empty((n,58),np.float32)
    seq=np.empty((n,42,49),np.float32)
    covered=np.zeros(n,bool)
    with zipfile.ZipFile(reader,"r") as z:
        need(sha("\n".join(sorted(z.namelist())).encode())==
             seal["zip_central_names_sha256"],
             "IDMT central directory changed since pre-annotation selection")
        for obj in selected:
            wavname=obj["wav"]
            ident=obj["wav_filename_sha256"]
            ii=np.flatnonzero(member==ident)
            payload=z.read(wavname) # CRC enforced
            need(sha(payload)==frozen[ident]["original_WAV_SHA256"],
                 "source WAV contents drifted since S302")
            audio,fmt=decode_external_wav(payload)
            need(fmt==frozen[ident]["waveform_preparation"],
                 "WAV normalization changed from official S302 input")
            if not len(ii):
                continue
            for i in ii:
                start=int(starts[i])
                need(0<=start<min(len(audio),44100*10),
                     "canonical V88 event not in first10s audio")
                spectrum,times=_frames(audio,start)
                d,activation=features_from_spectrogram(
                    spectrum,times,return_states=True)
                need(len(d)==58 and activation.shape==(42,49),
                     "historical S97 contract changed on new music")
                summary[i]=[d[k] for k in sorted(d)]
                seq[i]=np.log1p(activation)
                covered[i]=True
            print(json.dumps(dict(WAV_filename_sha256=ident,
                canonical_native_events=len(ii))),flush=True)
    need(covered.all() and np.isfinite(summary).all() and np.isfinite(seq).all(),
         "S302 canonical event audio features incomplete")
    flow=flow_profile(seq)
    morph,physics=audio_temporal_views(seq)
    need(flow.shape==(n,245) and morph.shape==(n,490) and
         physics.shape==(n,294),"historical original S97 feature geometry changed")
    out.mkdir(parents=True)
    dest=out/"S306_IDMT_dataset2_original_S97_on_frozen_native_events.npz"
    np.savez_compressed(dest,row_id=rid,member=member,start_samples=starts,
        incumbent_freeze_K=native["incumbent_K"],summary58=summary,
        sequence42x49=seq,flow245=flow,morphology490=morph,
        physics294=physics)
    report=dict(status="S306_ORIGINAL_AUDIO_S97_NATIVE_IDMT_DATASET2_NO_LABELS",
        events=n,WAV_count=16,
        source_S302_SHA256=sha(s302.read_bytes()),
        first_ten_seconds_only=True,
        S5_S18_source_production_not_yet_replayed=True,
        S275_same_native_event_output_not_yet_frozen=True,
        annotation_XML_MIDI_read=False,
        composition_level_independence_verified=False,
        source_original_checkpoint_training_unchanged=True,
        output_SHA256=sha(dest.read_bytes()),S80_official_promotion=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for name in ("s302","s302report","s276","out"):
        a.add_argument("--"+name,required=True,type=Path)
    x=a.parse_args();run(x.s302,x.s302report,x.s276,x.out)
