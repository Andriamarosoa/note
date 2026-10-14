"""S302: official original V84->V88/freezed baseline on IDMT dataset2 reserved 16 WAV.

Uses PRE-LABEL S276 ZIP member-name manifest only; downloads exactly those
five WAV, NEVER XML/MIDI. Replays unchanged S84 freeze_local_combo trained
stack. Save genuine native V88 onset+endpoints for later S275 same-row replay.
No musical-composition holdout claim or independent Exact-K from this stage.
"""
from __future__ import annotations
import argparse,hashlib,io,json,urllib.request,wave,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.s84_external_freeze_native_count import (
    proposal_args,runtime_records,runtime_stack,_load_frozen_stack,
    _predict_score_tracks,_extended_candidate_features,_clusters_window,
    CLUSTER_WINDOW_MS,_cluster_arrays,capture_timing,
    MAX_CANDIDATES,_spectral_maps_for_runtime,COVERED,
    build_model,predict,SEED)
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

SR=44100
SECONDS=10
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):
    return hashlib.sha256(b).hexdigest()
def pcm(samples):
    a=np.clip(np.round(samples*32768),-32768,32767).astype('<i2')
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
        w.writeframes(a.tobytes())
    return buffer.getvalue()

def decode_external_wav(raw):
    """Accommodate dataset2 reserve WAV layouts without inspecting annotations."""
    frequency,signal=wavfile.read(io.BytesIO(raw))
    native_channels=signal.shape[1] if signal.ndim==2 else 1
    need(native_channels in (1,2) and 20000<=int(frequency)<=96000,
         "unsupported external WAV sample rate/channel count")
    source_dtype=str(signal.dtype)
    if signal.dtype==np.int16:
        signal=signal.astype(np.float32)/32768.
    elif signal.dtype==np.int32:
        signal=signal.astype(np.float64)/2147483648.
    elif signal.dtype==np.uint8:
        signal=(signal.astype(np.float32)-128.)/128.
    elif np.issubdtype(signal.dtype,np.floating):
        signal=signal.astype(np.float64)
    else:
        raise ValueError("unknown external WAV PCM sample depth "+source_dtype)
    if native_channels>1:
        signal=signal.mean(axis=1)
    if frequency!=SR:
        g=gcd(int(frequency),SR)
        signal=resample_poly(signal,SR//g,int(frequency)//g)
    signal=np.clip(np.asarray(signal,dtype=np.float32),-1.0,1.0)
    need(len(signal)>=int(.35*SR) and np.isfinite(signal).all(),
         "invalid or too short normalized mono 44.1kHz waveform")
    meta=dict(original_sample_rate_hz=int(frequency),
              original_channels=int(native_channels),
              original_dtype=source_dtype,
              mono_downmix_applied=bool(native_channels>1),
              resampling_to_44100_applied=bool(int(frequency)!=SR))
    return signal,meta

def run(source_dir,weights,s276,out):
    need(not out.exists(),"S302 must not overwrite immutable original reference")
    seal=json.loads(s276.read_text())
    need(seal["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and seal["counts_WAV"]["dataset2"]==261
         and seal["XML_MIDI_annotation_contents_read"] is False
         and seal["WAV_file_contents_read"] is False
         and seal["promotion_permitted"] is False,
         "S302 requires original label-free S276 prospective reserve selection")
    selected=seal["candidate_groups"]["dataset2_unexposed_XML_paired_reserve"]
    need(len(selected)==16 and all(a["annotation_path_name_only"] for a in selected),
         "sixteen originally preselected dataset2 reserve WAVs must have by-name XML pairs")
    names=[a["wav"] for a in selected]
    ids=[a["wav_filename_sha256"] for a in selected]
    need(len(set(names))==len(set(ids))==16 and
         all(sha(name.encode())==identity for name,identity in zip(names,ids)),
         "S276 recorded WAV identities changed")
    need(weights.is_file(),"true unchanged freeze_local_combo checkpoint required")
    need(not set(ids).intersection(seal["dataset2_previous_48_wav_names_sha256"]),
         "S302 must not reuse first 48 already exposed Dataset2 WAVs")
    need(all("/dataset2/audio/" in name for name in names),
         "this second test requires only Dataset2 prospective reserve WAVs")
    with urllib.request.urlopen(urllib.request.Request(
         API,headers={"Accept":"application/json","User-Agent":"S302-original-V88-dataset2-reserve-no-annotations"}),
         timeout=60) as response:
        data=audit(json.load(response))
    reader=HttpRangeZIP(data["source_url"],data["size_bytes"])
    out.mkdir(parents=True)
    localzip=out/"TEMP_SEALED_AUDIO_NEVER_UPLOAD.zip"
    mapping={}
    proof=[]
    try:
        with zipfile.ZipFile(reader,'r') as z,zipfile.ZipFile(
             localzip,'w',compression=zipfile.ZIP_STORED) as tmp:
            original_paths=set(z.namelist())
            need(all(x in original_paths for x in names),
                 "originally preregistered 16 Dataset2 WAVs missing")
            need(sha("\n".join(sorted(z.namelist())).encode())==seal["zip_central_names_sha256"],
                 "S276 ZIP central directory identity drift")
            tracks=[]
            for i,(name,hashname) in enumerate(zip(names,ids)):
                raw=z.read(name) # CRC validated on WAV only
                decoded,format_meta=decode_external_wav(raw)
                print(json.dumps(dict(source_name_SHA256=hashname,source_format=format_meta)),flush=True)
                cut=pcm(decoded[:SECONDS*SR])
                base=f'dataset2_reserve_{i:02d}_{hashname[:12]}_comp'
                member=base+'.jams'
                wav=base+'_mix.wav'
                tmp.writestr(wav,cut)
                tracks.append(SimpleNamespace(annotation_member=member,
                    audio_zip=localzip,audio_member=wav))
                mapping.update({member:hashname,wav:hashname,base:hashname})
                proof.append(dict(original_archive_filename=name,
                    WAV_filename_SHA256=hashname,
                    original_WAV_SHA256=sha(raw),
                    reencoded_tensecond_PCM_SHA256=sha(cut),
                    original_audio_duration_seconds=len(decoded)/SR,
                    audio_excerpt_seconds=min(SECONDS,len(decoded)/SR),
                    waveform_preparation=format_meta))
        args=proposal_args(source_dir)
        floor,retention,encoder86,encoder87,model88=_load_frozen_stack(args)
        scored=_predict_score_tracks(args.base_model,tracks)
        records=runtime_records(tracks,scored,floor)
        need(len(records)>0,"original pretrained native V84 found no events")
        source_features,forward=runtime_stack(records,encoder86,encoder87,model88)
        features,fused=_extended_candidate_features(source_features,forward)
        clusters=_clusters_window(records,CLUSTER_WINDOW_MS)
        seq,mask,stats,_,_,truncated=_cluster_arrays(
            clusters,{},records,features,forward,fused)
        geo=capture_timing(clusters,records,fused,MAX_CANDIDATES)
        spectrogram=_spectral_maps_for_runtime(tracks,clusters,records,window=COVERED)
        need(len(clusters)>0 and len(seq)==len(clusters) and
             np.isfinite(spectrogram).all(),"native V8 runtime invalid")
        cache=dict(sequence=np.asarray(seq,np.float32),
            mask=np.asarray(mask,np.float32),stats=np.asarray(stats,np.float32),
            spectral=np.asarray(spectrogram,np.float32),
            members=np.asarray([cluster["member"] for cluster in clusters]),
            cluster_start_samples=np.asarray(geo["cluster_start_samples"],np.int64))
        incumbent=build_model("learned_gate",SEED)
        incumbent.load_weights(weights)
        prob,K=predict(incumbent,cache,np.arange(len(clusters)))
        need(prob.shape==(len(clusters),7) and K.shape==(len(clusters),) and
             np.isin(K,range(7)).all(),"official pretrained V88 K out of bounds")
        def match_member(x):
            if x in mapping:return mapping[x]
            stem=Path(str(x)).stem
            if stem in mapping:return mapping[stem]
            possibles={h for name,h in mapping.items() if str(x).startswith(name)}
            need(len(possibles)==1,"unmapped native cluster filename")
            return possibles.pop()
        member=np.array([match_member(v) for v in cache["members"].astype(str)])
        starts=np.asarray(geo["cluster_start_samples"],np.int64)
        full=np.asarray(geo["full_candidate_samples"],np.int64)
        offsets=np.asarray(geo["full_candidate_offsets"],np.int64)
        ends=full[offsets[1:]-1]
        need(starts.shape==ends.shape and
             np.all(ends>=starts) and
             np.all(ends-starts<=int(CLUSTER_WINDOW_MS*SR/1000)+1),
             "actual native V88 end samples invalid")
        row_id=np.array([m+":"+str(int(s)) for m,s in zip(member,starts)])
        need(len(set(row_id))==len(row_id),"duplicate event IDs in external cohort")
        path=out/"S302_official_V88_dataset2_rank49plus_16_WAV_predictions.npz"
        np.savez_compressed(path,row_id=row_id,member=member,
            start_samples=starts,end_samples=ends,incumbent_K=np.asarray(K,np.int8),
            incumbent_probabilities=np.asarray(prob,np.float32))
        report=dict(status="S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED",
            count_WAV=16,origin="IDMT-SMT-GUITAR_V2/dataset2",
            excerpt_first_ten_seconds_per_recording=True,
            original_native_event_count=len(row_id),
            original_candidate_records=len(records),
            predicted_K0to6_histogram=np.bincount(K,minlength=7).tolist(),
            S276_prior_zip_name_manifest_SHA256=sha(s276.read_bytes()),
            pretrained_official_weights_SHA256=sha(weights.read_bytes()),
            native_V84_V86_floor=float(floor),
            native_V88_retention=float(retention),
            WAV_source_read_only=[{k:v for k,v in rec.items()
                if k!="original_archive_filename"} for rec in proof],
            original_audio_fingerprints=proof,
            SHA_ranks_1_to_48_excluded=True,
            selected_dataset2_prospective_rank49plus=True,
            external_WAV_mono_resample_adaptation_disclosed=True,
            sealed_original_predictions_SHA256=sha(path.read_bytes()),
            XML_MIDI_annotated_contents_read=False,
            trained_original_weights_unchanged=True,
            S275_same_native_event_predictions_computed=False,
            full_true_K0to6_and_poly_ExactK_scored=False,
            song_composition_disjoint_from_training_verified=False,
            promotion_eligible=False)
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:report[k] for k in (
          "status","original_native_event_count","predicted_K0to6_histogram",
          "sealed_original_predictions_SHA256",
          "XML_MIDI_annotated_contents_read",
          "S275_same_native_event_predictions_computed",
          "song_composition_disjoint_from_training_verified")},
          indent=2),flush=True)
    finally:
        if localzip.exists():localzip.unlink()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("sources","weights","s276","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.sources,args.weights,args.s276,args.out)
