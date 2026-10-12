"""S233: genuine official freeze_local_combo baseline on unseen IDMT dataset2 ranks33-48.

Replays EXACT S84 original V84→V88 candidate stack and unmodified
freeze_local_combo.weights.h5 on 16 SHA-pinned WAVs, not the S230 proxy.
Native event clusters of this original official baseline are canonical
for any future paired S80 Exact-K score. Compare those cluster onset
samples with independent S232 native V84-only S232 proposals as an audit.
No XML/MIDI accessed; no truth; all original weights unchanged.
"""
from __future__ import annotations
import argparse,hashlib,io,json,urllib.request,wave,zipfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scripts.s84_external_freeze_native_count import (
    proposal_args,runtime_records,runtime_stack,_load_frozen_stack,
    _predict_score_tracks,_extended_candidate_features,_clusters_window,
    CLUSTER_WINDOW_MS,_cluster_arrays,capture_timing,
    MAX_CANDIDATES,_spectral_maps_for_runtime,COVERED,
    build_model,predict,SEED)
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s231_IDMT_rank33to48_seal_audio_only import select_wavs
from scripts.s178_idmt_audio_only_native_s35_heads import decode_wav
MAX_TRACKS=16
SECONDS=10
SR=44100
def need(value,msg):
    if not value:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode_pcm(samples):
    """Exactly same external int16 quantization used by S232 for native V84."""
    arr=np.clip(np.round(samples*32768.),-32768,32767).astype('<i2')
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
        w.writeframes(arr.tobytes())
    return buf.getvalue()
def run(source_dir,weight,sealed_report,s232_groups,out):
    need(not out.exists(),'official baseline must be immutable')
    pre=json.loads(sealed_report.read_text())
    need(pre['this_cohort_SHA256_filename_rank_start']==33 and
         pre['this_cohort_SHA256_filename_rank_end']==48 and
         pre['chosen_song_count']==16 and
         not pre['XML_note_annotation_content_read'],
         'baseline source must be S231 unexposed rank 33–48')
    allowed=pre['chosen_track_provenance']
    originals=[x['WAV_name_SHA256'] for x in allowed]
    need(len(set(originals))==16,'16 distinct original file identities')
    need(weight.is_file(),'real official freeze_local_combo trained checkpoint missing')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         'Accept':'application/json','User-Agent':'S233-IDMT-original-freeze-WAVonly'}),
          timeout=60) as z:
        archive=audit(json.load(z))
    reader=HttpRangeZIP(archive['source_url'],archive['size_bytes'])
    out.mkdir(parents=True)
    zipfile_out=out/'sealed_16_dataset2_PCM_external_tmp.zip'
    mapping={};track_shas=[]
    with zipfile.ZipFile(reader,'r') as remote,zipfile.ZipFile(
             zipfile_out,'w',compression=zipfile.ZIP_STORED) as local:
        selection=select_wavs(remote.namelist())
        need([sha(x.encode()) for x in selection]==originals,
             'original S231 filenames no longer match canonical benchmark')
        tracks=[]
        for index,name in enumerate(selection):
            data=remote.read(name)
            need(sha(data)==allowed[index]['recording_SHA256'],
                 'S231 sealed original external recording SHA changed')
            decoded=decode_wav(data)
            need(len(decoded)>2*SR,'original IDMT clip too short')
            processed=encode_pcm(decoded[:SECONDS*SR])
            basename=f'dataset2_{index:02d}_{originals[index][:12]}_comp'
            member=basename+'.jams'
            wav_member=basename+'_mix.wav'
            local.writestr(wav_member,processed)
            tracks.append(SimpleNamespace(annotation_member=member,
                audio_zip=zipfile_out,audio_member=wav_member))
            mapping[member]=originals[index]
            mapping[wav_member]=originals[index]
            mapping[basename]=originals[index]
            track_shas.append(dict(WAV_filename_SHA256=originals[index],
                original_WAV_SHA256=sha(data),reencoded_16bit_PCM_SHA256=sha(processed)))
    args=proposal_args(source_dir)
    floor,retention,enc86,enc87,model88=_load_frozen_stack(args)
    scored=_predict_score_tracks(args.base_model,tracks)
    records=runtime_records(tracks,scored,floor)
    need(len(records)>0,'original frozen V84/V88 produced no source records')
    view,forward=runtime_stack(records,enc86,enc87,model88)
    features,fused=_extended_candidate_features(view,forward)
    clusters=_clusters_window(records,CLUSTER_WINDOW_MS)
    seq,mask,stats,_,_,truncated=_cluster_arrays(
        clusters,{},records,features,forward,fused)
    geo=capture_timing(clusters,records,fused,MAX_CANDIDATES)
    spectral=_spectral_maps_for_runtime(tracks,clusters,records,window=COVERED)
    need(len(clusters)>0 and len(seq)==len(clusters) and
         len(geo['cluster_start_samples'])==len(clusters) and
         np.isfinite(spectral).all(),'S84 genuine original V8 full stack invalid')
    cache=dict(sequence=np.asarray(seq,np.float32),
        mask=np.asarray(mask,np.float32),stats=np.asarray(stats,np.float32),
        spectral=np.asarray(spectral,np.float32),
        members=np.asarray([c['member'] for c in clusters]),
        cluster_start_samples=np.asarray(geo['cluster_start_samples'],np.int64))
    model=build_model('learned_gate',SEED)
    model.load_weights(weight)
    probs,count=predict(model,cache,np.arange(len(clusters)))
    need(probs.shape==(len(clusters),7) and len(count)==len(clusters)
         and np.isin(count,range(7)).all() and np.isfinite(probs).all(),
         'official frozen baseline logits and ExactK impossible')
    names=cache['members'].astype(str)
    def match_member(x):
        if x in mapping:return mapping[x]
        root=Path(x).stem
        if root in mapping:return mapping[root]
        candidate=[sha for n,sha in mapping.items() if str(x).startswith(n)]
        need(len(set(candidate))==1,'native event member cannot be mapped to original WAV '+x)
        return candidate[0]
    sha_member=np.array([match_member(x) for x in names])
    times=cache['cluster_start_samples']
    uid=np.array([m+':'+str(int(s)) for m,s in zip(sha_member,times)])
    need(len(set(uid.tolist()))==len(uid),'native incumbent event ID collision')
    with np.load(s232_groups,allow_pickle=False) as z:
        need(set(z.files)=={'member','starts','ends'},'S232 native groups invalid')
        v84member=z['member'].astype(str)
        v84starts=z['starts'].copy()
    # Audit closeness to V84 proposals. Do not assume original V88 grouping
    # equal to preliminary V84-only onset candidate grouping.
    close=[]
    for member,start in zip(sha_member,times):
        other=v84starts[v84member==member]
        close.append(int(np.min(abs(other-start))) if len(other) else -1)
    deltas=np.asarray(close,int)
    match40=(deltas>=0)&(deltas<=int(.04*SR))
    np.savez_compressed(out/'S233_official_freeze_local_combo_IDMT_new_native_16songs.npz',
        row_id=uid,member=sha_member,start_samples=times,
        incumbent_K=np.asarray(count,np.int8),
        incumbent_probabilities=np.asarray(probs,np.float32),
        nearest_S232_V84_onset_distance_samples=deltas)
    # Uploaded only hashes and model predictions, never compressed raw audio.
    zipfile_out.unlink()
    report=dict(status='S233_ORIGINAL_OFFICIAL_FREEZE_ON_IDMT_SHA256_RANK33TO48',
        S231_original_audio_run=38299545404,
        S232_prior_original_native_V84_run=38301241411,
        chosen_WAV_count=MAX_TRACKS,
        original_native_V8_full_stack_used=True,
        original_freeze_local_combo_weights_used=True,
        original_incumbent_checkpoint_sha256=sha(weight.read_bytes()),
        original_V86_proposal_floor=float(floor),
        original_V88_retention_threshold=float(retention),
        original_candidate_record_count=len(records),
        original_official_native_event_count=len(clusters),
        candidate_K_histogram=np.bincount(count,minlength=7).tolist(),
        mapped_native_events_to_S232_original_V84_within_40ms=int(match40.sum()),
        unmatched_to_S232_original_V84_events=int((~match40).sum()),
        original_V84_only_groups_not_same_as_official_V88=True,
        full_baseline_prediction_immutable_SHA256=sha(
            (out/'S233_official_freeze_local_combo_IDMT_new_native_16songs.npz').read_bytes()),
        source_WAV_trust_SHA256=track_shas,
        baseline_predictions_locked_before_XML_content_read=True,
        new_audio_XML_content_read=False,
        new_audio_MIDI_content_read=False,
        new_data_model_fit=False,
        original_S80_same_event_predictions_completed=False,
        independent_paired_ExactK_scored=False,
        official_S80_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_WAV_trust_SHA256'},
      indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('sources','weights','s231','s232','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    run(a.sources,a.weights,a.s231,a.s232,a.out)
