"""S234: derive original S97 acoustic views on official freeze native IDMT event IDs.

Uses only the immutable S233 official freeze_local_combo native event
start positions, exact S231 SHA256-frozen WAV files, and original
acoustic extraction (58+42x49, flow245, morphology490, physics294).
No XML/MIDI, no model training or model predictions. Does not invent
original event end samples or 46D ownership geometry (needed for S5),
therefore DOES NOT claim complete S80 source compatibility.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit as frozen_source
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s231_IDMT_rank33to48_seal_audio_only import select_wavs
from scripts.s178_idmt_audio_only_native_s35_heads import decode_wav
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
BAD={'true_K','truth','label','labels','target','midi','correct','accuracy','regression','y'}
def req(v,m):
    if not v:raise ValueError(m)
def sha(blob):return hashlib.sha256(blob).hexdigest()
def process(s233file,s231file,out):
    req(not out.exists(),'S234 reference-audio features must remain immutable')
    pre=json.loads(s231file.read_text())
    req(pre['this_cohort_SHA256_filename_rank_start']==33 and
        pre['this_cohort_SHA256_filename_rank_end']==48 and
        not pre['XML_note_annotation_content_read'] and
        not pre['MIDI_content_read'],
        'not the presealed truly unlabelled IDMT rank 33-48 set')
    with np.load(s233file,allow_pickle=False) as z:
        fields={'row_id','member','start_samples','incumbent_K','incumbent_probabilities','nearest_S232_V84_onset_distance_samples'}
        req(set(z.files)==fields and not BAD.intersection(z.files),
            'S233 original freeze reference native prediction fields changed')
        rid=z['row_id'].astype(str)
        member=z['member'].astype(str)
        start=z['start_samples'].astype(np.int64)
        incumbent=z['incumbent_K'].astype(np.int8)
    req(len(start)>0 and len(set(rid))==len(rid) and
        len(member)==len(start) and np.isin(incumbent,range(7)).all() and
        all(r==m+':'+str(int(s)) for r,m,s in zip(rid,member,start)),
        'S233 official event ID association mismatch')
    filelist=pre['chosen_track_provenance']
    req(len(filelist)==16,'source 16 song provenances missing')
    original_shas=[f['WAV_name_SHA256'] for f in filelist]
    req(set(member).issubset(original_shas),
        'S233 event references unknown S231 audio recording')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         'Accept':'application/json','User-Agent':'S234-original-native-reference-acoustic-only'}),
         timeout=50) as response:
        src=frozen_source(json.load(response))
    reader=HttpRangeZIP(src['source_url'],src['size_bytes'])
    summary=np.empty((len(start),58),np.float32)
    sequence=np.empty((len(start),42,49),np.float32)
    seen=np.zeros(len(start),bool)
    with zipfile.ZipFile(reader,'r') as archive:
        filenames=select_wavs(archive.namelist())
        req([sha(name.encode()) for name in filenames]==original_shas,
            'file selection SHA256 changed')
        for filename,identity in zip(filenames,original_shas):
            ii=np.flatnonzero(member==identity)
            if not len(ii):continue
            payload=archive.read(filename)
            expected=filelist[original_shas.index(identity)]['recording_SHA256']
            req(sha(payload)==expected,'original S231 sealed WAV SHA256 changed')
            audio=decode_wav(payload)
            for i in ii:
                s=int(start[i])
                req(int(.14*44100)<s<len(audio)-int(.2*44100),
                    'real original native event outside audio')
                spectrum,times=_frames(audio,s)
                props,activ=features_from_spectrogram(
                       spectrum,times,return_states=True)
                req(len(props)==58 and activ.shape==(42,49),
                    'original historical S97 audio features incompatible')
                summary[i]=[props[k] for k in sorted(props)]
                sequence[i]=np.log1p(activ)
                seen[i]=True
            print(json.dumps({'file_SHA256':identity,'reference_native_event_count':len(ii)}),flush=True)
    req(seen.all() and np.isfinite(summary).all() and np.isfinite(sequence).all(),
        'one or more true S233 original reference events missing acoustic views')
    flow=flow_profile(sequence)
    morphology,physics=audio_temporal_views(sequence)
    req(flow.shape==(len(start),245) and
        morphology.shape==(len(start),490) and physics.shape==(len(start),294),
        'original S97 acoustic 245/490/294 geometry drift')
    out.mkdir(parents=True)
    dest=out/'S234_original_S97_audio_views_on_real_freeze_native_events.npz'
    np.savez_compressed(dest,row_id=rid,member=member,start_samples=start,
        incumbent_freeze_K=incumbent,
        summary58=summary,sequence42x49=sequence,flow245=flow,
        morphology490=morphology,physics294=physics)
    report=dict(status='S234_NATIVE_REFERENCE_EVENT_S97_SIGNAL_VIEWS_ONLY',
        original_S231_unlabelled_audio_run=38299545404,
        original_S233_official_baseline_run=38301633310,
        original_unseen_song_SHA256_ranks='33-48',
        original_official_freeze_native_event_count=len(start),
        original_58D_summary=True,original_42x49_sequence=True,
        original_245D_flow=True,original_490D_morphology=True,
        original_294D_physics=True,
        output_SHA256=sha(dest.read_bytes()),
        source_original_freeze_K_included_for_row_alignment_only=True,
        no_S5_owned_527D_without_original_geometry46=True,
        exact_original_geometry46_not_yet_available=True,
        original_S80_all_models_unseen_complete=False,
        annotation_XML_read=False,annotation_MIDI_read=False,
        original_training_labels_refit=False,
        independent_exactK_measured=False,
        official_S80_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s233','s231','out'):p.add_argument('--'+name,required=True,type=Path)
    x=p.parse_args()
    process(x.s233,x.s231,x.out)
