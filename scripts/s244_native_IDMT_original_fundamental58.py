"""S244: original fundamental-58 audio bank on EXACT S233 reference-native IDMT events.

Label-blind features required by original missing-five S1 S2 S3 S4/coherent
models. Reuse the real S234 event SHA identities and S235 fixed source WAV
provenance. Original fundamentals from actual spectrogram at exact V88
timestamps; no XML/MIDI/target/held-piece-oracle, no new training.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s231_IDMT_rank33to48_seal_audio_only import select_wavs
from scripts.s178_idmt_audio_only_native_s35_heads import decode_wav
from scripts.extract_v273_harmonic_trajectory import (
    _frames,features_from_spectrogram,FUNDAMENTAL_BANK)
def need(value,msg):
    if not value:raise ValueError(msg)
def digest(buf):return hashlib.sha256(buf).hexdigest()
def open_exact(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(fields),'S244 expected exact original label-free source '+str(path))
        return {k:z[k].copy() for k in z.files}
def run(s234file,s235file,s231file,out):
    need(not out.exists(),'original unseen label-blind fundamental features immutable')
    ref=open_exact(s234file,{'row_id','member','start_samples','incumbent_freeze_K',
        'summary58','sequence42x49','flow245','morphology490','physics294'})
    source=open_exact(s235file,{'row_id','member','start_samples','end_samples',
        'incumbent_K','incumbent_probabilities',
        'nearest_S232_V84_onset_distance_samples'})
    p=json.loads(s231file.read_text())
    ids=ref['row_id'].astype(str);n=len(ids)
    need(n==1034 and np.array_equal(ids,source['row_id']) and
         np.array_equal(ref['member'],source['member']) and
         np.array_equal(ref['start_samples'],source['start_samples']) and
         np.array_equal(ref['incumbent_freeze_K'],source['incumbent_K']),
         'original 1034 reference native events and S5 parents no longer match')
    need(p['this_cohort_SHA256_filename_rank_start']==33 and
         p['this_cohort_SHA256_filename_rank_end']==48 and
         not p['XML_note_annotation_content_read'] and not p['MIDI_content_read'],
         'S244 would see exposed corpus')
    provenance=p['chosen_track_provenance']
    need(len(provenance)==16,'original 16 unseen recordings')
    waveid=[z['WAV_name_SHA256'] for z in provenance]
    fundamental=np.zeros((n,58),np.float64)
    primary=np.zeros((n,58),np.float64)
    complete=np.zeros(n,bool)
    name_list=None
    with urllib.request.urlopen(urllib.request.Request(API,headers={
          'Accept':'application/json','User-Agent':'S244-original-fundamental-only'}),
          timeout=50) as resp:
        info=audit(json.load(resp))
    archive=HttpRangeZIP(info['source_url'],info['size_bytes'])
    with zipfile.ZipFile(archive) as z:
        selected=select_wavs(z.namelist())
        need([digest(str(x).encode()) for x in selected]==waveid,
             'S244 actual original WAV selection changed')
        for filename,record in zip(selected,provenance):
            hits=np.flatnonzero(ref['member'].astype(str)==record['WAV_name_SHA256'])
            if not len(hits):continue
            wav=z.read(filename)
            need(digest(wav)==record['recording_SHA256'],
                 'original IDMT true WAV digest mismatch')
            signal=decode_wav(wav)
            for ix in hits:
                spec,t=_frames(signal,int(ref['start_samples'][ix]))
                d=features_from_spectrogram(spec,t)
                f=features_from_spectrogram(spec,t,bank=FUNDAMENTAL_BANK)
                names=sorted(d)
                need(len(names)==58 and names==sorted(f),'original harmonic/fundamental 58 schema')
                if name_list is None:name_list=names
                need(names==name_list,'original harmonic feature names drift')
                primary[ix]=[d[k] for k in names]
                fundamental[ix]=[f[k] for k in names]
                complete[ix]=True
            print(json.dumps(dict(WAV_filename_SHA256=record['WAV_name_SHA256'],
                                  event_count=int(len(hits)))),flush=True)
    need(complete.all() and np.isfinite(primary).all() and np.isfinite(fundamental).all(),
         'original fundamental views missing on native IDMT events')
    normalized=np.max(np.abs(primary-ref['summary58'])/
                       (1+np.abs(ref['summary58'])))
    scaled_deviation=np.abs(primary-ref['summary58'])/(1+np.abs(ref['summary58']))
    strongest=np.unravel_index(np.argmax(scaled_deviation),scaled_deviation.shape)
    offenders=np.flatnonzero(np.max(scaled_deviation,axis=1)>=1e-3)
    print(json.dumps(dict(S244_PRIMARY_VS_S234_FORENSIC=True,
         max_scaled_difference=float(normalized),
         max_absolute_difference=float(np.max(np.abs(primary-ref['summary58']))),
         differing_original_events_over_1e3=int(len(offenders)),
         largest_event_id=str(ids[strongest[0]]),
         largest_feature=str(name_list[strongest[1]]),
         S234_value=float(ref['summary58'][strongest]),
         S244_value=float(primary[strongest]))),flush=True)
    need(float(normalized)<1e-3,
         'S244 original primary summaries differ too much from S234 source features')
    out.mkdir(parents=True)
    extra=out/'S244_original_fundamental58_on_1034_native_IDMT.npz'
    np.savez_compressed(extra,row_id=ids,features=primary,fundamental=fundamental,
                        feature_names=np.asarray(name_list))
    s5=out/'S244_original_S5_label_free_reference.npz'
    np.savez_compressed(s5,row_id=ids,freeze_K=ref['incumbent_freeze_K'])
    report=dict(status='S244_ORIGINAL_FUNDAMENTAL58_ON_1034_UNEXPOSED_NATIVE_EVENTS',
      original_reference_events=n,song_SHA256_rank_start=33,song_SHA256_rank_end=48,
      original_feature_count=58,original_fundamental_count=58,
      original_primary_max_scaled_deviation_from_S234=float(normalized),
      original_fundamental_audio_SHA256=digest(extra.read_bytes()),
      original_S5_reference_ID_SHA256=digest(s5.read_bytes()),
      no_XML_or_MIDI_content_read=True,no_new_training=True,
      S1_S4_and_coherent_not_yet_executed=True,
      S45_S46_not_yet_executed_on_1034=True,
      independent_ExactK_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('s234','s235','s231','out'):a.add_argument('--'+k,type=Path,required=True)
    args=a.parse_args();run(args.s234,args.s235,args.s231,args.out)
