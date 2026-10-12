"""S229: preregister untouched IDMT-SMT-GUITAR dataset2 AUDIO-only events.

Reads ZIP central DIRECTORY and exactly the 16 WAV selected by the
filename SHA256 ranking preregistered in S188. NEVER reads XML, MIDI,
other text/JSON, class labels, or targets. Extracts original S97
summary58/flow245/geometry3/temporal42x49/morphology490/physics294
for deterministic audio-only acoustic attack proposals. This is NOT
the original native freeze_local_combo event proposal algorithm; may
NOT be scored as final independent Exact-K until event-matching is
established and S5..S80 is generalized to these events. Fingerprint
all input audio and events before any labels can be accessed.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit as pinned_source
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s178_idmt_audio_only_native_s35_heads import decode_wav,acoustic_onsets
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s97_external_s45_original_features import geometry
FROZEN_WAV_COUNT=16
MAX_EVENTS_PER_TRACK=12
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def select_wavs(names):
    wav=sorted((n for n in names if '/dataset2/' in ('/'+n.lower())
                and n.lower().endswith('.wav')
                and not Path(n).name.startswith('._')
                and '__MACOSX' not in n),
               key=lambda x:sha(x.encode()))
    need(len(wav)>=FROZEN_WAV_COUNT,'need 16 original IDMT dataset2 WAVs')
    return wav[:FROZEN_WAV_COUNT]
def selftest():
    names=['a/dataset2/'+f'{i}.wav' for i in range(30)]+['a/dataset2/8.xml']
    selection=select_wavs(names)
    need(len(selection)==16 and selection==select_wavs(list(reversed(names))),
         'name hash selector must be permutation invariant')
    print('S229 PASS original 16 preregistered WAV selection; XML not opened')
def run(out):
    need(not out.exists(),'no overriding preregistered external IDs')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
            'Accept':'application/json','User-Agent':'S229-IDMT-AUDIO-ONLY/1.0'}),
            timeout=50) as response:
        source=pinned_source(json.load(response))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    ids=[];members=[];starts=[];ends=[];summaries=[];sequences=[]
    tracks=[];xml_names=[]
    with zipfile.ZipFile(reader,'r') as z:
        selected=select_wavs(z.namelist())
        # Recording names and SHA fingerprints only, never the metadata/annotations content.
        for name in selected:
            info=z.getinfo(name)
            need(info.file_size<70*1024*1024,'track exceeds preregistered audio bounds')
            payload=z.read(name)
            samples=decode_wav(payload)
            if len(samples)<=20000:raise ValueError('short external recording')
            onset=acoustic_onsets(samples,MAX_EVENTS_PER_TRACK)
            need(0<len(onset)<=MAX_EVENTS_PER_TRACK,
                 'no acoustic onsets for preregistered audio '+name)
            # Prefix identity in a fixed SHA256 of the archive filename:
            # no tuneable file mapping or labels used in event selection.
            file_hash=sha(name.encode())
            for s in onset:
                spec,t=_frames(samples,int(s))
                d,activation=features_from_spectrogram(spec,t,return_states=True)
                keys=sorted(d)
                need(len(keys)==58 and activation.shape==(42,49),
                     'real original acoustic dimensions inconsistent')
                ids.append(file_hash+':'+str(int(s)))
                members.append(file_hash)
                starts.append(int(s));ends.append(int(s))
                summaries.append([d[k] for k in keys])
                sequences.append(np.log1p(activation))
            tracks.append(dict(WAV_name_SHA256=file_hash,
                               original_archive_filename=name,
                               recording_SHA256=sha(payload),
                               archive_original_CRC32=hex(info.CRC),
                               selected_label_blind_onsets=len(onset)))
            print(json.dumps({k:v for k,v in tracks[-1].items()
                     if k!='original_archive_filename'}),flush=True)
        all_xml=[n for n in z.namelist() if
               '/dataset2/' in ('/'+n.lower()) and n.lower().endswith('.xml')]
        xml_names=all_xml
    ids=np.asarray(ids);members=np.asarray(members)
    starts=np.asarray(starts,np.int64);ends=np.asarray(ends,np.int64)
    summary=np.asarray(summaries,np.float32)
    sequences=np.asarray(sequences,np.float32)
    n=len(ids)
    need(n>=16 and len(set(map(str,ids)))==n and summary.shape==(n,58)
         and sequences.shape==(n,42,49) and np.isfinite(summary).all(),
         'IDMT independent WAV attack events inconsistent')
    flow=flow_profile(sequences)
    morph,physics=audio_temporal_views(sequences)
    geom=geometry(members,starts,ends)
    need(flow.shape==(n,245) and morph.shape==(n,490) and
         physics.shape==(n,294) and geom.shape==(n,3),
         'original S97 audio-only views fail for untouched external music')
    out.mkdir(parents=True)
    dest=out/'IDMT_dataset2_unseen_audio_only_original_S97_features.npz'
    np.savez_compressed(dest,row_id=ids,summary58=summary,flow245=flow,
       geometry3=geom,sequence42x49=sequences,morphology490=morph,
       physics294=physics,member=members,start_samples=starts,end_samples=ends)
    report=dict(status='S229_IDMT_DATASET2_UNTOUCHED_AUDIO_ONLY_S97_FEATURES_PRESEALED',
      archive_record_id=source['record_id'],source_size_bytes=source['size_bytes'],
      published_archive_md5=source['expected_md5'],
      deterministic_WAV_selection='first 16 dataset2 WAV names sorted by SHA256(filename)',
      sample_max_events_per_track=MAX_EVENTS_PER_TRACK,
      total_source_dataset2_XML_files_named_only=len(xml_names),
      chosen_song_count=FROZEN_WAV_COUNT,
      chosen_track_provenance=tracks,
      total_acoustic_onset_groups=n,
      generated_original_58_245_3_42x49_490_294_views=True,
      dataset_features_sha256=sha(dest.read_bytes()),
      source_reader_fetched_range_bytes=reader.network_bytes,
      source_reader_range_requests=reader.range_requests,
      README_WARNING='These RMS/flux events are NOT the original V8 native event groups; no paired exact-K benchmark is valid until native-event compatibility is established.',
      original_S5_to_S80_new_WAV_inference_completed=False,
      original_freeze_local_combo_on_these_events=False,
      WAV_audio_accessed=True,
      XML_note_annotation_content_read=False,
      MIDI_content_read=False,
      trained_models_refitted=False,
      independent_ExactK_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()
        if k!='chosen_track_provenance'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out is not None,'requires immutable output path')
        run(a.out)
