"""S232: NATIVE ORIGINAL V84/V86 GROUPS for independently selected IDMT dataset2 WAVs.

Uses SHA256(filename) first 16 from S188/S229. Loads *actual frozen*
V84 onset-only Keras predictor and original V86 candidate floor recovered
from hashed original sources. 10 first WAV seconds, native 40ms clusters,
not provisional S229 RMS-peak events. Does NOT read XML, MIDI or any truth.
This native event set is prerequisite to a fair paired baseline/S80 test.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from scripts.s177_preregister_external_polyphony_source import API,audit as source_audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s231_IDMT_rank33to48_seal_audio_only import select_wavs
from scripts.s178_idmt_audio_only_native_s35_heads import decode_wav
from scripts.s83_external_native_v84_groups import scores_from_frozen_v84
from scripts.train_v86_state_transition_proposals import (
    _candidate_groups,RECEPTIVE_FIELD,CHUNK_SIZE)
from scripts.evaluate_v90_cardinality_realization import _clusters_window
from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_MS
MAX_TRACKS=16
SECONDS=10
MAX_NATIVE_EVENTS=30
SR=44100
def need(c,msg):
    if not c:raise ValueError(msg)
def h(data):return hashlib.sha256(data).hexdigest()
def run(source_dir,sealed_audio,out):
    need(not out.exists(),'immutable native event dataset')
    report=json.loads(sealed_audio.read_text())
    need(report['chosen_song_count']==MAX_TRACKS and
         not report['XML_note_annotation_content_read'] and
         not report['MIDI_content_read'] and
         report['deterministic_WAV_selection'].startswith('dataset2 WAV filename SHA256 ranks 33-48'),
         'S231 original unseen ranks 33-48 audio-only selection not preserved')
    pinned=report['chosen_track_provenance']
    hashes=[a['WAV_name_SHA256'] for a in pinned]
    need(len(set(hashes))==MAX_TRACKS,'duplicate sealed source files')
    cfg=json.loads((source_dir/'v86'/'report.json').read_text())
    floor=float(cfg['configuration']['candidate_floor'])
    need(cfg['configuration'].get('candidate_floor_selected_on_locked_validation') is False and
         .01<floor<.4,'frozen original V86 threshold mismatch')
    weight=source_dir/'v84'/'control.epoch-01.keras'
    need(weight.is_file(),'frozen original V84 control checkpoint absent')
    predictor=V84OnsetOnlyKerasPredictor.from_path(
         str(weight),receptive_field=RECEPTIVE_FIELD)
    predictor.warm_up(CHUNK_SIZE)
    with urllib.request.urlopen(urllib.request.Request(
          API,headers={'Accept':'application/json','User-Agent':'S232-native-unseen-WAV-only'}),
          timeout=50) as r:
        source=source_audit(json.load(r))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    records=[];members=[];starts=[];ends=[];files=[]
    with zipfile.ZipFile(reader) as z:
        selected=select_wavs(z.namelist())
        need([h(n.encode()) for n in selected]==hashes,
             'S232 changed preregistered original 16 song identities')
        for i,name in enumerate(selected):
            data=z.read(name) # WAV only; never XML/MIDI
            need(h(data)==pinned[i]['recording_SHA256'],
                 'original sealed WAV content SHA mismatch')
            decoded=decode_wav(data)
            count=min(len(decoded),SECONDS*SR)
            need(count>2*SR,'source too short for native onset proposals')
            samples=np.clip(np.round(decoded[:count]*32768.),-32768,32767).astype(np.int16)
            streams=scores_from_frozen_v84(predictor,samples)
            proposal=_candidate_groups(streams,floor)
            local=[{'member':hashes[i],'arrangement':'comp','sample':int(p['sample'])}
                   for p in proposal]
            cluster=_clusters_window(local,CLUSTER_WINDOW_MS)
            usable=[]
            for cl in cluster:
                q=[int(local[k]['sample']) for k in cl['indices']]
                if not q:continue
                start,end=min(q),max(q)
                if start<=int(.15*SR) or end>=len(samples)-int(.2*SR):continue
                usable.append((start,end))
            selected_native=usable[:MAX_NATIVE_EVENTS]
            for s,e in selected_native:
                members.append(hashes[i]);starts.append(s);ends.append(e)
            files.append(dict(file_SHA256=hashes[i],
                 wav_SHA256=h(data),duration_processed_s=count/SR,
                 original_v84_candidate_count=len(proposal),
                 original_v86_native_cluster_count=len(cluster),
                 selected_native_event_count=len(selected_native)))
            print(json.dumps(files[-1]),flush=True)
    need(len(files)==MAX_TRACKS and len(starts)>0,
         'no original V84 source events from unseen songs')
    member=np.asarray(members);start=np.asarray(starts,np.int64)
    end=np.asarray(ends,np.int64)
    ids=np.asarray([f'{m}:{int(s)}' for m,s in zip(member,start)])
    need(len(set(ids.tolist()))==len(ids) and
         np.all((start>0)&(end>=start)&(end-start<=int(.04*SR))),
         'original native K events malformed')
    out.mkdir(parents=True)
    dest=out/'IDMT_dataset2_true_V84_native_unlabeled_groups.npz'
    np.savez_compressed(dest,member=member,starts=start,ends=end)
    manifest=dict(status='S232_IDMT_DATASET2_RANK33TO48_FROZEN_ORIGINAL_V84_V86_NATIVE_AUDIO_GROUPS',
      sealed_WAV_source_run_id=38299545404,
      selected_song_count=MAX_TRACKS,
      filename_sha256_rank_start=33,filename_sha256_rank_end=48,
      prior_XML_exposed_ranks_1to32_excluded=True,
      source_md5=source['expected_md5'],
      source_original_v84_checkpoint_sha=h(weight.read_bytes()),
      source_original_v86_floor=floor,
      original_frozen_V84_V86_grouping=True,
      original_V84_first_audio_seconds=SECONDS,
      native_event_count=len(ids),min_events_one_track=min(a['selected_native_event_count'] for a in files),
      original_tracks=files,
      event_ID_SHA256=h(('\n'.join(ids.tolist())).encode()),
      event_group_NPZ_SHA256=h(dest.read_bytes()),
      notes_vs_S229='S231 provisional RMS flux event groups are NOT comparable; this new native event group dataset is required for true matched freeze vs S80',
      untouched_IDMT_WAV_hashes_match_S231=True,
      XML_content_read=False,MIDI_content_read=False,
      model_weights_trained_on_unseen_IDMT=False,
      baseline_exactK_inference_ran=False,
      full_original_S80_inference_ran=False,
      independent_paired_ExactK_measured=False,
      official_S80_promotion=False)
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items()
        if k!='original_tracks'},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('sources','s229','out'):
        ap.add_argument('--'+name,required=True,type=Path)
    a=ap.parse_args()
    run(a.sources,a.s229,a.out)
