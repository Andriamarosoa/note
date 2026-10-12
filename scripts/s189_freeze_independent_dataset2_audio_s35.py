"""S189 prelabel-free frozen inference on IDMT dataset2 note-XML paired independent WAV files.

Select first 16 filename SHA256 ranks out of dataset2/audio; predictions and
matching policy are frozen BEFORE ANY XML annotation is read (S190).
Original S35 learned heads and S179 new 255-group adapter are unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s178_idmt_audio_only_native_s35_heads import (
    decode_wav,acoustic_onsets,views,predict_original_s35,sha_bytes)
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s179_s35_real_experts_255_new_groups import build as frozen_new_255
EVALUATION_PROTOCOL={
 'labels_XML_read_only_in_separate_later_S190':True,
 'sample_rate':44100,
 'event_join':'nearest XML event onset by absolute seconds; no reused label onsets within a track',
 'max_onset_difference_seconds':0.080,
 'tie_breaker':'earliest XML onset index',
 'ground_truth_cardinality':'count distinct active MIDI pitches at matched XML onset; do not derive K from chord names',
 'max_k_class':6,
 'if_annotation_missing_or_ambiguous':'do not score this predicted onset; report coverage',
 'scope':'individual acoustic onsets, not entire song',
 'eligibility':'all selected WAVs and all selected audio-only onsets fixed before labels',
 'S80_official_reference_required_for_promotion':True,
}
def need(q,m):
    if not q:raise ValueError(m)
def run(models,out):
    need(not out.exists(),'refuse to overwrite frozen pre-label benchmark')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
       'Accept':'application/json','User-Agent':'S80-S189-independent-note-audio'}),timeout=45) as r:
        source=audit(json.load(r))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    features=[];seq=[];name=[];sample_index=[];track_summary=[]
    with zipfile.ZipFile(reader,'r') as z:
        candidates=[n for n in z.namelist() if '/dataset2/audio/' in ('/'+n.lower())
                    and n.lower().endswith('.wav')]
        candidates.sort(key=lambda x:sha_bytes(x.encode()))
        selected=candidates[:16]
        need(len(selected)==16 and len(set(selected))==16,'pre-registered sixteen audio sources absent')
        for member in selected:
            info=z.getinfo(member)
            need(44100*2<=info.file_size<=44100*120*3,'unexpected WAV duration/encoding')
            payload=z.read(member)  # only WAV; verify CRC32
            x=decode_wav(payload)
            onsets=acoustic_onsets(x,max_events=8)
            for onset in onsets:
                spec,offset=_frames(x,int(onset))
                desc,act=features_from_spectrogram(spec,offset,return_states=True)
                need(len(desc)==58 and act.shape==(42,49),'original S97 numeric contract failed')
                features.append([desc[k] for k in sorted(desc)])
                seq.append(np.log1p(act))
                name.append(member);sample_index.append(int(onset))
            track_summary.append({'member':member,'WAV_SHA256':sha_bytes(payload),
              'selected_onsets':len(onsets),'source_file_crc32':hex(info.CRC)})
            print(json.dumps({'track_SHA256':sha_bytes(member.encode()),'events':len(onsets)}),flush=True)
    need(len(features)>=16,'insufficient frozen note-level audio samples')
    X=np.asarray(features,np.float32);S=np.asarray(seq,np.float32)
    flow=flow_profile(S)
    morphology,physics=audio_temporal_views(S)
    source_scores,checkpoints=predict_original_s35(models,views(X,flow,morphology,physics))
    names=tuple(source_scores)
    ids=np.asarray([sha_bytes((n+':'+str(t)).encode()) for n,t in zip(name,sample_index)])
    need(len(set(ids.tolist()))==len(ids),'selected event identity collision')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S189_frozen_original_S35_dataset2_audio_only.npz',
       row_id=ids,summary58=X,flow245=flow,morphology490=morphology,
       physics294=physics,sequence42x49=S,source_head_ids=np.asarray(names),
       source_probabilities=np.stack([source_scores[h] for h in names],axis=1),
       source_predicted_K=np.column_stack([source_scores[h].argmax(1) for h in names]).astype(np.int8))
    np.savez_compressed(out/'S189_frozen_event_index_no_labels.npz',
       row_id=ids,source_wav_path=np.asarray(name),
       acoustic_onset_sample=np.asarray(sample_index,dtype=np.int64))
    frozen_new_255(out/'S189_frozen_original_S35_dataset2_audio_only.npz',out/'new_255_experimental')
    r=json.loads((out/'new_255_experimental'/'report.json').read_text())
    report={
       'status':'S189_FROZEN_PRELABEL_DATASET2_NOTE_XML_SOURCE_HEADS_AND_255_ADAPTED_GROUPS',
       'dataset2_independent_WAVs':len(selected),
       'audio_only_selected_onsets':len(ids),
       'S35_unmodified_genuine_original_checkpoint_count':len(checkpoints),
       'S179_new_255_group_formula_unmodified':True,
       'experimental_poly_branch_eligible_events':r['new_S35_expert_poly_eligible_rows'],
       'source_parent_K0to6_histogram':r['new_parent_prediction_histogram_K0to6'],
       'WAV_checksum_provenance':track_summary,
       'audio_source_only_sha256':sha_bytes((out/'S189_frozen_original_S35_dataset2_audio_only.npz').read_bytes()),
       'frozen_event_index_sha256':sha_bytes((out/'S189_frozen_event_index_no_labels.npz').read_bytes()),
       'preregistered_label_matching_and_exactK_cardinality_policy':EVALUATION_PROTOCOL,
       'source_range_requests':reader.range_requests,
       'source_ZIP_bytes_transferred':reader.network_bytes,
       'XML_ground_truth_content_read':False,
       'new_WAV_bytes_uploaded':False,
       'ExactK_ground_truth_scored':False,
       'original_freeze_local_combo_baseline_run':False,
       'S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
      'status','dataset2_independent_WAVs','audio_only_selected_onsets',
      'S35_unmodified_genuine_original_checkpoint_count',
      'experimental_poly_branch_eligible_events',
      'source_parent_K0to6_histogram','XML_ground_truth_content_read')},indent=2),flush=True)
def selftest():
    sample=['b.wav','a.wav','c.wav']
    chosen=sorted(sample,key=lambda x:sha_bytes(x.encode()))
    need(chosen==sorted(chosen,key=lambda x:sha_bytes(x.encode())),'predeclared sort drift')
    need(EVALUATION_PROTOCOL['max_onset_difference_seconds']==.080,'frozen tolerance drift')
    print('S189 PASS deterministic WAV identity and 80ms frozen no-label matching policy')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.models and a.out,'original checkpoint and frozen output required')
        run(a.models,a.out)
