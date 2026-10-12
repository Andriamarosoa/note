"""S185 second, filename-disjoint IDMT guitar audio cohort, frozen 7 S35 experts + S179 group adapter.

This was defined after viewing S183 development predictions. No original
weights / threshold / policy refitting and no new annotations. The 10 new WAV
file identities are filename-hash ranks 11-20, disjoint from original 1-10.
This is not a measured independent Exact-K accuracy or S80 equivalence.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s178_idmt_audio_only_native_s35_heads import (
    decode_wav,acoustic_onsets,group_wav_names,views,predict_original_s35,sha_bytes)
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s179_s35_real_experts_255_new_groups import build as frozen_new_255

def need(x,m):
    if not x:raise ValueError(m)
def run(models,out,tracks=10,offset=10,max_events=10):
    need(not out.exists(),'refuse overwrite')
    need(tracks==10 and offset==10,'second heldout acoustic corpus frozen first ten excluded')
    with urllib.request.urlopen(urllib.request.Request(
      API,headers={'Accept':'application/json','User-Agent':'S185-predeclared-audio-holdout/1.0'}),
      timeout=60) as response:
        source=audit(json.load(response))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    names=[];moments=[];descriptors=[];seq=[];manifest=[]
    with zipfile.ZipFile(reader,'r') as z:
        first20,total=group_wav_names(z,20)
        original=first20[:10];nextten=first20[10:20]
        need(len(original)==len(nextten)==10 and not set(original)&set(nextten),
             'nonoverlap independent validation WAV identity')
        for name in nextten:
            info=z.getinfo(name)
            need(44100*2<=info.file_size<=44100*120*3,'source WAV encoding/duration')
            payload=z.read(name) # only WAV, ZipFile verifies original CRC
            samples=decode_wav(payload)
            selected=acoustic_onsets(samples,max_events)
            for sample in selected:
                spec,time=_frames(samples,int(sample))
                d,activated=features_from_spectrogram(spec,time,return_states=True)
                need(len(d)==58 and activated.shape==(42,49),'true S97 original feature dimensions')
                names.append(name);moments.append(int(sample))
                descriptors.append([d[k] for k in sorted(d)])
                seq.append(np.log1p(activated))
            manifest.append({'source_file_name_sha256':sha_bytes(name.encode()),
              'WAV_SHA256':sha_bytes(payload),'original_ZIP_CRC32':hex(info.CRC),
              'onsets_selected':len(selected)})
            print(json.dumps({'file_sha256':sha_bytes(name.encode()),
                              'audio_only_onsets':len(selected)}),flush=True)
    need(len(seq)>=10,'no available independent WAV onset evidence')
    X=np.asarray(descriptors,np.float32);S=np.asarray(seq,np.float32)
    flow=flow_profile(S)
    morphology,physics=audio_temporal_views(S)
    scores,weights=predict_original_s35(models,views(X,flow,morphology,physics))
    ids=np.asarray([sha_bytes((n+':'+str(t)).encode()) for n,t in zip(names,moments)])
    need(len(set(ids.tolist()))==len(ids),'sample identity collision')
    out.mkdir(parents=True)
    original_npz=out/'S185_second_IDMT_S35_audio_only.npz'
    head_ids=tuple(scores)
    np.savez_compressed(original_npz,row_id=ids,
        summary58=X,flow245=flow,morphology490=morphology,physics294=physics,
        sequence42x49=S,source_head_ids=np.asarray(head_ids),
        source_probabilities=np.stack([scores[h] for h in head_ids],axis=1),
        source_predicted_K=np.column_stack([scores[h].argmax(1) for h in head_ids]).astype(np.int8))
    frozen_new_255(original_npz,out/'frozen_new_255')
    second=json.loads((out/'frozen_new_255/report.json').read_text())
    parent=second['new_parent_prediction_histogram_K0to6']
    report={'status':'S185_DISJOINT_IDMT_SECOND_TEN_UNSEEN_WAV_FROZEN_S35_S179_LABEL_BLIND',
      'source_original_checksum_MD5_published_not_recalculated':source['expected_md5'],
      'source_archive_size_bytes':source['size_bytes'],
      'source_remote_ZIP_range_bytes':reader.network_bytes,
      'source_range_requests':reader.range_requests,
      'prior_S183_first_ten_track_names_excluded_by_deterministic_hash_ranks':True,
      'disjoint_source_WAV_count':len(manifest),
      'original_corpus_dataset4_available_WAVs':total,
      'second_wav_sha256_evidence':manifest,
      'audio_only_events':len(ids),
      'source_original_trained_pretrained_S35_models':len(weights),
      'predeclared_original_S35_expert_heads':list(head_ids),
      'source_7_expert_original_trained_weights_unmodified':True,
      'frozen_S179_new_eight_votes_255_group_rule_applied_as_is':True,
      'parent_prediction_histogram_K0to6':parent,
      'frozen_S179_K2_K3_K4_eligible_audio_event_count':second['new_S35_expert_poly_eligible_rows'],
      'old_historical_255_group_producer_equivalence':False,
      'unseen_external_audio_truth_MIDI_XML_read':False,
      'new_source_original_WAV_redistributed':False,
      'actual_freeze_local_combo_reference_run':False,
      'independent_ExactK_measured':False,
      'original_S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
      'status','source_remote_ZIP_range_bytes','disjoint_source_WAV_count',
      'audio_only_events','parent_prediction_histogram_K0to6',
      'frozen_S179_K2_K3_K4_eligible_audio_event_count',
      'original_S80_promoted')},indent=2))
def selftest():
    names=[str(i) for i in range(20)]
    a,b=names[:10],names[10:20]
    need(len(set(a)&set(b))==0,'second ten files same as development')
    print('S185 PASS unchanged frozen S35/S179 selected source rule and disjoint 10 later filename hashes')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.models and a.out,'original checkpoints and new cohort output required')
        run(a.models,a.out)
