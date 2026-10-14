"""S363: first prospective IDMT Dataset1 24 chord WAV Exact-K audit.

FIRST read of 12 XML annotations ONLY AFTER original source S358 and
both COMPLETE genuine S80 S352 variants are SHA sealed by S353.
Two different PREDECLARED target protocols are reported SEPARATELY:
(a) original GuitarSet native 20ms XML note-attacks per V88 group;
(b) physical PCM transient anchors S333, with unknown/unanchored
events ABSTAINING rather than fabricated K0.

Never train/refit after labels. Group-novelty is only filename-derived
and not certified composition-disjoint. Promotion remains gated.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s327_postfreeze_historical_attackK_IDMT_dataset2_964 import ledger,bootstrap
from scripts.s333_prelabel_physical_attack_reference_policy import label_audio_anchored_attacks
from scripts.s302_official_V88_dataset2_rank49plus_16_WAV_source_only import decode_external_wav
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
SR=44100
NATIVE={'row_id','member','start_samples','end_samples','incumbent_K','incumbent_probabilities'}
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def read(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'only original fully SHA sealed source model fields accepted')
        return {k:z[k].copy() for k in z.files}
def precheck(s276,s332,s332report,s333,s353,old,new):
    inventory=json.loads(s276.read_text())
    baseline=json.loads(s332report.read_text())
    geometry=json.loads(s333.read_text())
    frozen=json.loads(s353.read_text())
    need(inventory['status']=='S357_IDMT_DATASET1_24_NEW_CHORD_FAMILY_FILENAME_ONLY_PRELABEL'
         and inventory['all_XML_annotation_contents_read'] is False
         and baseline['count_WAV']==24
         and baseline['sealed_original_predictions_SHA256']==sha_file(s332)
         and baseline['XML_MIDI_annotated_contents_read'] is False
         and geometry['status']=='S333_PREXML_PHYSICAL_ATTACK_K_LABEL_POLICY_VERSION1'
         and geometry['test_data_XML_already_opened'] is False
         and geometry['source_model_predictions_never_used_to_select_audio_anchors']
         and frozen['status']=='S353_TWO_COMPLETE_S80_S274_CANDIDATES_PRELABEL_IDENTICAL_967_V88_EVENTS'
         and frozen['native_events']==967 and frozen['WAV_count']==24
         and frozen['original_baseline_freeze_SHA256']==sha_file(s332)
         and frozen['original_S80_SHA256']==sha_file(old)
         and frozen['S274_twofold_S80_SHA256']==sha_file(new)
         and frozen['truth_XML_or_MIDI_read'] is False
         and frozen['all_results_on_exact_same_original_V88_native_events'],
         'S363 original model source, XML-unread and policy precommit not SHA-verified')
    chosen=inventory['next_twentfour']
    need(len(chosen)==24 and len(set(x['wav_filename_sha256'] for x in chosen))==24
         and all(x['annotation_path_name_only'] for x in chosen),
         'prospective dataset1 chord cohort original WAV/XML identities changed')
    return inventory,baseline,geometry,frozen,chosen
def pair(y,old,new,mask,groups):
    return dict(
       all_eligible=ledger(y,old,new,mask),
       positive_K1plus=ledger(y,old,new,mask&(y>=1)),
       poly_K2plus=ledger(y,old,new,mask&(y>=2)),
       by_K={f'K{k}':ledger(y,old,new,mask&(y==k)) for k in range(7)},
       CI_positive_by_name_group=bootstrap(y,old,new,groups,mask&(y>=1)),
       CI_poly_by_name_group=bootstrap(y,old,new,groups,mask&(y>=2)),
       changed_native_rows=int(np.count_nonzero(old!=new)))
def run(s276,s332,s332report,s333,s353,oldfile,newfile,out):
    need(not out.exists(),'S363 immutable first prospective chord XML run')
    inventory,baseline,geometry,frozen,selected=precheck(
       s276,s332,s332report,s333,s353,oldfile,newfile)
    native=read(s332,NATIVE);old=read(oldfile,PRED);new=read(newfile,PRED)
    n=len(native['row_id'])
    need(n==967 and all(np.array_equal(native['row_id'],x['row_id']) for x in (old,new))
         and all(np.array_equal(native['incumbent_K'],x['incumbent_freeze_K'])
                for x in (old,new)),
         'the three competing predictions are not on original 967 V88 event IDs')
    y_hist=np.full(n,-1,dtype=np.int16)
    y_phys=np.full(n,-1,dtype=np.int16)
    composition=np.empty(n,dtype='<U128')
    seen=np.zeros(n,bool)
    official=native['incumbent_K'].astype(np.int8)
    c_old=old['s80_K'].astype(np.int8)
    c_new=new['s80_K'].astype(np.int8)
    snapshots={x['WAV_filename_SHA256']:x for x in baseline['original_audio_fingerprints']}
    need(len(snapshots)==24,'original 12 native WAV fingerprints missing')
    # FIRST XML BYTES READ: all reference, model, cohort and precommitted
    # physical alignment contract SHAs are checked above.
    with urllib.request.urlopen(urllib.request.Request(API,headers={
        'Accept':'application/json','User-Agent':'S363-first-Dataset1-XML-after-S353-SHA-sealed'}),timeout=60) as response:
        source=audit(json.load(response))
    archive=HttpRangeZIP(source['source_url'],source['size_bytes'])
    rows=[]
    with zipfile.ZipFile(archive,'r') as z:
        need(sha('\n'.join(sorted(z.namelist())).encode())==
             inventory['central_directory_sha256'],
             'IDMT unchanged official Zip listing mismatches prelabel S357')
        for item in selected:
            member=item['wav_filename_sha256']
            ix=np.flatnonzero(native['member'].astype(str)==member)
            need(len(ix)>0 and member in snapshots,'one prospective chord WAV absent')
            ordered=ix[np.argsort(native['start_samples'][ix],kind='stable')]
            starts=native['start_samples'][ordered]
            ends=native['end_samples'][ordered]
            raw=z.read(item['wav'])
            need(sha(raw)==snapshots[member]['original_WAV_SHA256'],
                 'original audio waveform sha256 different from S332')
            wave,meta=decode_external_wav(raw)
            need(meta==snapshots[member]['waveform_preparation'],
                 'S363 uses wrong original stereo/downmix/resample')
            annotation=z.read(item['annotation_path_name_only'])
            notes=parse_original_pitch_events(annotation)
            onset=np.asarray([float(note[0]) for note in notes],np.float64)
            within=(onset>=0)&(onset<min(10.0,len(wave)/SR))
            ignored_outside_clip=int(np.count_nonzero(~within))
            onset=onset[within]
            onset_samples=np.rint(onset*SR).astype(np.int64)
            historical,assigned=assign_onsets(onset_samples,starts,ends)
            physical,stats,_=label_audio_anchored_attacks(wave[:min(len(wave),SR*10)],
                                        onset,starts,ends)
            y_hist[ordered]=np.minimum(6,historical)
            # y_phys[i]=-1 if there is any unanchored expected note in
            # this WAV: absent confident evidence MUST NOT create K0.
            if stats['unanchored_unknown'] or stats['anchored_V88_unmatched']:
                y_phys[ordered]=-1
            else:
                y_phys[ordered]=np.minimum(6,physical)
            seen[ordered]=True
            root=PurePosixPath(item['wav']).stem.lower()
            composition[ordered]=root
            rows.append(dict(wav_filename_SHA256=member,
                XML_SHA256=sha(annotation),
                original_WAV_sha256=sha(raw),
                original_native_V88_group_count=len(ix),
                XML_note_count=len(notes),
                XML_note_onsets_outside_audio_excerpt=ignored_outside_clip,
                historical_attacks_assigned=int(np.count_nonzero(assigned>=0)),
                historical_attacks_missed=int(np.count_nonzero(assigned<0)),
                historical_attack_poly_groups=int(np.count_nonzero(historical>=2)),
                acoustic_anchor_evidence=stats,
                physical_score_whole_wav_eligible=bool(np.all(y_phys[ordered]>=0))))
    need(seen.all() and np.isin(y_hist,range(7)).all() and
         np.isin(y_phys,range(-1,7)).all(),
         'prospectively frozen truth evaluator misaligned native chord event IDs')
    all_hist=np.ones(n,bool)
    reliable_phys=y_phys>=0
    pairs={'S80_original_vs_official':(official,c_old),
           'S80_twofold_vs_official':(official,c_new),
           'S80_twofold_vs_S80_original':(c_old,c_new)}
    metrics={}
    for name,(baseline_K,new_K) in pairs.items():
        metrics[name]=dict(
            historical_XML20ms_if_all_attacks_annotated=pair(
                y_hist,baseline_K,new_K,all_hist,composition),
            preregistered_acoustic_peak_conditional=pair(
                y_phys,baseline_K,new_K,reliable_phys,composition))
    report=dict(
        status='S363_FIRST_POSTFROZEN_PROSPECTIVE_DATASET1_CHORDS_ATTACK_EXACTK_DUAL_PROTOCOL',
        native_V88_events=n,prospective_WAVs=24,
        S357_original_filename_selection_SHA256=sha_file(s276),
        S358_frozen_official_SHA256=sha_file(s332),
        S333_acoustic_protocol_SHA256=sha_file(s333),
        S353_both_complete_frozen_S80_SHA256=sha_file(s353),
        historical_if_XML_complete_truth_histogram=np.bincount(y_hist,minlength=7).tolist(),
        historical_if_XML_complete_poly_rows=int(np.count_nonzero(y_hist>=2)),
        physically_anchored_model_independent_rows=int(reliable_phys.sum()),
        physically_anchored_positive_K1plus_rows=int(np.count_nonzero(reliable_phys&(y_phys>=1))),
        physically_anchored_poly_K2plus_rows=int(np.count_nonzero(reliable_phys&(y_phys>=2))),
        comparisons=metrics,per_WAV_source_proofs=rows,
        source_models_unchanged=True,source_predictions_all_frozen_before_XML=True,
        XML_note_transcription_exhaustiveness_verified=False,
        negative_K0_truth_independently_certified=False,
        composition_song_disjointness_from_original_training_proven=False,
        acoustic_anchor_is_NOT_original_historical_K_definition=True,
        no_new_model_or_threshold_fit=True,
        official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'S363_postfrozen_967_two_distinct_protocols.npz',
       row_id=native['row_id'],member=native['member'],
       historical_attack_K_if_XML_complete=y_hist,
       acoustic_anchored_K_or_unknown=y_phys,
       official_freeze_K=official,S80_original_K=c_old,S80_twofold_K=c_new)
    for name,m in metrics.items():
        print(json.dumps({name:dict(
            original=m['historical_XML20ms_if_all_attacks_annotated']['all_eligible'],
            original_poly=m['historical_XML20ms_if_all_attacks_annotated']['poly_K2plus'],
            acoustic_positive=m['preregistered_acoustic_peak_conditional']['positive_K1plus'],
            acoustic_poly=m['preregistered_acoustic_peak_conditional']['poly_K2plus'])},indent=2),flush=True)
    print(json.dumps({k:v for k,v in report.items()
       if k not in ('comparisons','per_WAV_source_proofs')},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s276','s332','s332report','s333','s353','old','new','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.s276,a.s332,a.s332report,a.s333,
        a.s353,a.old,a.new,a.out)
