"""S305: freeze BEFORE XML the genuine historical new-attack K0..K6 protocol.

Works ONLY on the 16 IDMT dataset2 reserve WAVs originally selected by
S276 and official V88 source-only outputs frozen by S302. Reads ZERO XML
bytes and ZERO new candidate S80 predictions. This is a preregistration,
NOT a scored validation and not evidence of annotation completeness.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets,CLUSTER_SAMPLES
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(buf):return hashlib.sha256(buf).hexdigest()
def sha_file(path):return sha(Path(path).read_bytes())
def tune_hint(name):
    stem=Path(name).stem.lower()
    lick=re.search(r'lick0*(\d+)',stem)
    if lick:return 'LICK_'+lick.group(1)
    stem=re.sub(r'^[a-z]{2,3}_','',stem)
    stem=re.sub(r'_(fn|kn|mn|fs|pk|mu)$','',stem)
    return stem
def selftest():
    x=np.array([1000,4000],np.int64)
    y=np.array([1300,4200],np.int64)
    k,index=assign_onsets([1010,1070,3999,4205,9000],x,y)
    need(k.tolist()==[2,2] and index.tolist()==[0,0,1,1,-1],
         'original historical V9 attack assignment semantics changed')
    print('S305 predeclared original 20ms attack assignment selftest PASS')
def run(selection,report,source,out):
    need(not out.exists(),'do not mutate previously frozen candidate protocol')
    s=json.loads(selection.read_text())
    r=json.loads(report.read_text())
    need(s['status']=='S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED'
         and s['counts_WAV']['dataset2']==261
         and s['XML_MIDI_annotation_contents_read'] is False
         and r['status']=='S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED'
         and r['count_WAV']==16
         and r['SHA_ranks_1_to_48_excluded']
         and not r['XML_MIDI_annotated_contents_read']
         and not r['S275_same_native_event_predictions_computed']
         and not r['promotion_eligible']
         and r['S276_prior_zip_name_manifest_SHA256']==sha_file(selection)
         and r['sealed_original_predictions_SHA256']==sha_file(source),
         'new validation would not be before original source/score/selection freeze')
    with np.load(source,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','member','start_samples','end_samples',
           'incumbent_K','incumbent_probabilities'},
           'actual S302 original native V88 events expected and NO truth fields')
        d={k:z[k].copy() for k in z.files}
    n=len(d['row_id'])
    need(n==964 and len(set(map(str,d['row_id'])))==n and
         d['incumbent_probabilities'].shape==(n,7) and
         np.isin(d['incumbent_K'],range(7)).all(),
         'source-only S302 immutable event identity K0..6 out of contract')
    candidates=s['candidate_groups']['dataset2_unexposed_XML_paired_reserve']
    need(len(candidates)==16 and
         len(set(x['wav_filename_sha256'] for x in candidates))==16,
         'original S276 reserve 16 WAVs not found')
    frozen={x['WAV_filename_SHA256']:x for x in r['original_audio_fingerprints']}
    need(len(frozen)==16 and set(frozen)=={
         c['wav_filename_sha256'] for c in candidates},
         'S302 parent native model recorded different source audio')
    excluded=set(s['dataset2_previous_48_wav_names_sha256'])
    items=[]
    for row in candidates:
        h=row['wav_filename_sha256']
        need(h not in excluded and
             row['annotation_path_name_only'] and
             '/dataset2/annotation/' in row['annotation_path_name_only'].lower(),
             'already exposed track or missing XML member name')
        ix=np.flatnonzero(d['member'].astype(str)==h)
        need(len(ix)>0,'native S302 model has zero candidate events for selected WAV')
        ordered=ix[np.argsort(d['start_samples'][ix],kind='stable')]
        left=d['start_samples'][ordered]
        right=d['end_samples'][ordered]
        need(np.all(left[1:]>right[:-1]) and
             np.all(right>=left) and np.all((right-left)<=CLUSTER_SAMPLES),
             'cannot apply historical original V9 onset assignment')
        items.append(dict(
           WAV=Path(row['wav']).name,
           composition_family_name_only=tune_hint(row['wav']),
           original_SHA256_filename=h,
           original_SHA256_audio=frozen[h]['original_WAV_SHA256'],
           XML_name_precommitted=row['annotation_path_name_only'],
           source_only_native_V88_event_count=len(ix)))
    reportout=dict(
       status='S305_PRELABEL_ORIGINAL_TRAINED_20MS_ATTACK_K0K6_POLICY_SEALED',
       source_first_original_S276_WAV_manifest_sha256=sha_file(selection),
       source_original_S302_official_predict_sha256=sha_file(source),
       source_S302_report_sha256=sha_file(report),
       original_V88_native_events=n,
       source_original_S302_WAVs=16,
       source_official_original_freeze_K0to6_prediction_histogram=np.bincount(
           d['incumbent_K'],minlength=7).tolist(),
       current_exact_historical_training_K_target=
          'For each reference XML note event, use round(onsetSec*44100); assign once to closest original V88 group [start_samples,end_samples] if distance<=20ms, with earliest-group tie; target K=min(6,n_assigned_note_attacks)',
       historical_code='scripts/yourmt3_exactk_common.py::assign_onsets',
       historical_cluster_duration_samples=int(CLUSTER_SAMPLES),
       XML_pitch_offset_and_sustained_note_activity_ignored=True,
       scoring_protocol_precommitted_first=True,
       complete_XML_onsets_K0_false_negative_validation_required=True,
       corpus_music_composition_disjoint_proof_required=True,
       full_candidate_S80_predictions_must_be_sealed_before_XML=True,
       matched_80ms_S299_only_diagnostic_not_main_target=True,
       native_audio_and_annotated_prediction_joint_training_forbidden=True,
       per_file_composition_identity_from_name_only_unverified=True,
       SHA_filename_rank_1_to_48_excluded=True,
       per_WAV_preregistered=items,
       all_XML_contents_untouched=True,
       new_independent_ExactK_scored=False,
       official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(reportout,indent=2)+'\n')
    print(json.dumps({k:v for k,v in reportout.items()
           if k not in ('per_WAV_preregistered',)},indent=2),flush=True)
    print('RESERVED WAV NAMES',json.dumps([x['WAV'] for x in items]),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for name in ('selection','report','source','out'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        need(all(getattr(args,k) is not None for k in
            ('selection','report','source','out')),'required prelabel sources')
        selftest()
        run(args.selection,args.report,args.source,args.out)
