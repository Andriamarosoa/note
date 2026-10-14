"""S382 first NEW eight WAV XML read AFTER original GuitarSet-only K3 gate SHA frozen.

Compare PREDECLARED one source-derived 1147D early K3 specialist to
original trained S18 3/4 parent, NOT to full S80 (not yet replayed).
All rows original V88 native groups, original GuitarSet onset-matcher
20ms, WAV SHA verified after original source-only S374.
True K0 only *conditional* on annotation completeness; unknown
audio-physical anchors NOT fabricated as K0. No refit or threshold edits.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.yourmt3_exactk_common import assign_onsets
def sha(raw):return hashlib.sha256(raw).hexdigest()
def shafile(p):return sha(Path(p).read_bytes())
def need(ok,msg):
    if not ok:raise ValueError(msg)
NATIVE={'row_id','member','start_samples','end_samples','incumbent_K','incumbent_probabilities'}
PRED={'row_id','member','incumbent_original_freeze_K','original_S18_foldneutral_K',
  'early_new_K3_source_only_K','eligible_source_parent_K3to2',
  'expert_K3_probability_fourfold','mean_expert_K3_probability','expert_K3_action'}
def read(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==fields,'original prelabel frozen source schema drift')
        return {k:z[k].copy() for k in z.files}
def comp(truth,ref,pred,mask):
    ix=np.flatnonzero(mask)
    a=ref[ix]==truth[ix];b=pred[ix]==truth[ix]
    return dict(n=int(len(ix)),reference_correct=int(a.sum()),candidate_correct=int(b.sum()),
         candidate_corrections=int(np.count_nonzero(~a&b)),
         candidate_regressions=int(np.count_nonzero(a&~b)),
         candidate_net=int(b.sum()-a.sum()),
         ref_rate=(float(a.mean()) if len(ix) else None),
         candidate_rate=(float(b.mean()) if len(ix) else None))
def perclass(y,b,c):
    return {f'K{k}':comp(y,b,c,y==k) for k in range(7)}
def run(s372,s374,s374report,s380,s380report,out):
    need(not out.exists(),'first label read S382 cannot be overwritten')
    manifest=json.loads(s372.read_text())
    official=json.loads(s374report.read_text())
    gate=json.loads(s380report.read_text())
    need(manifest['status']=='S372_IDMT_DATASET1_FINAL_EIGHT_CHORD_FAMILIES_FILENAME_ONLY_PRELABEL'
        and len(manifest['final_eight'])==8 and not manifest['all_XML_annotation_contents_read']
        and not manifest['all_WAV_contents_read']
        and official['status']=='S374_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_8_WAV_UNLABELLED'
        and official['count_WAV']==8 and official['original_native_event_count']==255
        and official['sealed_original_predictions_SHA256']==shafile(s374)
        and not official['XML_MIDI_annotated_contents_read']
        and gate['status']=='S380_EIGHT_GUITARSET_TRAINED_EARLY_K3_S18_PRESCORE_SHA'
        and gate['native_events']==255 and gate['WAV_count']==8
        and gate['NO_IDMT_XML_MIDI_READ']
        and gate['source_predictions_SHA256']==shafile(s380)
        and gate['original_GuitarSet_selected_candidate']['view']=='source1147'
        and gate['original_GuitarSet_selected_candidate']['threshold']==.625
        and not gate['original_fold_models_trained_on_IDMT']
        and not gate['independent_ExactK_measured']
        and not gate['official_promotion'],
        'S382 source-original prelabel immutable K3 prediction/threshold SHA proofs absent')
    n=read(s374,NATIVE);p=read(s380,PRED)
    need(len(n['row_id'])==255 and np.array_equal(n['row_id'],p['row_id'])
        and np.array_equal(n['member'],p['member'])
        and np.array_equal(n['incumbent_K'],p['incumbent_original_freeze_K'])
        and p['expert_K3_probability_fourfold'].shape==(4,255),
        'S382 original reference, trained S18, K3 source predictions not aligned')
    reference=n['incumbent_K'];s18=p['original_S18_foldneutral_K']
    candidate=p['early_new_K3_source_only_K']
    need(np.count_nonzero(s18!=candidate)==np.count_nonzero(p['expert_K3_action'])
         and np.all(candidate[p['expert_K3_action']]==3),
         'prelabel-original source candidate changed rows not matching SHA lock')
    groups=manifest['final_eight']
    expected_sha={x['WAV_filename_SHA256']:x for x in official['original_audio_fingerprints']}
    need(len(expected_sha)==8,'must contain only eight original published WAV source SHAs')
    with urllib.request.urlopen(urllib.request.Request(
          API,headers={'Accept':'application/json','User-Agent':'S382-FIRST-XML-AFTER-IMMUTABLE-S380-SHA'}),
          timeout=60) as response:
        source=audit(json.load(response))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    Y=np.full(255,-1,np.int8)
    proof=[]
    with zipfile.ZipFile(reader,'r') as z:
        need(sha('\n'.join(sorted(z.namelist())).encode())==
             manifest['central_directory_sha256'],
             'S382 original published central directory changed')
        for item in groups:
            key=item['wav_filename_sha256']
            ix=np.flatnonzero(n['member'].astype(str)==key)
            need(len(ix)>0 and key in expected_sha,
                 'S382 original V88 native event source missing')
            ix=ix[np.argsort(n['start_samples'][ix],kind='stable')]
            wav=z.read(item['wav'])
            need(sha(wav)==expected_sha[key]['original_WAV_SHA256'],
                 'S382 exact WAV bytes changed after S374 source-only frozen benchmark')
            annotation=z.read(item['annotation_path_name_only'])
            notes=parse_original_pitch_events(annotation)
            ons=np.asarray([float(x[0]) for x in notes],np.float64)
            ons=ons[(ons>=0)&(ons<10.)]
            samples=np.rint(ons*44100).astype(np.int64)
            label,assigned=assign_onsets(samples,n['start_samples'][ix],n['end_samples'][ix])
            Y[ix]=np.minimum(6,label).astype(np.int8)
            proof.append(dict(WAV_name_SHA256=key,XML_SHA256=sha(annotation),
                original_wav_SHA256=sha(wav),native_events=len(ix),
                total_original_notes=len(notes),
                annotated_onsets_in_first_10s=len(samples),
                unmatched_onsets=int(np.count_nonzero(assigned<0)),
                K3_original_groups=int(np.count_nonzero(label==3)),
                eligible_parent3to2=int(np.count_nonzero(p['eligible_source_parent_K3to2'][ix])),
                prelabel_expert_actions=int(np.count_nonzero(p['expert_K3_action'][ix]))))
    need(np.isin(Y,range(7)).all(),
         'S382 first eight XML labels not completely assigned to original 255 V88 groups')
    metrics=dict(
       old_source_S18_vs_GuitarSet_K3_expert=dict(
           full=comp(Y,s18,candidate,np.ones(255,bool)),
           positive=comp(Y,s18,candidate,Y>=1),
           poly=comp(Y,s18,candidate,Y>=2),
           by_true_K=perclass(Y,s18,candidate)),
       official_freeze_vs_early_K3_expert=dict(
           full=comp(Y,reference,candidate,np.ones(255,bool)),
           positive=comp(Y,reference,candidate,Y>=1),
           poly=comp(Y,reference,candidate,Y>=2),
           by_true_K=perclass(Y,reference,candidate)))
    report=dict(status='S382_FIRST_UNSEEN_EIGHT_IDMT_K3_SPECIALIST_POSTFROZEN_EVALUATION',
       original_new_WAV_count=8,original_native_event_count=255,
       first_XML_read_after_expert_trained_and_predictions_frozen=True,
       original_S372_names_only_sha=shafile(s372),
       original_S374_source_SHA=shafile(s374),
       original_S380_premodel_SHA=shafile(s380),
       original_source_models_trained_only_on_GuitarSet=True,
       old_GuitarSet_dev_selected_view='source1147',
       old_GuitarSet_dev_selected_threshold=.625,
       old_train_only_posterior_probability_aggregation='four original heldfold-trained source models mean',
       original_real_K_histogram_if_XML_complete=np.bincount(Y,minlength=7).tolist(),
       original_source_gate_candidates_changed=int(np.count_nonzero(p['expert_K3_action'])),
       comparisons=metrics,WAV_identity_XML_proofs=proof,
       original_S80_full_tail_not_replayed_on_last8=True,
       XML_negative_K0_exhaustiveness_not_certified=True,
       source_song_composition_disjointness_from_GuitarSet_not_proven=True,
       cannot_make_promotional_accuracy_claim=True,
       no_postlabel_optimization_permitted=True,
       original_freeze_local_combo_unchanged=True,
       S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'S382_scored_eight_original_native_255_events.npz',
       row_id=n['row_id'],member=n['member'],
       original_XML20ms_K_if_exhaustive=Y,
       original_frozen_reference_K=reference,
       original_S18_foldneutral_K=s18,
       new_GuitarSet_K3_expert_K=candidate,
       prelabel_K3_action=p['expert_K3_action'])
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','WAV_identity_XML_proofs')},indent=2))
    print('S382 real original trained prelabel source paired score',json.dumps(metrics,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s372','s374','s374report','s380','s380report','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    run(a.s372,a.s374,a.s374report,a.s380,a.s380report,a.out)
