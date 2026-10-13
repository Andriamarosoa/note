"""S387 ORIGINAL S18 3/4 + prelabel S376 trained K3 head last eight IDMT WAVs.

ALL weights originally trained ONLY on GuitarSet; MUST consume S376
GuitarSet-only selected view+threshold fixed BEFORE any eight XML was
opened. Original S5 4 fold raw+prior probabilities feed corresponding
trained S376 expert per fold; average their K3 scores as a new and
PREDECLARED deployment rule (distribution shift from GuitarSet single
OOF fold disclosed). Invoke ONLY when original reference K3 and
S18 piece-neutral 3/4 predicts K2. No truth, no XML, no fit.
Compare to true original S18 parent (not complete S80); do not claim
S80 promoted, any independent K3 precision until XML first read.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
AUDIO={'row_id','member','start_samples','incumbent_freeze_K','summary58',
        'sequence42x49','flow245','morphology490','physics294'}
OWNED={'row_id','member','start_samples','end_samples','incumbent_freeze_K',
       'original_PR16_geometry46','original_S5_input_owned527'}
S5={'row_id','incumbent_freeze_K','fold_ids',
    'original_S5_raw_probability_by_fold',
    'original_S5_prior_probability_by_fold',
    'original_S5_K_by_fold','all_four_folds_agree'}
S18={'row_id','freeze_K','original_fold_ids','original_S18_by_fold_K',
     'S18_piece_neutral_consensus_K','S18_quorum_of_3_supported',
     'S18_vote_strength','S18_abstention_to_freeze'}
FOLDS=(0,1,2,4)
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'strict prelabel original source schema drift: '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(audio,owned,s5,s18,report,models,out):
    need(not out.exists(),'cannot overwrite eight WAV prelabel expert results')
    info=json.loads(report.read_text())
    chosen=info['selected_for_truly_unseen_external_S372_8WAV']
    need(info['status']=='S376_FOLDHELD_GUITARSET_SOURCE_S5_POSTERIOR_K3_AUDIO_EXPERT'
        and info['trained_on_original_GuitarSet_only']
        and info['any_IDMT_WAV_XML_read'] is False
        and chosen is not None and chosen['view'] in ('source118','source1147')
        and chosen['threshold']>=.1 and chosen['threshold']<=.8
        and len(chosen['original_trained_fold_model_SHA256'])==4
        and not info['official_promotion'],
        'no precommitted positive train-only GuitarSet K3 expert; fail closed')
    a=load(audio,AUDIO);b=load(owned,OWNED)
    c=load(s5,S5);d=load(s18,S18)
    ids=a['row_id'];n=len(ids)
    need(n>0 and len(set(map(str,a['member'])))==32 and all(np.array_equal(ids,z['row_id']) for z in (b,c,d))
         and np.array_equal(a['incumbent_freeze_K'],b['incumbent_freeze_K'])
         and np.array_equal(a['incumbent_freeze_K'],c['incumbent_freeze_K'])
         and np.array_equal(a['incumbent_freeze_K'],d['freeze_K'])
         and np.array_equal(c['fold_ids'],FOLDS)
         and np.array_equal(d['original_fold_ids'],FOLDS)
         and c['original_S5_raw_probability_by_fold'].shape==(4,n,7)
         and c['original_S5_prior_probability_by_fold'].shape==(4,n,7)
         and np.isfinite(a['summary58']).all(),
         'original 255 V88 true pretrained event source not aligned')
    ref=a['incumbent_freeze_K'].astype(np.int8)
    parent=d['S18_piece_neutral_consensus_K'].astype(np.int8)
    eligible=(ref==3)&(parent==2)
    per_fold=np.empty((4,n),np.float32)
    summary=a['summary58'];geom=b['original_PR16_geometry46']
    for fi,fold in enumerate(FOLDS):
        path=models/f"{chosen['view']}_true_original_fold{fold}_K3_expert.joblib"
        need(path.is_file() and sha(path)==chosen['original_trained_fold_model_SHA256'][fi],
             'GuitarSet trained K3 expert checkpoint SHA does not match S376 original train-only selection')
        X=np.column_stack([summary,geom,
          c['original_S5_raw_probability_by_fold'][fi],
          c['original_S5_prior_probability_by_fold'][fi]]).astype(np.float32)
        if chosen['view']=='source1147':
            X=np.column_stack([X,a['flow245'],a['morphology490'],a['physics294']]).astype(np.float32)
        need(X.shape==(n,1147 if chosen['view']=='source1147' else 118)
             and np.isfinite(X).all(),'GuitarSet 118D or 1147D external K3 expert feature mismatch')
        clf=joblib.load(path)
        need(clf.n_features_in_==X.shape[1] and 1 in clf.classes_,
             'trained source model incompatible with original V88 acoustic features')
        per_fold[fi]=clf.predict_proba(X)[:,list(clf.classes_).index(1)]
    mean=per_fold.mean(0)
    threshold=float(chosen['threshold'])
    action=eligible&(mean>=threshold)
    candidate=parent.copy();candidate[action]=3
    need(np.all((candidate==parent)|(action&(parent==2)&(ref==3)&(candidate==3))),
         'K3 head acted on wrong source parent')
    out.mkdir(parents=True)
    dest=out/'S387_32WAV_dataset2_original_S18_vs_early_K3_specialist_PRESCORE.npz'
    np.savez_compressed(dest,row_id=ids,member=a['member'],
        incumbent_original_freeze_K=ref,
        original_S18_foldneutral_K=parent,
        early_new_K3_source_only_K=candidate,
        eligible_source_parent_K3to2=eligible,
        expert_K3_probability_fourfold=per_fold,
        mean_expert_K3_probability=mean,
        expert_K3_action=action)
    result=dict(status='S387_NEW32_DATASET2_GUITARSET_TRAINED_EARLY_K3_S18_PRESCORE_SHA',
        native_events=n,WAV_count=len(set(map(str,a['member']))),
        original_frozen_V88_reference_SHA256=sha(owned),
        original_S18_source_SHA256=sha(s18),
        original_S5_trained_source_SHA256=sha(s5),
        original_audio_58_245_490_294_SHA256=sha(audio),
        trained_GuitarSet_S376_report_SHA256=sha(report),
        original_GuitarSet_selected_candidate=chosen,
        deployment_rule_NEW_and_NOT_original_OOF='mean of four genuine fold-specialized GuitarSet-trained K3 expert probabilities then fixed development threshold',
        original_fold_models_trained_on_IDMT=False,
        original_prelabel_S18_K_histogram=np.bincount(parent,minlength=7).tolist(),
        resulting_prelabel_candidate_K_histogram=np.bincount(candidate,minlength=7).tolist(),
        eligible_incumbentK3_S18K2_events=int(eligible.sum()),
        new_prelabel_K3_actions=int(action.sum()),
        source_predictions_SHA256=sha(dest),
        NO_IDMT_XML_MIDI_READ=True,independent_ExactK_measured=False,
        complete_S80_replayed_on_8WAV=False,
        official_promotion=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('audio','owned','s5','s18','report','models','out'):
        p.add_argument('--'+k,required=True,type=Path)
    z=p.parse_args();run(z.audio,z.owned,z.s5,z.s18,z.report,z.models,z.out)
