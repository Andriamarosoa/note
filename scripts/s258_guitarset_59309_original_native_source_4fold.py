"""S258: ORIGINAL GuitarSet 59309 native audio feature joins for counterfactual
4-fold reconstructed S80. Load untouched four original PR16 train feature
folds; source features and original OOF PR16 truth are never modified.

This is DEVELOPMENT/IN-SAMPLE for other trained folds, not an independent
validation and not the original piece-held OOF model.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.v273_s4_owned527 import build_owned527
from scripts.s97_external_s45_original_features import geometry
FOLDS=(0,1,2,4);N=59309
def require(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(source,pr16,out):
    require(not out.exists(),'immutable original source and labels')
    original={}
    keys=('global_index','member','starts','ends','k','baseline','fold','summary','sequence','geometry','feature_names')
    for fold in FOLDS:
        fp=source/f'fold{fold}'/f'features-fold-{fold}.npz'
        require(fp.is_file(),'missing original distinct GuitarSet fold acoustic archive')
        with np.load(fp,allow_pickle=False) as z:
            require(set(z.files)==set(keys),'GuitarSet original native acoustic fold schema')
            a={key:z[key].copy() for key in keys}
        n=len(a['global_index'])
        require(all(len(a[k])==n for k in keys[:-1]) and
            np.all(a['fold']==fold) and a['summary'].shape==(n,58) and
            a['geometry'].shape==(n,46) and a['sequence'].shape==(n,42,49),
            'GuitarSet native exact 527D feature fold alignment error')
        original[fold]=a
    idx=np.concatenate([original[f]['global_index'] for f in FOLDS])
    order=np.argsort(idx,kind='stable')
    require(len(idx)==N and len(np.unique(idx))==N,'59309 original unmodified native groups')
    arrays={key:np.concatenate([original[f][key] for f in FOLDS],axis=0)[order]
            for key in keys[:-1]}
    require(all(np.array_equal(original[0]['feature_names'],original[f]['feature_names'])
             for f in FOLDS),'original feature order 58D changed')
    with np.load(pr16/'predictions.npz',allow_pickle=False) as z:
        require(set(z.files)=={'global_index','true_K','baseline_K','fold','variant_ids','predictions'},
                'original PR16 36 vote OOF schema')
        ref={key:z[key].copy() for key in z.files}
    with np.load(pr16/'probabilities.npz',allow_pickle=False) as z:
        require(set(z.files)=={'global_index','variant_ids','probabilities'},
                'original PR16 12 heldfold probability schema')
        p={key:z[key].copy() for key in z.files}
    # PR16 historical archives are ordered by original held-fold, whereas
    # native source features were explicitly sorted by global_index.
    # Align on the immutable numerical event ID, NEVER by array position.
    pr_order=np.argsort(ref['global_index'],kind='stable')
    q_order=np.argsort(p['global_index'],kind='stable')
    ref={key:(value[pr_order] if len(value)==N else value) for key,value in ref.items()}
    p={key:(value[q_order] if len(value)==N else value) for key,value in p.items()}
    checks=dict(ids=np.array_equal(arrays['global_index'],ref['global_index']),
         p_ids=np.array_equal(arrays['global_index'],p['global_index']),
         baseline=np.array_equal(arrays['baseline'],ref['baseline_K']),
         true_K=np.array_equal(arrays['k'],ref['true_K']),
         fold=np.array_equal(arrays['fold'],ref['fold']))
    print(json.dumps(dict(S258_IMMUTABLE_EVENT_ID_JOIN_PROOF=checks,
         mismatch_counts={k:int(np.count_nonzero(
         arrays[{ 'baseline':'baseline','true_K':'k','fold':'fold','ids':'global_index','p_ids':'global_index'}[k]]
         != ({'baseline':ref['baseline_K'],'true_K':ref['true_K'],
             'fold':ref['fold'],'ids':ref['global_index'],'p_ids':p['global_index']}[k])))
         for k in checks})),flush=True)
    require(all(checks.values()),
            'same 59309 exact GuitarSet native OOF ids, true pitch K, and incumbent freeze')
    row=np.asarray(arrays['global_index']).astype(str)
    base=arrays['baseline'].astype(np.int8)
    need_count={f:int(np.count_nonzero(arrays['fold']==f)) for f in FOLDS}
    require(sum(need_count.values())==59309 and int(np.count_nonzero(arrays['k']>=2))==7385,
            'GuitarSet official original scoring denominator changed')
    # The four trained folds are ALL scored on these same 59309 original
    # items in subsequent steps. Some models have already trained on them;
    # this is never an OOF score or future promotion evidence.
    summary=arrays['summary'].astype(np.float32)
    sequ=arrays['sequence'].astype(np.float32)
    geom46=arrays['geometry'].astype(np.float32)
    geom3=geometry(arrays['member'],arrays['starts'],arrays['ends'])
    flow=np.empty((N,245),np.float32)
    morph=np.empty((N,490),np.float32)
    phys=np.empty((N,294),np.float32)
    for i in range(0,N,512):
        j=min(N,i+512)
        flow[i:j]=flow_profile(sequ[i:j])
        morph[i:j],phys[i:j]=audio_temporal_views(sequ[i:j])
    owned=build_owned527(summary,geom46,sequ,base)
    require(owned.shape==(N,527) and np.isfinite(owned).all(),'historic real S5 owned527 matrix')
    out.mkdir(parents=True)
    audio=out/'S258_GuitarSet_59309_original_V88_S97_audio.npz'
    np.savez_compressed(audio,row_id=row,member=arrays['member'],
         start_samples=arrays['starts'],end_samples=arrays['ends'],
         incumbent_freeze_K=base,summary58=summary,
         sequence42x49=sequ,flow245=flow,
         morphology490=morph,physics294=phys,
         original_PR16_geometry46=geom46,geometry3=geom3)
    p527=out/'S258_GuitarSet_59309_owned527.npz'
    np.savez_compressed(p527,row_id=row,member=arrays['member'],
         start_samples=arrays['starts'],end_samples=arrays['ends'],
         incumbent_freeze_K=base,original_PR16_geometry46=geom46,
         original_S5_input_owned527=owned)
    ppr=out/'S258_GuitarSet_59309_true_OOF_PR16.npz'
    np.savez_compressed(ppr,row_id=row,global_index=arrays['global_index'],
         member=arrays['member'],start_samples=arrays['starts'],
         end_samples=arrays['ends'],incumbent_freeze_K=base,
         pr16_policy_K=ref['predictions'],pr16_network_probabilities=p['probabilities'],
         pr16_vote_count=np.stack([(ref['predictions']==k).sum(1) for k in range(7)],axis=1),
         pr16_policy22_K=ref['predictions'][:,22])
    y=out/'S258_GuitarSet_59309_true_K_ONLY_FOR_FINAL_AUDIT.npz'
    np.savez_compressed(y,row_id=row,true_K=arrays['k'],original_fold=arrays['fold'],
        original_freeze_K=base)
    report=dict(status='S258_HISTORIC_GUITARSET_59309_COMPLETE_NATIVE_ACOUSTIC_FOURFOLD_SOURCE',
        original_events=N,original_true_poly_K2plus_events=7385,
        historical_PR16_true_OOF_baseline_unchanged=True,
        original_fold_distribution={str(k):v for k,v in need_count.items()},
        original_59k_audio_SHA256=sha(audio),
        original_59k_owned527_SHA256=sha(p527),
        original_59k_PR16_SHA256=sha(ppr),
        original_59k_truth_SHA256=sha(y),
        truth_is_separated_from_all_predictor_archives=True,
        all_4fold_outof_fold_predictions_on_every_row_not_yet_generated=True,
        future_all_fold_train_contamination_MUST_disclose=True,
        independent_accuracy_measured=False,S80_new_score_measured=False,
        reference_official_model='freeze_local_combo',official_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','pr16','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.source,a.pr16,a.out)
