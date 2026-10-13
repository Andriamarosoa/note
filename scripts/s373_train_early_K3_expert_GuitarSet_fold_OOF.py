"""S373 NEW EARLY K2/K3 K3-rescue expert, trained ONLY on original GuitarSet.

PREREGISTERED BEFORE any S372 holdout WAV/XML read. Training labels
solely ORIGINAL S258 59309 GuitarSet reference folds; candidate
S18 fourfold stage from ORIGINAL genuine archived S260 checkpoints.
STRICT fold f: train on other GuitarSet folds (and exclude any
song/member shared with validation), predict ONLY fold f. No
in-sample fold candidate output is fed to the model. At external
deployment, original source S18 fold-neutral 3/4 votes is a
distribution shift; OOF gains are development ONLY and DO NOT
validate S80 performance.

Two fixed views: low 104D summary58+geometry46; expanded
1133D = low104 + flow245 + morphology490 + physics294.
Fixed ExtraTrees architecture and P(true K3)>=0.80 conservative
guard. NEVER select trained variant or threshold by opened IDMT XML.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
FOLDS=(0,1,2,4);N=59309;THRESHOLD=.80;SEED=373
AUDIO_FIELDS={'row_id','member','start_samples','end_samples','incumbent_freeze_K',
  'summary58','sequence42x49','flow245','morphology490',
  'physics294','original_PR16_geometry46','geometry3'}
TRUTH_FIELDS={'row_id','true_K','original_fold','original_freeze_K'}
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def read(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==fields, 'S373 source schema mismatch: '+str(path))
        return {k:z[k].copy() for k in z.files}
def construct_model(fold,variant):
    return ExtraTreesClassifier(n_estimators=128,max_depth=12,
      min_samples_leaf=5,max_features='sqrt',bootstrap=False,
      class_weight='balanced',n_jobs=2,random_state=SEED+int(fold)+
      (1000 if variant=='expanded1133' else 0))
def tally(label,base,guard,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return dict(n=0,base_correct=0,guard_correct=0,
       corrections=0,regressions=0,net=0)
    ref=base[ix]==label[ix];new=guard[ix]==label[ix]
    return dict(n=len(ix),base_correct=int(ref.sum()),guard_correct=int(new.sum()),
      corrections=int((~ref&new).sum()),regressions=int((ref&~new).sum()),
      net=int(new.sum()-ref.sum()))
def run(audio_path,truth_path,s18_path,out):
    need(not out.exists(),'never overwrite frozen independent development evidence')
    a=read(audio_path,AUDIO_FIELDS)
    truth=read(truth_path,TRUTH_FIELDS)
    with np.load(s18_path,allow_pickle=False) as z:
        need({'row_id','freeze_K','fold_ids','S18_K'}.issubset(z.files),
           'historical 133 trained source S18 archive incomplete')
        model_ids=z['row_id'].copy(); fold_ids=z['fold_ids'].copy()
        four_stage=z['S18_K'].copy()
    ids=a['row_id'];reference=a['incumbent_freeze_K'].astype(int)
    y=truth['true_K'].astype(int);fold=truth['original_fold'].astype(int)
    members=a['member'].astype(str)
    need(len(ids)==N and len(np.unique(ids))==N
         and np.array_equal(ids,truth['row_id'])
         and np.array_equal(ids,model_ids)
         and np.array_equal(reference,truth['original_freeze_K'])
         and np.array_equal(reference,read(audio_path,AUDIO_FIELDS)['incumbent_freeze_K'])
         and np.array_equal(fold_ids,FOLDS) and four_stage.shape==(4,N)
         and np.isin(y,range(7)).all() and
         all(np.count_nonzero(fold==f)>1000 for f in FOLDS),
         'source old GuitarSet authentic heldfold/true K invariant broken')
    subset=np.flatnonzero(reference==3)
    need(len(subset)>80,'too few source reference K3 to fit an expert')
    low=np.column_stack([a['summary58'][subset],a['original_PR16_geometry46'][subset]]).astype(np.float32)
    extra=np.column_stack([low,a['flow245'][subset],a['morphology490'][subset],a['physics294'][subset]]).astype(np.float32)
    need(low.shape==(len(subset),104) and extra.shape==(len(subset),1133)
         and np.isfinite(low).all() and np.isfinite(extra).all(),
         'original acoustic 104D/1133D source inputs not reproducible')
    # Only one original model per row was held out of THAT row's GuitarSet fold.
    id_to_fold={f:i for i,f in enumerate(FOLDS)}
    row_fold_idx=np.asarray([id_to_fold[int(f)] for f in fold],int)
    true_heldfold_S18=four_stage[row_fold_idx,np.arange(N)]
    trigger=(reference==3)&(true_heldfold_S18==2)
    need(trigger.sum()>5,'too few original heldfold K3->K2 validation cases')
    arows=np.arange(len(subset))
    fsub=fold[subset]; msub=members[subset]; ysub=(y[subset]==3).astype(int)
    trsub=trigger[subset]
    variants={'base104':low,'expanded1133':extra}
    trained={}
    reportvariants={}
    out.mkdir(parents=True)
    for name,X in variants.items():
        oof=np.full(len(subset),np.nan,np.float32)
        fold_proofs=[]
        for held in FOLDS:
            train=(fsub!=held)
            val=(fsub==held)
            # Remove any acoustic SOURCE SONG from training if also
            # used in original heldfold validation (stronger than fold split).
            val_members=set(msub[val])
            overlap_count=len(set(msub[train]).intersection(val_members))
            train &= ~np.isin(msub,list(val_members))
            need(train.sum()>20 and val.sum()>10
                 and len(np.unique(ysub[train]))==2,
                 f'insufficient non-leaky GuitarSet K3/class-notK3 in fold {held}')
            est=construct_model(held,name)
            est.fit(X[train],ysub[train])
            probs=est.predict_proba(X[val])[:,list(est.classes_).index(1)]
            oof[val]=probs.astype(np.float32)
            fp=out/f'{name}_trained_without_original_fold{held}.joblib'
            joblib.dump(est,fp,compress=3)
            fold_proofs.append(dict(fold=held,heldout_events=int(val.sum()),
                 training_events=int(train.sum()),
                 positive_K3_train=int(ysub[train].sum()),
                 positive_K3_held=int(ysub[val].sum()),
                 source_member_overlap_count_before_excluding=overlap_count,
                 source_member_overlap_remaining=0,
                 trained_model_SHA256=sha(fp),
                 model_target='GuitarSet true_K==3 among ORIGINAL freeze_K==3 only',
                 source_audio_dimensions=X.shape[1]))
        need(np.isfinite(oof).all() and (oof>=0).all() and (oof<=1).all(),
             'heldfold OOF posterior predictions missing')
        # Nonoracle candidate: only S18 heldfold K2, immutable
        # original freeze K3 and the fixed .8 3rd note posterior.
        route=trsub&(oof>=THRESHOLD)
        actual_changed_rows=subset[route]
        base=true_heldfold_S18.copy()
        expert=base.copy();expert[actual_changed_rows]=3
        mask=trigger
        byK={f'K{k}':tally(y,base,expert,mask&(y==k)) for k in range(7)}
        changed_truth=np.bincount(y[actual_changed_rows],minlength=7)
        detail=dict(view=name,feature_width=X.shape[1],folds=fold_proofs,
          all_original_reference_K3_events=len(subset),
          eligible_S18_true_heldfold_K2_under_reference_K3=int(trigger.sum()),
          eligible_true_K_hist=np.bincount(y[trigger],minlength=7).tolist(),
          fixed_predeclared_P3_threshold=THRESHOLD,
          triggered_new_K3_actions=int(route.sum()),
          actions_trueK_hist=changed_truth.tolist(),
          entire_GuitarSet_event_count=N,
          paired_truth_for_trigger=tally(y,base,expert,mask),
          paired_for_all_59309=tally(y,base,expert,np.ones(N,bool)),
          by_trueK_on_trigger=byK,
          original_perfold_trained_model_NOT_ALL_FOLD_IN_SAMPLE=True,
          song_member_group_overlap_excluded=True,
          early_heldfold_S18_scores_not_full_foldneutral_S80=True,
          not_new_external_validation=True)
        reportvariants[name]=detail
        np.savez_compressed(out/f'{name}_source_heldfold_OOF_59309_reference3_only.npz',
            row_id=ids[subset],original_fold=fsub,
            original_S18_OOF_K=true_heldfold_S18[subset],
            eligible_trigger=trsub,heldfold_pred_probability_true_K3=oof,
            fixed_trigger_above_threshold=route)
    summary=dict(status='S373_EARLY_K3_AUDIO_DISCRIMINATOR_GUITARSET_FOLDS_ONLY',
       historic_original_GuitarSet_59309=True,source_audio_SHA256=sha(audio_path),
       source_original_truth_SHA256=sha(truth_path),
       source_original_S18_fourfold_SHA256=sha(s18_path),
       fold_members_are_excluded_across_train_validation=True,
       training_only_GUITARSET_59309_NO_IDMT_AUDIO_OR_XML=True,
       gate_only_original_reference_K3_and_original_heldfold_S18_K2=True,
       alternative_candidate_selections_forbidden_after_IDMT_score=True,
       fixed_P3_threshold=THRESHOLD,trained_variants=reportvariants,
       future_external_unseen_WAVs_assigned_to_S372_precommitted=True,
       full_S80_deployment_not_yet_tested=True,
       original_freeze_local_combo_reference_unchanged=True,
       official_S80_promotion=False)
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='trained_variants'},indent=2))
    for name,v in reportvariants.items():
        print('DEV SOURCE OOF',name,json.dumps({k:val for k,val in v.items()
               if k not in ('folds','by_trueK_on_trigger')},indent=2))
        print('DEV per-K',name,json.dumps(v['by_trueK_on_trigger'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('audio','truth','s18','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.audio,a.truth,a.s18,a.out)
