"""S390: strictly cross-fitted NEW upstream K5 acoustic candidate producer.

No existing S80 model is modified. Target only the leading K5->K4 failure.
Train from ORIGINAL S258 audio features and annotations; train subset:
original genuinely held-piece historic S80 OOF predicted K4. For outer fold f,
train only OTHER folds excluding ALL validation WAV members, then predict fold f.
This keeps the NEW producer OOF, not the reconstructed S80 S272 (which has
training overlap). All thresholds are PREREGISTERED; no selected winner.
GuitarSet is already a development dataset: no independent promotion claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

FOLDS = (0, 1, 2, 4)
N = 59309
SEED = 390
CUTS = (0.25, 0.35, 0.50, 0.65, 0.80)
AUDIO = {'row_id','member','start_samples','end_samples','incumbent_freeze_K',
         'summary58','sequence42x49','flow245','morphology490',
         'physics294','original_PR16_geometry46','geometry3'}
TRUTH = {'row_id','true_K','original_fold','original_freeze_K'}
S272 = {'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
        'original_PR16_policy22_K','K1_by_fold_threshold_action',
        'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path, fields):
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == fields, 'source schema drift at '+str(path))
        return {k:z[k].copy() for k in z.files}

def measure(y, base, pred, scope):
    i = np.flatnonzero(scope)
    a = base[i] == y[i]
    b = pred[i] == y[i]
    return dict(n=int(len(i)),baseline_correct=int(a.sum()),
                candidate_correct=int(b.sum()),
                corrections=int((~a&b).sum()),
                regressions=int((a&~b).sum()),
                net=int(b.sum()-a.sum()))

def model(fold, view):
    return ExtraTreesClassifier(
        n_estimators=128, max_depth=12, min_samples_leaf=4,
        max_features='sqrt', class_weight='balanced',
        n_jobs=2, random_state=SEED+fold+(1000 if view=='audio1133' else 0))

def run(audiofile, truthfile, historicalfile, s272file, yourfile, output):
    require(not output.exists(), 'S390 output must be immutable')
    a = read(audiofile, AUDIO)
    t = read(truthfile, TRUTH)
    s = read(s272file, S272)
    with np.load(historicalfile, allow_pickle=False) as z:
        require({'global_index','true_K','s80_K','fold'}.issubset(z.files),
                'genuine historical HELD-PIECE OOF S80 source missing')
        old = {k:z[k].copy() for k in ('global_index','true_K','s80_K','fold')}
    with np.load(yourfile, allow_pickle=False) as z:
        require({'global_index','k','predicted','baseline','fold'}.issubset(z.files),
                'YourMT3+ comparison source incomplete')
        your = {k:z[k].copy() for k in ('global_index','k','predicted','baseline','fold')}
    ids = a['row_id'].astype(str)
    order = np.argsort(old['global_index'].astype(np.int64))
    yorder = np.argsort(your['global_index'].astype(np.int64))
    y = t['true_K'].astype(np.int8)
    f = t['original_fold'].astype(int)
    historic = old['s80_K'][order].astype(np.int8)
    incumbent = s['s80_K'].astype(np.int8)
    members = a['member'].astype(str)
    require(len(ids) == N and len(set(ids)) == N
            and np.array_equal(ids,t['row_id'].astype(str))
            and np.array_equal(ids,s['row_id'].astype(str))
            and np.array_equal(ids,old['global_index'][order].astype(str))
            and np.array_equal(ids,your['global_index'][yorder].astype(str)),
            'S258 / original OOF / S272 / YourMT3+ native row identifiers drifted')
    require(np.array_equal(y,old['true_K'][order])
            and np.array_equal(y,your['k'][yorder])
            and np.array_equal(f,old['fold'][order])
            and np.array_equal(f,your['fold'][yorder])
            and np.array_equal(t['original_freeze_K'],a['incumbent_freeze_K'])
            and np.array_equal(t['original_freeze_K'],s['incumbent_freeze_K'])
            and np.array_equal(t['original_freeze_K'],your['baseline'][yorder]),
            'original held-piece truth/fold/reference alignment failure')
    require(set(map(int,np.unique(f))) == set(FOLDS)
            and int((y==5).sum()) == 355
            and int((historic==y).sum()) == 49281
            and int((incumbent==y).sum()) == 53327
            and int((your['predicted'][yorder]==y).sum()) == 51328,
            'official original benchmark expected counts drifted')
    low = np.column_stack((a['summary58'],a['original_PR16_geometry46'])).astype(np.float32)
    wide = np.column_stack((low,a['flow245'],a['morphology490'],a['physics294'])).astype(np.float32)
    require(low.shape==(N,104) and wide.shape==(N,1133)
            and np.isfinite(low).all() and np.isfinite(wide).all(),
            '104D / 1133D source acoustic views invalid')
    views = {'audio104':low,'audio1133':wide}
    output.mkdir(parents=True)
    reports = {}
    for name,X in views.items():
        oof = np.full(N,np.nan,dtype=np.float32)
        fold_proofs=[]
        for hold in FOLDS:
            val = f == hold
            train = (f != hold) & (historic == 4)
            held_members = set(members[val])
            overlaps = len(set(members[train]).intersection(held_members))
            train &= ~np.isin(members,list(held_members))
            positives = int((y[train]==5).sum())
            negatives = int((y[train]!=5).sum())
            require(val.sum()>1000 and train.sum()>100
                    and positives>=10 and negatives>=50
                    and not set(members[train]).intersection(held_members),
                    'fold '+str(hold)+' insufficient independent train members or K5')
            clf=model(hold,name)
            clf.fit(X[train],(y[train]==5).astype(np.int8))
            score=clf.predict_proba(X[val])[:,list(clf.classes_).index(1)]
            require(len(score)==int(val.sum()) and np.isfinite(score).all(),
                    'OOF K5 model probability invalid')
            oof[val]=score.astype(np.float32)
            dest=output/(name+'_trained_without_fold'+str(hold)+'.joblib')
            joblib.dump(clf,dest,compress=3)
            fold_proofs.append(dict(fold=hold,training_rows=int(train.sum()),
                training_K5=positives,training_nonK5=negatives,
                heldout_rows=int(val.sum()),heldout_trueK5=int((y[val]==5).sum()),
                member_intersections_before_exclusion=overlaps,
                member_intersections_after_exclusion=0,
                model_sha256=sha(dest)))
        require(np.isfinite(oof).all() and (oof>=0).all() and (oof<=1).all(),
                'not all 59309 examples have truly OOF posterior')
        audit=[]
        for cut in CUTS:
            original_gate=(historic==4)&(oof>=cut)
            real_gate=(incumbent==4)&(oof>=cut)
            historical_new=historic.copy()
            candidate_new=incumbent.copy()
            historical_new[original_gate]=5
            candidate_new[real_gate]=5
            item=dict(threshold=cut,
                historic_actions=int(original_gate.sum()),
                reconstructed_S80_actions=int(real_gate.sum()),
                historical_all=measure(y,historic,historical_new,np.ones(N,bool)),
                historical_K5=measure(y,historic,historical_new,y==5),
                historical_K4=measure(y,historic,historical_new,y==4),
                reconstructed_all_DESCRIPTIVE=measure(y,incumbent,candidate_new,np.ones(N,bool)),
                reconstructed_poly_DESCRIPTIVE=measure(y,incumbent,candidate_new,y>=2),
                reconstructed_K5_DESCRIPTIVE=measure(y,incumbent,candidate_new,y==5),
                reconstructed_K4_DESCRIPTIVE=measure(y,incumbent,candidate_new,y==4),
                reconstructed_by_K_DESCRIPTIVE={
                    'K'+str(k):measure(y,incumbent,candidate_new,y==k) for k in range(7)},
                historical_by_fold={
                    str(hold):measure(y,historic,historical_new,f==hold) for hold in FOLDS})
            require(item['historical_all']['net']==item['historical_K5']['net']+
                    item['historical_K4']['net']+
                    sum(measure(y,historic,historical_new,y==k)['net']
                        for k in (0,1,2,3,6)), 'historical correction accounting drift')
            audit.append(item)
        np.savez_compressed(output/(name+'_crossfit_prediction_NO_TRUTH.npz'),
                            row_id=ids, original_fold=f,
                            posterior_K5_oof=oof)
        reports[name]=dict(feature_width=X.shape[1],
                          eligible_historical_S80_predK4=int((historic==4).sum()),
                          eligible_S272_S80_predK4=int((incumbent==4).sum()),
                          original_hist_K5_predK4=int(((y==5)&(historic==4)).sum()),
                          reconstructed_K5_predK4=int(((y==5)&(incumbent==4)).sum()),
                          fold_proofs=fold_proofs,predeclared_grid_results=audit)
    report=dict(status='S390_K5_SOURCE_ACOUSTIC_PRODUCER_TRUE_FOLD_HELD_DEVELOPMENT',
        source_runs={'S258':38311525987,'S80_historic_OOF':38187186049,
                     'S80_reconstructed_S272':38314291222,
                     'YourMT3_plus':37605163312},
        source_hashes={'audio':sha(audiofile),'truth':sha(truthfile),
                       'historical_OOF':sha(historicalfile),'S272':sha(s272file),
                       'YourMT3':sha(yourfile)},
        original_event_count=N,original_true_K5_count=355,
        training_subset_rule='historical S80 held-piece OOF predicted K4 only',
        true_group_and_fold_disjoint_new_model=True,
        fold_assignments=(0,1,2,4),
        thresholds_fixed_before_experiment=list(CUTS),
        model_selection_from_outer_fold_truth_forbidden=True,
        reports=reports,
        original_S80_checkpoint_changed=False,
        proposed_new_K5_head_not_deployed=True,
        S272_training_overlap_not_pure_OOF=True,
        GuitarSet_previously_exposed_not_independent=True,
        new_audio_musical_holdout_evaluation_performed=False,
        official_promotion=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,item in reports.items():
        print('S390 '+name+' OOF TRAIN GROUP CHECK',json.dumps(item['fold_proofs']))
        for row in item['predeclared_grid_results']:
            print('S390 '+name+' THRESH',row['threshold'],
                  'real S80 K5 fixes',row['reconstructed_K5_DESCRIPTIVE']['corrections'],
                  'K4 regressions',row['reconstructed_K4_DESCRIPTIVE']['regressions'],
                  'real S80 all net',row['reconstructed_all_DESCRIPTIVE']['net'],
                  'historical OOF net',row['historical_all']['net'],flush=True)
    print('S390 PASS NO REFERENCE CHANGE NO IDMT LABELS READ',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('audio','truth','historical','s272','your','out'):
        p.add_argument('--'+k,type=Path,required=True)
    x=p.parse_args()
    run(x.audio,x.truth,x.historical,x.s272,x.your,x.out)
