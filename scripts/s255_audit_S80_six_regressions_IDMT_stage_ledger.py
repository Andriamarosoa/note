"""S255 post-hoc ROOT CAUSE of six independent XML-labelled S80 regressions.

Absolutely no model fitting, threshold tuning, checkpoint edits, or re-use
of this already revealed IDMT split as a future holdout. Compare actual
S239 fold S18, S247/249 S46/S69, S250 S70, S251 S71/S72 and S252
S79/S80 decisions at the SAME true official V88 event IDs on all 1034
source predictions. Only report label-matched causal stage/edge ledger.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(path,expected):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(expected),'unexpected stage field/shape '+str(path))
        return {k:z[k].copy() for k in z.files}
def run(xml,s239,s247,s249,s250,s251,s252,out):
    need(not out.exists(),'immutable post-label evidence')
    anno=read(xml,{'row_id','member','start_samples','XML_true_K_or_minus1',
        'matched_offset_ms','frozen_official_freeze_K','frozen_full_S80_K'})
    parent=read(s239,{'row_id','freeze_K','original_fold_ids',
        'original_S18_by_fold_K','S18_piece_neutral_consensus_K',
        'S18_quorum_of_3_supported','S18_vote_strength','S18_abstention_to_freeze'})
    specialists=read(s247,{'row_id','variant_ids','original_freeze_K',
        'S18_parent_K','original_19_branch_folds','S46_193_policy_K',
        'per_fold_specialist_K','active_fold_count',
        'parent_incompatible_specialists','contradictory_specialists'})
    s69=read(s249,{'row_id','S18_parent_K','balanced_s46_K','balanced_protected_K',
        'polymax_s46_K','polymax_protected_K','balanced_fold_veto',
        'balanced_veto_three_folds','polymax_fold_veto','polymax_veto_three_folds'})
    s70=read(s250,{'row_id','s69_polymax_K','s69_balanced_K','s18_K',
        'proposed_poly_K','proposed_poly_fold_K','proposed_poly_fold_active',
        'proposed_poly_vote_accepted','downward_poly_K','downward_poly_fold_K',
        'downward_poly_fold_active','downward_poly_vote_accepted'})
    risk=read(s251,{'row_id','s69_polymax_K','s70_proposed_poly_K',
        's71_risk_K','s71_poly_K','s71_veto_three_folds','s72_veto_three_folds',
        's71_fold_veto','s72_fold_veto','original_pr16_geometry46',
        's72_risk_all_K','s72_risk_poly_guard_K','s72_poly_all_K',
        's72_poly_poly_guard_K'})
    full=read(s252,{'row_id','incumbent_freeze_K','s72_K','s79_K',
        's80_K','original_PR16_policy22_K','K1_by_fold_threshold_action',
        'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'})
    ids=anno['row_id'];n=len(ids)
    need(n==1034 and all(np.array_equal(ids,z['row_id'])
         for z in (parent,specialists,s69,s70,risk,full)),
         'native event joined stage source across S233 S239 S247 S252 mismatch')
    base=anno['frozen_official_freeze_K']
    y=anno['XML_true_K_or_minus1'].astype(int)
    good=y>=1
    need(np.array_equal(base,full['incumbent_freeze_K']) and
         np.array_equal(anno['frozen_full_S80_K'],full['s80_K']) and
         np.array_equal(parent['S18_piece_neutral_consensus_K'],specialists['S18_parent_K']) and
         np.array_equal(s69['polymax_protected_K'],s70['s69_polymax_K']) and
         np.array_equal(s70['proposed_poly_K'],risk['s70_proposed_poly_K']) and
         np.array_equal(risk['s72_poly_poly_guard_K'],full['s72_K']),
         'this annotation is not for exact ORIGINAL frozen S80 source graph')
    names=['freeze','S18','S46_polymax','S69_veto','S70_original4',
           'S71_original4','S72_original4','S79_K1','S80_final']
    cols=np.column_stack((base,parent['S18_piece_neutral_consensus_K'],
         s69['polymax_s46_K'],s69['polymax_protected_K'],
         s70['proposed_poly_K'],risk['s71_risk_K'],
         risk['s72_poly_poly_guard_K'],full['s79_K'],full['s80_K'])).astype(int)
    need(cols.shape==(1034,9) and np.isin(cols,range(7)).all(),
         'original frozen policy chain Exact-K class range')
    transitions={}
    for k in range(1,len(names)):
        prior=cols[:,k-1];after=cols[:,k];change=prior!=after
        subset=change&good
        improve=int(np.count_nonzero(subset&(prior!=y)&(after==y)))
        regress=int(np.count_nonzero(subset&(prior==y)&(after!=y)))
        transitions[names[k]]=dict(all_1034_changed=int(change.sum()),
            changed_and_XML_matched=int(subset.sum()),
            stage_corrections_on_matched=improve,
            stage_regressions_on_matched=regress,
            stage_delta_exactK_matched=int(np.count_nonzero(good&(after==y))-
                                           np.count_nonzero(good&(prior==y))))
    final=full['s80_K'];changed=(base!=final);paired=changed&good
    regression=np.flatnonzero(paired&(base==y)&(final!=y))
    correction=np.flatnonzero(paired&(base!=y)&(final==y))
    neither=np.flatnonzero(paired&(base!=y)&(final!=y))
    need((len(regression),len(correction),len(neither))==(6,3,7),
         'postfreeze correction ledger must exactly match S254 evidence')
    def record(i):
        i=int(i)
        steps=[names[t] for t in range(1,len(names)) if cols[i,t]!=cols[i,t-1]]
        return dict(row_id=str(ids[i]),WAV_name_SHA256=str(anno['member'][i]),
            onset_s=float(anno['start_samples'][i]/44100.),
            true_K=int(y[i]),official_freeze_K=int(base[i]),S80_final_K=int(final[i]),
            staged_K={z:int(v) for z,v in zip(names,cols[i])},
            transition_steps=steps,
            original_S18_four_fold_K=parent['original_S18_by_fold_K'][:,i].astype(int).tolist(),
            original_S18_vote_strength=int(parent['S18_vote_strength'][i]),
            original_S18_no_three_vote_abstain=bool(parent['S18_abstention_to_freeze'][i]),
            source_S69_veto=bool(s69['polymax_veto_three_folds'][i]),
            source_S70_fold_gate=bool(s70['proposed_poly_vote_accepted'][i]),
            source_S71_veto=bool(risk['s71_veto_three_folds'][i]),
            source_S72_veto=bool(risk['s72_veto_three_folds'][i]))
    result=dict(status='S255_POSTHOC_INDEPENDENT_REGRESSION_CAUSAL_STAGE_AUDIT',
        original_S254_XML_paired_run=38308757247,
        source_original_1034_native_events=1034,matched_XML_events=int(good.sum()),
        frozen_S80_changed_native_events=int(changed.sum()),
        frozen_S80_matched_changes=int(paired.sum()),
        original_S80_corrections=int(len(correction)),
        original_S80_regressions=int(len(regression)),
        original_S80_changed_both_incorrect=int(len(neither)),
        all_pipeline_stage_effects=transitions,
        six_rootcause_regression_cases=[record(i) for i in regression],
        three_original_corrected_cases=[record(i) for i in correction],
        seven_both_incorrect_cases=[record(i) for i in neither],
        HEADROOM_WARNING='Posthoc 16 labelled WAV files used for DIAGNOSIS only. Any tuned rule is invalid for an independent retest on the same WAVs.',
        no_models_refit=True,no_thresholds_changed=True,
        no_song_disjointness_claim=True,
        full_ExactK_K0toK6_not_measured=True,S80_promoted=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
           if k not in ('six_rootcause_regression_cases','three_original_corrected_cases',
                        'seven_both_incorrect_cases')},indent=2),flush=True)
    print(json.dumps({'first_regression_example':result['six_rootcause_regression_cases'][0]},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('xml','s239','s247','s249','s250','s251','s252','out'):
        p.add_argument('--'+x,required=True,type=Path)
    a=p.parse_args();run(a.xml,a.s239,a.s247,a.s249,a.s250,a.s251,a.s252,a.out)
