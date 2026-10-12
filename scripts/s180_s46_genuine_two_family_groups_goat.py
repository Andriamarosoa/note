"""S180 genuine S46 frozen selection on the TWO complete S46 groups in partial 11-family GOAT S45.

No padding of missing five original sources. This runs 96 frozen S46 variants
on original-trained 209 Ridge risk outputs, but eleven-family source votes
ARE NOT the original full-sixteen vote contract, so NOT original S46/S80.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.audit_v273_s46_classwise_compromise import (
    FAMILY_GROUPS,LAM_POLY,CUT_POLY,VOTES,STRUCTURE,select)
EXPECTED_GROUPS=('unique_poly','ordering')
def check(c,m):
    if not c:raise ValueError(m)
def run(source,out):
    check(not out.exists(),'S180 immutable output')
    with np.load(source,allow_pickle=False) as z:
        check(set(z.files)=={'row_id','S18_parent_K','original_family_ids','source_K',
          'S35_new_K','S35_selected_actions','agreement','trained_Ridge_risks',
          'partial_eleven_selected_head','partial_eleven_candidate_K'},
          'strictly 11 original-trained source GOAT with no truth')
        raw={k:z[k].copy() for k in z.files}
    ids=raw['row_id'];old=raw['S18_parent_K'].astype(np.int8)
    fam=list(map(str,raw['original_family_ids']))
    proposals=raw['source_K'].astype(np.int8)
    risk=raw['trained_Ridge_risks'].astype(np.float32)
    votes=raw['agreement'].astype(np.int16)
    n=len(ids)
    check(n==57 and proposals.shape==(n,11) and risk.shape==(n,11,2)
          and votes.shape==(n,11),'S173 true new GOAT 11-family shape')
    check(len(fam)==11 and len(set(fam))==11 and np.isin(old,range(7)).all()
          and np.isin(proposals,range(7)).all(),'original family identities or K invalid')
    active=proposals!=old[:,None]
    check(np.isfinite(risk[active]).all() and np.isnan(risk[~active]).all(),
          'partial 11 trained Ridge risk contract')
    check(all(set(FAMILY_GROUPS[group]).issubset(fam) for group in EXPECTED_GROUPS),
          'two independently complete genuine S46 sub-families absent')
    check(not set(FAMILY_GROUPS['historic_acoustic']).issubset(fam),
          'historic acoustic S46 family accidentally claimed ready')
    output={'series18_parent_adapted':old.copy()}
    summary={}
    for group in EXPECTED_GROUPS:
        cols=[fam.index(name) for name in FAMILY_GROUPS[group]]
        for lam in LAM_POLY:
            for cut in CUT_POLY:
                for minvote in VOTES:
                    for structure in STRUCTURE:
                        key=f'series46_partial11__{group}__polyLambda{lam:g}__cut{cut:g}__votes{minvote}__{structure}'
                        new,j=select(old,proposals,risk,votes,cols,lam,cut,minvote,structure)
                        check(np.isin(new,range(7)).all() and
                              np.array_equal(new[j<0],old[j<0]),
                              'S46 protected votes/policy violated')
                        output[key]=new
                        delta=new!=old
                        summary[key]={'changed_vs_adapted_S18':int(delta.sum()),
                            'source_K0to6_histogram':np.bincount(new,minlength=7).tolist(),
                            'selected_original_source_families':
                                {fam[k]:int((j==k).sum()) for k in cols if np.any(j==k)}}
    check(len(output)==97 and len(summary)==96,
          '2 real original S46 family groups, 96 variants + parent')
    tally=np.asarray([v['changed_vs_adapted_S18'] for v in summary.values()])
    out.mkdir(parents=True)
    np.savez_compressed(out/'S180_partial_S46_two_groups_97_preds.npz',
        row_id=ids,variant_ids=np.asarray(list(output)),
        predictions=np.column_stack(list(output.values())).astype(np.int8))
    report={
       'status':'S180_REAL_11_SOURCE_96_S46_SUBGROUP_POLICIES_RUN_ON_GOAT_EXPERIMENT',
       'source_S173_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'native_GOAT_events':n,'original_S45_ridge_checkpoints_employed':209,
       'families_present':fam,'families_count':len(fam),
       'two_fully_available_S46_groups':list(EXPECTED_GROUPS),
       'S46_variants_excluding_adapted_S18':len(summary),
       'S46_max_changes_over_variants':int(tally.max()),
       'S46_nontrivial_variants_with_any_change':int((tally>0).sum()),
       'S46_all_policy_selected_changes':summary,
       'unavailable_full_original_S46_groups':['all16','historic_acoustic'],
       'no_fabricated_missing_family_votes':True,
       'vote_contract':'actual 11-family consensus, DIFFERENT from original 16-family S46 agreement',
       'original_full_S46_equivalence':False,
       'original_parent_S18_equivalence':False,
       'GOAT_annotations_MIDI_or_true_K_read':False,
       'independent_ExactK_accuracy_measured':False,
       'official_S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in (
       'status','native_GOAT_events','families_count',
       'S46_variants_excluding_adapted_S18','S46_max_changes_over_variants',
       'S46_nontrivial_variants_with_any_change','official_S80_promoted')},indent=2))
def selftest():
    check(len(FAMILY_GROUPS)==4 and
          len(LAM_POLY)*len(CUT_POLY)*len(VOTES)*len(STRUCTURE)==48,
          'historic S46 policy cardinality drift')
    check(sum(len(FAMILY_GROUPS[k]) for k in EXPECTED_GROUPS)==6,
          'source group composition drift')
    print('S180 PASS 2 real complete three-producer S46 groups, 96 frozen policy variants; not original all16')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        check(a.source and a.out,'S173 real source mandatory')
        run(a.source,a.out)
