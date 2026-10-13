"""S347: frozen S274 two-compatible-fold S46 gate on genuine 477 unseen V88 rows.

Replay the PREDECLARED twofold S274 strict quorum rule. This is a
counterfactual deployment rule, not re-fitting any of the trained original
S45/S46 models. Compare original S346 and new S347 on exactly same rows.
DO NOT read annotations or report independent performance before S80
frozen and composition-level novelty proven.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s274_audit_s46_twofold_gate_GuitarSet import twofold_gate
S346={'row_id','variant_ids','original_freeze_K','S18_parent_K',
      'original_19_branch_folds','S46_193_policy_K','per_fold_specialist_K',
      'active_fold_count','parent_incompatible_specialists','contradictory_specialists'}
S338={'row_id','freeze_K','original_fold_ids','original_S18_by_fold_K',
      'S18_piece_neutral_consensus_K','S18_quorum_of_3_supported',
      'S18_vote_strength','S18_abstention_to_freeze'}
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,"S347 unknown source fields/labels forbidden")
        return {k:z[k].copy() for k in z.files}
def run(s346,s338,out):
    need(not out.exists(),"S347 immutable source-only S46 gate")
    old=load(s346,S346)
    s18=load(s338,S338)
    ids=old['row_id'];n=len(ids)
    need(n==477 and np.array_equal(ids,s18['row_id'])
         and np.array_equal(old['S18_parent_K'],
                            s18['S18_piece_neutral_consensus_K'])
         and np.array_equal(old['original_freeze_K'],s18['freeze_K'])
         and old['S46_193_policy_K'].shape==(477,193)
         and tuple(map(int,s18['original_fold_ids']))==(0,1,2,4)
         and tuple(map(int,old['original_19_branch_folds']))==
         tuple([0]*5+[1]*4+[2]*5+[4]*5),
         "not exact historical source 19 branches and new 477 native events")
    new,supporters,active=twofold_gate(
        old['S46_193_policy_K'],old['per_fold_specialist_K'],
        s18['original_S18_by_fold_K'],old['S18_parent_K'],
        old['active_fold_count'])
    before=old['S46_193_policy_K']!=old['S18_parent_K'][:,None]
    after=new!=old['S18_parent_K'][:,None]
    need(np.all(~after|before) and np.all((new==old['S46_193_policy_K'])|
              (new==old['S18_parent_K'][:,None])),
         "S347 cannot introduce new interventions beyond frozen S346")
    out.mkdir(parents=True)
    oldarchive=out/'S347_original_S346_193_S46_native_477.npz'
    np.savez_compressed(oldarchive,**old)
    counter=dict(old)
    counter['S46_193_policy_K']=new
    result=out/'S347_original_source_counterfactual_S274_gate_477.npz'
    np.savez_compressed(result,**counter)
    report=dict(status='S347_SOURCE_ONLY_ORIGINAL_S46_S274_TWOFOLD_COHORT_477',
       native_events=477,policies=193,
       S346_source_sha256=sha(s346),
       S338_original_S18_source_sha256=sha(s338),
       original_intervention_cells=int(before.sum()),
       onefold_interventions_suppressed=int(np.count_nonzero(before&~after)),
       two_or_more_fold_interventions_retained=int(after.sum()),
       affected_events=int(np.count_nonzero(np.any(
           new!=old['S46_193_policy_K'],axis=1))),
       original_S346_snapshot_sha256=sha(oldarchive),
       stricter_S274_counterfactual_sha256=sha(result),
       S274_trained_rule_unchanged=True,
       historical_S45_weights_refitted=False,
       new_S275_full_S80_not_yet_replayed=True,
       true_K_or_XML_MIDI_read=False,
       song_composition_independence_not_yet_proven=True,
       independent_ExactK_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for a in ("s346","s338","out"):p.add_argument("--"+a,required=True,type=Path)
    a=p.parse_args();run(a.s346,a.s338,a.out)
