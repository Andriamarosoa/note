"""S397: actual FULL S70 -> S71 -> S72 -> S79 -> S80 replay with safe
label-free K5->K4 S70 suppression. This is a development hypothesis derived
from exposed GuitarSet S395/S396, NOT independent, NOT promotion.

Only suppress original active 5->4 S70 proposals before downstream heads.
Use the UNMODIFIED original S271 and S272 functions to really recompute all
downstream trained heads, K1 critics and K0/K2 tail with unchanged weights.
No oracle from truth, no posthoc alteration of final S80 predicted K.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s271_original_S71_S72_GuitarSet_59309 import run as run_S271
from scripts.s272_full_new_S80_GuitarSet_59309 import run as run_S272

KEYS={'row_id','s69_polymax_K','s69_balanced_K','s18_K',
      'proposed_poly_K','proposed_poly_fold_K','proposed_poly_fold_active',
      'proposed_poly_vote_accepted',
      'downward_poly_K','downward_poly_fold_K','downward_poly_fold_active',
      'downward_poly_vote_accepted'}
def ensure(v,m):
    if not v:raise ValueError(m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(source_s70,s69,s97,s71models,s72models,s46,pr16,s80models,existing_s80,out):
    ensure(not out.exists(),'S397 immutable experiment cannot overwrite any run')
    with np.load(source_s70,allow_pickle=False) as z:
        ensure(set(z.files)==KEYS,'original archived S70 source fields drift')
        source={k:z[k].copy() for k in KEYS}
    with np.load(existing_s80,allow_pickle=False) as z:
        original_ids=z['row_id'].copy()
        orig_pred=z['s80_K'].copy()
        orig_parent=z['s72_K'].copy()
    n=len(source['row_id'])
    ensure(n==59309 and np.array_equal(source['row_id'],original_ids),
           'original full S80 and S70 native event identity drift')
    old_parent=source['s69_polymax_K'].astype(np.int8)
    proposed=source['proposed_poly_K'].astype(np.int8)
    accepted=source['proposed_poly_vote_accepted']
    fold_action=source['proposed_poly_fold_active']
    fold_k=source['proposed_poly_fold_K']
    ensure(proposed.shape==(n,) and accepted.shape==(n,)
           and fold_action.shape==(4,n) and fold_k.shape==(4,n),
           'original four S70 fold posterior decision shape drift')
    # A fully predeclared label-blind policy change at the S70 boundary:
    # protect any K5 original S69 parent against downgrading to K4.
    # Never rely on whether final S80 formerly predicted K4.
    selected=(old_parent==5)&(proposed==4)&accepted
    ensure(int(selected.sum())>100,
           'expected legitimate active S70 5->4 instances absent')
    ensure(np.all(fold_action[:,selected].sum(axis=0)>=3),
           'cannot cancel insufficient non-deterministic S70 votes')
    copy={k:v.copy() for k,v in source.items()}
    copy['proposed_poly_K'][selected]=5
    copy['proposed_poly_vote_accepted'][selected]=False
    copy['proposed_poly_fold_K'][:,selected]=5
    copy['proposed_poly_fold_active'][:,selected]=False
    # Exact consistency with intact S69 parent everywhere suppression occurs.
    ensure(np.array_equal(copy['proposed_poly_K'][selected],old_parent[selected])
           and not np.any(copy['proposed_poly_vote_accepted'][selected])
           and not np.any(copy['proposed_poly_fold_active'][:,selected])
           and np.array_equal(copy['downward_poly_K'],source['downward_poly_K']),
           'S70 conditional suppression mangled original proposal')
    out.mkdir(parents=True)
    gated=out/'S397_s70_guarded_fourfold_truthfree.npz'
    np.savez_compressed(gated,**copy)
    # Actual trained models re-executed, not a final-S80 classification flip.
    run_S271(s69,gated,s97,s71models,s72models,out/'S397_S271_FULL_REPLAY')
    ran_s271=out/'S397_S271_FULL_REPLAY'/'S271_original_S71_S72_native_GUITARSET.npz'
    run_S272(ran_s271,s46,pr16,s97,s80models,out/'S397_S272_FULL_REPLAY')
    ran_s272=out/'S397_S272_FULL_REPLAY'/'S272_full_original_S80_unseen_GUITARSET_59309_frozen.npz'
    with np.load(ran_s272,allow_pickle=False) as z:
        ensure(np.array_equal(original_ids,z['row_id']), 'S397 final S80 source IDs mismatch')
        new=z['s80_K'].copy()
        new_parent=z['s72_K'].copy()
        s79=z['s79_K'].copy()
    with np.load(ran_s271,allow_pickle=False) as z:
        new_s70=z['s70_proposed_poly_K'].copy()
        new_s71=z['s71_poly_K'].copy()
        new_s72=z['s72_poly_poly_guard_K'].copy()
    ensure(np.array_equal(new_parent,new_s72) and
           np.array_equal(copy['proposed_poly_K'],new_s70),
           'original replay lineage changed between S70,S71,S72,S80')
    mutations=np.flatnonzero(orig_pred!=new)
    # We did not read a single truth/label here, and the final ledger contains
    # no score, correctness, or class identity.
    report=dict(status='S397_SOURCE_S70_K5_DOWNCAST_GUARD_FULL_S80_REPLAY_LABEL_FREE',
       original_source_event_count=n,guard_rule='S69 K==5, original S70 accepted K==4; suppress S70 action',
       precommitted_guard_actions=int(selected.sum()),
       source_unchanged_samples=int((~selected).sum()),
       S71_original_4_checkpoints_reexecuted=True,
       S72_original_4_checkpoints_reexecuted=True,
       S79_original_4_checkpoints_reexecuted=True,
       S80_original_K0_K2_tail_reexecuted=True,
       full_S80_replay_output_changes=int(len(mutations)),
       changed_K_distribution_before=np.bincount(orig_pred[mutations],minlength=7).tolist(),
       changed_K_distribution_after=np.bincount(new[mutations],minlength=7).tolist(),
       source_sha256={'original_S70':digest(source_s70),
                       'S69':digest(s69),'S97':digest(s97),
                       'S46_193':digest(s46),'PR16':digest(pr16),
                       'original_S80':digest(existing_s80),
                       'guarded_S70':digest(gated),
                       'replayed_S271':digest(ran_s271),
                       'replayed_S80':digest(ran_s272)},
       labels_used=False,
       final_prediction_posthoc_modified=False,
       S80_reference_model_weights_unchanged=True,
       GuitarSet_not_independent_training_overlap=True,
       fully_new_external_music_test_completed=False,
       reference_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('S397 PASS real full S70-S80 label-free replay; original S80 unchanged',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-s70','s69','s97','s71models','s72models','s46',
              'pr16','s80models','existing-s80','out'):
        p.add_argument('--'+k,required=True,type=Path)
    z=p.parse_args()
    run(z.source_s70,z.s69,z.s97,z.s71models,z.s72models,z.s46,
        z.pr16,z.s80models,z.existing_s80,z.out)
