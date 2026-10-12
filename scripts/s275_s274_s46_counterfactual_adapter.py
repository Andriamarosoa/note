"""S275: adapt frozen S274 no-label S46 bank to unchanged S269->S272 runtimes.

Only source-K columns are swapped; original checkpoint descriptors/folds remain
intact. This archive is a COUNTERFACTUAL input, never an original-S268 claim.
No fitting, threshold selection, truth or IDMT annotations at this boundary.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

S268 = {"row_id","variant_ids","original_freeze_K","S18_parent_K",
        "original_19_branch_folds","S46_193_policy_K","per_fold_specialist_K",
        "active_fold_count","parent_incompatible_specialists","contradictory_specialists"}
S274 = {"row_id","variant_ids","S18_parent_K","original_S46_K",
        "S274_twofold_S46_K","eligible_fold_count",
        "support_for_original_proposal"}
def need(v,msg):
    if not v: raise ValueError(msg)
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==keys,"unexpected label/source field: "+str(p))
        return {k:z[k].copy() for k in z.files}
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(original, stricter, out):
    need(not out.exists(),"never overwrite frozen S275 counterfactual")
    old=read(original,S268);proposal=read(stricter,S274)
    n=len(old['row_id'])
    need(n==59309 and old['S46_193_policy_K'].shape==(59309,193)
         and all(np.array_equal(old[k],proposal[k]) for k in
                 ('row_id','variant_ids','S18_parent_K'))
         and np.array_equal(old['S46_193_policy_K'],proposal['original_S46_K']),
         'S274 must be paired to exact original S268 bank')
    parent=old['S18_parent_K'][:,None]
    new=proposal['S274_twofold_S46_K']
    changed=old['S46_193_policy_K']!=parent
    survived=new!=parent
    need(np.all(~survived|changed) and
         np.all((new==old['S46_193_policy_K'])|(new==parent)),
         'S275 can only abstain, not invent K changes')
    new_source=dict(old)
    new_source['S46_193_policy_K']=new
    out.mkdir(parents=True)
    path=out/'S275_counterfactual_s268_schema_twofold_S46.npz'
    np.savez_compressed(path,**new_source)
    manifest=dict(
        status='S275_S274_COUNTERFACTUAL_S46_BANK_FOR_ORIGINAL_FROZEN_S69_S80',
        event_count=n, S46_policy_count=193,
        original_S268_SHA256=digest(original),
        prelabel_S274_SHA256=digest(stricter),
        candidate_SHA256=digest(path),
        original_changed_policy_event_cells=int(changed.sum()),
        retained_policy_event_cells=int(survived.sum()),
        source_model_weights_unchanged=True,
        original_S219_source_lineage_preserved=True,
        this_policy_bank_is_not_original_S268=True,
        labels_or_midi_read=False,
        downstream_S269_S272_completed=False,
        independent_accuracy_measured=False,
        official_promotion=False)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('original','stricter','out'):
        a.add_argument('--'+k,required=True,type=Path)
    p=a.parse_args();run(p.original,p.stricter,p.out)
