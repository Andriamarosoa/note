"""S396: read-only structural K5->K4 guard counterfactual on exact S80 stages.

PREREGISTERED three guards, no held-label inputs to guard itself.
This is an IN-SAMPLE development counterfactual, not full neural re-run
after changing upstream S70. Source models/weights/reference unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def require(x,m):
    if not x:raise ValueError(m)
def score(y,old,new,mask):
    a=old[mask]==y[mask];b=new[mask]==y[mask]
    return dict(n=int(mask.sum()),correct_old=int(a.sum()),
      correct_new=int(b.sum()),corrections=int((~a&b).sum()),
      regressions=int((a&~b).sum()),net=int(b.sum()-a.sum()))
def run(s46,s271,s80,your,out):
    require(not out.exists(),'never replace previous counterfactual evidence')
    a=load(s46);r=load(s271);s=load(s80);t=load(your)
    order=np.argsort(t['global_index'].astype(np.int64))
    ids=s['row_id'].astype(str);y=t['k'][order].astype(int)
    base=s['s80_K'].astype(int);s18=a['S18_parent_K'].astype(int)
    parent=r['s69_polymax_K'].astype(int)
    p70=r['s70_proposed_poly_K'].astype(int)
    require(len(ids)==59309 and np.array_equal(ids,a['row_id'].astype(str))
        and np.array_equal(ids,r['row_id'].astype(str))
        and np.array_equal(ids,t['global_index'][order].astype(str))
        and np.array_equal(r['s72_poly_poly_guard_K'],s['s72_K'])
        and int(((y==5)&(base==4)).sum())==224,
        'original score and stage invariants')
    # Three fixed structural rules drawn from S395. No truth in action.
    gate70=(parent==5)&(p70==4)&(base==4)
    gate69=(s18==5)&(parent==4)&(base==4)
    gates={'rescue_S70_5to4':gate70,'rescue_S69_5to4':gate69,
           'rescue_either':gate70|gate69}
    out.mkdir()
    reports={}
    for name,g in gates.items():
        cand=base.copy();cand[g]=5
        reports[name]=dict(actions=int(g.sum()),
           selected_true_hist=np.bincount(y[g],minlength=7).tolist(),
           global_score=score(y,base,cand,np.ones(len(y),bool)),
           poly_score=score(y,base,cand,y>=2),
           by_true_K={f'K{k}':score(y,base,cand,y==k) for k in range(7)},
           fold_ledger={str(k):score(y,base,cand,t['fold'][order]==k)
                        for k in (0,1,2,4)})
    report=dict(status='S396_S70_S69_K5_TO_K4_DIRECT_GUARD_DEV_COUNTERFACTUAL',
      original_event_count=59309,original_true_K5=int((y==5).sum()),
      original_S80_correct=int((base==y).sum()),
      original_S80_weights_changed=False,
      counterfactual_does_NOT_reexecute_downstream_S71_S72=True,
      selected_gate_conditions_are_label_blind=True,
      audit_groups_selected_after_exposure_to_GuitarSet_labels=True,
      training_overlap_in_original_S80=True,
      independent_holdout_measurement=False,
      promotion=False,
      hypotheses=reports)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('S396 PASS original model unchanged: posthoc stage guard no promotion',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('s46','s271','s80','your','out'):
        p.add_argument('--'+arg,required=True,type=Path)
    v=p.parse_args();run(v.s46,v.s271,v.s80,v.your,v.out)
