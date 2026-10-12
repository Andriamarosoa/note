"""S181 independent numerical verifier of S180 96 decision vectors.

No import of S46 policy selector. Recomputes every utility and mask using
NumPy, compares 96 policies exactly and identifies all changed GOAT event
identities without reading MIDI/truth. Never claims correction.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.audit_v273_s46_classwise_compromise import FAMILY_GROUPS,LAM_POLY,CUT_POLY,VOTES,STRUCTURE
GROUPS=('unique_poly','ordering')
def require(q,m):
    if not q:raise ValueError(m)
def strict(path,keys):
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==set(keys),'independent S181 strict no-label array '+str(path))
        return {k:z[k].copy() for k in z.files}
def independent_decision(parent,proposal,risks,votes,source_ids,lam,cut,minimum,structure):
    row=np.arange(len(parent))
    keep=parent[:,None]
    eligible=(proposal!=keep)&(votes>=minimum)
    include=np.zeros(proposal.shape[1],bool);include[source_ids]=True
    eligible &= include[None]
    if structure=='neighbor_and_no_drop_to_K1':
        eligible &= ((keep<2)|((np.abs(proposal.astype(np.int16)-keep.astype(np.int16))==1)&(proposal>=2)))
    else:require(structure=='open','policy structure unknown')
    u=risks[:,:,0]-np.where(keep>=2,lam,1.)*risks[:,:,1]
    u[~eligible]=-np.inf
    u[~np.isfinite(u)]=-np.inf
    best=np.argmax(u,axis=1)
    winning=u[row,best]
    accept=np.isfinite(winning)&(winning>np.where(parent>=2,cut,.02))
    return np.where(accept,proposal[row,best],parent).astype(np.int8),np.where(accept,best,-1)
def run(s173,s180,out):
    require(not out.exists(),'no overwrite historical proof')
    a=strict(s173,('row_id','S18_parent_K','original_family_ids','source_K',
      'S35_new_K','S35_selected_actions','agreement','trained_Ridge_risks',
      'partial_eleven_selected_head','partial_eleven_candidate_K'))
    b=strict(s180,('row_id','variant_ids','predictions'))
    ids=a['row_id']
    require(np.array_equal(ids,b['row_id']) and len(ids)==57,'source event alignment drift')
    fam=list(map(str,a['original_family_ids']))
    names=list(map(str,b['variant_ids']))
    pred=b['predictions']
    parent=a['S18_parent_K']
    cand=a['source_K'];risks=a['trained_Ridge_risks'];votes=a['agreement']
    require(pred.shape==(57,97) and names[0]=='series18_parent_adapted'
        and np.array_equal(pred[:,0],parent),'S180 frozen baseline changed')
    mismatches=0;head_usage={f:0 for f in fam};actions=[]
    i=0
    for group in GROUPS:
        selected=[fam.index(f) for f in FAMILY_GROUPS[group]]
        for lam in LAM_POLY:
            for cut in CUT_POLY:
                for v in VOTES:
                    for kind in STRUCTURE:
                        key=f'series46_partial11__{group}__polyLambda{lam:g}__cut{cut:g}__votes{v}__{kind}'
                        i+=1
                        require(names[i]==key,'S46 variant ID/order drift')
                        k,index=independent_decision(parent,cand,risks.copy(),votes,
                                                      selected,lam,cut,v,kind)
                        mismatches+=int(np.sum(k!=pred[:,i]))
                        for row in np.flatnonzero(index>=0):
                            h=fam[index[row]]
                            head_usage[h]+=1
                            actions.append((str(ids[row]),int(parent[row]),int(k[row]),h))
    require(i==96 and mismatches==0,'independent S46 mask/risk replay mismatch')
    unique=sorted(set(actions))
    report={'status':'S181_INDEPENDENT_EXACT_REPLAY_S180_96_PARTIAL_S46_POLICIES',
       'S173_SHA256':hashlib.sha256(s173.read_bytes()).hexdigest(),
       'S180_SHA256':hashlib.sha256(s180.read_bytes()).hexdigest(),
       'S46_policies_independently_replayed':96,
       'policy_decision_mismatches':mismatches,
       'distinct_action_event_source_K_triples':len(unique),
       'distinct_events_with_any_selected_change':len({z[0] for z in unique}),
       'changed_events':[
          {'row_id':row,'before_K':before,'proposed_K':after,'family':f}
          for row,before,after,f in unique],
       'selected_source_family_usage_across_96_policies':head_usage,
       'full_16_source_votes_available':False,
       'real_original_S18_equivalence_proven':False,
       'annotation_truth_or_MIDI_read':False,
       'independent_ExactK_accuracy_measured':False,
       'S80_promoted':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))
def selftest():
    p=np.array([1,3,2],np.int8)
    can=np.array([[0,2],[2,4],[1,3]],np.int8)
    scores=np.array([[[.7,.1],[.6,.1]],[[.4,.5],[.8,.2]],[[.4,.3],[.5,.1]]],np.float32)
    votes=np.ones((3,2),np.int8)
    a,j=independent_decision(p,can,scores,votes,[0,1],2.,0.,0,'open')
    require(len(a)==3 and len(j)==3,'isolated NumPy risk mask')
    print('S181 PASS independent frozen S46 decision semantics without importing source select')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s173','s180','out'):p.add_argument('--'+k,type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,k) for k in ('s173','s180','out')),'both sources and output needed')
        run(a.s173,a.s180,a.out)
