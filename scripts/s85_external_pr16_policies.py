"""S85: native, label-free reconstruction of all 36 PR16 policies.

Applies the exact policy definitions from summarize_v273_open_cohort.py:
12 neural heads x mixing alphas (0.6, 0.8, 1.0) with frozen K parent.
Never reads MIDI, annotations or true_K. Alignment is by audio and sample.
This is NOT the missing S46/S72/S80 stack, and not an accuracy test.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np

ALPHAS=(.6,.8,1.)
def require(test,message):
    if not test:raise ValueError(message)
def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def original_pr16_policies(probs,base):
    """Byte-exact interpretation of original PR16 36-policy summary code."""
    n=len(base)
    require(probs.shape==(n,12,7) and base.shape==(n,), 'policy dimensions')
    require(np.isin(base,range(7)).all(), 'invalid frozen parent K')
    original=np.eye(7,dtype=np.float32)[base]
    cols=[]
    for j in range(12):
        for alpha in ALPHAS:
            mixed=alpha*probs[:,j]+(1-alpha)*original
            winner=mixed.argmax(1)
            oldscore=mixed[np.arange(n),base]
            newscore=mixed[np.arange(n),winner]
            cols.append(np.where(newscore>oldscore,winner,base).astype(np.int8))
    return np.column_stack(cols)
def canon_name(member):
    m=re.fullmatch(r'00_GOATexample([12])_comp\.jams',str(member))
    require(m is not None,'unknown external member: '+str(member))
    return 'example'+m.group(1)+'_di.wav'
def run(groups,prob_path,incumbent_path,output):
    require(not output.exists(),'audit output exists')
    g=npz(groups);p=npz(prob_path);z=npz(incumbent_path)
    require(set(g)=={'member','starts','ends'},'native groups schema')
    require(set(p)=={'member','start_samples','probabilities','producer_names'},'PR16 inputs schema')
    require(np.array_equal(g['member'],p['member']) and
        np.array_equal(g['starts'],p['start_samples']),'native group/prediction mismatch')
    require(p['probabilities'].shape==(len(g['starts']),12,7),'PR16 head count drift')
    names=np.asarray([canon_name(v) for v in z['member']])
    keys=[(str(v),int(s)) for v,s in zip(names,z['start_samples'])]
    require(len(keys)==len(set(keys)),'duplicate frozen reference native event IDs')
    ref_by={key:int(k) for key,k in zip(keys,z['incumbent_K'])}
    event_keys=[(str(m),int(s)) for m,s in zip(g['member'],g['starts'])]
    require(len(event_keys)==len(set(event_keys)),'duplicate external event IDs')
    require(all(k in ref_by for k in event_keys),'PR16 groups not found in official frozen reference')
    old=np.asarray([ref_by[k] for k in event_keys],np.int8)
    predictions=original_pr16_policies(p['probabilities'],old)
    require(predictions.shape==(len(old),36),'policy population invalid')
    require(np.all(np.isin(predictions,range(7))),'invalid PR16 K')
    vote=np.stack([(predictions==k).sum(1) for k in range(7)],1)
    require(np.all(vote.sum(1)==36),'invalid PR16 votes')
    output.mkdir(parents=True)
    np.savez_compressed(output/'pr16_external_36_policies.npz',
       member=g['member'],start_samples=g['starts'],end_samples=g['ends'],
       incumbent_freeze_K=old,pr16_policy_K=predictions,
       pr16_vote_count=vote,pr16_policy22_K=predictions[:,22],
       pr16_network_probabilities=p['probabilities'])
    report={'status':'EXTERNAL_NATIVE_PR16_36_POLICY_RECONSTRUCTION_COMPLETE',
       'independent_model_validation':False,
       'external_truth_used':False,'full_S80_executed':False,
       'formula':'alpha*q+(1-alpha)*onehot(original_freeze_K); argmax only if strictly better',
       'mixing_alphas':list(ALPHAS),'neural_heads':12,'policy_columns':36,
       'matched_events':len(old),'original_frozen_events':len(keys),
       'policy22_network_index':7,'policy22_alpha':.8,
       'proposed_counts_policy22':np.bincount(predictions[:,22],minlength=7).tolist(),
       'baseline_counts_matched':np.bincount(old,minlength=7).tolist(),
       'policy22_changes':int(np.sum(predictions[:,22]!=old)),
       'output_sha256':digest(output/'pr16_external_36_policies.npz'),
       'remaining_blockers':['external S45 16 original proposal families not deployed',
           'external S46 193 decision policies not deployed',
           'external S69-S72 cascaded meta selectors not deployed',
           'S80 four fold-specific K1 critics require predeclared deployment aggregation']}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--groups',type=Path,required=True)
    a.add_argument('--probabilities',type=Path,required=True)
    a.add_argument('--incumbent',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    v=a.parse_args()
    run(v.groups,v.probabilities,v.incumbent,v.output)
