"""S145: audit S41 zero interventions on new audio, with NO labels.

Distinguish lack of non-KEEP proposals from zero gate preference.
Uses immutable S144 scores, trained S38/S40 proposals; no inference
retraining, no arbitrary parameter tuning or GOAT truth/MIDI.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def require(v,m):
    if not v:raise ValueError(m)
def run(source,out):
    require(not out.exists(),'immutable audit')
    with np.load(source,allow_pickle=False) as z:
        required={'row_id','parent_K','proposal_K','gate_scores',
                  'proposed_K','S38_four_expert_probabilities',
                  'S40_cnn_probabilities','selected_route'}
        require(set(z.files)==required,'unexpected label or schema')
        ids=z['row_id'].astype(str)
        parent=z['parent_K'].astype(np.int8)
        proposal=z['proposal_K'].astype(np.int8)
        gate=z['gate_scores'].astype(np.float64)
        result=z['proposed_K'].astype(np.int8)
        chosen=z['selected_route'].astype(np.int8)
        s38=z['S38_four_expert_probabilities'].astype(np.float64)
        s40=z['S40_cnn_probabilities'].astype(np.float64)
    require(len(ids)==57 and len(set(ids))==57,'native 57 keys')
    require(proposal.shape==(57,6) and gate.shape==(57,6) and
            s38.shape==(57,4,7) and s40.shape==(57,7),'original S41 6 input heads')
    require(np.array_equal(proposal[:,0],parent),'KEEP must equal parent')
    require(np.array_equal(chosen,gate.argmax(1)),'original S41 argmax drift')
    require(np.array_equal(result,parent),'expected S144 frozen keep result')
    require(np.allclose(gate.sum(1),1,atol=1e-5),'trained model score normalization')
    source_diff=proposal[:,1:]!=parent[:,None]
    eligible=parent>=1
    proposed_any=source_diff.any(1)
    nonkeep_best=gate[:,1:].max(1)
    keep=gate[:,0]
    margin=keep-nonkeep_best
    # Report factual ranks; do not create new gated decisions.
    details=[]
    for head in range(1,6):
        p=proposal[:,head]
        other=(p!=parent)
        details.append({'head_index':head,'different_K':int(other.sum()),
             'different_K_with_parent_K1plus':int((other&eligible).sum()),
             'times_this_head_has_highest_nonKEEP_score':
                int((gate[:,1:].argmax(1)==head-1).sum()),
             'mean_gate_weight':float(gate[:,head].mean())})
    report={
      'status':'S145_S41_LABEL_FREE_ABSTENTION_CAUSE_AUDIT',
      'GOAT_real_native_rows':len(parent),
      'saved_trained_checkpoint_source':'S144 original S38 16 + S40 4 + S41 19',
      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'parent_K0_event_count':int((parent==0).sum()),
      'parent_K1plus_eligible_event_count':int(eligible.sum()),
      'events_with_at_least_one_nonKEEP_source_proposal':int(proposed_any.sum()),
      'eligible_events_with_nonKEEP_source_proposal':int((proposed_any&eligible).sum()),
      'events_nonKEEP_head_score_above_KEEP':int((nonkeep_best>keep).sum()),
      'eligible_events_nonKEEP_head_score_above_KEEP':int(((nonkeep_best>keep)&eligible).sum()),
      'KEEP_argmax':int((chosen==0).sum()),
      'actual_changed_K':int((result!=parent).sum()),
      'keep_gate_weight_mean':float(keep.mean()),
      'keep_gate_weight_min':float(keep.min()),
      'keep_gate_weight_max':float(keep.max()),
      'KEEP_margin_over_best_nonKEEP_median':float(np.median(margin)),
      'KEEP_margin_over_best_nonKEEP_min':float(margin.min()),
      'KEEP_margin_over_best_nonKEEP_p95':float(np.quantile(margin,.95)),
      'heads':details,
      'failure_attribution':'gate_veto' if int((proposed_any&eligible).sum())>0
                            and int((chosen!=0).sum())==0 else 'no_eligible_alternative',
      'original_S80_chain_complete':False,
      'truth_MIDI_read':False,'promotion':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
def selftest():
    base=np.array([0,1],np.int8)
    p=np.array([[0,1,2],[1,1,2]],np.int8)
    change=p[:,1:]!=base[:,None]
    assert change.shape==(2,2) and change.any(1).tolist()==[True,True]
    print('S145 PASS: eligible family proposal vs gate preference audit')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--input',type=Path)
    a.add_argument('--output',type=Path)
    a.add_argument('--self-test',action='store_true')
    x=a.parse_args()
    if x.self_test:selftest()
    else:
        require(x.input and x.output,'frozen S144 input required')
        run(x.input,x.output)
