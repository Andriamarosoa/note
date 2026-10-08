"""Audit competing scores for duplicate actions before consensus training."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.verify_v273_selector_repair_artifacts import named_pairs, require, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    proof=json.loads((Path(__file__).resolve().parents[1]/'analysis/evidence/v273-group127/verification.json').read_text())
    result=dict(source_run=37777844182,independent_validation=False,
        diagnostic_mean_pool_is_not_retrained=True,arms={});cases=[]
    for arm in ['direct','global','local']:
        path=args.artifacts/arm/'predictions.npz'
        require(digest(path)==proof['arms'][arm]['predictions_sha256'],'archive differs')
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        lookup={v:i for i,v in enumerate(d['global_index'])}
        at=np.array([lookup[v] for v in d['eligible_global_index']])
        y=d['true_K'][at];b=d['frozen_baseline_K'][at]
        pred=d['predicted_K'][at];single=d['singletons_only_K'][at]
        proposals=d['group_proposal'];cor=d['group_outcome_probability'][:,:,0]
        risk=d['baseline_correct_probability'];row=np.arange(len(y));mask=d['chosen_group_mask']
        count=np.stack([(proposals==k).sum(1) for k in range(2,7)],1)
        lo=np.stack([np.min(np.where(proposals==k,cor,np.inf),1) for k in range(2,7)],1)
        hi=np.stack([np.max(np.where(proposals==k,cor,-np.inf),1) for k in range(2,7)],1)
        mean=np.stack([np.where(proposals==k,cor,0).sum(1)/np.maximum(count[:,k-2],1) for k in range(2,7)],1)
        change=np.arange(2,7)[None]!=b[:,None];active=(count>1)&change
        signs=active&(lo<risk[:,None])&(hi>risk[:,None])
        added_reg=(single==y)&(pred!=y);added_cor=(single!=y)&(pred==y)
        singleton_has_target=(proposals[:,[0,1,3,7,15,31,63]]==pred[:,None]).any(1)
        scores=np.where((count>0)&change,mean-risk[:,None],-np.inf)
        best=scores.argmax(1);mean_pred=np.where(scores[row,best]>0,best+2,b)
        entry=dict(predictions_sha256=digest(path),eligible_rows=len(y),
            duplicate_action_events=int(active.any(1).sum()),
            same_action_correction_probability_range_quantiles=dict(zip(['median','p90','max'],np.quantile((hi-lo)[active],[.5,.9,1]).tolist())),
            events_with_positive_and_negative_gain_for_same_verdict=int(signs.any(1).sum()),
            summed_max_action_correction_probability_exceeds_base_wrong_probability=int((np.where(change&(count>0),hi,0).sum(1)>1-risk+1e-6).sum()),
            added_corrections=int(added_cor.sum()),added_regressions=int(added_reg.sum()),
            added_regressions_with_target_already_available_as_singleton=int((added_reg&singleton_has_target).sum()),
            diagnostic_mean_probability_redecode=named_pairs(y,b,mean_pred))
        result['arms'][arm]=entry
        if arm=='global':
            for i in np.flatnonzero(added_cor|added_reg):
                k=int(pred[i]);chosen=max(int(mask[i])-1,0)
                cases.append(dict(global_index=int(d['eligible_global_index'][i]),fold=int(d['fold'][at[i]]),
                    effect='regression' if added_reg[i] else 'correction',true_K=int(y[i]),frozen_K=int(b[i]),
                    singleton_K=int(single[i]),all_groups_K=k,chosen_mask=int(mask[i]),
                    singleton_already_offers_target=bool(singleton_has_target[i]),
                    chosen_gain=float(d['group_expected_gain'][i,chosen]) if mask[i]>0 else 0.,
                    min_same_target_correction_probability=float(lo[i,k-2]),max_same_target_correction_probability=float(hi[i,k-2]),
                    baseline_correct_probability=float(risk[i]),groups_offering_target=int(count[i,k-2])))
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'global-groups-vs-singletons.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(cases)
    result['case_file']=dict(path=path.name,rows=len(cases),sha256=digest(path))
    (args.output/'duplicate-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({a:v['events_with_positive_and_negative_gain_for_same_verdict'] for a,v in result['arms'].items()}))


if __name__=='__main__':main()
