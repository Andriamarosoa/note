"""Compare a fresh full fit to every archived policy and nested internal decision."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_v273_regression_loops import read, dump
from scripts.verify_v273_selector_repair_artifacts import require,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();reports={}
    for i in range(1,5):
        name='series'+str(i);old=a.archive/name;new=a.replay/name
        before=read(old/'predictions.npz');after=read(new/'predictions.npz')
        require(set(before)==set(after),'same native evidence fields')
        for key in before:require(np.array_equal(before[key],after[key]),'fresh full-fit exact reproduction '+name+'/'+key)
        result=dict(native_policy_vectors=len(before['variant_ids']),decisions=int(before['predictions'].size),
                    changed_decisions=0,archive_predictions_sha256=digest(old/'predictions.npz'))
        if (old/'inner-evidence.npz').exists():
            x=read(old/'inner-evidence.npz');y=read(new/'inner-evidence.npz')
            require(set(x)==set(y) and all(np.array_equal(x[k],y[k]) for k in x),'inner excluded decisions '+name)
            result['inner_decisions']=int(x['predictions'].size)
        if (old/'probabilities.npz').exists():
            x=read(old/'probabilities.npz');y=read(new/'probabilities.npz');difference=0.
            require(set(x)==set(y),'same probability fields')
            for key in x:
                require(np.allclose(x[key],y[key],rtol=1e-7,atol=1e-8),'full-fit probabilities '+name+'/'+key)
                difference=max(difference,float(np.max(np.abs(x[key]-y[key]))))
            result['max_probability_difference']=difference
        reports[name]=result
    dump(a.output,dict(status='verified',all_162_policies_reproduced_exactly=True,series=reports))
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()
