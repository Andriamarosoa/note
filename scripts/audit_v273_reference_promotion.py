"""S74 promotion audit: sealed independent holdout required (never promotes automatically)."""
from __future__ import annotations
import argparse,json,hashlib,zipfile,io
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)

def check(ok,why):
    if not ok: raise ValueError(why)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def pair(y,old,new):
    prev=old==y;curr=new==y
    return {"count":int(len(y)),
            "correct_reference":int(prev.sum()),
            "correct_candidate":int(curr.sum()),
            "gain":int(curr.sum()-prev.sum()),
            "corrections":int((~prev&curr).sum()),
            "regressions":int((prev&~curr).sum())}

def evaluate(ref,proposal,output,features=None,holdout=None,manifest=None):
    check(not output.exists(),'cannot overwrite a prior audit')
    a=load(ref);b=load(proposal)
    for field in ('global_index','true_K','fold'):
        check(np.array_equal(a[field],b[field]),'unaligned '+field)
    ids=a['global_index'];y=a['true_K'];fold=a['fold']
    old=a['baseline_K'];new=b['s72_poly_poly_guard_K']
    check(len(ids)==len(set(map(int,ids)))==59309,'historical cohort drift')
    check(set(fold.tolist())==set(FOLDS) and int((y>=2).sum())==7385,'fold/class count drift')
    for field in (y,old,new):
        check(np.isin(field,range(7)).all(),'invalid K')
    check(int((old==y).sum())==48454 and int((new==y).sum())==49238,
          'frozen reference/candidate drift')
    report={"promotion_status":"BLOCKED_MISSING_INDEPENDENT_EVIDENCE",
            "reference":"freeze_local_combo",
            "candidate":"S72_poly_poly_guard",
            "development":{"overall":pair(y,old,new),
                           "poly":pair(y[y>=2],old[y>=2],new[y>=2]),
                           "by_K":{str(k):pair(y[y==k],old[y==k],new[y==k])
                                   for k in range(7)},
                           "by_fold":{str(k):pair(y[fold==k],old[fold==k],new[fold==k])
                                      for k in FOLDS}},
            "holdout":{"status":"MISSING"},
            "sources":{"reference_sha256":sha(ref),"candidate_sha256":sha(proposal)}}
    if features is not None:
        assets=[]
        for outer in FOLDS:
            files=list(features.rglob(f'features-fold-{outer}.npz'))
            check(len(files)==1,'missing feature fold '+str(outer))
            file=files[0];rep=json.loads(file.with_name('report.json').read_text())
            check(rep.get('fold')==outer and rep.get('fold3_used') is False and
                  rep.get('player05_used') is False,'unexpected holdout exposure')
            check(rep.get('feature_sha256')==sha(file),'bad feature hash')
            with np.load(file,allow_pickle=False) as feat:
                ix=feat['global_index']
                check(len(np.unique(ix))==len(ix),'repeated feature IDs')
                check(np.all(np.isin(ix,ids)),'foreign feature ID')
            assets.append({"fold":outer,"sha256":sha(file)})
        report["sources"]["features"]=assets
    if holdout is not None or manifest is not None:
        check(holdout is not None and manifest is not None,
              'holdout and manifest must be supplied together')
        m=json.loads(manifest.read_text())
        for flag in ('sealed_before_selection','producers_excluded_holdout',
                     'selector_excluded_holdout','upstream_models_nested'):
            check(m.get(flag) is True,'failed independent provenance: '+flag)
        check(sha(holdout)==m.get('test_sha256'),'holdout test hash drift')
        h=load(holdout)
        required=('global_index','true_K','incumbent_K','candidate_K','member')
        check(all(k in h for k in required),'holdout arrays incomplete')
        hi=h['global_index'];hy=h['true_K'];ho=h['incumbent_K'];hn=h['candidate_K']
        check(len(hi)>0 and len(set(hi.tolist()))==len(hi)
              and not np.isin(hi,ids).any(),'holdout identity overlap')
        members=set(map(str,h['member'].tolist()))
        check(members==set(m.get('holdout_members',[])) and bool(members),
              'holdout recording names not sealed')
        all_test=pair(hy,ho,hn)
        poly_test=pair(hy[hy>=2],ho[hy>=2],hn[hy>=2])
        check(poly_test['count']>0,'missing polyphonic holdout')
        passed=all_test['gain']>0 and poly_test['gain']>0
        report["holdout"]={"status":"PASS" if passed else "FAIL",
                           "overall":all_test,"poly":poly_test}
        report["promotion_status"]="READY_FOR_MANUAL_REVIEW" if passed else "FAILED_HOLDOUT"
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('promotion_status','holdout')},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--features',type=Path)
    p.add_argument('--holdout',type=Path)
    p.add_argument('--manifest',type=Path)
    a=p.parse_args()
    evaluate(a.reference,a.candidate,a.output,a.features,a.holdout,a.manifest)
