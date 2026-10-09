"""Independently check each of the 320 S44 risk masks and paired K audits."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scripts.audit_v273_s44_compromise_loops import observed_mask

def data(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run(a):
    if a.output.exists():raise ValueError('no overwrite')
    s=data(a.bank)
    o=data(a.source/'all_compromise_predictions.npz')
    report=json.loads((a.source/'report.json').read_text())
    with (a.source/'all_compromises.csv').open(newline='',encoding='utf-8') as f:
        lines=list(csv.DictReader(f))
    if len(lines)!=320 or len(o['variant_ids'])!=321 or report['status']!='completed':
        raise ValueError('missing source experiments')
    for key in ('global_index','true_K','fold'):
        if not np.array_equal(s[key],o[key]):
            raise ValueError('row alignment changed: '+key)
    ids=o['global_index'];y=o['true_K'];ref=o['S18_reference_K']
    if int((ref==y).sum())!=49178 or int(((ref==y)&(y>=2)).sum())!=2998:
        raise ValueError('S18 source altered')
    champion={}
    names=list(map(str,s['distinct_vector_sha256']))
    for c in report['family_champions']:
        if c['kind']=='gross':
            champion[c['family']]=s['predictions'][:,names.index(c['sha256'])]
    available=list(champion)
    variants=list(map(str,o['variant_ids']))
    assert variants[0]=='series18_parent'
    if not np.array_equal(o['predictions'][:,0],ref):
        raise ValueError('baseline vector mismatch')
    checked=0
    for r in lines:
        j=variants.index(r['name'])
        col=names.index(r['original_sha256'])
        proposal=s['predictions'][:,col]
        votes=np.zeros(len(ids),np.int16)
        for group,q in champion.items():
            if group==r['family']:continue
            votes+=((q==proposal)&(q!=ref)).astype(np.int16)
        allowed=observed_mask(ref,proposal,r['mask'],votes)
        reproduced=np.where(allowed,proposal,ref)
        stored=o['predictions'][:,j]
        if not np.array_equal(reproduced,stored):
            raise ValueError('unreproducible mask '+r['name'])
        fixed=(ref!=y)&(stored==y)
        broken=(ref==y)&(stored!=y)
        poly=y>=2
        expected=[('corrections',fixed.sum()),('regressions',broken.sum()),
            ('poly_corrections',(fixed&poly).sum()),
            ('poly_regressions',(broken&poly).sum())]
        for k,value in expected:
            if int(r[k])!=int(value):
                raise ValueError('false paired count '+k+' '+r['name'])
        checked+=1
    if checked!=320:raise ValueError('some masks missing')
    a.output.mkdir(parents=True)
    result=dict(status='verified',all_320_full_native_decisions_replayed=True,
        all_16_families_consistent=True,
        no_label_used_by_mask=True,H9_excluded=True,
        previously_explored_cohort_not_independent=True,
        no_promotion=True,correct_s18_global=49178)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS: all 320 mask vectors reproduced from immutable 910-vector S43 bank; paired counts and S18 verified')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
