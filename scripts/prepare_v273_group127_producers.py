"""Freeze all fourteen cross-fit expert producers once for all comparison arms.

The first run exposed <=1.02e-6 differences in logistic probabilities between
runners, with unchanged proposals/audits. This cache removes that numerical
confound without changing any model, fold, loss or training hyperparameter.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_selector_design import aligned_positions, load_feature_rows
from scripts.v273_selector_contract import ProducerPool, matrices, FOLDS, require, id_digest
from scripts.yourmt3_exactk_common import digest


def cache_key(folds):
    return 'fit_'+'_'.join(map(str,folds))


def save_producers(pool, output):
    require(not output.exists(),'refusing cache overwrite')
    data=dict(eligible_global_index=pool.ids,truth=pool.y,baseline=pool.b,fold=pool.fold)
    for size in (1,2,3):
        for key in itertools.combinations(FOLDS,size):
            positions=np.flatnonzero(~np.isin(pool.fold,key))
            p=pool.predict(key,positions,set(FOLDS)-set(key))
            dense=np.full((len(pool.ids),6,5),np.nan,np.float32)
            dense[positions]=p;data[cache_key(key)]=dense
    require(len(pool.manifests)==14,'producer inventory')
    output.mkdir(parents=True)
    np.savez_compressed(output/'producers.npz',**data)
    manifest=dict(producers=list(pool.manifests.values()),cache_sha256=digest(output/'producers.npz'),
        eligible_ids_sha256=id_digest(pool.ids), producer_sets=14,
        note='Each probability row is available only outside its producer training folds.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


class FrozenProducerPool(ProducerPool):
    def __init__(self,x,truth,baseline,folds,ids,directory):
        super().__init__(x,truth,baseline,folds,ids)
        manifest=json.loads((directory/'manifest.json').read_text())
        require(digest(directory/'producers.npz')==manifest['cache_sha256'],'producer cache digest')
        with np.load(directory/'producers.npz',allow_pickle=False) as z:
            self.cache={k:z[k] for k in z.files}
        for key,actual in [('eligible_global_index',self.ids),('truth',self.y),('baseline',self.b),('fold',self.fold)]:
            require(np.array_equal(self.cache[key],actual),'producer cache alignment '+key)
        self.manifests={tuple(p['train_folds']):p for p in manifest['producers']}
        require(len(self.manifests)==14,'cache producer count')
        self.cache_sha256=manifest['cache_sha256']
        for key,p in self.manifests.items():
            take=np.isin(self.fold,key)&(self.y>=2)
            require(np.array_equal(p['train_global_ids'],self.ids[take]),'cached fit IDs')
            require(p['train_id_sha256']==id_digest(self.ids[take]),'cached producer provenance')

    def predict(self,train_folds,positions,forbidden_folds):
        key=tuple(sorted(map(int,train_folds)));positions=np.asarray(positions,int)
        require(not set(key)&set(forbidden_folds),'forbidden cached producer fold')
        require(not np.isin(self.fold[positions],key).any(),'in-sample cached producer use')
        require(key in self.manifests,'unknown cached producer')
        result=self.cache[cache_key(key)][positions]
        require(np.isfinite(result).all(),'missing out-of-fold cached predictions')
        return result.copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.reference,allow_pickle=False) as z:
        ids,y,b,f,eligible=(z[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    pos=aligned_positions(ids,eligible)
    rows,_=load_feature_rows(args.features,ids,y,b,f,pos)
    x,_,_=matrices(rows)
    manifest=save_producers(ProducerPool(x,y[pos],b[pos],f[pos],eligible),args.output)
    (args.output/'source-reference-sha256.txt').write_text(digest(args.reference)+'\n')
    print(json.dumps(dict(status='completed',producer_sets=14,cache_sha256=manifest['cache_sha256'])),flush=True)


if __name__=='__main__':main()
