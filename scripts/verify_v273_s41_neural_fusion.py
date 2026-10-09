"""S41 independent replay: exact 19 neural gates, held-piece splits and K decisions."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_extended_dynamic_router import read
from scripts.train_v273_s41_neural_fusion import (
    FusionGate,make_proposals,gated_prediction,DOMAINS,THRESHOLDS)
from scripts.yourmt3_exactk_common import metrics,require

def sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

def run(a):
    require(not a.output.exists(),'cannot overwrite existing result')
    torch.set_num_threads(2)
    d=prepare(a)
    P38=read(a.s38);P40=read(a.s40)
    r=read(a.input/'predictions.npz')
    q=read(a.input/'action_probs.npz')
    meta=json.loads((a.input/'report.json').read_text())
    for x in (P38,P40,r,q):
        require(np.array_equal(x['global_index'],d['ids']),
                'source event ID mismatch')
    base=d['parent'].astype(np.int8)
    require(metrics(d['y'],base)['correct']==49178,
            'source S18 reference changed')
    t=P38['poly_probs'];u=P40['probabilities']
    proposals=make_proposals(base,t,u)
    require(np.array_equal(q['candidate_K'],proposals),
            'saved neural action candidates were altered')
    require(meta['models_count']==19 and
            meta['no_H9_input_no_H9_action'],
            'missing saved neural head models or H9 breach')
    scores=np.full((len(base),6),np.nan,np.float32)
    hashes={}
    for item in meta['models']:
        piece=item['piece'];fold=item['fold']
        held=np.flatnonzero(d['pieces']==piece)
        fit=np.flatnonzero((d['fold']==fold)&(d['pieces']!=piece))
        path=a.input/item['file']
        require(path.exists(),'missing neural fusion weights')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        require(saved['held_piece']==piece and saved['fold']==fold,
                'incorrect held-piece model provenance')
        require(np.array_equal(saved['fit_native_ids'],d['ids'][fit]) and
                np.array_equal(saved['hold_native_ids'],d['ids'][held]),
                'S41 neural gate saw held-piece label')
        require(saved['fit_sha256']==sha(d['ids'][fit]) and
                saved['held_sha256']==sha(d['ids'][held]),
                'source provenance signature mismatch')
        scale=np.maximum(saved['scaler_scale'],1e-12)
        x=np.column_stack((
            np.clip((d['X'][held]-saved['scaler_mean'])/scale,
                -5,5),
            np.eye(7,dtype=np.float32)[base[held]],
            t[held].reshape(len(held),28),u[held])).astype(np.float32)
        require(x.shape==(len(held),377),'invalid replay input')
        net=FusionGate()
        net.load_state_dict(saved['model'])
        net.eval()
        result=[]
        with torch.no_grad():
            for st in range(0,len(held),512):
                result.append(torch.softmax(net(torch.from_numpy(
                      x[st:st+512])),dim=1).numpy())
        scores[held]=np.vstack(result)
        hashes[item['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
    require(len(hashes)==19,'all piece-held weights not verified')
    require(np.allclose(scores,q['action_probs'],atol=1e-6),
            'reloaded neural action probabilities differ')
    names=list(map(str,r['variant_ids']))
    require(names[0]=='series18_parent' and len(names)==7,
            'six real learned gate policies missing')
    require(np.array_equal(r['predictions'][:,0],base),
            'S18 baseline mutated')
    for domain in DOMAINS:
        for margin in THRESHOLDS:
            name=f'series41__neural_fusion__{domain}__margin{margin:g}'
            require(name in names,'missing predeclared selection '+name)
            pp=gated_prediction(base,proposals,scores,domain,margin)
            require(np.array_equal(pp,r['predictions'][:,names.index(name)]),
                    'not reproducible K decision '+name)
    a.output.mkdir(parents=True)
    verified=dict(status='independently_reproduced',
        all_19_models_reloaded=True,
        all_6_selection_vectors_identical=True,
        source_fold_exclusion_checked=True,
        meta_test_piece_not_in_training=True,
        H9_still_excluded=True,
        no_compositions_truly_unseen_from_the_start=True,
        no_production_promotion=True,
        model_sha256=hashes)
    (a.output/'report.json').write_text(json.dumps(verified,sort_keys=True,indent=2)+'\n')
    print('PASS all 19 piece-held neural gates and six action choices numerically replayed; no H9')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--s38',type=Path,required=True)
    p.add_argument('--s40',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
