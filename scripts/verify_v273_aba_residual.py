"""Independent numeric replay of series20 conditional A-B-A residual selector."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require
from scripts.loop_v273_native_risk import read
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_aba_residual import ResidualAB, proposals

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()

def verify(a):
    require(not a.output.exists(),'no overwrite in verification')
    torch.set_num_threads(2)
    d=prepare(a)
    v=read(a.input/'predictions.npz')
    p=read(a.input/'probabilities.npz')
    r=json.loads((a.input/'report.json').read_text())
    require(r['status']=='completed' and not r['independent_validation'],
            'report provenance changed')
    for k in ('global_index',):
        require(np.array_equal(v[k],d['ids']) and np.array_equal(p[k],d['ids']),
                'different original IDs')
    require(np.array_equal(v['true_K'],d['y']) and
            np.array_equal(v['fold'],d['fold']),'truth/fold changed')
    A=np.asarray(p['A_probs'],np.float32)
    B=np.asarray(p['B_compatibility'],np.float32)
    require(A.shape==(59309,5,7) and B.shape==(59309,4,7),
            'training output missing')
    policies=proposals(A,d['parent'])
    policies['freeze_parent']=d['freeze']
    names=list(map(str,v['variant_ids']))
    require(set(policies)==set(names) and len(names)==27,
            'expected 27 predeclared policies')
    for name,pred in policies.items():
        require(np.array_equal(pred,v['predictions'][:,names.index(name)]),
                'policy decoder could not regenerate '+name)
    check={}
    for name,pred in policies.items():
        m=metrics(d['y'],pred)
        expected=r['policies'][name]['metrics']
        require(m['correct']==expected['correct'] and
                m['poly']['correct']==expected['poly']['correct'],
                'wrong benchmark score '+name)
        check[name]=dict(metrics=m,
            vs_S18=paired(d['y'],d['parent'],pred),
            vs_freeze=paired(d['y'],d['freeze'],pred))
    require(len(r['weights'])==19,'missing piece-held checkpoints')
    max_A=max_B=0
    sha={}
    for rec in r['weights']:
        piece=rec['piece'];f=rec['fold']
        held=d['pieces']==piece
        fit=(d['fold']==f)&~held
        require(int(held.sum())==rec['held_rows'] and
                int(fit.sum())==rec['fit_rows'],'wrong split sample counts')
        require(piece not in rec['fit_pieces'],'leaked training piece')
        require(rec['fit_sha256']==hashlib.sha256(
            d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            'training IDs modified')
        require(rec['held_sha256']==hashlib.sha256(
            d['ids'][held].astype('<i8').tobytes()).hexdigest(),
            'test IDs modified')
        file=a.input/'models'/rec['model_file']
        require(file.is_file(),'missing fitted model')
        sha[file.name]=digest(file)
        saved=torch.load(file,weights_only=False,map_location='cpu')
        require(saved['piece']==piece and saved['fold']==f,
                'model provenance does not match')
        require(np.array_equal(saved['fit_ids'],d['ids'][fit]) and
                np.array_equal(saved['held_ids'],d['ids'][held]),
                'the model was fit on incorrect IDs')
        xx=np.clip((d['X'][held]-saved['scaler_mean'])/
                   saved['scaler_scale'],-5,5).astype(np.float32)
        original=d['parent'][held]
        idx=np.flatnonzero(held)
        net=ResidualAB()
        net.load_state_dict(saved['model']);net.eval()
        with torch.no_grad():
            for st in range(0,len(xx),512):
                data=torch.from_numpy(xx[st:st+512])
                base=torch.from_numpy(original[st:st+512].astype(np.int64))
                a_pred,b_pred,pred=net(data,base,passes=4)
                nob=net(data,base,passes=4,no_B=True)[2][-1]
                pa=np.stack([*[t.numpy() for t in pred],nob.numpy()],axis=1)
                pb=np.stack([torch.sigmoid(t).numpy() for t in b_pred],axis=1)
                ix=idx[st:st+len(data)]
                max_A=max(max_A,float(np.max(np.abs(pa-A[ix]))))
                max_B=max(max_B,float(np.max(np.abs(pb-B[ix]))))
    require(max_A<5e-5 and max_B<5e-5,
            'independent forward pass disagrees with original CI')
    result=dict(status='verified',independent_validation=False,
        original_data_already_exposed=True,model_promoted=False,
        all_27_policy_decisions_equal=True,all_19_piece_excluded_models_replayed=True,
        max_A_probability_absdiff=max_A,max_B_compatibility_absdiff=max_B,
        model_sha256=sha,policies=check)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# S20 independent numerical replay of conditional A-B-A model','',
        '**No independent validation on new music; this verifies the original CI.**',
        f'Max A output deviation: {max_A:.9g}',
        f'Max B output deviation: {max_B:.9g}',
        f'Fitted models SHA256 verified: {len(sha)}',
        '27 saved decisions regenerated independently from probabilities.',
        '', '| Variant | Global | Poly | Corrections vs S18 | Regressions vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name,info in check.items():
        m=info['metrics'];pp=info['vs_S18']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{pp['corrections']} | {pp['regressions']} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--features',type=Path,required=True)
    a.add_argument('--s18',type=Path,required=True)
    a.add_argument('--input',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    verify(a.parse_args())
