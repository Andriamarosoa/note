"""Verify fitted B-first recurrent models and all saved decisions independently."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_Bfirst_recurrent import TrainedBfirst
from scripts.train_v273_aba_residual import proposals
from scripts.yourmt3_exactk_common import metrics,paired,require

def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as file:
        for block in iter(lambda:file.read(2<<20),b''):
            h.update(block)
    return h.hexdigest()

def verify(args):
    require(not args.output.exists(),'no overwrite of prior evidence')
    torch.set_num_threads(2)
    d=prepare(args)
    raw=read(args.input/'probabilities.npz')
    preds=read(args.input/'predictions.npz')
    rep=json.loads((args.input/'report.json').read_text())
    for z in (raw,preds):
        require(np.array_equal(z['global_index'],d['ids']),'native identity changed')
    require(np.array_equal(preds['true_K'],d['y']) and
        np.array_equal(preds['fold'],d['fold']),'label/fold alignment changed')
    require(raw['A_probs'].shape==(59309,5,7) and
        raw['B_compatibility'].shape==(59309,4,7),
        'no Bfirst source distributions')
    require(len(rep['fit_models'])==19,'missing 19 checkpoints')
    A=raw['A_probs'];B=raw['B_compatibility']
    stored={name:preds['predictions'][:,i] for i,name in
            enumerate(map(str,preds['variant_ids']))}
    out={name.replace('series20__','series25__'):vec
         for name,vec in proposals(A,d['parent']).items()}
    out['freeze_reference']=d['freeze'].copy()
    out['Bbefore_A0__raw']=B[:,0,:].argmax(1).astype(np.int8)
    require(len(out)==28 and set(out)==set(stored),'decision inventory drift')
    for name,vec in out.items():
        require(np.array_equal(vec,stored[name]),
                'saved decoder no longer reproduces '+name)
    max_A,max_B=0.,0.
    sha={}
    for rec in rep['fit_models']:
        piece=rec['piece'];f=int(rec['fold'])
        held=d['pieces']==piece
        fit=(d['fold']==f)&~held
        require(piece not in rec['train_pieces'] and
            int(held.sum())==rec['held_rows'] and
            int(fit.sum())==rec['fit_rows'],'piece split violated')
        require(rec['fit_sha256']==hashlib.sha256(
            d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            'fit identities drift')
        require(rec['held_sha256']==hashlib.sha256(
            d['ids'][held].astype('<i8').tobytes()).hexdigest(),
            'evaluation identities drift')
        file=args.input/'models'/rec['model_file']
        require(file.is_file(),'model missing')
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        require(checkpoint['piece']==piece and
                checkpoint['fold']==f and
                np.array_equal(checkpoint['fit_ids'],d['ids'][fit]) and
                np.array_equal(checkpoint['held_ids'],d['ids'][held]),
                'wrong checkpoint provenance')
        net=TrainedBfirst()
        net.load_state_dict(checkpoint['model'])
        net.eval()
        xi=np.flatnonzero(held)
        xx=np.clip((d['X'][held]-checkpoint['scaler_mean'])/
            checkpoint['scaler_scale'],-5,5).astype(np.float32)
        parent=d['parent'][held]
        sha[file.name]=digest(file)
        with torch.no_grad():
            for j in range(0,len(xx),512):
                q=torch.from_numpy(xx[j:j+512])
                k=torch.from_numpy(parent[j:j+512].astype(np.int64))
                As,Bs,Ps=net(q,k,passes=4)
                noB=net(q,k,passes=4,no_B=True)[2][-1]
                a=np.stack([*[v.numpy() for v in Ps],noB.numpy()],axis=1)
                b=np.stack([torch.sigmoid(v).numpy() for v in Bs],axis=1)
                at=xi[j:j+len(q)]
                max_A=max(max_A,float(np.max(np.abs(a-A[at]))))
                max_B=max(max_B,float(np.max(np.abs(b-B[at]))))
    require(max_A<5e-5 and max_B<5e-5,
            'replayed checkpoint output mismatch')
    results={}
    for name,vec in out.items():
        m=metrics(d['y'],vec)
        require(m['correct']==rep['policies'][name]['metrics']['correct'] and
                m['poly']['correct']==rep['policies'][name]['metrics']['poly']['correct'],
                'score drift '+name)
        results[name]=dict(metrics=m,vs_S18=paired(d['y'],d['parent'],vec),
            vs_freeze=paired(d['y'],d['freeze'],vec))
    summary=dict(status='verified',independent_validation=False,
        all_19_learned_Bfirst_models_replayed=True,
        all_28_predictions_exactly_decoded=True,
        max_A_difference=max_A,max_B_difference=max_B,
        models_SHA256=sha,scores=results,
        original_cohort_previously_exposed=True,
        promote=False)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    lines=['# S25 — independent numerical replay of trained B-first order',
        '',f'Models reloaded: {len(sha)}',
        f'Max error A: {max_A:.8g}; B: {max_B:.8g}',
        'All 28 prediction vectors exactly reproduced. No new-music validation.',
        '', '| Policy | Global | Poly | Corrections vs S18 | Regressions vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name,v in results.items():
        m=v['metrics'];p=v['vs_S18']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{p['corrections']} | {p['regressions']} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--s18',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args())
