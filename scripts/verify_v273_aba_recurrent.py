"""Independent numeric replay of each held-piece S19 learned recurrent model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.yourmt3_exactk_common import metrics, paired, require, FOLDS
from scripts.loop_v273_native_risk import read
from scripts.train_v273_aba_recurrent import (
    NAMES,SEED,RecurrentSelector,prepare)

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def verify(a):
    require(not a.output.exists(),'do not overwrite provenance')
    torch.set_num_threads(2)
    d=prepare(a)
    reported=json.loads((a.input/'report.json').read_text())
    require(reported['status']=='completed' and
            not reported['independent_validation'],'report status unexpected')
    prob=read(a.input/'probabilities.npz')
    pred=read(a.input/'predictions.npz')
    ids=d['ids'];y=d['y']
    require(np.array_equal(prob['global_index'],ids) and
            np.array_equal(pred['global_index'],ids) and
            np.array_equal(pred['true_K'],y),'native identity changed')
    require(np.array_equal(pred['fold'],d['fold']),'fold alignment changed')
    require(list(map(str,prob['variant_ids']))==list(NAMES),
            'unexpected output stage inventory')
    old_p=np.asarray(prob['A_probs'],np.float32)
    old_b=np.asarray(prob['B_compatibility'],np.float32)
    require(old_p.shape==(len(y),5,7) and old_b.shape==(len(y),4,7),
            'recurrent message matrix malformed')
    catalog=list(map(str,pred['variant_ids']))
    for i,name in enumerate(NAMES):
        require(name in catalog,'stage prediction missing '+name)
        require(np.array_equal(old_p[:,i].argmax(1),
             pred['predictions'][:,catalog.index(name)]),
             'stored decisions disagree with probabilities '+name)
    require(len(reported['fit_records'])==19,'not all 19 pieces evaluated')
    all_model_digests={}
    max_A_diff=0.
    max_B_diff=0.
    total_changed=0
    for rec in reported['fit_records']:
        piece=rec['piece'];fold=rec['fold']
        held=d['pieces']==piece
        fit=(d['fold']==fold)&~held
        require(int(held.sum())==rec['held_rows'] and
                int(fit.sum())==rec['fit_rows'],
                'piece count drift')
        require(piece not in rec['train_pieces'] and
                all(d['fold'][d['pieces']==p][0]==fold
                    for p in rec['train_pieces']), 'training piece/fold leak')
        require(rec['fit_sha256']==hashlib.sha256(
            ids[fit].astype('<i8').tobytes()).hexdigest(),
            'fit IDs changed')
        require(rec['held_sha256']==hashlib.sha256(
            ids[held].astype('<i8').tobytes()).hexdigest(),
            'held IDs changed')
        f=a.input/'models'/rec['model_file']
        require(f.is_file(),'model weight missing')
        all_model_digests[f.name]=sha_file(f)
        saved=torch.load(f,map_location='cpu',weights_only=False)
        require(saved['piece']==piece and saved['fold']==fold,
                'wrong-model metadata')
        require(np.array_equal(saved['fit_ids'],ids[fit]) and
                np.array_equal(saved['held_ids'],ids[held]),
                'serialized model trained on unexpected events')
        mean=np.asarray(saved['scaler_mean'],np.float32)
        scale=np.asarray(saved['scaler_scale'],np.float32)
        require(mean.shape==(335,) and np.all(scale>0),'scaler not fitted')
        x=np.clip((d['X'][held]-mean)/scale,-5,5).astype(np.float32)
        model=RecurrentSelector()
        model.load_state_dict(saved['state_dict'])
        model.eval()
        ix=np.flatnonzero(held)
        with torch.no_grad():
            for start in range(0,len(ix),512):
                chunk=torch.from_numpy(x[start:start+512])
                al,bl,ap=model(chunk,passes=4)
                nob=model(chunk,passes=4,no_B=True)[2][-1]
                got_A=np.stack([*[t.numpy() for t in ap],nob.numpy()],axis=1)
                got_B=np.stack([torch.sigmoid(t).numpy() for t in bl],axis=1)
                i=ix[start:start+len(chunk)]
                max_A_diff=max(max_A_diff,float(np.max(np.abs(got_A-old_p[i]))))
                max_B_diff=max(max_B_diff,float(np.max(np.abs(got_B-old_b[i]))))
                total_changed+=int((got_A[:,0].argmax(1)!=
                                     got_A[:,3].argmax(1)).sum())
    require(max_A_diff<5e-5 and max_B_diff<5e-5,
            'trained checkpoint fails to reproduce saved messages')
    model_summary={}
    for name in catalog:
        p=pred['predictions'][:,catalog.index(name)]
        m=metrics(y,p)
        old=reported['policies'][name]['metrics']
        require(m['correct']==old['correct'] and
                m['poly']['correct']==old['poly']['correct'],
                'independent full-cohort score mismatches '+name)
        model_summary[name]=dict(metrics=m,
            paired_vs_series18=paired(y,d['parent'],p),
            paired_vs_freeze=paired(y,d['freeze'],p))
    print(json.dumps(dict(status='PASS',models=len(all_model_digests),
        max_A_difference=max_A_diff,max_B_difference=max_B_diff,
        across_pass_K_changed=total_changed)),flush=True)
    result=dict(status='replayed',independent_validation=False,
        previously_exposed_cohort=True,trained_piece_excluded=True,
        all_19_recurrent_models_numerically_replayed=True,
        max_A_probability_difference=max_A_diff,
        max_B_compatibility_difference=max_B_diff,
        one_to_four_pass_changed_predictions=total_changed,
        weight_sha256=all_model_digests,
        policies=model_summary)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    lines=['# S19 independent checkpoint/probability replay','',
        'All 19 models loaded and re-evaluated; no refit and no additional label-based policy choice.',
        '**Development data, not a new independent composition-level validation.**',
        f'Max A probability error: {max_A_diff:.9g}',
        f'Max B compatibility error: {max_B_diff:.9g}',
        f'Changed argmax from pass1 to pass4: {total_changed}', '',
        '| Variant | Global | Poly | Corrected vs S18 | Regessed vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name,v in model_summary.items():
        m=v['metrics'];pa=v['paired_vs_series18']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | "
            f"{pa['corrections']} | {pa['regressions']} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--s18',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    verify(ap.parse_args())
