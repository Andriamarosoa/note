"""S24: B first, then A, repeatedly. Frozen S20 weights, same piece and sound.

A-first and B-first are contrasted at equal A passes, not silently
conflated. B is the weaker *auxiliary role*, not assumed low accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_aba_residual import ResidualAB,onehot_prior
from scripts.loop_v273_native_risk import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

CUTS=(.10,.25,.40,.60)
A_NAMES=('Bfirst_A1','Bfirst_A2','Bfirst_A3','Bfirst_A4')
ORIGINAL=('Afirst_A1','Afirst_A2','Afirst_A3','Afirst_A4')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()

def infer_reverse(net,xx,parent,passes=4,remove_first_B=False):
    """B observes the inherited parent state *before* A's first analysis."""
    prior=onehot_prior(parent).to(dtype=xx.dtype,device=xx.device)
    base=torch.log(prior)
    old=prior
    first=net.B(torch.cat((xx,prior,prior),dim=1))
    message=torch.tanh(first) if not remove_first_B else torch.zeros_like(first)
    B=[first]
    A=[]
    for t in range(passes):
        logits=base+net.A(torch.cat((xx,prior,old,message),dim=1))
        p=torch.softmax(logits,dim=1)
        A.append(p)
        old=p
        b=net.B(torch.cat((xx,prior,p),dim=1))
        B.append(b)
        message=torch.tanh(b)
    return A,B

def selftest():
    torch.manual_seed(241024)
    m=ResidualAB()
    torch.nn.init.normal_(m.A[-1].weight,std=.03)
    x=torch.randn(11,335)
    y=torch.tensor([0,1,2,3,4,5,6,0,1,2,3])
    m.eval()
    with torch.no_grad():
        A,B=infer_reverse(m,x,y,4)
        first=m(x,y,passes=4)[2]
        A0,_=infer_reverse(m,x,y,4,remove_first_B=True)
        assert len(A)==4 and len(B)==5
        assert torch.allclose(A0[0],first[0],atol=1e-7)
        assert not torch.allclose(A[0],first[0])
        assert torch.allclose(A[0].sum(1),torch.ones(11))
        assert A[1].shape==(11,7)
    print("PASS: B-first changes A1, remove-B reverts to A-first, 4 A passes")

def policies(A,original,Bfirst,parent,freeze):
    output={'series18_parent':parent.copy(),'freeze_parent':freeze.copy()}
    for i,n in enumerate(ORIGINAL):
        output[n+'__raw']=original[:,i].argmax(1).astype(np.int8)
    b=Bfirst.argmax(1).astype(np.int8)
    output['Bfirst_B0__raw']=b
    for i,n in enumerate(A_NAMES):
        p=A[:,i,:]
        choice=p.argmax(1).astype(np.int8)
        output[n+'__raw']=choice
        before=p[np.arange(len(p)),parent]
        margin=p[np.arange(len(p)),choice]-before
        for c in CUTS:
            change=(choice!=parent)&(margin>c)
            output[n+f'__margin{c:g}']=np.where(change,choice,parent).astype(np.int8)
    require(len(output)==27,'2 parents + 4 Afirst + B + 4*(raw+4 gates)')
    return output

def analyse(args):
    require(not args.output.exists(),'cannot overwrite original evidence')
    t0=time.monotonic()
    torch.set_num_threads(2)
    d=prepare(args)
    source=read(args.s20_probs)
    require(np.array_equal(source['global_index'],d['ids']),'S20 ids misaligned')
    archived=np.asarray(source['A_probs'],np.float32)
    require(archived.shape==(len(d['ids']),5,7),'old A-pass shape')
    r=json.loads((args.s20/'report.json').read_text())
    require(len(r['weights'])==19 and r['status']=='completed',
            '19 original checkpoints needed')
    n=len(d['y'])
    q=np.full((n,4,7),np.nan,np.float32)
    firstB=np.full((n,7),np.nan,np.float32)
    max_original_error=0.
    model_digests={}
    records=[]
    for rec in r['weights']:
        piece=rec['piece'];f=rec['fold']
        held=d['pieces']==piece
        fit=(d['fold']==f)&~held
        require(held.any() and int(held.sum())==rec['held_rows'],
                'held sample count changed')
        require(int(fit.sum())==rec['fit_rows'] and
                piece not in rec['fit_pieces'],'split leakage')
        require(rec['fit_sha256']==hashlib.sha256(
            d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            'training identities changed')
        require(rec['held_sha256']==hashlib.sha256(
            d['ids'][held].astype('<i8').tobytes()).hexdigest(),
            'held identities changed')
        file=args.s20/'models'/rec['model_file']
        require(file.is_file(),'model missing '+str(file))
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        require(checkpoint['piece']==piece and checkpoint['fold']==f and
                np.array_equal(checkpoint['fit_ids'],d['ids'][fit]) and
                np.array_equal(checkpoint['held_ids'],d['ids'][held]),
                'checkpoint provenance invalid')
        x=np.clip((d['X'][held]-checkpoint['scaler_mean'])/
            checkpoint['scaler_scale'],-5,5).astype(np.float32)
        p=d['parent'][held].astype(np.int64)
        net=ResidualAB();net.load_state_dict(checkpoint['model']);net.eval()
        ix=np.flatnonzero(held)
        with torch.no_grad():
            for start in range(0,len(x),512):
                xx=torch.from_numpy(x[start:start+512])
                yy=torch.from_numpy(p[start:start+512])
                original=net(xx,yy,passes=4)[2]
                reverse,bs=infer_reverse(net,xx,yy,passes=4)
                j=ix[start:start+len(xx)]
                for k in range(4):
                    error=np.max(np.abs(original[k].numpy()-archived[j,k]))
                    max_original_error=max(max_original_error,float(error))
                    q[j,k]=reverse[k].numpy()
                firstB[j]=torch.sigmoid(bs[0]).numpy()
        model_digests[rec['model_file']]=digest(file)
        records.append(dict(piece=piece,fold=f,held=int(held.sum()),
             checkpoint_SHA256=model_digests[rec['model_file']]))
        print(json.dumps(dict(piece=piece,fold=f,complete=len(records),
                max_original_error=max_original_error,
                seconds=round(time.monotonic()-t0,1))),flush=True)
    require(max_original_error<5e-5,'cannot reproduce A-first S20 output')
    require(np.isfinite(q).all() and np.isfinite(firstB).all(),
            'missing B-first inference')
    require(np.allclose(q.sum(2),1,atol=1e-5),
            'B-first invalid normalizations')
    originals=archived[:,:4,:]
    changes=[int((q[:,i].argmax(1)!=originals[:,i].argmax(1)).sum())
             for i in range(4)]
    average_shift=[float(np.abs(q[:,i]-originals[:,i]).mean()) for i in range(4)]
    require(average_shift[0]>1e-7,
            'initial reverse message did not influence A')
    out=policies(q,originals,firstB,d['parent'],d['freeze'])
    audits={}
    for name,pred in out.items():
        x=paired(d['y'],d['parent'],pred)
        audits[name]=dict(metrics=metrics(d['y'],pred),
            versus_S18=x,
            versus_freeze=paired(d['y'],d['freeze'],pred),
            versus_YourMT3=paired(d['y'],d['yourmt3'],pred),
            neutral_changes_vs_S18=int(np.sum(
                (pred!=d['parent'])&(pred!=d['y'])&(d['parent']!=d['y']))),
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                pred[d['fold']==f]),vs_S18=paired(d['y'][d['fold']==f],
                d['parent'][d['fold']==f],pred[d['fold']==f])) for f in FOLDS})
    passage={}
    for i in range(4):
        original=out[ORIGINAL[i]+'__raw']
        reverse=out[A_NAMES[i]+'__raw']
        matrix=np.zeros((7,7),dtype=np.int64)
        np.add.at(matrix,(original,reverse),1)
        passage[str(i+1)]=dict(changed=changes[i],
            absolute_probability_shift=average_shift[i],
            relative_to_Afirst=paired(d['y'],original,reverse),
            by_true_K={str(k):dict(
                Afirst_correct=int(((d['y']==k)&(original==d['y'])).sum()),
                Bfirst_correct=int(((d['y']==k)&(reverse==d['y'])).sum()),
                reverse_saves=int(((d['y']==k)&(original!=d['y'])&
                         (reverse==d['y'])).sum()),
                reverse_regresses=int(((d['y']==k)&(original==d['y'])&
                         (reverse!=d['y'])).sum()))
                for k in range(7)},
            transitions_Afirst_to_Bfirst=matrix.tolist())
    bw=out['Bfirst_B0__raw']
    ba=out['Bfirst_A1__raw']
    weak=dict(accuracy=metrics(d['y'],bw),
        B_wrong_to_A_correct=int(((bw!=d['y'])&(ba==d['y'])).sum()),
        B_correct_to_A_wrong=int(((bw==d['y'])&(ba!=d['y'])).sum()),
        B_wrong_and_A_wrong=int(((bw!=d['y'])&(ba!=d['y'])).sum()),
        Afirst1_correct=int((out['Afirst_A1__raw']==d['y']).sum()))
    ranked=sorted(audits,key=lambda name:(audits[name]['metrics']['correct'],
        audits[name]['metrics']['poly']['correct']),reverse=True)
    protect=[name for name in out if name.startswith('Bfirst_A')
        and audits[name]['versus_S18']['global']['corrections']>0
        and audits[name]['versus_S18']['global']['regressions']==0
        and audits[name]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',source='S20 fixed trained checkpoints',
        independent_validation=False,
        development_cohort_already_exposed=True,
        new_training=False,
        reversed_order_at_fixed_weights=True,
        B_is_multilabel_auxiliary_not_ranked_accuracy=True,
        test_piece_excluded_from_training=True,
        original_reproduction_max_error=max_original_error,
        B_worse_first_diagnostic=weak,
        comparisons_by_A_passage=passage,
        models=records,model_count=len(records),
        all_policies=audits,
        zero_loss_research_candidates=protect,
        best_global=ranked[0],
        no_auto_promotion=True,
        seconds=round(time.monotonic()-t0,2))
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,
        sort_keys=True)+'\n')
    np.savez_compressed(args.output/'probabilities.npz',
        global_index=d['ids'],first_B_compatibility=firstB,
        Bfirst_A_probs=q,Afirst_A_probs=originals)
    np.savez_compressed(args.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.array(list(out)),predictions=np.column_stack(list(out.values())))
    textlines=['# Série24 — commencer par la tête B, à poids identiques','',
        'Mêmes poids, normalisations et événements que S20. Pas de fit nouveau.',
        '**B est une tête auxiliaire BCE, pas une « mauvaise sélection » attestée avant mesure.**',
        'Développement déjà exploré ; pas de validation indépendante ni promotion.',
        f'19 modèles d origine reproduits, erreur maximum: {max_original_error:.9g}.',
        '', '| Ordre à nombre égal de A | A→B correct global | B→A correct global | A→B poly | B→A poly | Changements de K |',
        '|---|---:|---:|---:|---:|---:|']
    for i in range(4):
        orig=audits[ORIGINAL[i]+'__raw']['metrics']
        rev=audits[A_NAMES[i]+'__raw']['metrics']
        textlines.append(f"| A passage {i+1} | {orig['correct']} | {rev['correct']} | "
            f"{orig['poly']['correct']} | {rev['poly']['correct']} | {changes[i]} |")
    textlines+=['', '## B seule puis A après B',
        f"B seule corrects: {weak['accuracy']['correct']}",
        f"B erronée puis A correcte: {weak['B_wrong_to_A_correct']}",
        f"B correcte puis A erronée: {weak['B_correct_to_A_wrong']}",
        f"Effet de l'ordre sur la première distribution A: {average_shift[0]:.9g}",
        '', '## Toutes les décisions et corrections/régressions',
        '| Politique | Exact global | Exact poly | Corrigées vs S18 | Cassées vs S18 | Regressions restantes vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ranked:
        a=audits[name];m=a['metrics'];vs=a['versus_S18']['global'];fr=a['versus_freeze']['global']
        textlines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {vs['corrections']} | "
            f"{vs['regressions']} | {fr['regressions']} |")
    textlines+=['',f"Strict no-loss candidates: {len(protect)}",
        'Input selection is label-blind and does not use YourMT3+ outputs.',
        'No production model modified.']
    (args.output/'report.md').write_text('\n'.join(textlines)+'\n')
    print('\n'.join(textlines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s20',type=Path)
    p.add_argument('--s20-probs',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features and args.s18 and args.s20 and
             args.s20_probs and args.output,'missing original model sources')
        analyse(args)
