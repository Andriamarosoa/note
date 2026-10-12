"""S26: B-first proposals promoted only when the original A-first order agrees.

Ninety pre-fixed label-free multi-order consensus gates. No new fit.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

ORDER={
 'p2':(('r',1),('s',1)),
 'p3':(('r',2),('s',2)),
 'p4':(('r',3),('s',3)),
 'p2p4':(('r',1),('s',1),('r',3),('s',3)),
 'r2r4_s4':(('r',1),('r',3),('s',3)),
 'r1r2_s2':(('r',0),('r',1),('s',1))
}
MODES=('two_A_conf','B0_supports','B0_challenges_parent')
CUTS=(.10,.20,.30,.40,.50)

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def policy(predictions,name):
    keys=list(map(str,predictions['variant_ids']))
    require(name in keys,'policy not found '+name)
    return predictions['predictions'][:,keys.index(name)].astype(np.int8)

def make(r,s,B,parent):
    n=len(parent)
    require(r.shape==s.shape==(n,5,7),'A source matrix shape')
    require(B.shape==(n,4,7),'B source shape')
    out={'series18_parent':parent.copy()}
    masks={}
    ii=np.arange(n)
    for group,members in ORDER.items():
        cur=[]
        for who,pass_id in members:
            cur.append((r if who=='r' else s)[:,pass_id,:])
        candidate=cur[0].argmax(1).astype(np.int8)
        agree=candidate!=parent
        margins=[]
        for p in cur:
            agree &= (p.argmax(1)==candidate)
            margins.append(p[ii,candidate]-p[ii,parent])
        all_margin=np.min(np.column_stack(margins),axis=1)
        B0_argmax=B[:,0,:].argmax(1)
        for mode in MODES:
            mask=agree.copy()
            if mode=='B0_supports':
                mask&=(B[ii,0,candidate]>B[ii,0,parent])
            elif mode=='B0_challenges_parent':
                mask&=(B0_argmax!=parent)
            for cut in CUTS:
                name=f'series26__{group}__{mode}__margin>{cut:g}'
                yes=mask&(all_margin>cut)
                out[name]=np.where(yes,candidate,parent).astype(np.int8)
                masks[name]=yes
    require(len(out)==91,'90 grids plus reference')
    return out,masks

def selftest():
    n=11
    y=np.arange(n)%7
    p=np.zeros((n,5,7),np.float32)
    for i in range(5):
        p[np.arange(n),i,(y+1)%7]=1
    B=np.ones((n,4,7),np.float32)*.2
    B[np.arange(n),0,(y+1)%7]=.8
    out,mask=make(p,p,B,y.astype(np.int8))
    assert len(out)==91 and len(mask)==90
    assert all(np.array_equal(v,(y+1)%7) for key,v in out.items()
        if key!='series18_parent')
    B[:,:,:]=.2
    out,_=make(p,p,B,y.astype(np.int8))
    assert np.array_equal(out['series26__p2__B0_supports__margin>0.5'],y)
    print('PASS: 90 genuine multi-order consensus policies and reversible B challenges')

def audit(args):
    require(not args.output.exists(),'never overwrite existing audit')
    p20=read(args.s20_predictions); p25=read(args.s25_predictions)
    r=read(args.s25_probabilities);s=read(args.s20_probabilities)
    for z in (p20,p25,r,s):
        require(np.array_equal(z['global_index'],p25['global_index']),
                'source identity misaligned')
    y=p25['true_K'];fold=p25['fold'];ids=p25['global_index']
    require(np.array_equal(p20['true_K'],y) and np.array_equal(p20['fold'],fold),
            'fold or truth labels drift')
    require(len(y)==59309 and set(fold.tolist())==set(FOLDS),
            'native cohort mismatch')
    parent=policy(p25,'series18_parent')
    require(np.array_equal(parent,policy(p20,'series18_parent')),
            'different learned parents between orders')
    require(metrics(y,parent)['correct']==49178 and
            metrics(y,parent)['poly']['correct']==2998,
            'S18 reference drift')
    freeze=policy(p25,'freeze_reference')
    require(np.array_equal(freeze,policy(p20,'freeze_parent')),
            'freeze source drift')
    ordered=np.asarray(r['A_probs'],np.float32)
    standard=np.asarray(s['A_probs'],np.float32)
    B=np.asarray(r['B_compatibility'],np.float32)
    out,masks=make(ordered,standard,B,parent)
    out['freeze_parent']=freeze.copy()
    require(len(out)==92,'fixed policy matrix changed')
    results={}
    for name,pred in out.items():
        good=(pred==y)
        comp=paired(y,parent,pred)
        changed=(pred!=parent)
        results[name]=dict(metrics=metrics(y,pred),
            vs_S18=comp,vs_freeze=paired(y,freeze,pred),
            corrected=int((changed&(pred==y)&(parent!=y)).sum()),
            regressed=int((changed&(pred!=y)&(parent==y)).sum()),
            neutral=int((changed&(pred!=y)&(parent!=y)).sum()),
            changed=int(changed.sum()),
            by_true_K={str(k):dict(total=int((y==k).sum()),
                correct=int(((y==k)&good).sum()),
                corrected=int(((y==k)&changed&(pred==y)&(parent!=y)).sum()),
                regressed=int(((y==k)&changed&(pred!=y)&(parent==y)).sum()))
                for k in range(7)},
            by_fold={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                vs_S18=paired(y[fold==f],parent[fold==f],pred[fold==f]))
                for f in FOLDS})
    ranking=sorted(results,key=lambda key:(
        results[key]['metrics']['correct'],
        results[key]['metrics']['poly']['correct']),reverse=True)
    safe=[k for k in out if k.startswith('series26__') and
        results[k]['corrected']>0 and results[k]['regressed']==0 and
        results[k]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',
        data_is_development_already_exposed=True,
        independent_validation=False,no_new_training=True,
        all_policies_label_blind=True,
        parent_s18_unchanged=True,
        source_runs=dict(Afirst=37858383973,Bfirst=37862417522),
        policy_count=len(out),original_90_variants=90,
        audits=results,strict_no_loss_research_candidates=safe,
        best_global=ranking[0])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',
        global_index=ids,true_K=y,fold=fold,
        variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    for name in ranking[:3]:
        pred=out[name]
        with (args.output/(name.replace('>','_gt_')+'__changed.csv')).open(
                'w',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['native_id','fold','truth','parent_S18',
                'new_K','outcome'])
            for i in np.flatnonzero(pred!=parent):
                result='corrected' if pred[i]==y[i] else (
                    'regressed' if parent[i]==y[i] else 'neutral')
                writer.writerow([int(ids[i]),int(fold[i]),int(y[i]),
                    int(parent[i]),int(pred[i]),result])
    lines=['# S26 — confirmation croisée A d abord / B d abord','',
        '90 politiques sans vérité en entrée ; cohorte de développement déjà exposée.',
        '', '| Politique | Global | Poly | Corrections vs S18 | Régressions vs S18 | Neutres |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ranking:
        v=results[key];m=v['metrics']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {v['corrected']} | "
            f"{v['regressed']} | {v['neutral']} |")
    lines+=['',f"Best global: {ranking[0]}",
        f"Strict no-loss candidates: {len(safe)}",
        'Complete source probabilities and models remain in S20/S25 archives.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--s20-predictions',type=Path)
    ap.add_argument('--s20-probabilities',type=Path)
    ap.add_argument('--s25-predictions',type=Path)
    ap.add_argument('--s25-probabilities',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        require(a.s20_predictions and a.s20_probabilities and
            a.s25_predictions and a.s25_probabilities and a.output,
            'four required sources')
        audit(a)
