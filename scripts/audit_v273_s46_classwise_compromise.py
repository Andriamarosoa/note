"""S46 source-K-aware polyphony protection, no learned model refitting.

Audits 192 preregistered policies using frozen S45 risk estimates,
separating observable source K>=2 from K<=1. Truth used for audit only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

LAM_POLY=(2.,4.,8.)
CUT_POLY=(0.,.05,.10,.20)
VOTES=(0,1)
STRUCTURE=('open','neighbor_and_no_drop_to_K1')
FAMILY_GROUPS={
 'all16':None,
 'unique_poly':('S40_temporal_CNN','S19_recurrent_ABA','S38_morphology'),
 'historic_acoustic':('S1_acoustic','S2_source','S3_repair','S4_repair',
                      'historical_repair_coherent'),
 'ordering':('S24_Bfirst','S25_Bfirst','S19_recurrent_ABA')
}

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def select(old,proposal,risk,votes,selected,lam,cut,minvote,structure):
    n,p=proposal.shape
    available=np.zeros(p,bool)
    available[np.asarray(selected,dtype=np.int64)]=True
    eligible=available[None,:]&(proposal!=old[:,None])&(votes>=minvote)
    if structure=='neighbor_and_no_drop_to_K1':
        eligible &= ( (old[:,None]<2) |
            ((np.abs(proposal.astype(np.int16)-old[:,None].astype(np.int16))==1)&
             (proposal>=2)) )
    elif structure!='open':
        raise ValueError('invalid predetermined structural policy')
    lamvec=np.where(old>=2,float(lam),1.0)[:,None]
    cutvec=np.where(old>=2,float(cut),.02)
    u=risk[:,:,0]-lamvec*risk[:,:,1]
    u=np.where(eligible&np.isfinite(u),u,-np.inf)
    best=u.argmax(1)
    gain=u[np.arange(n),best]
    accepted=np.isfinite(gain)&(gain>cutvec)
    return np.where(accepted,proposal[np.arange(n),best],old).astype(np.int8),np.where(accepted,best,-1).astype(np.int8)

def selftest():
    old=np.array([2,3,1],np.int8)
    cand=np.array([[1,3],[4,2],[2,1]],np.int8)
    risk=np.zeros((3,2,2),np.float32)
    risk[:,:,0]=.8
    v=np.ones((3,2),np.int16)
    x,j=select(old,cand,risk,v,[0,1],2,.1,1,'neighbor_and_no_drop_to_K1')
    assert x.tolist()==[3,4,2]
    assert j.tolist()==[1,0,0]
    assert len(FAMILY_GROUPS)*len(LAM_POLY)*len(CUT_POLY)*len(VOTES)*len(STRUCTURE)==192
    print('PASS: fixed 192 poly conservative source-K gates, no truth/H9')

def run(a):
    if a.output.exists():raise ValueError('immutable native experiment')
    q=read(a.s45risk)
    src=read(a.s45pred)
    for key in ('global_index',):
        if not np.array_equal(q[key],src[key]):raise ValueError('source IDs differ')
    ids=q['global_index'];truth=src['true_K']
    if len(ids)!=59309:raise ValueError('wrong source cohort')
    variants=list(map(str,src['variant_ids']))
    if variants[0]!='series18_parent':raise ValueError('no S18 reference')
    base=src['predictions'][:,0]
    if ((base==truth).sum()!=49178 or
        ((base==truth)&(truth>=2)).sum()!=2998):
        raise ValueError('certified S18 changed')
    fam=list(map(str,q['family_ids']))
    if len(fam)!=16 or any('h9' in f.lower() or 'yourmt3' in f.lower() for f in fam):
        raise ValueError('forbidden or incomplete family')
    proposal=q['proposal_K']
    risks=q['risks']
    agree=q['independent_votes']
    if (proposal.shape!=(len(ids),16) or
        risks.shape!=(len(ids),16,2) or agree.shape!=(len(ids),16)):
        raise ValueError('S45 source risk bank incomplete')
    result={'series18_parent':base.copy()}
    indices={}
    for group,names in FAMILY_GROUPS.items():
        if names is None:indices[group]=list(range(16))
        else:
            if not all(x in fam for x in names):raise ValueError('missing family '+str(names))
            indices[group]=[fam.index(f) for f in names]
        for lam in LAM_POLY:
            for cut in CUT_POLY:
                for mv in VOTES:
                    for structure in STRUCTURE:
                        name=f'series46__{group}__polyLambda{lam:g}__cut{cut:g}__votes{mv}__{structure}'
                        out,_=select(base,proposal,risks,agree,
                            indices[group],lam,cut,mv,structure)
                        result[name]=out
    if len(result)!=193:raise ValueError('192 tests missing')
    scores={}
    for name,vec in result.items():
        diff=(base!=vec)
        fixes=(base!=truth)&(vec==truth)&diff
        loss=(base==truth)&(vec!=truth)&diff
        neutral=(base!=truth)&(vec!=truth)&diff
        poly=(truth>=2)
        scores[name]=dict(
            global_exact=float((vec==truth).mean()),
            poly_exact=float(((vec==truth)&poly).sum()/poly.sum()),
            correct_global=int((vec==truth).sum()),
            correct_poly=int(((vec==truth)&poly).sum()),
            corrections=int(fixes.sum()),regressions=int(loss.sum()),
            net=int(fixes.sum()-loss.sum()),
            poly_corrections=int((fixes&poly).sum()),
            poly_regressions=int((loss&poly).sum()),
            poly_net=int((fixes&poly).sum()-(loss&poly).sum()),
            neutral=int(neutral.sum()),
            by_true_K={str(k):dict(fixed=int((fixes&(truth==k)).sum()),
                lost=int((loss&(truth==k)).sum())) for k in range(7)},
            by_source_K={str(k):dict(fixed=int((fixes&(base==k)).sum()),
                lost=int((loss&(base==k)).sum())) for k in range(7)},
            by_fold={str(f):dict(fixed=int((fixes&(src['fold']==f)).sum()),
                lost=int((loss&(src['fold']==f)).sum())) for f in (0,1,2,4)})
    rank_global=sorted(scores,key=lambda k:(-scores[k]['global_exact'],
                                             -scores[k]['poly_exact']))
    rank_poly=sorted(scores,key=lambda k:(-scores[k]['poly_exact'],
                                           -scores[k]['global_exact']))
    candidates=[k for k in result if k.startswith('series46__')]
    safe=[k for k in candidates if scores[k]['corrections']>0
          and scores[k]['regressions']==0 and
          scores[k]['correct_poly']>=2998]
    poly_gain=[k for k in candidates if scores[k]['poly_net']>0]
    pareto=[]
    for key in candidates:
        s=scores[key]
        if not any(scores[r]['poly_corrections']>=s['poly_corrections'] and
                   scores[r]['poly_regressions']<=s['poly_regressions'] and
                  (scores[r]['poly_corrections']>s['poly_corrections'] or
                   scores[r]['poly_regressions']<s['poly_regressions'])
                   for r in candidates):
            pareto.append(key)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz',
        global_index=ids,fold=src['fold'],true_K=truth,
        variant_ids=np.asarray(list(result)),
        predictions=np.column_stack(list(result.values())))
    report=dict(status='completed',S18_unchanged=True,H9_excluded=True,
        source_is_frozen_S45_no_model_retraining=True,
        candidate_pool_groups={name:[fam[i] for i in values]
                               for name,values in indices.items()},
        exactly_192_fixed_policies=True,
        no_true_K_in_any_gate=True,
        tested_on_previously_explored_cohort=True,
        source_models_historical_OOF_incomplete=True,
        no_production_promotion=True,
        strict_zero_loss_candidates=safe,
        positive_poly_net_variants=poly_gain,
        pareto_poly_names=pareto,
        top_global=rank_global[:25],
        top_poly=rank_poly[:25],
        audits=scores)
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,
        indent=2)+'\n')
    lines=['# S46 — apprentissage gelé, conditions de risque de régression par origine K',
        '', '192 politiques à partir des 16 familles spécialisées S45 ; aucune vérité au moment du choix.',
        f"Critère zéro régression / gain poly : {len(safe)}; "
         f"politiques à net poly > 0 : {len(poly_gain)}",
        f"Pareto des corrections poly: {len(pareto)} options.",
        '**Développement historique exposé : aucun gain validé sur composition inédite, aucune promotion.**',
        '', '| Meilleures options poly | Global | Poly | Corr | Régr | Corr poly | Régr poly |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in rank_poly[:20]:
        s=scores[key]
        lines.append(f"| {key} | {s['global_exact']*100:.4f}% | "
            f"{s['poly_exact']*100:.4f}% | {s['corrections']} | "
            f"{s['regressions']} | {s['poly_corrections']} | {s['poly_regressions']} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s45pred',type=Path)
    p.add_argument('--s45risk',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        if not args.s45pred or not args.s45risk or not args.output:
            raise ValueError('source S45 heads and outputs required')
        run(args)
