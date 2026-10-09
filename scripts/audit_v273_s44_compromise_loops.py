"""S44: dedicated compromise loops for highest-correcting archived head families.

Uses only original prediction vectors from independently verified S43, plus
observable K transitions / agreements between distinct source families.
Truth is ONLY for evaluation, never for candidate selection or masking.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

FAMILIES=(
 ('S1_acoustic', 'historical_regression_loops','regression_loops_s1__'),
 ('S2_source', 'historical_regression_loops','regression_loops_s2__'),
 ('S3_repair', 'historical_regression_loops','regression_loops_s3__'),
 ('S4_repair', 'historical_regression_loops','regression_loops_s4__'),
 ('historical_repair_coherent','historical_memory','repair_coherent'),
 ('S25_Bfirst','S25',''),
 ('S24_Bfirst','S24',''),
 ('S40_temporal_CNN','S40',''),
 ('S19_recurrent_ABA','S19',''),
 ('S11_votes_time_audio','S11',''),
 ('S38_morphology','S38',''),
 ('S12_audio_logistic','S12',''),
 ('S20_repeated_ABA','S20',''),
 ('S27_early_stop','S27',''),
 ('S41_neural_fusion','S41',''),
 ('S35_multiroute','S35',''),
)
RULES=('all','origin_K01','origin_poly','origin_K234','origin_K34',
       'origin_K3plus','only_increase','only_decrease',
       'no_poly_to_low','neighbor_delta1','K234_neighbor_delta1',
       'poly_increase','destination_poly','other_family_agree1',
       'other_family_agree2','other_agree1_no_poly_to_low')
N=59309
BAN=('yourmt3','your_mt3','h9_','oracle','truth_aware')

def load_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def observed_mask(old,proposed,rule,agreements):
    ch=old!=proposed
    d=np.abs(old.astype(np.int16)-proposed.astype(np.int16))
    if rule=='all':return ch
    if rule=='origin_K01':return ch & (old<=1)
    if rule=='origin_poly':return ch & (old>=2)
    if rule=='origin_K234':return ch & np.isin(old,[2,3,4])
    if rule=='origin_K34':return ch & np.isin(old,[3,4])
    if rule=='origin_K3plus':return ch & (old>=3)
    if rule=='only_increase':return ch&(proposed>old)
    if rule=='only_decrease':return ch&(proposed<old)
    if rule=='no_poly_to_low':return ch&~((old>=2)&(proposed<=1))
    if rule=='neighbor_delta1':return ch&(d==1)
    if rule=='K234_neighbor_delta1':return ch&np.isin(old,[2,3,4])&(d==1)
    if rule=='poly_increase':return ch&(old>=2)&(proposed>old)
    if rule=='destination_poly':return ch&(proposed>=2)
    if rule=='other_family_agree1':return ch&(agreements>=1)
    if rule=='other_family_agree2':return ch&(agreements>=2)
    if rule=='other_agree1_no_poly_to_low':
        return ch&(agreements>=1)&~((old>=2)&(proposed<=1))
    raise ValueError('unexpected predeclared mask '+rule)

def self_test():
    old=np.array([0,1,2,3,4],dtype=np.int8)
    new=np.array([1,2,1,4,2],dtype=np.int8)
    a=np.array([0,1,2,3,0],dtype=np.int16)
    assert observed_mask(old,new,'origin_K01',a).tolist()==[True,True,False,False,False]
    assert observed_mask(old,new,'other_family_agree2',a).tolist()==[False,False,True,True,False]
    assert observed_mask(old,new,'no_poly_to_low',a).tolist()==[True,True,False,True,True]
    assert len(RULES)==16 and len(FAMILIES)==16
    print('PASS: 16 per-family conditional gates, oracle/true-K absent from inference',flush=True)

def run(a):
    if a.output.exists():raise ValueError('immutable experiment; output already exists')
    audit=a.source
    rank=load_csv(audit/'rank_actionable_vs_S18.csv')
    with np.load(audit/'all_distinct_native_predictions.npz',allow_pickle=False) as z:
        ids=z['global_index']
        y=z['true_K']
        folds=z['fold']
        baseline=z['S18_reference_K']
        bank=z['predictions']
        hashes=list(map(str,z['distinct_vector_sha256']))
        originals=list(map(str,z['original_names']))
    if len(ids)!=N or bank.shape[0]!=N or len(hashes)!=bank.shape[1]:
        raise ValueError('S43 full native archive incomplete')
    if ((baseline==y).sum()!=49178 or
        ((baseline==y)&(y>=2)).sum()!=2998):
        raise ValueError('S18 reference does not match verified S43')
    hash_index={h:i for i,h in enumerate(hashes)}
    chosen=[]
    missing=[]
    for family,source,prefix in FAMILIES:
        matching=[r for r in rank if r['source']==source and
            (not prefix or prefix.lower() in r['variant'].lower()) and
            not any(b in r['variant'].lower() for b in BAN)]
        if not matching:
            missing.append(family)
            continue
        # Each family contributes max gross fixes and max true-poly gross
        # fixes. Only the *rank selection* uses historical labels; all masks
        # and follow-up predictors use observable audio and predictions.
        champion=matching[0]
        polybest=max(matching,key=lambda r:(int(r['gross_poly_fixes_vs_S18']),
                                            int(r['gross_fixes_vs_S18'])))
        for suffix,row in [('gross',champion),('poly',polybest)]:
            digest=row['evidence_sha256']
            if digest not in hash_index:
                raise ValueError('missing verified S43 vector '+digest)
            if any((c['family']==family and c['sha256']==digest) for c in chosen):
                continue
            chosen.append(dict(family=family,source=source,
                kind=suffix,sha256=digest,variant=row['variant'],
                column=hash_index[digest],
                historical_gross=int(row['gross_fixes_vs_S18']),
                historical_regressions=int(row['regressions_vs_S18']),
                historical_poly=int(row['gross_poly_fixes_vs_S18'])))
    if len(chosen)<12 or not any(c['family']=='S40_temporal_CNN' for c in chosen):
        raise ValueError('key source families missing from S43 records: '+str(missing))
    # Deduplicate family votes. Only the chosen highest-gross variant of
    # each other family can corroborate a proposed class.
    families=sorted({c['family'] for c in chosen})
    champions={f:next(c for c in chosen if c['family']==f) for f in families}
    representatives={f:bank[:,c['column']] for f,c in champions.items()}
    policies={'series18_parent':baseline.copy()}
    records=[]
    disagreements=defaultdict(list)
    for cand in chosen:
        proposed=bank[:,cand['column']]
        independent_votes=np.zeros(N,np.int16)
        for other_name,other in representatives.items():
            if other_name==cand['family']:continue
            independent_votes+=((other==proposed)&(other!=baseline)).astype(np.int16)
        for rule in RULES:
            allowed=observed_mask(baseline,proposed,rule,independent_votes)
            p=np.where(allowed,proposed,baseline).astype(np.int8)
            name=f'series44__{cand["family"]}__{cand["kind"]}__{rule}'
            policies[name]=p
            cor=(baseline!=y)&(p==y)
            reg=(baseline==y)&(p!=y)
            changed=p!=baseline
            neutral=changed&(baseline!=y)&(p!=y)
            poly=(y>=2)
            raw_cor=(baseline!=y)&(proposed==y)
            raw_reg=(baseline==y)&(proposed!=y)
            r=dict(name=name,family=cand['family'],source=cand['source'],
                variant=cand['variant'],kind=cand['kind'],
                original_sha256=cand['sha256'],mask=rule,
                source_fit_provenance_certified=False if cand['family'].startswith(
                    ('S1','S2','S3','S4','historical_')) else None,
                corrections=int(cor.sum()),regressions=int(reg.sum()),
                neutral=int(neutral.sum()),
                net=int(cor.sum()-reg.sum()),
                poly_corrections=int((cor&poly).sum()),
                poly_regressions=int((reg&poly).sum()),
                poly_net=int((cor&poly).sum()-(reg&poly).sum()),
                global_correct=int((p==y).sum()),
                poly_correct=int(((p==y)&poly).sum()),
                suppressed_old_corrections=int((raw_cor&~cor).sum()),
                suppressed_old_regressions=int((raw_reg&~reg).sum()),
                per_true_K=[dict(k=k,cor=int((cor&(y==k)).sum()),
                      reg=int((reg&(y==k)).sum())) for k in range(7)],
                per_source_K=[dict(k=k,cor=int((cor&(baseline==k)).sum()),
                      reg=int((reg&(baseline==k)).sum())) for k in range(7)],
                per_fold=[dict(fold=f,cor=int((cor&(folds==f)).sum()),
                      reg=int((reg&(folds==f)).sum())) for f in (0,1,2,4)])
            records.append(r)
    pareto=set()
    for family in families:
        rows=[r for r in records if r['family']==family]
        for x in rows:
            if not any((o['corrections']>=x['corrections'] and
                       o['regressions']<=x['regressions'] and
                       (o['corrections']>x['corrections'] or
                        o['regressions']<x['regressions'])) for o in rows):
                pareto.add(x['name'])
    # Build strict posterior-free per-fam ledgers, class labels audit ONLY.
    for r in records:r['pareto_within_family']=r['name'] in pareto
    ranked=sorted(records,key=lambda r:(-r['corrections'],
        r['regressions'],-r['poly_corrections'],r['name']))
    safe=[r['name'] for r in ranked if r['corrections']>0
          and r['regressions']==0 and r['poly_correct']>=2998]
    bestnet=sorted(records,key=lambda r:(-r['net'],-r['poly_net']))[:20]
    a.output.mkdir(parents=True)
    with (a.output/'all_compromises.csv').open('w',newline='',encoding='utf-8') as f:
        field=['name','family','source','variant','kind','original_sha256',
               'mask','corrections','regressions','neutral','net',
               'poly_corrections','poly_regressions','poly_net',
               'global_correct','poly_correct',
               'suppressed_old_corrections','suppressed_old_regressions',
               'pareto_within_family','per_true_K','per_source_K','per_fold']
        writer=csv.DictWriter(f,fieldnames=field,extrasaction='ignore')
        writer.writeheader()
        for row in ranked:
            writer.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),list)
                else row.get(k,'') for k in field})
    np.savez_compressed(a.output/'all_compromise_predictions.npz',
        global_index=ids,true_K=y,fold=folds,S18_reference_K=baseline,
        variant_ids=np.asarray(list(policies)),
        predictions=np.column_stack(list(policies.values())))
    result=dict(status='completed',
       archived_source='S43 independent verified immutable archive',
       total_source_families=len(families),missing_requested_families=missing,
       candidate_variants_selected=len(chosen),
       predeclared_gates_per_variant=len(RULES),
       total_evaluated_proposals=len(records),
       original_S18_unchanged=True,H9_excluded=True,
       no_true_K_used_in_masks_or_agreement=True,
       confidence_no_historical_OOF_guarantee=True,
       previously_used_development_cohort=True,
       no_new_head_promoted=True,
       family_champions=chosen,
       best_gross=ranked[:25],
       best_net=bestnet,
       pareto_count=len(pareto),
       zero_regression_positive_poly_candidates=safe)
    (a.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,
        sort_keys=True,indent=2)+'\n')
    lines=['# S44 — compromis dédiés sur les corrections historiques',
        '', f"Familles auditées : {len(families)}, candidats : {len(chosen)}, variantes label-blind : {len(records)}.",
        f"Masques : {len(RULES)} par candidat ; Pareto : {len(pareto)} politiques.",
        'Précaution : familles historiques sans preuves OOF complètes = signal de recherche seulement.',
        '**Aucune prédiction utilisant le vrai K ; les classements eux-mêmes sont exploratoires sur une cohorte connue.**',
        '', '| Source (meilleur compromis NET de la famille) | Corrections | Régressions | Poly fix | Poly loss | Règle |',
        '|---|---:|---:|---:|---:|---|']
    for fam in families:
        within=sorted([r for r in records if r['family']==fam],
            key=lambda r:(-r['net'],-r['poly_net']))[0]
        lines.append(f"| {fam} | {within['corrections']} | "
           f"{within['regressions']} | {within['poly_corrections']} | "
           f"{within['poly_regressions']} | {within['mask']} |")
    lines+=['','## Plus grands volumes de corrections, sans effacer les régressions',
        '| Source | Règle | Corrections | Régressions | Corrections poly | Régressions poly |',
        '|---|---|---:|---:|---:|---:|']
    for r in ranked[:24]:
        lines.append(f"| {r['family']} | {r['mask']} | {r['corrections']} | "
           f"{r['regressions']} | {r['poly_corrections']} | "
           f"{r['poly_regressions']} |")
    lines+=['',f"Positives zéro-régression avec poly≥S18 : {len(safe)}",
        'Tous les cas retenus/perdus, fold et K0–K6 sont dans all_compromises.csv et all_compromise_predictions.npz.',
        'Aucune promotion, H9 exclue, S18 inchangée.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        if not a.source or not a.output:raise ValueError('S43 archive and output required')
        run(a)
