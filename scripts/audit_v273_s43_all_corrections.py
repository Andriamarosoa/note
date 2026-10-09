"""S43: exhaustive retrievable native correction audit from every preserved loop.

RANK POSITIVE CORRECTIONS FIRST, regardless of net. All variants survive in
a CSV. Equal vectors are aliases, not new heads. No truth-driven gate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

N=59309
FOLDS=(0,1,2,4)
AVOID=('yourmt3','your_mt3','h9_','oracle','ideal_selector',
       'target_oracle','truth_aware')
VECTOR_KEYS=('predictions','predicted_K','candidate_predictions',
             'all_predictions','all_candidates','candidate_K','decision_K')
IDENTITY_KEYS=('global_index','native_global_index','global_indices',
               'ids','native_ids')
TRUTH_KEYS=('true_K','native_truth','truth','target_K')
META_KEYS={'global_index','native_global_index','global_indices',
           'ids','native_ids','fold','native_fold','true_K','native_truth',
           'truth','target_K','variant_ids','variant_names','source_ids',
           'labels','baseline','native_baseline','freeze_K','parent_K',
           'yourmt3_K','yourmt3_K0','yourmt3_K1'}
SOURCE_REGISTRIES=[
    ('historical_regression_loops',
     'analysis/evidence/v273-regression-loops/candidate-registry.json',
     'added_candidates'),
    ('historical_memory',
     'analysis/evidence/v273-audit-memory/variants/registry.json',
     'variants')
]
NATIVE_FILES=[
    ('historical_regression_loops',
     'analysis/evidence/v273-regression-loops/all-candidate-decisions.npz'),
    ('historical_regression_posthoc',
     'analysis/evidence/v273-regression-loops/posthoc-audit/appended-candidates.npz'),
    ('historical_memory',
     'analysis/evidence/v273-audit-memory/variants/all-variant-decisions.npz'),
]

def jdump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,
                               indent=2,sort_keys=True)+'\n')

def filehash(file):
    h=hashlib.sha256()
    with file.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def digest(pred):
    return hashlib.sha256(np.asarray(pred,dtype=np.int8).tobytes()).hexdigest()

def metrics(pred,truth,source):
    change=pred!=source
    fix=change&(pred==truth)&(source!=truth)
    regress=change&(pred!=truth)&(source==truth)
    neutral=change&(pred!=truth)&(source!=truth)
    return fix,regress,neutral

def rows_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({key:(json.dumps(v,ensure_ascii=False,
                     sort_keys=True) if isinstance(v,(dict,list,tuple)) else v)
                 for key,v in r.items()})

def data(file):
    with np.load(file,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def extract_views(z,ids,truth,file):
    """Return (name,vector) list, rejecting unprovable index alignment."""
    n=len(ids)
    source_id=next((np.asarray(z[k]) for k in IDENTITY_KEYS if k in z),None)
    source_y=next((np.asarray(z[k]) for k in TRUTH_KEYS if k in z),None)
    if source_id is None:
        # A repository-tracked file is only usable if its row-order manifest
        # can be validated elsewhere, not solely because row count matches.
        raise ValueError('no native global_index; row alignment unproven')
    if source_id.shape!=(n,) or not np.array_equal(source_id,ids):
        raise ValueError('different native global_index order or cohort')
    if source_y is not None and not np.array_equal(source_y,truth):
        raise ValueError('different true-K labels')
    variants=None
    for key in ('variant_ids','variant_names','candidate_ids','names'):
        if key in z:
            variants=list(map(str,np.asarray(z[key]).tolist()))
            break
    key=next((k for k in VECTOR_KEYS if k in z and
              np.asarray(z[k]).ndim in (1,2)),None)
    if key is None:
        guesses=[]
        for k,v in z.items():
            v=np.asarray(v)
            if (k not in META_KEYS and
                ('pred' in k.lower() or 'decision' in k.lower()) and
                v.ndim in (1,2)):
                guesses.append(k)
        if len(guesses)==1:key=guesses[0]
    if key is None:
        raise ValueError('no named numeric prediction matrix')
    v=np.asarray(z[key])
    if v.ndim==1:
        if len(v)!=n:raise ValueError('one decision vector with wrong length')
        v=v[:,None]
    elif v.shape[0]!=n:
        if v.shape[1]==n:v=v.T
        else:raise ValueError(f'prediction rows={v.shape} != {n}')
    if v.dtype.kind not in 'biu':
        raise ValueError(f'prediction dtype {v.dtype} is not integer K0..K6')
    if not np.all((v>=0)&(v<=6)):
        raise ValueError('predictions contain invalid K outside 0..6')
    if variants is None:
        variants=[f'{key}__column{j:04d}' for j in range(v.shape[1])]
    elif len(variants)!=v.shape[1]:
        raise ValueError('variant_ids count differs from matrix columns')
    return [(name,v[:,j].astype(np.int8,copy=False))
            for j,name in enumerate(variants)]

def find_reference(root,collection,inputs):
    src=inputs['native_baseline'].astype(np.int8)
    y=inputs['native_truth'].astype(np.int8)
    ids=inputs['native_global_index']
    fold=inputs['native_fold']
    assert y.shape==src.shape==ids.shape==fold.shape==(N,)
    assert set(map(int,np.unique(fold)))==set(FOLDS)
    assert int((src==y).sum())==48454
    assert int(((src==y)&(y>=2)).sum())==2530
    possible=list((collection/'S18').rglob('predictions.npz'))
    if len(possible)!=1:
        raise ValueError(f'exactly one S18 artifact required, found {len(possible)}')
    z=data(possible[0])
    names=extract_views(z,ids,y,possible[0])
    exact=[(name,p) for name,p in names
           if int((p==y).sum())==49178 and
           int(((p==y)&(y>=2)).sum())==2998 and
           int(metrics(p,y,src)[0].sum())==2934 and
           int(metrics(p,y,src)[1].sum())==2210]
    if not exact:raise ValueError('S18 exact 49,178/2,998 baseline missing')
    preferred=next(((name,p) for name,p in exact
          if 'parent' in name or 'k43' in name),exact[0])
    return y,ids,fold,src,preferred[1],dict(name=preferred[0],
                                  file=str(possible[0]))

def source_files(root,collection):
    for series in sorted(collection.iterdir()):
        if not series.is_dir():continue
        if not re.match(r'^S\d{2}[ab]?$',series.name):continue
        for file in sorted(series.rglob('*.npz')):
            if file.name in ('predictions.npz','prediction.npz',
                             'all-candidate-decisions.npz',
                             'all-variant-decisions.npz'):
                yield series.name,file
    for series,path in NATIVE_FILES:
        file=root/path
        if file.exists():yield series,file

def registry_sources(root):
    result=[]
    for name,relative,field in SOURCE_REGISTRIES:
        file=root/relative
        if not file.exists():
            result.append(dict(source=name,file=relative,
                state='not_available'))
            continue
        obj=json.loads(file.read_text())
        rows=obj.get(field,[])
        if not isinstance(rows,list):
            result.append(dict(source=name,file=relative,
                state='registry_not_array'))
            continue
        for rec in rows:
            ref=rec.get('versus_freeze',{}).get('global',{})
            fix=int(ref.get('corrections',-1))
            loss=int(ref.get('regressions',-1))
            if fix<0:continue
            result.append(dict(source=name,file=relative,
                state='metric_only_not_a_new_trained_head',
                name=rec.get('variant_id','unknown'),
                baseline='freeze',gross_fixes=fix,
                gross_regressions=loss,
                net=fix-loss,
                identical_prediction_alias_of=rec.get('identical_prediction_alias_of'),
                from_native_registry=True,
                historical_overlap_not_claimed_if_vector_missing=True))
    return result

def run(a):
    root=a.root
    archive=a.collection
    out=a.output
    if out.exists():raise ValueError('will not overwrite prior S43 research')
    inp=data(root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz')
    y,ids,fold,freeze,s18,ref=find_reference(root,archive,inp)
    out.mkdir(parents=True)
    provenance=json.loads((archive/'source_manifest.json').read_text())
    excluded=[]
    vector_rows=[]
    by_hash={}
    rawvectors={}
    per_vector_audits={}
    covered_all=np.zeros(len(y),bool)
    covered_poly=np.zeros(len(y),bool)
    all_fix_count=0
    original_vectors=0
    for family,file in source_files(root,archive):
        try:
            z=data(file)
            variants=extract_views(z,ids,y,file)
            for variant,pred in variants:
                if any(x in variant.lower() for x in AVOID):
                    excluded.append(dict(source=family,file=str(file),
                        variant=variant,reason='forbidden oracle or YourMT3+ head'))
                    continue
                hashkey=digest(pred)
                fixed_s18,lost_s18,neutral_s18=metrics(pred,y,s18)
                fixed_fr,lost_fr,neutral_fr=metrics(pred,y,freeze)
                if not (fixed_s18.any() or fixed_fr.any()):
                    # Required to preserve all actual zero-positive vectors
                    # for complete audits, not candidate heads.
                    pass
                origin='head_vector'
                duplicate=by_hash.get(hashkey)
                if duplicate is None:
                    by_hash[hashkey]=f'{family}::{variant}'
                    rawvectors[hashkey]=pred.copy()
                    original_vectors+=1
                else:origin='alias_identical_vector'
                all_fix_count+=int(fixed_s18.sum())
                covered_all|=fixed_s18
                covered_poly|=(fixed_s18&(y>=2))
                cor_by_k=[int(np.sum(fixed_s18&(y==k))) for k in range(7)]
                reg_by_k=[int(np.sum(lost_s18&(y==k))) for k in range(7)]
                per_fold=[
                    dict(fold=f,corrections=int(np.sum(fixed_s18&(fold==f))),
                        regressions=int(np.sum(lost_s18&(fold==f))))
                    for f in FOLDS]
                row=dict(source=family,variant=str(variant),
                    evidence_file=str(file.relative_to(root) if file.is_relative_to(root)
                                      else file.relative_to(archive)),
                    evidence_sha256=hashkey,
                    original_or_duplicate=origin,
                    alias_of=duplicate or '',
                    gross_fixes_vs_S18=int(fixed_s18.sum()),
                    regressions_vs_S18=int(lost_s18.sum()),
                    neutral_vs_S18=int(neutral_s18.sum()),
                    net_vs_S18=int(fixed_s18.sum()-lost_s18.sum()),
                    gross_poly_fixes_vs_S18=int(np.sum(fixed_s18&(y>=2))),
                    poly_regressions_vs_S18=int(np.sum(lost_s18&(y>=2))),
                    net_poly_vs_S18=int(np.sum((pred==y)&(y>=2))-
                                        np.sum((s18==y)&(y>=2))),
                    gross_fixes_vs_freeze=int(fixed_fr.sum()),
                    regressions_vs_freeze=int(lost_fr.sum()),
                    net_vs_freeze=int(fixed_fr.sum()-lost_fr.sum()),
                    global_correct=int((pred==y).sum()),
                    poly_correct=int(np.sum((pred==y)&(y>=2))),
                    per_true_K_fixes=cor_by_k,
                    per_true_K_regressions=reg_by_k,
                    per_fold=per_fold,
                    previously_implemented_no_new_head_promotion=True)
                vector_rows.append(row)
                if duplicate is None:
                    per_vector_audits[hashkey]=(fixed_s18.copy(),lost_s18.copy())
        except Exception as exc:
            excluded.append(dict(source=family,file=str(file),
                reason=f'{type(exc).__name__}: {str(exc)[:450]}'))
        print(json.dumps(dict(source=family,file=file.name,
                  all_candidates=len(vector_rows),distinct=original_vectors,
                  exclusions=len(excluded))),flush=True)

    require_total=len(vector_rows)
    if require_total<100:
        raise ValueError(f'insufficient native historical vectors {require_total}')
    # Register historical metrics even if the NPZ had missing IDs, and flag
    # registry-only (never pretend overlap was measured).
    registry=registry_sources(root)
    registry_by_id={r.get('name'):r for r in registry if 'name' in r}
    for row in vector_rows:
        source_info=registry_by_id.get(row['variant'])
        if source_info is not None:
            assert source_info['gross_fixes']==row['gross_fixes_vs_freeze']
            assert source_info['gross_regressions']==row['regressions_vs_freeze']
            source_info['state']='native_predictions_crosschecked'
            source_info['gross_fixes_vs_S18']=row['gross_fixes_vs_S18']
            source_info['gross_regressions_vs_S18']=row['regressions_vs_S18']
    ranking=sorted(vector_rows,key=lambda r:(
        -r['gross_fixes_vs_S18'],-r['gross_poly_fixes_vs_S18'],
         -r['net_vs_S18'],r['source'],r['variant']))
    rank_fr=sorted(vector_rows,key=lambda r:(
        -r['gross_fixes_vs_freeze'],
        -r['net_vs_freeze'],r['source'],r['variant']))
    # Never intermix reference spaces; save two distinct complete tables.
    rows_csv(out/'rank_all_vs_S18.csv',ranking)
    rows_csv(out/'rank_all_vs_freeze.csv',rank_fr)
    rows_csv(out/'historical_metric_registry.csv',registry)
    rows_csv(out/'excluded_or_incomparable_sources.csv',excluded)
    seen=set()
    distinct_rank=[]
    for row in ranking:
        h=row['evidence_sha256']
        if h not in seen:
            seen.add(h)
            distinct_rank.append(row)
    top=distinct_rank[:128]
    # Actual measured overlap of corrected events among distinct top heads.
    overlap=[]
    alltop=np.zeros((N,),np.uint16)
    for row in top:
        c,_=per_vector_audits[row['evidence_sha256']]
        alltop+=c.astype(np.uint16)
    current=np.zeros(N,bool)
    for idx,row in enumerate(top):
        changed,reg=per_vector_audits[row['evidence_sha256']]
        unique=int(((alltop==1)&changed).sum())
        novel=int((changed&~current).sum())
        novel_poly=int((changed&~current&(y>=2)).sum())
        ov=int((changed&current).sum())
        current|=changed
        overlap.append(dict(rank=idx+1,source=row['source'],
            variant=row['variant'],
            gross_corrections=int(changed.sum()),
            regressions=int(reg.sum()),
            exclusive_among_top_128=unique,
            marginal_new_over_ranked_prior=novel,
            overlap_with_prior=ov,
            marginal_new_poly_over_ranked_prior=novel_poly,
            cumul_union_correctable_events=int(current.sum()),
            cumulative_union_is_truth_oracle_not_deployable=True))
    rows_csv(out/'head_correction_overlap.csv',overlap)
    # Dedicated transition audit for best 128 (not only positive net).
    granular={}
    for row in top:
        v=rawvectors[row['evidence_sha256']]
        fc,rc=per_vector_audits[row['evidence_sha256']]
        changes=v!=s18
        trans=[]
        for a0 in range(7):
            for a1 in range(7):
                if a0==a1:continue
                mask=changes&(s18==a0)&(v==a1)
                n=int(mask.sum())
                if n==0:continue
                fixes=int((fc&mask).sum())
                breaks=int((rc&mask).sum())
                trans.append(dict(source_K=a0,candidate_K=a1,
                    events=n,corrections=fixes,regressions=breaks,
                    neutral=n-fixes-breaks,
                    empirical_mean_utility_on_historical_data=(fixes-breaks)/n))
        granular[f"{row['source']}::{row['variant']}"]=trans
    jdump(out/'transition_audit_top128.json',granular)

    # Rank best branch per series, even if negative or absent.
    groups=defaultdict(list)
    for row in vector_rows:groups[row['source']].append(row)
    best=[]
    for fam in sorted(groups):
        ordered=sorted(groups[fam],key=lambda r:-r['gross_fixes_vs_S18'])
        best.append(dict(series=fam,variant_count=len(ordered),
            distinct_vectors=len({r['evidence_sha256'] for r in ordered}),
            max_gross_corrections=ordered[0]['gross_fixes_vs_S18'],
            associated_regressions=ordered[0]['regressions_vs_S18'],
            associated_net=ordered[0]['net_vs_S18'],
            best_variant=ordered[0]['variant'],
            poly_corrections=ordered[0]['gross_poly_fixes_vs_S18'],
            poly_regressions=ordered[0]['poly_regressions_vs_S18']))
    best.sort(key=lambda r:-r['max_gross_corrections'])
    rows_csv(out/'best_per_loop.csv',best)

    # Keep corrections of single families with a negative global net;
    # rank potential new selectors by correction volume and diversity.
    proposer=[]
    for idx,row in enumerate(distinct_rank[:256]):
        fix,lost=per_vector_audits[row['evidence_sha256']]
        observed=np.asarray(row['per_true_K_fixes'])
        counts=[d['corrections'] for d in row['per_fold']]
        # No true labels may be used in a deployed gate: these are
        # retrospective audit annotations, not a user-facing control.
        exceptional=row['gross_fixes_vs_S18']>=80
        allfold=sum(c>0 for c in counts)
        if not exceptional:continue
        family=row['source'].lower()
        headtype=('temporal_morphology' if 'S40' in row['source'] else
          'neural_fusion' if 'S41' in row['source'] else
          'acoustic_k_expert' if 'S38' in row['source'] else
          'historical_corrector_or_selector')
        mask=fix&(y>=2)
        topK=int(np.argmax(observed))
        proposer.append(dict(head_id=f'S43_ARCHIVE_{len(proposer)+1:03d}',
            variant=row['variant'],source=row['source'],
            rank_by_gross_corrections=idx+1,
            type=headtype,
            action='consider_new_specialized_selection',
            full_native_prediction_saved=True,
            historic_corrections=row['gross_fixes_vs_S18'],
            historic_regressions=row['regressions_vs_S18'],
            historic_poly_corrections=int(mask.sum()),
            leading_true_K_diagnostic=topK,
            folds_with_any_correction=allfold,
            allow_prediction_using_true_K=False,
            possible_input_features=[
              'original_spectrum_42x49','birth_attack_morphology',
              'S18_current_K','proposed_K',
              'relative_harmonic_energy','audited_path_history'
            ],
            prevent_promotion=True,
            no_held_fold_validation_of_new_head_yet=True,
            available_vectors_hash=row['evidence_sha256'],
            alias_count=sum(r['evidence_sha256']==row['evidence_sha256']
                             for r in vector_rows)-1))
        if len(proposer)>=100:break
    jdump(out/'candidate_head_registry.json',dict(
        schema='S43_ARCHIVED_HEAD_PROPOSALS_NOT_PRODUCTION',
        score_data_uses_held_truth_for_audit_only=True,
        maximum_candidates=len(proposer),
        proposal_heads=proposer,
        banned_expert='H9_YourMT3',
        retained_modules_existing_S35=18,
        input_source_only_no_new_weights_trained=True,
        enabled_for_inference=False))

    # Count source runs with no prediction data; preserve missing artifacts.
    found={r['source'] for r in vector_rows}
    missing=[x for x in provenance['sources']
             if x['series'] not in found]
    uncovered_summary=[]
    for x in missing:
        uncovered_summary.append(dict(series=x['series'],
           status=x['status'],reason=x.get('warnings',[]),
           data_files=x.get('prediction_archives',[])))
    jdump(out/'uncovered_series.json',uncovered_summary)
    report=dict(status='completed',
        reference_freeze_exact=48454,reference_S18_exact=49178,
        reference_freeze_poly_exact=2530,
        reference_S18_poly_exact=2998,
        reference_S18_cor_vs_freeze=2934,
        reference_S18_reg_vs_freeze=2210,
        all_native_rows=N,poly_rows=int((y>=2).sum()),
        source_total_declared=len(provenance['sources']),
        source_series_with_valid_vectors=len([v for v in found if v.startswith('S')]),
        source_series_without_valid_vectors=uncovered_summary,
        total_valid_native_variants=len(vector_rows),
        distinct_prediction_vectors=len(by_hash),
        total_archived_legacy_registry_entries=len(registry),
        registry_entries_native_crosschecked=sum(
              x.get('state')=='native_predictions_crosschecked' for x in registry),
        source_alignment_or_schema_exceptions=len(excluded),
        max_fix_S18=ranking[0] if ranking else None,
        max_fix_freeze=rank_fr[0] if rank_fr else None,
        best_by_source=best,
        retrospective_all_variants_oracle_union_correctable_S18=
            int(covered_all.sum()),
        retrospective_poly_oracle_union_correctable_S18=
            int(covered_poly.sum()),
        oracle_is_not_a_prediction=True,
        no_new_correction_selection_model_trained=True,
        produced_candidate_head_proposals=len(proposer),
        never_seen_music_validation=False,
        no_model_promotion=True,
        H9_excluded_from_candidate_head_registry=True,
        source_reference=ref)
    jdump(out/'report.json',report)

    lines=['# S43 — Classement de toutes les corrections réellement récupérables',
        '', '**Comptage par correction brute, sans filtrer les régressions ni choisir les vrais labels lors de l inférence.**',
        f'Archives séries valides: {report["source_series_with_valid_vectors"]}/{report["source_total_declared"]}; '
        f'variantes: {len(vector_rows)}, vecteurs distincts: {len(by_hash)}.',
        f'Registres historiques: {len(registry)} variantes, dont '
        f'{report["registry_entries_native_crosschecked"]} recoupées sur valeurs natives.',
        'Références distinctes: S18 49 178/59 309, freeze 48 454/59 309; '
        'S18 vs freeze +2934/−2210.',
        '', '## Meilleure source de corrections vs S18 (classement brut)',
        '', '| Rang | Boucle | Variante | Corrections | Régressions | Poly corrigés | Poly perdus |',
        '|---|---|---|---:|---:|---:|---:|']
    for i,r in enumerate(best[:30],1):
        lines.append(f"| {i} | {r['series']} | {r['best_variant']} | "
            f"{r['max_gross_corrections']} | {r['associated_regressions']} | "
            f"{r['poly_corrections']} | {r['poly_regressions']} |")
    lines+=['','## Couverture et distinction des corrections',
        f"Union-oracle NON DÉPLOYABLE, toutes variantes: {int(covered_all.sum())} événements erronés S18 disposant d'au moins une bonne proposition ; poly {int(covered_poly.sum())}.",
        f"Union-oracle NON DÉPLOYABLE des 128 premiers: {int(current.sum())}.",
        'Les corrections « unique » par vecteur et « ajoutées par rang » figurent dans head_correction_overlap.csv.',
        f"Sources incomplètes/non comparables: {len(uncovered_summary)} séries; "
        f"fichiers au schéma incompatible: {len(excluded)}.",
        'Ne pas transformer ce classement en résultat de modèle. '
        'Chaque future tête doit apprendre une porte depuis les observables sur folds tenus à l écart.',
        f'Candidats d adaptation non activés: {len(proposer)}, liste détaillée dans candidate_head_registry.json.',
        'H9/YourMT3 exclue ; anciens producteurs conservés ; aucune promotion.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
