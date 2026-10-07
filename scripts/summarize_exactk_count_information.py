"""Independently rescore diagnostic probes and persist row-level evidence."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import sklearn

ARMS=('candidate_only','native_channel_norm','native_raw','native_raw_multires')
FOLDS=(0,1,2,4)

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def metrics(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7)
    n=cm.sum(); poly_n=cm[2:].sum()
    return {'rows':int(n),'correct':int(np.trace(cm)),'exact':float(np.trace(cm)/n),
        'poly_rows':int(poly_n),'poly_correct':int(np.diag(cm)[2:].sum()),
        'poly_exact':float(np.diag(cm)[2:].sum()/poly_n),
        'under':int(np.tril(cm,-1).sum()),'over':int(np.triu(cm,1).sum()),
        'poly_under':int(sum(cm[i,:i].sum() for i in range(2,7))),
        'poly_over':int(sum(cm[i,i+1:].sum() for i in range(2,7))),
        'by_k':{str(k):{'rows':int(cm[k].sum()),'correct':int(cm[k,k]),
                        'exact':float(cm[k,k]/cm[k].sum())} for k in range(7)},
        'confusion_true_by_predicted':cm.tolist()}

def main(a):
    root=a.input;a.output.mkdir(parents=True,exist_ok=True)
    with np.load(root/'cohort.npz') as d:data={k:d[k] for k in d.files}
    y,fold,members=data['k'],data['fold'],data['member']
    comp=np.asarray([Path(str(m)).stem.split('_',1)[1].rsplit('_',1)[0] for m in members])
    groups=np.unique(comp)
    assert len(groups)==19
    predictions={'freeze_local_combo':data['baseline'],'yourmt3_plus':data['predicted']}
    original_reports={}
    for arm in ARMS:
        with np.load(root/(arm+'-predictions.npz')) as d:
            pred,prob=d['predicted'],d['probability']
            assert pred.shape==y.shape and prob.shape==(len(y),7)
            np.testing.assert_array_equal(pred,prob.argmax(1))
            assert np.allclose(prob.sum(1),1,atol=1e-5) and np.isfinite(prob).all()
            predictions[arm]=pred
        r=json.loads((root/(arm+'-report.json')).read_text())
        verified=metrics(y,pred)
        assert verified['correct']==r['metrics']['correct']
        assert verified['poly_correct']==r['metrics']['poly']['correct']
        assert all(metrics(y[fold==f],pred[fold==f])['correct']==r['by_fold'][str(f)]['metrics']['correct'] for f in FOLDS)
        original_reports[arm]=r
    boot_groups=np.random.default_rng(7102026).integers(0,len(groups),(10000,len(groups)))
    comparisons={}
    for before,after in [('candidate_only','native_channel_norm'),('native_channel_norm','native_raw'),
                         ('native_raw','native_raw_multires'),('freeze_local_combo','native_raw_multires')]:
        x,z=predictions[before],predictions[after]
        delta=(z==y).astype(int)-(x==y).astype(int)
        record={}
        for name,mask in [('global',np.ones(len(y),bool)),('poly',y>=2)]:
            n=np.asarray([np.sum(mask&(comp==g)) for g in groups])
            v=np.asarray([delta[mask&(comp==g)].sum() for g in groups])
            boot=100*v[boot_groups].sum(1)/n[boot_groups].sum(1)
            record[name]={'net':int(delta[mask].sum()),'gain_pp':float(100*delta[mask].mean()),
                'fixed':int((mask&(x!=y)&(z==y)).sum()),'regressed':int((mask&(x==y)&(z!=y)).sum()),
                'composition_bootstrap_95_interval_pp':np.quantile(boot,[.025,.975]).tolist(),
                'gain_pp_by_fold':{str(f):float(100*delta[mask&(fold==f)].mean()) for f in FOLDS}}
        comparisons[before+'__to__'+after]=record
    base,teacher=predictions['freeze_local_combo'],predictions['yourmt3_plus'];poly=y>=2
    complementarity={'both_correct':int((poly&(base==y)&(teacher==y)).sum()),
        'only_freeze_correct':int((poly&(base==y)&(teacher!=y)).sum()),
        'only_yourmt3_correct':int((poly&(base!=y)&(teacher==y)).sum()),
        'both_wrong':int((poly&(base!=y)&(teacher!=y)).sum()),
        'oracle_union_exact':float(np.mean(((base==y)|(teacher==y))[poly])),
        'oracle_is_not_achieved_score':True}
    results={'status':'independently_rescored','automatic_promotion':False,'date':'2026-10-07',
        'reference_run':37605163312,'source_commit':'aebfe9f8cb6dc2c0dc391f773fd331756e4c5a45',
        'protocol':json.loads((root/'protocol.json').read_text()),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
        'rows':len(y),'poly_rows':int(poly.sum()),'tracks':len(np.unique(members)),
        'compositions':len(groups),'folds':list(FOLDS),
        'metrics':{k:metrics(y,p) for k,p in predictions.items()},
        'metrics_by_fold':{str(f):{k:metrics(y[fold==f],p[fold==f]) for k,p in predictions.items()} for f in FOLDS},
        'comparisons':comparisons,'complementarity':complementarity,
        'cache_provenance':json.loads((root/'cache_provenance.json').read_text()),
        'audio_provenance':json.loads((root/'audio_provenance.json').read_text()),
        'bootstrap':{'unit':'composition','resamples':10000,'seed':7102026,
                     'interpretation':'descriptive development-set interval, conditional on one training seed; no multiplicity adjustment'},
        'limitations':['Existing repeatedly inspected development folds, not fresh generalization evidence.',
            'Information probes are boosted trees on deterministic summaries, not a retraining of freeze_local_combo.',
            'Raw versus normalized changes both available level information and numerical representation.',
            'Native cached upstream feature producers are shared with the historical reference.',
            'YourMT3 pretrained overlap is unresolved; its outputs are never probe inputs or targets.',
            'Multi-resolution poly gain bootstrap interval includes zero.',
            'Best probe loses 397 K0 correct decisions relative to reference and remains at zero K6 accuracy.',
            'Probe versus reference gain is not a pure input ablation: learner and weighting differ.']}
    npz=a.output/'exactk-count-information-predictions.npz'
    np.savez_compressed(npz,global_index=data['global_index'],member=members,fold=fold,
        starts=data['starts'],k=y,**{k:v.astype(np.int8) for k,v in predictions.items()})
    results['predictions_sha256']=digest(npz)
    (a.output/'exactk-count-information-report.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    labels={'freeze_local_combo':'Référence freeze_local_combo','candidate_only':'Sonde : candidats seuls',
        'native_channel_norm':'Sonde : candidats + spectre normalisé',
        'native_raw':'Sonde : candidats + spectre brut',
        'native_raw_multires':'Sonde : brut + FFT 1024/2048'}
    lines=['# Audit des informations utiles au comptage polyphonique','',
        'Audit local terminé le 7 octobre 2026. Comptage K directement supervisé ; aucune identification de note produite.',
        '59 309 groupes, dont 7 385 polyphoniques, 190 enregistrements et 19 compositions. Folds 0, 1, 2, 4 ; fold 3 et player 05 exclus.',
        'Les paramètres et les quatre bras ont été fixés avant lecture des scores. Les compositions du fold évalué sont exclues de l’apprentissage de chaque sonde.',
        '', '## Résultats des sondes', '',
        '| Modèle | Exacts poly | Exact poly | Exact global |','|---|---:|---:|---:|']
    for name,label in labels.items():
        m=results['metrics'][name]
        lines.append(f"| {label} | {m['poly_correct']}/7385 | {100*m['poly_exact']:.4f}% | {100*m['exact']:.4f}% |")
    lines+=['','Une sonde est ici un classifieur de comptage HistGradientBoosting, avec 120 itérations et 15 feuilles maximum par arbre, identique dans les quatre bras. Elle ne remplace pas la référence.',
        '', '## Informations établies', '',
        '1. Les indices spectraux apportent +212 comptes poly exacts par rapport aux seuls résumés des candidats (+2,871 points).',
        '2. Conserver les valeurs brutes des trois canaux apporte +168 comptes poly exacts (+2,275 points) et +198 comptes globaux. Le gain poly est positif sur chacun des quatre folds. Intervalle descriptif à 95% par bootstrap des 19 compositions : +1,316 à +3,269 points.',
        '3. Ajouter des fenêtres FFT réelles de 1024 et 2048 échantillons, toujours dans les mêmes 4096 échantillons disponibles, apporte +82 comptes poly exacts (+1,110 point). Intervalle descriptif : -0,125 à +2,281 points ; le signal est plus faible.',
        '', 'LayerNormalization sur les trois canaux centre leur moyenne locale et normalise leur dispersion. Une branche conservant les niveaux et leur dispersion, en complément de la branche normalisée, est donc une piste concrète. Le présent test ne sépare pas parfaitement préservation de l’information et facilité d’apprentissage de la représentation.',
        '', '## Comparaison complète avec la référence', '',
        '| K vrai | Effectif | freeze_local_combo | Sonde brut + multirésolution | Solde exacts |',
        '|---:|---:|---:|---:|---:|']
    for k in range(7):
        x=results['metrics']['freeze_local_combo']['by_k'][str(k)]
        z=results['metrics']['native_raw_multires']['by_k'][str(k)]
        lines.append(f"| {k} | {x['rows']} | {100*x['exact']:.3f}% | {100*z['exact']:.3f}% | {z['correct']-x['correct']:+d} |")
    p=results['comparisons']['freeze_local_combo__to__native_raw_multires']['poly']
    lines += ['',f"Bilan poly : {p['fixed']} corrections, {p['regressed']} régressions, +{p['net']} net. Sous-comptages poly : 3630 → 3303 ; surcomptages poly : 1225 → 1407.",
        'Le gain poly de +1,963 point reste modeste. Il est positif sur trois folds ; le fold 4 recule de 0,594 point. K0 perd 397 réponses correctes et K6 reste à 0%. Ce résultat ne satisfait pas encore l’objectif d’une forte amélioration.',
        '', '## L’essai de normalisation antérieur est pris en compte', '',
        'Le run 36490993251 avait déjà comparé, sur le composant natif et une validation interne, la normalisation et une division fixe par 12 : +47 comptes poly exacts mais -95 K1 et -11 globaux. Il avait été rejeté. Cet audit confirme un signal dans les valeurs brutes avec un autre apprenant et quatre folds ; il ne transforme pas cet ancien échec en succès.',
        'La modification à évaluer doit préserver la branche normalisée et ajouter les informations manquantes, avec validation appariée. Aucun score de ce futur réseau n’est annoncé.',
        '', '## Comparaison descriptive avec YourMT3+', '',
        'YourMT3+ corrige 2430 erreurs poly de la référence mais en perd 932 que celle-ci réussissait. Le choix parfait entre les deux atteindrait 67,163% ; ce chiffre est un oracle utilisant la vérité, pas un score obtenu par un système. Les prédictions YourMT3+ ne servent jamais à entraîner ces sondes.',
        '', '## Reproduction', '',
        'Environnement mesuré : Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 ; deux threads par apprentissage.',
        'Sources : release `v273-window-pair-36351028493` (`native-window-bundle.zip`), configuration du commit `aebfe9f8cb6dc2c0dc391f773fd331756e4c5a45`, audio GuitarSet mono pickup mix vérifié par MD5, et artifact `yourmt3-exactk-summary` du run 37605163312.',
        'Les 190 reconstructions de contrôle du spectre d’origine concordent avec le cache float16 à la précision attendue (erreur absolue maximale 0,003942). Les nouvelles FFT n’accèdent jamais à des échantillons après début du groupe +2788.',
        'Scripts : `scripts/audit_exactk_count_information.py` (étapes cache, audio, train) et `scripts/summarize_exactk_count_information.py`. Empreintes, matrices de confusion, métriques par fold et protocole sont dans le rapport JSON. Les décisions par ligne sont conservées dans le NPZ.',
        '', '```sh',
        'python scripts/audit_exactk_count_information.py cache --summary SUMMARY.zip --bundle BUNDLE --config CONFIG.json --output RESULTS',
        'python scripts/audit_exactk_count_information.py audio --summary SUMMARY.zip --bundle BUNDLE --audio audio_mono-pickup_mix.zip --output RESULTS',
        'python scripts/audit_exactk_count_information.py train --arm candidate_only --output RESULTS',
        'python scripts/audit_exactk_count_information.py train --arm native_channel_norm --output RESULTS',
        'python scripts/audit_exactk_count_information.py train --arm native_raw --output RESULTS',
        'python scripts/audit_exactk_count_information.py train --arm native_raw_multires --output RESULTS',
        'python scripts/summarize_exactk_count_information.py --input RESULTS --output analysis',
        '```', '', 'Aucune modification de la référence, aucune évaluation du fold 3 ou du player 05, aucune promotion automatique.']
    (a.output/'README-exactk-count-information.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'verified','rows':len(y),'compositions':len(groups),
        'paired_best':results['comparisons']['freeze_local_combo__to__native_raw_multires'],
        'prediction_bytes':npz.stat().st_size},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
