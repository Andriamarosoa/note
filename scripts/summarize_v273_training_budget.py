"""Verify all saved budget predictions and publish a bounded causal verdict."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_learning_bottleneck import metrics
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.test_v273_training_budget import interpret_budget
from scripts.train_boundaries import group_stem
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import array_hash, require


def check_metrics(actual, recorded):
    """Keep counts exact, allowing only floating point implementation noise."""
    if isinstance(actual,dict):
        require(set(actual)==set(recorded),'metric fields differ')
        for key in actual:
            check_metrics(actual[key],recorded[key])
    elif isinstance(actual,float):
        np.testing.assert_allclose(actual,recorded,rtol=1e-9,atol=1e-10)
    else:
        require(actual==recorded,'saved metrics do not match raw predictions')


def paired_change(before, after):
    for key in ('global_index', 'member', 'k'):
        np.testing.assert_array_equal(before[key], after[key])
    k=before['k']
    a=before['predicted']==k
    b=after['predicted']==k
    poly=k>=2
    def count(mask):
        return dict(rows=int(mask.sum()), before_correct=int((mask&a).sum()),
            after_correct=int((mask&b).sum()), corrected=int((mask&~a&b).sum()),
            regressed=int((mask&a&~b).sum()), delta=int((mask&b).sum()-(mask&a).sum()))
    groups=np.asarray([group_stem(str(member)) for member in before['member']])
    return dict(all=count(np.ones(len(k),bool)), poly=count(poly),
        by_k={str(value):count(k==value) for value in range(7)},
        poly_by_composition={group:count(poly&(groups==group)) for group in np.unique(groups)})


def verify(root, arm, cfg):
    validate_original_inventory(root)
    r=json.loads((root/'report.json').read_text())
    require(r['status']=='completed' and r['weighting']==arm, 'incomplete/wrong arm')
    require(r['outer_rows_evaluated']==0 and not r['output_corrector'] and
            not r['automatic_promotion'] and r['optimizer_state_restored_exactly'], 'wrong experiment')
    require(r['initial_epoch']==8 and r['final_epoch']==16 and
            r['original_optimizer_iterations']==2712 and r['batches_per_epoch']==339,
            'wrong training budget')
    require([x['epoch'] for x in r['history']]==list(range(9,17)), 'missing training epochs')
    require(set(r['checkpoints'])=={'8','12','16'}, 'missing checkpoints')
    predictions={}
    for epoch in (8,12,16):
        cp=r['checkpoints'][str(epoch)]
        require(cp['optimizer_iterations']==epoch*339, 'wrong update count')
        require(cp['weights_sha256']==digest(root/f'epoch-{epoch:02d}.weights.h5'), 'checkpoint mismatch')
        predictions[epoch]={}
        for split, rows, poly in (('fit',43357,5274),('validation',15952,2111)):
            with np.load(root/f'epoch-{epoch:02d}-{split}.npz',allow_pickle=False) as z:
                p={key:np.asarray(z[key]) for key in z.files}
            require(len(p['k'])==rows and int((p['k']>=2).sum())==poly, 'wrong population')
            require(len(np.unique(p['global_index']))==rows, 'duplicate rows')
            expected_hash=r['fit_indices_sha256' if split=='fit' else 'val_indices_sha256']
            require(array_hash(p['global_index'])==expected_hash, 'partition hash differs')
            folds={cfg['member_folds'][str(member)] for member in p['member']}
            require(folds==({1,2,4} if split=='fit' else {0}), 'outer or wrong inner fold')
            np.testing.assert_array_equal(p['predicted'],p['probability'].argmax(1))
            actual=metrics(p['k'],p['probability'],np.asarray(r['class_weights']))
            check_metrics(actual,cp['splits'][split])
            predictions[epoch][split]=p
        require(not np.intersect1d(predictions[epoch]['fit']['global_index'],
                                  predictions[epoch]['validation']['global_index']).size, 'partition overlap')
    comparisons={}
    for epoch in (12,16):
        change={split:paired_change(predictions[8][split],predictions[epoch][split])
                for split in ('fit','validation')}
        delta={split:change[split]['poly']['delta'] for split in change}
        label=interpret_budget(delta['fit'],delta['validation'])
        require(r['comparisons'][str(epoch)]==dict(poly_correct_delta=delta,interpretation=label),
                'training and independent conclusions differ')
        comparisons[str(epoch)]=dict(paired=change,interpretation=label)
    return dict(report=r,comparisons=comparisons),predictions


VERDICTS={
    'more_training_improves_seen_and_internal_validation':
        "Les mises à jour supplémentaires améliorent l'apprentissage et la validation interne. "
        "Le point à huit époques laisse donc un gain accessible à architecture constante dans cette expérience.",
    'better_fit_does_not_improve_internal_validation':
        "Les exemples vus sont mieux appris, mais la validation interne ne progresse pas. "
        "Prolonger ainsi ne résout pas la faible performance de validation.",
    'internal_gain_without_better_aggregate_fit':
        "La validation interne progresse sans gain agrégé sur les exemples vus. "
        "Ce résultat ne confirme pas un simple manque d'apprentissage du corpus vu.",
    'no_internal_gain_demonstrated':
        "Cette prolongation ne démontre aucun gain de validation interne. "
        "Elle ne prouve pas que tout autre budget ou réglage serait inutile."
}


def summarize(args):
    cfg=load_config(args.config)
    result=dict(status='verified',primary_comparison='16 versus 8',outer_rows_evaluated=0,
                automatic_promotion=False,arms={})
    predictions={}
    for arm in ('uniform','weighted'):
        result['arms'][arm],predictions[arm]=verify(args.root/f'budget-{arm}',arm,cfg)
    a,b=[result['arms'][arm]['report'] for arm in ('uniform','weighted')]
    for key in ('epoch_orders','original_seed','resume_dropout_seed','fit_indices_sha256',
                'val_indices_sha256','bundle_sha256','config_sha256','source_sha'):
        require(a[key]==b[key], 'paired arms differ: '+key)
    require(a['config_sha256']==digest(args.config),'wrong validation configuration')
    for epoch in (8,12,16):
        for split in ('fit','validation'):
            for key in ('global_index','member','k'):
                np.testing.assert_array_equal(predictions['uniform'][epoch][split][key],
                                              predictions['weighted'][epoch][split][key])
    text=['# Test du budget d’apprentissage : 8, 12 et 16 époques', '',
        'Résultats vérifiés à partir des probabilités brutes. Comparaison principale : **16 contre 8**.',
        'Aucune donnée du fold externe 3 évaluée ; aucun correcteur ; aucune promotion officielle.', '',
        '| Objectif | Époques | Exact K poly apprentissage | Exact K poly validation |',
        '|---|---:|---:|---:|']
    for arm in ('uniform','weighted'):
        r=result['arms'][arm]['report']
        for epoch in ('8','12','16'):
            scores=r['checkpoints'][epoch]['splits']
            cells=[f"{scores[s]['poly_correct']}/{scores[s]['poly_rows']} ({100*scores[s]['poly_exact']:.2f} %)"
                   for s in ('fit','validation')]
            text.append(f"| {arm} | {epoch} | {' | '.join(cells)} |")
    for arm in ('uniform','weighted'):
        c=result['arms'][arm]['comparisons']['16']
        p=c['paired']['validation']['poly']
        text.extend(['',f'## Objectif {arm}', '',VERDICTS[c['interpretation']], '',
            f"Validation polyphonique : {p['corrected']} erreurs corrigées, {p['regressed']} décisions dégradées, "
            f"solde **{p['delta']:+d}/{p['rows']}**.", '',
            '| K vrai | Effectif validation | Exacts à 8 | Exacts à 16 | Différence |',
            '|---:|---:|---:|---:|---:|'])
        for k,d in c['paired']['validation']['by_k'].items():
            text.append(f"| {k} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} |")
        text.extend(['','| Composition interne | Groupes poly | Exacts à 8 | Exacts à 16 | Différence |',
            '|---|---:|---:|---:|---:|'])
        for g,d in c['paired']['validation']['poly_by_composition'].items():
            text.append(f"| {g} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} |")
    text.extend(['','## Portée du verdict','',
        "Une seule graine de reprise et une seule partition interne : ces variations sont descriptives, "
        "sans affirmation de significativité ou de généralisation externe. Le point 12 est intermédiaire ; "
        "il ne remplace pas le critère principal à 16.", '',
        "Adam et ses 2 712 mises à jour initiales sont restaurés exactement. Le dropout redémarre avec "
        "la graine déclarée ; il ne s’agit pas d’un entraînement ininterrompu reproduit bit à bit. "
        "Les deux objectifs utilisent les mêmes exemples et ordres de lots.", '',
        "Ce test porte sur le composant natif de comptage, pas sur toute la chaîne V27.3. "
        "Il ne suffit pas à départager toutes les autres causes : représentation, capacité, "
        "ambiguïté de la cible ou optimisation. La référence officielle reste inchangée.", ''])
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'diagnosis.json',result)
    (args.output/'report.md').write_text('\n'.join(text))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    summarize(p.parse_args())
