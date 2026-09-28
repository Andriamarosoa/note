"""Recompute the paired normalization verdict from all raw predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_v273_learning_bottleneck import metrics
from scripts.rebuild_v273_sources import digest,write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.summarize_v273_training_budget import check_metrics,paired_change
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import array_hash,epoch_order,require
from scripts.v273_spectral_normalization import ARMS,SEED,DROPOUT_SEED,EPOCHS,CHECKPOINTS,interpretation


def verify(root,arm,cfg,config_path,gate):
    validate_original_inventory(root)
    r=json.loads((root/'report.json').read_text())
    require(r['status']=='completed' and r['arm']==arm,'incomplete or wrong arm')
    require(r['epochs']==EPOCHS and r['primary_epoch']==12 and
        r['seed']==SEED and r['dropout_seed']==DROPOUT_SEED and r['weighting']=='uniform' and
        r['batch_size']==128 and r['frames']==31 and r['learning_rate']==0.0002,'protocol changed')
    require(r['outer_rows_evaluated']==0 and not r['automatic_promotion'] and
            not r['output_corrector'] and not r['historical_resume'],'wrong scope')
    require(r['config_sha256']==digest(config_path),'wrong configuration')
    require(r['source_sha']==gate['source_sha'] and gate['status']=='passed','preflight differs')
    require(r['launch_sha256']==gate['launch_sha256'],'launch protocol differs')
    for key in ('common_initial_sha256','full_initial_sha256','parameters','normalizer_class'):
        require(r[key]==gate['arms'][arm][key],'preflight identity differs: '+key)
    require(r['class_weights']==[1.]*7,'class weights changed')
    require([h['epoch'] for h in r['history']]==list(range(1,13)),'missing epochs')
    require(set(r['checkpoints'])==set(map(str,CHECKPOINTS)),'missing endpoints')
    require(len(r['epoch_orders'])==12,'missing batch order audit')
    prediction={}
    identity=None
    for epoch in CHECKPOINTS:
        cp=r['checkpoints'][str(epoch)]
        require(cp['optimizer_iterations']==epoch*339,'wrong update count')
        require(cp['weights_sha256']==digest(root/f'epoch-{epoch:02d}.weights.h5'),'checkpoint hash differs')
        prediction[epoch]={}
        for split,rows,poly,folds in (('fit',43357,5274,{1,2,4}),('validation',15952,2111,{0})):
            with np.load(root/f'epoch-{epoch:02d}-{split}.npz',allow_pickle=False) as z:
                p={key:np.asarray(z[key]) for key in z.files}
            require(len(p['k'])==rows and int((p['k']>=2).sum())==poly,'population differs')
            require(len(np.unique(p['global_index']))==rows,'duplicate rows')
            require(array_hash(p['global_index'])==r['fit_indices_sha256' if split=='fit' else 'val_indices_sha256'],
                    'partition hash differs')
            require({cfg['member_folds'][str(m)] for m in p['member']}==folds,'outer or wrong inner data')
            np.testing.assert_array_equal(p['predicted'],p['probability'].argmax(1))
            check_metrics(metrics(p['k'],p['probability'],np.ones(7)),cp['splits'][split])
            if identity is not None:
                for key in ('global_index','k','member'):
                    np.testing.assert_array_equal(p[key],identity[split][key])
            prediction[epoch][split]=p
        identity=prediction[epoch]
        require(not np.intersect1d(identity['fit']['global_index'],identity['validation']['global_index']).size,
                'fit/validation overlap')
        log=r['history'][epoch-1]
        m=cp['splits']['validation']
        np.testing.assert_allclose(log['val_loss'],m['nll'],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(log['val_poly_exact'],m['poly_exact'],atol=1e-6)
        for k in range(7):
            np.testing.assert_allclose(log[f'val_k{k}_exact'],m['by_true_k'][str(k)]['exact'],atol=1e-6)
    for epoch,order in enumerate(r['epoch_orders']):
        require(order==dict(epoch=epoch+1,sha256=array_hash(epoch_order(identity['fit']['global_index'],SEED,epoch))),
                'batch order changed')
    return r,prediction


def audit(args):
    require(not args.output.exists(),'audit output exists')
    cfg=load_config(args.config)
    validate_original_inventory(args.preflight)
    gate=json.loads((args.preflight/'report.json').read_text())
    result=dict(status='verified',experiment='native_spectral_normalization',primary_epoch=12,
        outer_rows_evaluated=0,automatic_promotion=False,arms={},comparisons={})
    predictions={}
    for arm in ARMS:
        result['arms'][arm],predictions[arm]=verify(args.root/f'norm-{arm}',arm,cfg,args.config,gate)
        require(result['arms'][arm]['preflight_sha256']==digest(args.preflight/'report.json'),'wrong preflight bytes')
    a,b=[result['arms'][arm] for arm in ARMS]
    for key in ('seed','dropout_seed','common_initial_sha256','epoch_orders','class_weights',
                'fit_indices_sha256','val_indices_sha256','bundle_sha256','config_sha256','source_sha'):
        require(a[key]==b[key],'paired arms differ: '+key)
    require(a['parameters']-b['parameters']==6,'unexpected model difference')
    for epoch in CHECKPOINTS:
        comp={split:paired_change(predictions['channel_norm'][epoch][split],
                                 predictions['fixed_scale'][epoch][split]) for split in ('fit','validation')}
        low=comp['validation']['by_k']
        verdict=interpretation(comp['validation']['poly']['delta'],{k:v['delta'] for k,v in low.items()})
        result['comparisons'][str(epoch)]=dict(paired=comp,interpretation=verdict)
    result['primary_verdict']=result['comparisons']['12']['interpretation']
    result['limitations']=['One seed and one repeatedly inspected internal development split.',
        'The intervention changes normalization and removes six normalization affine parameters.',
        'Native component only; no full V27.3 or outer-fold score.',
        'Mathematical preservation of inputs does not by itself prove a count performance gain.',
        'No significance claim and no automatic checkpoint selection or promotion.']
    lines=['# Normalisation spectrale : comparaison native vérifiée','',
        'Comparaison principale : **mise à l’échelle fixe contre normalisation actuelle, à 12 époques**.',
        'Deux entraînements frais sans pondération. Aucun correcteur et aucune donnée du fold externe 3 évaluée.','',
        '| Entrée | Époques | Exact K poly apprentissage | Exact K poly validation |',
        '|---|---:|---:|---:|']
    for arm in ARMS:
        for epoch in CHECKPOINTS:
            cp=result['arms'][arm]['checkpoints'][str(epoch)]['splits']
            cells=[f"{cp[s]['poly_correct']}/{cp[s]['poly_rows']} ({100*cp[s]['poly_exact']:.2f} %)" for s in ('fit','validation')]
            lines.append(f"| {arm} | {epoch} | {' | '.join(cells)} |")
    messages={
        'no_polyphonic_validation_gain':'Aucun gain polyphonique de validation à l’échéance fixée. Cette correction n’est pas validée dans ce test.',
        'polyphonic_gain_with_low_k_regressions':'Le score polyphonique progresse, avec des régressions parmi K=1, 2 ou 3. Résultat mixte : la correction ne satisfait pas le critère de non-régression sur les petits K.',
        'polyphonic_gain_without_low_k_regression_on_this_split':'Le score polyphonique progresse sans recul de K=1, 2 ou 3 sur cette partition. Candidat positif sur ce seul test de développement, à confirmer avant toute promotion.'}
    lines.extend(['','## Verdict', '',messages[result['primary_verdict']], '',
        '| K vrai | Groupes de validation | Exacts avec normalisation | Exacts avec échelle fixe | Différence |',
        '|---:|---:|---:|---:|---:|'])
    comp=result['comparisons']['12']['paired']['validation']
    for k,d in comp['by_k'].items():
        lines.append(f"| {k} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} |")
    p=comp['poly']
    lines.extend(['',f"Polyphonie : {p['corrected']} erreurs corrigées, {p['regressed']} bonnes décisions perdues, solde {p['delta']:+d}.", '',
        '| Composition interne | Groupes poly | Exacts avec normalisation | Exacts avec échelle fixe | Différence |',
        '|---|---:|---:|---:|---:|'])
    for name,d in comp['poly_by_composition'].items():
        lines.append(f"| {name} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} |")
    lines.extend(['','## Contrôles et limites','',
        'Poids initiaux communs identiques, mêmes données, partitions, ordre des lots, objectif et graines. '
        'Les archives et inventaires sont vérifiés ; tous les scores sont recalculés à partir des probabilités. '
        'Les points 4 et 8 sont descriptifs et ne remplacent pas le critère principal à 12.','',
        'Le changement remplace LayerNormalization sur les trois canaux par leur division fixe par 12 ; '
        'les six paramètres gamma/beta de cette normalisation disparaissent. Les autres couches restent identiques. '
        'Ce test ne sépare pas tous les effets de conditionnement numérique de la préservation des amplitudes.','',
        'Une seule graine et une validation interne déjà inspectée : résultat exploratoire, sans affirmation '
        'de significativité ni de généralisation externe. Aucun modèle n’est promu. V27.3 reste la référence officielle.',''])
    args.output.mkdir(parents=True)
    write_json(args.output/'diagnosis.json',result)
    (args.output/'report.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','preflight','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
