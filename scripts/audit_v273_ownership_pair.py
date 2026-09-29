"""Recompute the paired ownership verdict, including low-K and competition strata."""
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
from scripts.v273_ownership_experiment import (ARMS,SEED,DROPOUT_SEED,EPOCHS,CHECKPOINTS,
    interpretation,geometry_for_track)


def verify(root,arm,cfg,config_path,gate,geometry_rows,geometry_report):
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
    require(r['geometry_sha256']==geometry_report['geometry_sha256'] and
            r['geometry_column']==ARMS.index(arm) and r['same_sample_support_for_both_arms'],
            'geometry treatment differs')
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
            require(p['probability'].shape==(rows,7) and np.isfinite(p['probability']).all() and
                    (p['probability']>=0).all() and (p['probability']<=1).all(),'invalid probabilities')
            np.testing.assert_allclose(p['probability'].sum(1),1.,atol=2e-6)
            np.testing.assert_array_equal(p['predicted'],p['probability'].argmax(1))
            ids=p['global_index']
            np.testing.assert_array_equal(p['member'],geometry_rows['member'][ids])
            np.testing.assert_array_equal(p['lost_eligible_samples'],geometry_rows['lost_eligible_samples'][ids])
            require((p['lost_eligible_samples']>=0).all(),'unprepared geometry row')
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


def verify_geometry(root,cfg,config,gate):
    validate_original_inventory(root)
    r=json.loads((root/'report.json').read_text())
    require(r['status']=='verified' and r['config_sha256']==digest(config) and
            r['source_sha']==gate['source_sha'] and r['launch_sha256']==gate['launch_sha256'],
            'wrong geometry provenance')
    require(r['label_free_builder'] and r['full_untruncated_candidates'] and
            r['all_sample_prefix_checks_passed'] and r['outer_tracks_processed']==0 and
            r['outer_rows_evaluated']==0,'wrong geometry scope')
    g=np.load(root/'geometry.npy',mmap_mode='r',allow_pickle=False)
    require(array_hash(g)==r['geometry_sha256'],'geometry array changed')
    with np.load(root/'rows.npz',allow_pickle=False) as z: rows={key:z[key] for key in z.files}
    require(g.shape==(len(rows['member']),31,2),'wrong geometry shape')
    covered=np.zeros(len(g),bool);seen=set()
    for track in r['tracks']:
        member=track['member'];path=root/track['timing_file']
        require(member not in seen and cfg['member_folds'][member]!=3 and
                digest(path)==track['timing_sha256'],'wrong timing source')
        with np.load(path,allow_pickle=False) as z:
            require(z['member'].tolist()==[member],'wrong track identity')
            ids=z['global_index'];o=z['full_candidate_offsets'];v=z['full_candidate_samples']
            require(o[0]==0 and o[-1]==len(v) and np.all(np.diff(o)>0),'invalid full timing')
            groups=[v[a:b] for a,b in zip(o[:-1],o[1:])]
        require(not covered[ids].any() and len(np.unique(ids))==len(ids),'duplicate track rows')
        np.testing.assert_array_equal(rows['member'][ids],np.repeat(member,len(ids)))
        np.testing.assert_array_equal(rows['start'][ids],[x[0] for x in groups])
        reconstructed=geometry_for_track(groups)
        for actual,expected in zip(reconstructed,(g[ids],rows['lost_eligible_samples'][ids],
                rows['proposal_complete_through'][ids],rows['decision_sample_support'][ids])):
            np.testing.assert_array_equal(actual,expected)
        covered[ids]=True;seen.add(member)
    require(seen=={m for m,f in cfg['member_folds'].items() if f!=3},'wrong geometry track inventory')
    inner=np.asarray([cfg['member_folds'][str(m)]!=3 for m in rows['member']])
    np.testing.assert_array_equal(covered,inner)
    require(int(covered.sum())==59309 and np.isnan(g[~covered]).all() and
            (rows['lost_eligible_samples'][~covered]==-1).all(),'outer input or incomplete rows')
    return r,rows


def comparison(before,after):
    out=paired_change(before,after)
    np.testing.assert_array_equal(before['lost_eligible_samples'],after['lost_eligible_samples'])
    k=before['k'];a=before['predicted'];b=after['predicted']
    changed=before['lost_eligible_samples']>0
    out['by_competition']={}
    for name,mask in (('no_competition',~changed),('competition',changed)):
        pa={key:value[mask] for key,value in before.items()}
        pb={key:value[mask] for key,value in after.items()}
        out['by_competition'][name]=paired_change(pa,pb)
    out['error_direction_by_k']={str(value):dict(
        over_before=int(((k==value)&(a>k)).sum()),over_after=int(((k==value)&(b>k)).sum()),
        under_before=int(((k==value)&(a<k)).sum()),under_after=int(((k==value)&(b<k)).sum()))
        for value in range(7)}
    return out


def audit(args):
    require(not args.output.exists(),'audit output exists')
    cfg=load_config(args.config);validate_original_inventory(args.preflight)
    gate=json.loads((args.preflight/'report.json').read_text())
    require(gate['status']=='passed' and gate['identical_architecture_and_weights'] and
            gate['same_input_same_prediction'],'model gate failed')
    geometry,rows=verify_geometry(args.geometry,cfg,args.config,gate)
    result=dict(status='verified',experiment='native_ownership_context',primary_epoch=12,
        outer_rows_evaluated=0,automatic_promotion=False,arms={},comparisons={},
        geometry_reconstructed_from_all_full_proposals=True,
        geometry_report_sha256=digest(args.geometry/'report.json'))
    predictions={}
    for arm in ARMS:
        r,p=verify(args.root/f'ownership-{arm}',arm,cfg,args.config,gate,rows,geometry)
        require(r['preflight_sha256']==digest(args.preflight/'report.json'),'wrong preflight bytes')
        require(r['geometry_report_sha256']==result['geometry_report_sha256'],'wrong geometry bytes')
        result['arms'][arm]=r;predictions[arm]=p
    a,b=[result['arms'][arm] for arm in ARMS]
    for key in ('seed','dropout_seed','common_initial_sha256','full_initial_sha256',
                'parameters','normalizer_class','epoch_orders','class_weights',
                'fit_indices_sha256','val_indices_sha256','bundle_sha256','config_sha256',
                'source_sha','geometry_sha256','geometry_report_sha256'):
        require(a[key]==b[key],'paired arms differ: '+key)
    for epoch in CHECKPOINTS:
        comp={split:comparison(predictions[ARMS[0]][epoch][split],predictions[ARMS[1]][epoch][split])
              for split in ('fit','validation')}
        verdict=interpretation(comp['validation']['poly']['delta'],
            {k:v['delta'] for k,v in comp['validation']['by_k'].items()})
        result['comparisons'][str(epoch)]=dict(paired=comp,interpretation=verdict)
    result['primary_verdict']=result['comparisons']['12']['interpretation']
    result['limitations']=[
        'One seed and one repeatedly inspected internal development split; no significance claim.',
        'Both arms have one extra geometry channel: local_only is not the historical four-input model.',
        'Frame fractions compress exact sample masks and may alias distinct boundary arrangements.',
        'Native count component only; no full V27.3 or outer-fold score, no automatic promotion.',
        'Proposal support bounds are verified; end-to-end streaming latency has not been measured.',
        'Competition strata describe where inputs differ, not a causal attribution of each error.']
    lines=['# Contexte des groupes voisins : comparaison native vérifiée','',
        'Comparaison principale à **12 époques** : zone locale contre zone attribuée après concurrence des voisins.',
        'Deux modèles identiques, mêmes poids initiaux, données, lots, objectif et graines. '
        'Aucun correcteur et aucune évaluation externe.','',
        '| Entrée | Époques | Exact K poly apprentissage | Exact K poly validation |',
        '|---|---:|---:|---:|']
    for arm in ARMS:
        for epoch in CHECKPOINTS:
            cp=result['arms'][arm]['checkpoints'][str(epoch)]['splits']
            cells=[f"{cp[s]['poly_correct']}/{cp[s]['poly_rows']} ({100*cp[s]['poly_exact']:.2f} %)"
                   for s in ('fit','validation')]
            lines.append(f"| {arm} | {epoch} | {' | '.join(cells)} |")
    messages={
        'no_polyphonic_validation_gain':'Aucun gain polyphonique à l’échéance fixée. Cette solution n’est pas validée par ce test.',
        'polyphonic_gain_with_low_k_regressions':'Gain polyphonique, mais régression sur au moins un des petits K=1, 2, 3. Résultat mixte ; critère non satisfait.',
        'polyphonic_gain_without_low_k_regression_on_this_split':'Gain polyphonique sans recul de K=1, 2, 3 sur cette partition interne. Candidat positif dans ce test exploratoire ; aucune promotion.'}
    comp=result['comparisons']['12']['paired']['validation']
    lines+=['','## Verdict','',messages[result['primary_verdict']],'',
        '| K vrai | Groupes | Exacts local | Exacts voisins | Différence | Surcomptés local → voisins |',
        '|---:|---:|---:|---:|---:|---:|']
    for k,d in comp['by_k'].items():
        e=comp['error_direction_by_k'][k]
        lines.append(f"| {k} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} | {e['over_before']} → {e['over_after']} |")
    p=comp['all'];poly=comp['poly']
    lines+=['',f"Tous K : {p['before_correct']}/{p['rows']} → {p['after_correct']}/{p['rows']}, solde {p['delta']:+d}.",
        f"Polyphonie : {poly['corrected']} erreurs corrigées, {poly['regressed']} bonnes décisions perdues, solde {poly['delta']:+d}.",'',
        '| Concurrence géométrique | Groupes poly | Exacts local | Exacts voisins | Différence |',
        '|---|---:|---:|---:|---:|']
    for name,stratum in comp['by_competition'].items():
        d=stratum['poly']
        lines.append(f"| {name} | {d['rows']} | {d['before_correct']} | {d['after_correct']} | {d['delta']:+d} |")
    lines+=['','## Contrôles et limites','',
        'Toutes les probabilités sauvegardées sont rescorrées ; les masques temporels sont reconstruits '
        'depuis les propositions complètes des 190 pistes internes. Les points 4 et 8 sont descriptifs.','',
        'Les deux modèles possèdent un canal géométrique et 288 paramètres de plus que le modèle historique. '
        'Ce témoin local est donc nouveau. Seule la concurrence des voisins diffère entre les deux branches. '
        'Les fractions par trame compriment les frontières exactes ; elles ne garantissent pas une représentation sans perte.','',
        'Une seule graine, partition interne déjà inspectée, aucune mesure de latence de bout en bout. '
        'Les strates de concurrence ne donnent pas une attribution causale erreur par erreur. '
        'V27.3 reste la référence officielle à 42,6019 %, métrique non directement comparable à ce test natif.','']
    args.output.mkdir(parents=True)
    write_json(args.output/'diagnosis.json',result)
    (args.output/'report.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','preflight','geometry','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
