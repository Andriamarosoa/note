"""S704: iterative full acoustic heads with one actually emitted note stream.

Three warm-start learning budgets, two mixtures and 48 causal selectors per
    mixture. Only composition-disjoint CAL selects. HELD/DEV are evaluated once
after the checkpoint/policy freeze. No modification of official LIVE files.
"""
import argparse
from dataclasses import asdict
import gc
import itertools
import json
from pathlib import Path
import tempfile
import time
import zipfile
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR, HOP, Selection, load_heads, feature_track
from causal_note.s704_joint_heads import (
    BATCH, JointPolicy, acoustic_features, count_input, decode_joint,
    probabilities, count_probabilities,
)
from scripts.s613_prepare_strict_live import labels, download, digest
from scripts.s613_train_strict_live import real_notes, decode
from scripts.s672_causal_offset_timing_grid import (
    ROOT, MODEL, raw_guitar_audio, native_posteriors, note_instances,
    native_rhythm_clock, hazard_predictions, MAX_AGE_FRAMES,
)
from scripts.s704_joint_metrics import metric, aggregate, target_rank, SCORES

ITERATIONS = (100, 220, 400)
POLICIES = {
    f'on{on}_act{act}_off{off}_rel{rel}_k{count}_mix{mix}':
        JointPolicy(onset=on, sustain_on=act, sustain_off=off, release=rel,
                    count_confidence=count, blend=mix)
    for on, act, off, rel, count, mix in itertools.product(
        (.30, .45, .60), (.30, .50), (.15, .30), (.65, .85), (1.1, .70), (.70, 1.))
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def targets(ends, truth):
    t = ends/SR
    y = np.zeros((len(t), 49, 3), np.uint8)
    k = np.zeros(len(t), np.int32)
    for n in truth:
        p, start, stop = n['pitch']-40, n['time'], n['time']+n['duration']
        y[(t>=start) & (t<start+.060), p, 0] = 1
        y[(t>=start) & (t<stop), p, 1] = 1
        y[(t>=stop) & (t<stop+.060), p, 2] = 1
        k += (t>=start) & (t<stop)
    return y, np.minimum(k, 6)


def sample_training(x, post, y, seed):
    rng = np.random.default_rng(seed)
    boundary = (y[:,:,0] | y[:,:,2]).astype(bool)
    hard = (post[:,:,0]>.12) | (post[:,:,1]>.25)
    rate = np.where(boundary, 1., np.where(y[:,:,1], .80, np.where(hard,.30,.04)))
    selected = rng.random(rate.shape) < rate
    # Sampling correction preserves the stream distribution, rather than
    # presenting an artificial 50/50 event population as calibrated odds.
    return x[selected].copy(), y[selected].copy(), (1/rate[selected]).astype(np.float32)


def available_event(e, samples):
    e = dict(e)
    sample = min(samples, ((e['decision_sample']+BATCH*HOP-1)//(BATCH*HOP))*(BATCH*HOP))
    e['available_sample'] = sample
    e['time'] = sample/SR
    return e


def causal_s675_events(ends, features, post, k, native, samples, model):
    """Vectorized evaluation of causal timers, never retrospective fallback.

    A native release schedules a fallback 24 hops LATER. Hazard, a known
    reattack, max age or EOF may fire first. The future existence/absence of
    a hazard never chooses an earlier original native release timestamp.
    """
    notes = note_instances(native, ends)
    pack = dict(ends=ends, feats=features, post=post, k=k, notes=notes,
                clock=native_rhythm_clock(native, ends))
    frames, hazards = hazard_predictions(pack, model)
    ons = [e for e in native if e['type']=='note_on']
    events = []
    for i, n in enumerate(notes):
        events.append(available_event(ons[i], samples))
        options = [(len(ends), 'stream_end'), (n['start']+MAX_AGE_FRAMES, 's675_max_age')]
        if n['next_start'] < len(ends):
            options.append((n['next_start'], 's675_restrike'))
        if n['native_end'] is not None:
            options.append((n['native_end']+24, 's675_causal_fallback_grace'))
        hit = np.flatnonzero(hazards[i] >= .60)
        if len(hit):
            options.append((int(frames[i][hit[0]]), 's675_causal_hazard'))
        frame, reason = min(options, key=lambda z:z[0])
        at = int(ends[frame]) if frame<len(ends) else samples
        off = dict(type='note_off', pitch=n['pitch'], decision_sample=at,
                   available_sample=at, time=at/SR, reason=reason, velocity=0)
        events.append(available_event(off, samples))
    return sorted(events, key=lambda e:(e['available_sample'], e['decision_sample'],
                                       0 if e['type']=='note_off' else 1, e['pitch']))


def record(spec, archive, heads, selection, offset_model=None):
    # This function takes no annotations. Evaluation labels are opened only
    # after the corresponding candidate events have been generated/saved.
    audio, sha = raw_guitar_audio(spec, archive)
    ends, feat = feature_track(audio, block_size=4096)
    post, k = native_posteriors(feat, heads)
    native = decode(ends, post, k, selection, len(audio))
    base = {'native': [available_event(e,len(audio)) for e in native]}
    if offset_model is not None:
        base['native_s675_causal'] = causal_s675_events(
            ends, feat, post, k, native, len(audio), offset_model)
    chosen = np.arange(BATCH-1, len(ends), BATCH)
    x = acoustic_features(feat,post,k)[chosen].copy()
    cx = count_input(feat[chosen],post[chosen],k[chosen])
    return dict(spec=spec, sha256=sha, samples=len(audio), ends=ends,
                decision_ends=ends[chosen], x=x, cx=cx,
                native_post=post[chosen], native_k=k[chosen], baseline=base)


def short(r):
    return {k:r[k] for k in SCORES}


def report_split(rows, names):
    result = {n:aggregate([r['metrics'][n] for r in rows]) for n in names}
    result['by_texture'] = {texture:{n:aggregate([r['metrics'][n] for r in rows
                          if r['name'].endswith('_'+texture)]) for n in names}
                          for texture in ('comp','solo')}
    return result


def run(args):
    from scripts.s699_player_held_full_train_K import expanded_train_specs
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite an experiment')
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    frozen = json.loads((ROOT/'selection_frozen.json').read_text())
    assert digest(MODEL.read_bytes()) == frozen['model_sha256']
    assert digest((ROOT/'protocol.json').read_bytes()) == frozen['protocol_sha256']
    heads = load_heads(MODEL, expected_sha256=frozen['model_sha256'])
    selection = Selection(**heads.metadata['selection'])
    from causal_note.s672_live_offset import S672TrialNoteOff
    offset = S672TrialNoteOff.load(args.offset_checkpoint, heads).model
    originals = [s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='train']
    dev = [s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='dev']
    compositions = sorted({s['composition'] for s in originals})
    fit_comps, cal_comps, held_comps = set(compositions[:4]), {compositions[4]}, set(compositions[5:])
    assert len(compositions)==9
    report = dict(schema='S704_ONE_CAUSAL_EVENT_STREAM_LOOPS', promoted=False,
        official_LIVE_changed=False, TEST_opened=False, DEV_previously_exposed=True,
        held_previously_exposed=True, base_sha256=frozen['model_sha256'],
        fit_compositions=sorted(fit_comps), cal_compositions=sorted(cal_comps),
        held_compositions=sorted(held_comps), learning_budgets=ITERATIONS,
        policies={n:asdict(p) for n,p in POLICIES.items()}, thresholds={s:.80 for s in SCORES},
        criterion='EVERY score strictly >80%; same emitted stream, full/active K, pitch F1',
        note='New heads fit composition-disjoint FIT; native upstream heads and S675 have historical TRAIN exposure. '
             'HELD/DEV are research diagnostics, not a fresh external test. No live CPU certification.',
        rounds=[])

    def checkpoint():
        report['elapsed_s'] = round(time.perf_counter()-started,2)
        save(args.out/'report.json', report)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root/'data').mkdir(); (root/'cal').mkdir()
        for key, name in (('GuitarSet_audio','GuitarSet_audio.zip'),('GuitarSet_annotations','GuitarSet_annotations.zip')):
            source = protocol['guitar_archives'][key]
            download(source['url'],root/'data'/name,source['size'],source['sha256'])
        with zipfile.ZipFile(root/'data/GuitarSet_audio.zip') as archive, zipfile.ZipFile(root/'data/GuitarSet_annotations.zip') as ann:
            expanded = expanded_train_specs(originals, archive, ann)
            fit = [s for s in expanded if s['composition'] in fit_comps]
            cal = [s for s in expanded if s['composition'] in cal_comps]
            held = [s for s in expanded if s['composition'] in held_comps]
            assert (len(fit),len(cal),len(held)) == (64,16,64)
            xs, ys, ws, cxs, cys, proof = [],[],[],[],[],[]
            for i, spec in enumerate(fit):
                item = record(spec,archive,heads,selection)
                truth = real_notes(labels(spec,root)['notes'],item['samples']/SR)
                y, ky = targets(item['decision_ends'],truth)
                x,y,w = sample_training(item['x'],item['native_post'],y,704+i)
                xs.append(x); ys.append(y); ws.append(w); cxs.append(item['cx']); cys.append(ky)
                proof.append(dict(name=spec['name'],composition=spec['composition'],sha256=item['sha256'],rows=len(y)))
                print('S704 PREPARE FIT',i+1,len(fit),spec['name'],len(y),flush=True)
            x,y,w = np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
            cx,cy = np.concatenate(cxs),np.concatenate(cys)
            del xs,ys,ws,cxs,cys,item; gc.collect()
            report['training'] = dict(rows=len(x),dimensions=x.shape[1],positive=y.sum(0).tolist(),
                count_rows=len(cx),count_dimensions=cx.shape[1],sources=proof)
            checkpoint()
            count = HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=31,learning_rate=.065,
                min_samples_leaf=80,l2_regularization=20,early_stopping=False,random_state=704)
            count.fit(cx,cy)
            del cx,cy; gc.collect()
            print('S704 COUNT TRAINED',flush=True)
            cal_meta=[]
            for i,spec in enumerate(cal):
                item=record(spec,archive,heads,selection,offset)
                item['truth']=real_notes(labels(spec,root)['notes'],item['samples']/SR)
                item['count']=count_probabilities(count,item.pop('cx'))
                path=root/'cal'/f'{i}.joblib'; joblib.dump(item,path,compress=0)
                cal_meta.append(path)
                print('S704 PREPARE CAL',i+1,len(cal),spec['name'],flush=True)
            del item; gc.collect()
            models=[HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,learning_rate=.065,
                min_samples_leaf=80,l2_regularization=12,early_stopping=False,warm_start=True,
                random_state=704+j) for j in range(3)]
            best_key=None
            for budget in ITERATIONS:
                for j,model in enumerate(models):
                    model.set_params(max_iter=budget)
                    model.fit(x,y[:,j],sample_weight=w*np.where(y[:,j],(6.,1.6,5.)[j],1.))
                    print('S704 TRAINED',budget,'head',j,flush=True)
                rows={n:[] for n in POLICIES}
                baserows=[]
                for i,path in enumerate(cal_meta):
                    item=joblib.load(path)
                    p=probabilities(models,item['x'])
                    base={n:metric(e,item['truth'],item['ends']) for n,e in item['baseline'].items()}
                    baserows.append(dict(name=item['spec']['name'],metrics=base))
                    for n,policy in POLICIES.items():
                        blend=policy.blend*p+(1-policy.blend)*item['native_post']
                        events=decode_joint(item['decision_ends'],blend,item['count'],policy,item['samples'])
                        rows[n].append(metric(events,item['truth'],item['ends']))
                    print('S704 CAL',budget,i+1,len(cal_meta),flush=True)
                scores={n:aggregate(v) for n,v in rows.items()}
                best=max(scores,key=lambda n:target_rank(scores[n]))
                base=report_split(baserows,('native','native_s675_causal'))
                round_report=dict(iterations=budget,best=best,scores=scores,baseline=base,
                    all_targets_above_80=scores[best]['all_targets_above_80'])
                report['rounds'].append(round_report)
                key=target_rank(scores[best])
                if best_key is None or key>best_key:
                    best_key=key
                    report['frozen_candidate']=dict(iterations=budget,policy=best,CAL=scores[best])
                    joblib.dump(dict(models=models,count=count,policy=asdict(POLICIES[best]),
                        dimensions=x.shape[1],batch_hops=BATCH),args.out/'s704_joint_heads.joblib',compress=3)
                checkpoint()
                print('S704 ROUND',budget,best,json.dumps(short(scores[best])),flush=True)
                if scores[best]['all_targets_above_80']:
                    break
            del x,y,w,rows,item,models; gc.collect()
            bundle=joblib.load(args.out/'s704_joint_heads.joblib')
            policy=JointPolicy(**bundle['policy'])
            report['checkpoint_sha256']=digest((args.out/'s704_joint_heads.joblib').read_bytes())
            save(args.out/'selection_frozen.json',report['frozen_candidate'])
            print('S704 FROZEN',json.dumps(report['frozen_candidate']),flush=True)
            checkpoint()
            for phase,specs in (('HELD',held),('DEV',dev)):
                rows=[]
                with (args.out/(phase.lower()+'_events.jsonl')).open('w') as f:
                    for i,spec in enumerate(specs):
                        item=record(spec,archive,heads,selection,offset)
                        p=probabilities(bundle['models'],item['x'])
                        k=count_probabilities(bundle['count'],item['cx'])
                        blended=policy.blend*p+(1-policy.blend)*item['native_post']
                        events=decode_joint(item['decision_ends'],blended,k,policy,item['samples'])
                        event_dict=dict(item['baseline'],candidate=events)
                        # Save predictions before this evaluation source's labels.
                        (args.out/'latest_prediction_before_labels.json').write_text(json.dumps(dict(name=spec['name'],events=event_dict)))
                        truth=real_notes(labels(spec,root)['notes'],item['samples']/SR)
                        scores={n:metric(e,truth,item['ends']) for n,e in event_dict.items()}
                        base,trial=scores['native_s675_causal'],scores['candidate']
                        paired=dict(recovered=len(set(trial['matched_truth_ids'])-set(base['matched_truth_ids'])),
                            lost=len(set(base['matched_truth_ids'])-set(trial['matched_truth_ids'])),
                            lost_long=len(set(base['long_onset_truth_ids'])-set(trial['long_onset_truth_ids'])),
                            lost_correct_offsets=len(set(base['correct_offset_truth_ids'])-set(trial['correct_offset_truth_ids'])))
                        rows.append(dict(name=spec['name'],composition=spec['composition'],metrics=scores,paired=paired))
                        f.write(json.dumps(dict(name=spec['name'],sha256=item['sha256'],samples=item['samples'],
                            truth=truth,events=event_dict))+'\n'); f.flush()
                        print('S704',phase,i+1,len(specs),spec['name'],json.dumps(short(scores['candidate'])),flush=True)
                report[phase]=report_split(rows,('native','native_s675_causal','candidate'))
                report[phase]['paired']={k:sum(r['paired'][k] for r in rows) for k in rows[0]['paired']}
                base,trial=report[phase]['native_s675_causal'],report[phase]['candidate']
                report[phase]['non_regression']=bool(all(trial[s]>=base[s] for s in SCORES)
                    and trial['tp']>=base['tp'] and report[phase]['paired']['lost_long']==0
                    and report[phase]['paired']['lost_correct_offsets']==0)
                save(args.out/(phase.lower()+'_audit.json'),rows)
                checkpoint()
            report['all_targets_above_80']=all(report[p]['candidate']['all_targets_above_80'] for p in ('HELD','DEV'))
            report['non_regression_passed']=all(report[p]['non_regression'] for p in ('HELD','DEV'))
            assert digest(MODEL.read_bytes())==frozen['model_sha256']
            checkpoint()
            print('S704 FINAL',json.dumps({k:report[k] for k in ('frozen_candidate','HELD','DEV','all_targets_above_80','promoted')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--offset-checkpoint',type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
